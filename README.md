# Weight-only 4-bit 分组量化器（线性层）

纯 NumPy 实现的线性层 weight-only 4-bit 分组对称量化：真半字节打包、权重 max-abs
定 scale 与校准驱动 clipping 搜索两条路线、两种推理路径（反量化参照 + 直接按组解包
累加）、带版本与 CRC 校验的保存格式，以及可复算的对照实验。核心量化与 packing
代码不调用任何现成量化器。

## 1. 量化定义

权重形状 `W: (out_features, in_features)`。**沿每个输出通道的输入维度**做长度为
`group_size` 的连续分组；行内不足一组时补零列（padding 数量记录在元数据中）。

- 对称整数量化级：**q ∈ [-7, 7]**（4-bit 有符号，不用 -8）。
- 每组一个正 scale：max-abs 路径下 `scale = max|w_g| / 7`（零组 scale 严格为 0，
  码全 0，且在 float32 scale 往返后仍精确重建为全零）。
- 量化：`q = clip(round(clip(w, -t, t) / scale), -7, 7)`；反量化：`w_hat = q * scale`。
- scale 以 float32 存储（格式同样支持 float16/float64），在**量化之前**就完成
  dtype 转换，因此校准搜索优化的舍入与推理实际执行的舍入完全一致，坐标下降因此
  对目标函数保持单调。

### 1.1 半字节编码与字节内顺序（真 4-bit 存储）

有符号整数用**符号-幅值（sign-magnitude）**编码进 4-bit：

| 编码位 | 含义 |
|---|---|
| bit3 | 符号位（1 为负） |
| bit2..bit0 | 幅值 |-q|，0..7 |

例如：`1 -> 0x1`、`-2 -> 0xA`、`7 -> 0x7`、`-7 -> 0xF`、`0 -> 0x0`、`-1 -> 0x9`。

**字节内顺序为低 nibble 在前（little-nibble）**：第 `2i` 个元素放字节的 bit0..3，
第 `2i+1` 个元素放 bit4..7。例如数组 `[1,-2,7,-7,0,-1]` 打包为
`[0xA1, 0xF7, 0x90]`。

**奇数尾部规则与行对齐**：元素数为奇数时，最后一个字节的高 nibble 补 `0x0`（解码为 0）。每个输出通道的 nibble 行按**字节边界对齐**：行宽 k_padded 为奇数时该尾 nibble 仅作对齐占位，使下一行仍从某字节的低 nibble 开始；直接推理的任意范围 reader 始终在该行自己的字节窗口内寻址。
行级布局为每个输出通道分配 `ceil(k_padded/2)` 个连续 uint8。直接推理路径用
`unpack_nibble_range(start, stop)` 支持任意（含奇数）起止 nibble，因此一组从某
字节高 nibble 开始、在另一字节低 nibble 结束时不会多读或错序（组大小为奇数时会
实际触发该情形）。存储中**不存在** “int8 张量冒充 4-bit”：打包后真实占用
`ceil(n/2)` 字节。

## 2. Scale / clipping：max-abs 与校准驱动

候选阈值集合在实验前固定于代码常量 `CANDIDATE_CLIP_FRACTIONS`：

```
(1.0, 0.9375, 0.875, 0.8125, 0.75, 0.6875, 0.625, 0.5625, 0.5)
```

某组阈值 `t = f * max|w_g|`，三个方法：

- `maxabs`：不做 clipping 搜索，`f=1`，`t = 7*scale`。
- `weight_clip`：每组独立选择使**该组权重重建 SSE** 最小的固定候选。
- `calibrated`：**分组坐标搜索（grouped coordinate descent）**，目标是有限校准
  输入上的**线性层输出误差**
  `MSE = mean_{n,o} ((Q(W) x_cal - W x_cal)_no)^2`（bias 在差值中抵消，不参与
  目标）。坐标为每个 (行, 组) 的阈值候选；逐组枚举 9 个候选，只接受严格降低总
  MSE 的移动，扫描全部坐标直到一轮无改动或达到上限。因此目标单调不增，确定性地
  收敛到该固定候选网格上的逐坐标最优点。阈值只从 `x_cal` 拟合，评估集仅用于
  汇报。

### 2.1 为什么“逐组最小化权重误差”一般不等于最小化输出误差

输出误差 `E = X (Q(W)-W)^T`。对输出通道 `o`，令 `Δ_g = Q(w_{og}) - w_{og}`：

```
MSE_o = Σ_g E[X_g Δ_g]^2
      = Σ_g  Δ_g^T Σ_gg  Δ_g              (组内，由输入协方差加权)
      + Σ_{g≠h} Δ_g^T Σ_gh  Δ_h           (跨组耦合项)
```

- 逐组权重 SSE 等价于把 `Σ_gg` 当成单位阵、并完全丢弃跨组耦合项
  `E[X_g^T X_h]`。
- 当输入特征相关（`Σ_gh ≠ 0`）时，各组量化误差会在输出端相干叠加/抵消；输出最优
  的阈值会牺牲部分权重重建（给某坐标更大 clipping 误差），去换整体输出误差下降。
- 当不同输入坐标方差差异大时，即使不相关，权重 SSE 对方差小的坐标“重视过度”，
  两个目标的逐组最优也会不同（见测试
  `test_weight_clipping_not_optimal_for_output_with_input_variance`）。

## 3. 推理路径

1. **反量化参照** `forward_dequant`：完整解包并重建 `W_hat`，再做 `X @ W_hat^T + b`。
2. **直接按组解包累加** `forward_packed`：每次调用显式解包半字节，然后逐组累加
   `out += X_group @ (scale_g * codes_group)^T`，不构造完整浮点权重矩阵做单次 GEMM。
   另保留 `forward_packed_scalar`：严格按每个 (输出行, 组) 调用任意范围 nibble
   reader 累加，作为解包寻址的交叉验证实现。

三条路径均在 float64 下累加、float32 加 bias。实验与测试中三者最大差在 float64
舍入量级（实测约 `6.4e-14`，见 `results.json`）。

## 4. 保存格式（版本化 + CRC）

`save_linear` / `load_linear` 使用自定义小端二进制格式：

- 47 字节定长 header：magic `INT4QNT`、**version=1**、header_size、flags（bias
  存在位）、scale dtype id（1=fp16/2=fp32/3=fp64）、**group_size**、in_features、
  out_features、**k_padded**、**input_pad（padding 列数）**、num_groups、保留字、
  **CRC-32**（对“crc 置零的 header + 全部 payload”计算）。
- payload：scales（原始字节）→ packed uint8 nibbles →（可选）float32 bias。
- `load_linear` 校验 magic、版本、header 长度、dtype id、维度自洽性
  （`k_padded = in + pad`、`k_padded % group_size == 0`、组数一致、padding 范围）、
  flags/保留字段、payload 长度与 CRC；任一不符抛 `QuantFormatError`。测试覆盖：
  截断、magic 损坏、版本非法、元数据损坏、payload 损坏、空文件、长度不符。

## 5. 真实存储记账

`storage_bytes` 统计 **header + scales + packed nibbles + bias**，并单独列出两类
padding：组边界补零列（`input_padding_nibbles`）与奇数宽行的尾 nibble
（`tail_pad_nibbles`）。实验层（O=64, K=128, G=16, fp32 scale, fp32 bias）：

```
header=47B  scales=2048B  packed=4096B  bias=256B  TOTAL=6447B
fp32 稠密基线 (W+b)=33024B   总比例=0.195   折合 6.296 bit/权重
```

裸 nibble 确实是 4 bit/权重，但小 group（每组一个 fp32 scale）使 scale 开销等于
权重本体（各 4096B/2048B 量级），这正是必须计入 scale/header/padding 的原因；
裸报“4× 压缩”会严重失真。

## 6. 实验（可复算）

数据（`experiments/run_experiment.py`，固定种子 `20260930`，运行前即固定候选集）：

- 层：O=64，K=128，group_size=16；权重 ~N(0, 0.3²)，另有 3% 位置注入
  ±(2.5..4) 的 **outlier**（相对基底约 10×）。
- 输入：8 个隐变量经随机 loading 矩阵混合 + 小噪声，特征**强相关**
  （平均 |相关系数| 约 0.29），所以跨组耦合项真实存在。
- 校准 1024 条；两个评估各 1024 条：`iid` 同分布；`shift` 放大部分隐变量方差并
  引入校准期未出现的新隐变量方向（协变量漂移）。
- `eval_leak`：在 shifted 评估集上做坐标搜索，仅作为“用评估集挑阈值”的泄漏对照，
  **不是合法方法**。

复算：

```bash
python -m pytest tests/ -q          # 17 个测试
python experiments/run_experiment.py   # 结果同时写入 experiments/results.json
```

### 6.1 权重误差 / 输出误差 / 字节数（本次运行实测）

| 方法 | weight MSE | out MSE cal | out MSE iid | out MSE shift | 总存储 |
|---|---:|---:|---:|---:|---:|
| maxabs | 0.00725 | 0.9419 | 0.9167 | 3.0217 | 6447B (6.30 bit/w) |
| weight_clip | 0.00708 | 0.9048 | 0.8864 | 2.8628 | 6447B |
| calibrated | 0.00991 | **0.2551** | **0.2476** | **2.5883** | 6447B |
| eval_leak（非法对照） | 0.01032 | 0.4928 | 0.4719 | 1.0441 | 6447B |

观察：

- **校准驱动的权重 MSE 反而更高**（0.00991 vs 0.00725），但同分布输出 MSE 降低
  约 **73%**（0.92 -> 0.25），直接验证了“最小化权重误差 ≠ 最小化输出误差”。
- `weight_clip` 只比 max-abs 略好（组内 SSE 目标不含输入协方差），远不如输出
  驱动搜索——相关输入下跨组项不可忽略。
- 本次漂移下校准方法仍最好（2.59 vs 3.02），但收益相对 IID 明显收窄；漂移足够
  大时校准可能劣于 max-abs，这是过拟合资态统计的真实风险，工程上需用代表性校准
  集 / 正则化或混合目标。
- `eval_leak` 把 shift MSE 压到 1.04（说明搜索有效），却让 IID MSE 升到 0.47
  （calibrated 为 0.25）：量化展示了**事后用评估集挑阈值属于数据泄漏**，得到的
  数字不能泛化。

### 6.2 计时（含解包成本，不编造加速比）

1024×128 batch，中位数秒；调用顺序随机交叉以压低系统噪声，每次 packed 调用都
重新执行半字节解包：

| 路径 | 量级（本次机器） |
|---|---:|
| 稠密 fp BLAS `X@W.T+b`（基线） | ~1.7 ms |
| dequant 后矩阵乘 | ~6–53 ms（含整表解包） |
| 直接按组解包累加（向量化） | ~72–102 ms（约比 BLAS 慢 40–60×） |
| 严格逐 (o,g) 标量解包累加 | ~0.6–0.7 s（最慢，仅作正确性交叉验证） |

没有融合 SIMD/反量化内核时，NumPy 级显式解包**更慢是预期且如实报告的**；本项目
不声称任何 wall-clock 加速。实际部署需要 fused nibble-unpack GEMM 内核才能兑现
带宽/延迟收益。所有绝对时间随机器波动，以 `results.json` 中本次运行为准。

## 7. 代码结构

- `int4_quant/packing.py`：sign-magnitude nibble 编解码、低 nibble 在前打包、
  奇数尾补 0x0、任意范围 nibble reader。
- `int4_quant/quantize.py`：分组/padding、零组处理、固定候选集、max-abs、逐组
  权重 SSE clipping、坐标下降输出 MSE 搜索。
- `int4_quant/linear.py`：`QuantizedLinear`；`forward_dequant` /
  `forward_packed` / `forward_packed_scalar`，bias 在 float32 加入。
- `int4_quant/format.py`：版本化 header、CRC、严格元数据校验、存储记账。
- `tests/test_quantizer.py`：17 个 unittest 风格 pytest 测试。
- `experiments/run_experiment.py` + `results.json`：可复算实验与原始数据。
