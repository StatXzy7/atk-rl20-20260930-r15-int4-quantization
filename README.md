# wq4 — weight-only 4-bit group quantization for linear layers

Pure-NumPy implementation of a weight-only 4-bit group quantizer for a
linear layer `y = x Wᵀ + b`.  No existing quantizer is used for the core
quantization or packing paths.  Only `numpy` (runtime) and `pytest`
(tests) are required.

```
wq4/
  packing.py        nibble pack/unpack (the true 4-bit storage codec)
  quantizer.py      grouping, max-abs and weight-MSE clipping scale search
  calibration.py    calibration-driven coordinate search on output error
  linear.py         QuantizedLinear: dequant + fused unpack-accumulate paths
  serialization.py  WQ4 v1 container: header, checksum, validation
tests/              47 tests (pytest)
experiments/        reproducible experiment + results.json
```

## Quantization scheme

- `W` is `(out_features, in_features)`; each output row is split along
  the **input dimension** into consecutive groups of `group_size`.
- Symmetric 15-level integer range **[-7, 7]** (not `[-8, 7]`):
  `q = clip(round(w / s), -7, 7)`, dequant `ŵ = s·q`.
- Each group stores one **positive** scale (`<f4` default, `<f2`
  optional).  Quantization uses the *stored* (dtype-rounded) scale so
  dequantization is exactly consistent.
- **Zero groups** (all-zero group, e.g. from padding) are handled
  separately: reserved sentinel `scale == 0.0` with all-zero codes;
  dequant yields exact zeros.  Loaders reject negative scales and reject
  `scale == 0` paired with non-zero codes.
- If `in_features % group_size != 0`, the last group is **padded** with
  zero weights; the padding width is recorded in the header and padded
  columns always quantize to code 0.

## Nibble encoding (true 4-bit storage, no int8 tensor)

Each code `q ∈ [-7, 7]` is stored as a 4-bit two's-complement nibble
`n = q & 0xF` (`7 → 0x7`, `0 → 0x0`, `-1 → 0xF`, `-7 → 0x9`).  The
decoder is total: `n ≥ 8 → n − 16` (encoder never emits `0x8`; it would
decode to −8).

- **Byte order:** within a group, codes pack two per byte in
  input-dimension order — code `2k` into the **low** nibble of byte `k`,
  code `2k+1` into the **high** nibble:
  `byte[k] = nib(code[2k]) | (nib(code[2k+1]) << 4)`.
- **Odd tail:** if `group_size` is odd, the group's last byte holds the
  final code in its low nibble and `0x0` in its high nibble (ignored by
  decoders).
- Codes are laid out as `(out_features, num_groups, ceil(group_size/2))`
  uint8.  This packed array is the only code payload — storage cost is a
  genuine 4 bits/weight.  int8 appears only transiently inside compute.

## Scale selection: max-abs vs calibration-driven clipping

Candidate thresholds are a **fixed a priori** set
`CLIP_RATIOS = {1.00, 0.95, …, 0.50}` (`wq4/quantizer.py`); each
candidate scale is `ratio · max|w_group| / 7`.  Three methods:

- `maxabs` — `s = max|w| / 7`, no clipping.
- `clip-wmse` — per group, independently pick the ratio minimizing
  **weight** MSE `Σ(w − ŵ)²`.
- `clip-calib` — **group-wise coordinate search** minimizing the layer
  **output** error on a finite calibration set:
  `f(ŵ) = ‖X(w − ŵ)‖²`.  Starting from max-abs, each sweep visits every
  group and picks the candidate minimizing the exact objective with all
  other groups fixed (via `‖r‖² − 2uᵀd + dᵀ(XⱼᵀXⱼ)d`).  Strict
  improvement + fixed visit order ⇒ deterministic, monotone
  non-increasing, never worse than max-abs on the calibration objective.
  Only the calibration split is used — never the evaluation set.

**Why per-group weight-MSE ≠ output error.**  With `e = w − ŵ` and Gram
matrix `H = XᵀX`, the output error is the quadratic form
`f = Σᵢ Hᵢᵢ eᵢ² + 2 Σᵢ<ⱼ Hᵢⱼ eᵢeⱼ`.  Independent per-group weight-MSE
minimization is only equivalent when `H ∝ I` (white, uncorrelated,
equal-variance inputs).  With correlated features, (i) errors on
high-energy/highly-correlated coordinates weigh more, and (ii)
cross-group terms `Hᵢⱼeᵢeⱼ` couple groups, so one group's best clip
depends on the others' quantization.  The experiment below shows
`clip-calib` achieving *worse* weight error but *better* output error —
the two objectives genuinely differ.

## Inference paths

- `forward_dequant(x)` — reference: dequantize to dense fp32 `Ŵ`,
  matmul, add bias.
- `forward_fused(x)` — per-group unpack-accumulate: unpack each group's
  nibbles → signed codes → scale → accumulate partial dot products;
  only a transient `(out_features, group_size)` buffer exists.  Bias
  included.
- Both compute identical math in different summation orders; declared
  agreement tolerance `rtol=1e-4, atol=1e-5` (tests + experiment assert
  it; observed max diff ≈ 1.1e-05 on the shifted eval batch).

## Container format (WQ4 v1)

```
[0:4]   magic b"WQ4" + version byte 0x01
[4:8]   uint32 LE header JSON length
[8:..]  header JSON: format, version, out_features, in_features,
        group_size, padding, scale_dtype, bias_dtype, has_bias,
        scales/codes/bias_nbytes, payload_sha256
[..]    payload = scales | packed codes | bias(fp32)
```

Loading rejects (with `FormatError`): bad magic, unsupported version,
truncated/invalid header, inconsistent sizes or padding, unsupported
dtypes, payload length mismatch, trailing garbage, SHA-256 mismatch,
non-finite/negative scales, zero-scale groups with non-zero codes.

**Storage accounting counts everything** — header, scales, codes,
padding, bias — not just bare weights (`QuantizedLinear.storage_report`).

## Experiment (reproducible)

`python3 experiments/run_experiment.py` (seed 20260930; writes
`experiments/results.json`).  Setup: `W` 64×100 fp32 with 1% outliers
(×25); inputs from an anisotropic rank-12 factor model (correlated);
eval split has reversed/amplified spectrum + mean shift (distribution
shift); `group_size=32` (100 % 32 ≠ 0 ⇒ 28 padding columns, counted).

| method    | weight_err | out_err calib | out_err eval | total bytes | bits/weight | ratio vs fp32 |
|-----------|-----------|---------------|--------------|-------------|-------------|---------------|
| maxabs    | 0.1382    | 0.1206        | 0.1342       | 5682        | 7.10        | 0.2198        |
| clip-wmse | 0.1368    | 0.1181        | 0.1306       | 5682        | 7.10        | 0.2198        |
| clip-calib| **0.1458*** | **0.0972**  | **0.1243**   | 5682        | 7.10        | 0.2198        |

*deliberately worse weight error, better output error — the point above.

Byte breakdown per model: codes 4096 + scales 1024 + header 306 + bias
256 = **5682 total** vs 25856 fp32.  Effective 7.10 bits/weight — far
from the naive "4 bits" claim because scales (2.5 b/w), padding, header
and bias are real costs.  Distribution shift degrades all methods
(eval > calib) but the calibration-driven ranking survives.

Timing on the 2048×100 eval batch (median of 15, ms, unpack cost
**included** in both quantized paths): fused unpack-accumulate 8.13,
dequantize+matmul 3.02, cached-dequant matmul 1.36 (amortized reference,
no unpack), original fp32 matmul 1.46.  Pure-NumPy unpacking without a
dedicated kernel is **slower** than cached BLAS matmul — no speedup is
claimed; the win here is memory (4.6× smaller), not latency.

## Tests

`python3 -m pytest tests/ -q` — 47 tests covering: nibble layout /
odd-tail / two's-complement decode, round-trip error bounds, zero-group
sentinel, padding, determinism, coordinate-search monotonicity, fused vs
dequant agreement within tolerance, bias handling, format round-trip,
and 10 corruption/incompatibility rejection cases.
