"""Reproducible comparison experiment for 4-bit grouped linear quantization.

Scenario (seed fixed at the top; no data-dependent choices anywhere else):
  * 64 output channels, 128 inputs, group size 16 along the input dimension.
  * Weights contain sparse ~10x outliers.
  * Inputs are strongly correlated (low-rank latent structure), so group
    errors are correlated through E[X_g^T X_h].
  * Two evaluation splits are reported:
      - eval_iid  : same distribution as calibration.
      - eval_shift: latent variances change and latent directions unseen at
                    calibration time appear (covariate shift).

Compared methods
  * maxabs      : per-group scale = max-abs / 7, no clipping search.
  * weight_clip : each group independently minimizes *weight* SSE over the
                  candidate threshold grid fixed before the experiment.
  * calibrated  : grouped coordinate descent on calibration OUTPUT MSE over
                  the same fixed grid; only X_cal is used to fit.
  * eval_leak   : coordinate search run on X_eval_shift (NOT a valid method;
                  a leakage control showing how much "picking thresholds on
                  the eval set" would cheat).

Timings always include nibble unpacking. Without a fused SIMD kernel the
pure-NumPy unpack paths are expected to be slower than a dense BLAS matmul;
no speedup is claimed. All figures land in experiments/results.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from int4_quant import QuantizedLinear, storage_bytes  # noqa: E402

SEED = 20260930
OUT_FEATURES = 64
IN_FEATURES = 128
GROUP_SIZE = 16
N_CAL = 1024
N_EVAL = 1024
N_LATENT = 8
N_EXTRA_LATENT = 4

HERE = Path(__file__).resolve().parent


def make_weights(rng: np.random.Generator):
    W = rng.standard_normal(size=(OUT_FEATURES, IN_FEATURES)) * 0.3
    outlier_mask = rng.random(W.shape) < 0.03
    W[outlier_mask] = rng.choice([-1.0, 1.0], size=outlier_mask.sum()) * (
        2.5 + rng.random(outlier_mask.sum()) * 1.5
    )
    bias = rng.standard_normal(size=OUT_FEATURES) * 0.1
    return W, bias


def make_inputs(rng, n, loading_core, loading_extra, shifted):
    s = rng.standard_normal(size=(n, N_LATENT))
    if shifted:
        s = s.copy()
        s[:, :4] *= 1.6
        s_extra = rng.standard_normal(size=(n, N_EXTRA_LATENT)) * 1.3
        z = s @ loading_core + s_extra @ loading_extra
    else:
        z = s @ loading_core
    z += rng.standard_normal(size=z.shape) * 0.05
    return z


def weight_mse(layer, W):
    deq = layer.dequantize_weights()[:, : IN_FEATURES]
    return float(np.mean((deq - W) ** 2))


def out_mse(layer, x, y_ref):
    return float(np.mean((layer.forward_packed(x) - y_ref) ** 2))


def timed_interleaved(jobs, repeats=25):
    """Time callables interleaved in shuffled order to fight system noise.

    Returns median seconds keyed by job name. All packed paths pay the nibble
    unpacking cost on every call.
    """
    names = list(jobs)
    for fn in jobs.values():
        fn()  # warmup every callable first
    samples = {name: [] for name in names}
    for rep in range(repeats):
        order = rng_perm.permutation(len(names))
        for idx in order:
            name = names[idx]
            t0 = time.perf_counter()
            jobs[name]()
            samples[name].append(time.perf_counter() - t0)
    return {name: float(np.median(vals)) for name, vals in samples.items()}


rng_perm = np.random.default_rng(0)


def main():
    rng = np.random.default_rng(SEED)
    W, bias = make_weights(rng)
    loading_core = rng.standard_normal(size=(N_LATENT, IN_FEATURES)) / np.sqrt(N_LATENT)
    loading_extra = rng.standard_normal(size=(N_EXTRA_LATENT, IN_FEATURES)) / np.sqrt(N_EXTRA_LATENT)

    x_cal = make_inputs(rng, N_CAL, loading_core, loading_extra, shifted=False)
    x_iid = make_inputs(rng, N_EVAL, loading_core, loading_extra, shifted=False)
    x_shift = make_inputs(rng, N_EVAL, loading_core, loading_extra, shifted=True)
    y_cal = x_cal @ W.T + bias
    y_iid = x_iid @ W.T + bias
    y_shift = x_shift @ W.T + bias

    def mean_abs_corr(x):
        c = np.corrcoef(x, rowvar=False)
        iu = np.triu_indices(IN_FEATURES, k=1)
        return float(np.mean(np.abs(c[iu])))

    methods = {
        "maxabs": dict(method="maxabs"),
        "weight_clip": dict(method="weight_clip"),
        "calibrated": dict(method="calibrated", x_cal=x_cal),
        "eval_leak": dict(method="calibrated", x_cal=x_shift),
    }

    results = {
        "config": {
            "seed": SEED,
            "out_features": OUT_FEATURES,
            "in_features": IN_FEATURES,
            "group_size": GROUP_SIZE,
            "n_cal": N_CAL,
            "n_eval": N_EVAL,
            "n_latent": N_LATENT,
            "n_extra_latent_shift": N_EXTRA_LATENT,
            "mean_abs_input_corr_cal": mean_abs_corr(x_cal),
            "mean_abs_input_corr_iid": mean_abs_corr(x_iid),
            "mean_abs_input_corr_shift": mean_abs_corr(x_shift),
        },
        "methods": {},
    }

    layers_built = {}
    for name, kw in methods.items():
        layer = QuantizedLinear.from_weights(W, GROUP_SIZE, bias=bias, **kw)
        layers_built[name] = layer
        stats = storage_bytes(layer).as_dict()

        yp_cal = layer.forward_packed(x_cal)
        yp_iid = layer.forward_packed(x_iid)
        yp_shift = layer.forward_packed(x_shift)
        ys_shift = layer.forward_packed_scalar(x_shift)
        yd_shift = layer.forward_dequant(x_shift)
        path_gap = float(
            max(
                np.abs(yp_shift - yd_shift).max(),
                np.abs(yp_shift - ys_shift).max(),
            )
        )

        # Timing is done once after all layers are built (below) so calls
        # interleave across methods; placeholders filled in afterwards.
        results["methods"][name] = {
            "_layer_placeholder": True,
            "weight_mse": weight_mse(layer, W),
            "output_mse_cal": float(np.mean((yp_cal - y_cal) ** 2)),
            "output_mse_eval_iid": float(np.mean((yp_iid - y_iid) ** 2)),
            "output_mse_eval_shift": float(np.mean((yp_shift - y_shift) ** 2)),
            "output_rmse_eval_shift": float(np.sqrt(np.mean((yp_shift - y_shift) ** 2))),
            "max_path_gap_packed_vs_dequant_vs_scalar": path_gap,
            "storage": stats,
            "timing_seconds_shift_batch_1024": None,
            "fit_sweeps": (
                len(layer.fit.mse_history) - 1
                if layer.fit.mse_history is not None
                else 0
            ),
            "cal_search_mse_history": layer.fit.mse_history,
        }

    # Interleaved, order-randomized timings (same operations for every method).
    timing_jobs = {}
    for name, layer in layers_built.items():
        timing_jobs[f"{name}:packed"] = lambda L=layer: L.forward_packed(x_shift)
        timing_jobs[f"{name}:scalar"] = lambda L=layer: L.forward_packed_scalar(x_shift)
        timing_jobs[f"{name}:dequant"] = lambda L=layer: L.forward_dequant(x_shift)
    timing_jobs["dense"] = lambda: x_shift @ W.T + bias
    med = timed_interleaved(timing_jobs, repeats=25)
    for name in layers_built:
        results["methods"][name].pop("_layer_placeholder", None)
        results["methods"][name]["timing_seconds_shift_batch_1024"] = {
            "packed_direct_median": med[f"{name}:packed"],
            "packed_per_group_scalar_median": med[f"{name}:scalar"],
            "dequant_then_matmul_median": med[f"{name}:dequant"],
            "dense_fp32_matmul_median": med["dense"],
            "packed_vs_dense_median_ratio": med[f"{name}:packed"] / med["dense"],
        }

    out_json = HERE / "results.json"
    out_json.write_text(json.dumps(results, indent=2))

    cfg = results["config"]
    print("=" * 100)
    print(
        f"4-bit grouped quantizer  O={OUT_FEATURES} K={IN_FEATURES} "
        f"G={GROUP_SIZE} N={N_CAL}"
    )
    print(
        f"mean |corr(inputs)|  cal={cfg['mean_abs_input_corr_cal']:.3f}  "
        f"iid={cfg['mean_abs_input_corr_iid']:.3f}  "
        f"shift={cfg['mean_abs_input_corr_shift']:.3f}"
    )
    print("=" * 100)
    print(
        f"{'method':<12}{'weight_mse':>11}{'mse_cal':>11}"
        f"{'mse_iid':>11}{'mse_shift':>11}{'bits/w':>9}"
    )
    print("-" * 100)
    for name, rec in results["methods"].items():
        print(
            f"{name:<12}{rec['weight_mse']:>11.5f}{rec['output_mse_cal']:>11.4f}"
            f"{rec['output_mse_eval_iid']:>11.4f}{rec['output_mse_eval_shift']:>11.4f}"
            f"{rec['storage']['effective_bits_per_weight']:>9.3f}"
        )
    print("-" * 100)
    s = results["methods"]["maxabs"]["storage"]
    print(
        "storage: "
        f"header={s['header_bytes']}B scales={s['scales_bytes']}B "
        f"packed_nibbles={s['packed_weight_bytes']}B "
        f"(tail_pad_nibbles={s['tail_pad_nibbles']}, "
        f"input_pad_nibbles={s['input_padding_nibbles']}) "
        f"bias={s['bias_bytes']}B TOTAL={s['total_bytes']}B "
        f"(fp32 baseline={s['dense_fp32_bytes']}B, "
        f"ratio={s['total_over_fp32_ratio']:.3f})"
    )
    gap = max(
        r["max_path_gap_packed_vs_dequant_vs_scalar"]
        for r in results["methods"].values()
    )
    print(f"path equivalence max|packed-dequant-scalar| = {gap:.3e}")
    print("timing (median seconds for 1024 x 128 batch, includes unpack):")
    for name, rec in results["methods"].items():
        t = rec["timing_seconds_shift_batch_1024"]
        print(
            f"  {name:<12} packed={t['packed_direct_median']*1e3:8.3f}ms "
            f"scalar={t['packed_per_group_scalar_median']*1e3:8.2f}ms "
            f"dequant={t['dequant_then_matmul_median']*1e3:7.3f}ms "
            f"dense(baseline)={t['dense_fp32_matmul_median']*1e3:7.3f}ms "
            f"packed/dense={t['packed_vs_dense_median_ratio']:7.1f}x"
        )
    print(
        "NOTE: eval_leak fits thresholds on shifted evaluation data and is "
        "shown only to quantify evaluation-set leakage; it is not a valid method."
    )
    print("Full numbers in", out_json)


if __name__ == "__main__":
    main()
