"""Reproducible experiment: WQ4 weight-only 4-bit group quantization.

Fixed seed, fixed candidate thresholds (wq4.CLIP_RATIOS, frozen before
the experiment).  The clipping methods are chosen on the CALIBRATION
split only; the eval split (with distribution shift) is used purely for
reporting.  Run:  python3 experiments/run_experiment.py
"""
import json
import os
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from wq4 import CLIP_RATIOS, QuantizedLinear  # noqa: E402

SEED = 20260930
OUT_FEATURES, IN_FEATURES, GROUP_SIZE = 64, 100, 32  # 100 % 32 != 0 -> padding
N_CAL, N_EVAL = 512, 2048
RANK = 12
TIMING_REPEATS = 15


def build_data():
    rng = np.random.default_rng(SEED)
    # Weights: Gaussian body + 1% heavy outliers (x25).
    W = rng.normal(0.0, 0.06, (OUT_FEATURES, IN_FEATURES)).astype(np.float32)
    W[rng.random(W.shape) < 0.01] *= 25.0
    bias = rng.normal(0.0, 0.05, OUT_FEATURES).astype(np.float32)
    # Correlated inputs: anisotropic low-rank factor model + noise.
    A = rng.normal(0.0, 1.0, (RANK, IN_FEATURES)) / np.sqrt(RANK)
    cal_spectrum = np.linspace(0.4, 2.0, RANK)
    Zc = rng.normal(0.0, 1.0, (N_CAL, RANK)) * cal_spectrum
    X_cal = (Zc @ A + 0.05 * rng.normal(0.0, 1.0, (N_CAL, IN_FEATURES))).astype(
        np.float32
    )
    # Eval: distribution shift -- reversed+amplified spectrum, mean shift.
    eval_spectrum = cal_spectrum[::-1] * 1.4
    Ze = rng.normal(0.0, 1.0, (N_EVAL, RANK)) * eval_spectrum
    mean_shift = 0.3 * rng.normal(0.0, 1.0, IN_FEATURES)
    X_eval = (
        Ze @ A + mean_shift + 0.05 * rng.normal(0.0, 1.0, (N_EVAL, IN_FEATURES))
    ).astype(np.float32)
    return W, bias, X_cal, X_eval


def rel_err(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(a) + 1e-30))


def time_median(fn, repeats=TIMING_REPEATS):
    fn()  # warmup
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ts))


def main():
    W, bias, X_cal, X_eval = build_data()
    Y_cal = X_cal @ W.T + bias
    Y_eval = X_eval @ W.T + bias
    w_norm = np.linalg.norm(W)

    print(f"seed={SEED}  W=({OUT_FEATURES}x{IN_FEATURES}) fp32  group_size={GROUP_SIZE}"
          f"  clip candidates (fixed a priori)={CLIP_RATIOS}")
    print(f"calib: {X_cal.shape}  eval(shifted): {X_eval.shape}\n")

    rows = []
    for method in ("maxabs", "clip-wmse", "clip-calib"):
        ql = QuantizedLinear.from_float(
            W, bias, group_size=GROUP_SIZE, method=method, calib_X=X_cal
        )
        W_hat = ql.weight.dequantize()
        rep = ql.storage_report()
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "m.wq4")
            ql.save(path)
            file_bytes = os.path.getsize(path)
        assert file_bytes == rep["total_bytes"]
        rows.append(
            {
                "method": method,
                "weight_rel_err": float(np.linalg.norm(W - W_hat) / w_norm),
                "out_rel_err_calib": rel_err(Y_cal, ql.forward_fused(X_cal)),
                "out_rel_err_eval": rel_err(Y_eval, ql.forward_fused(X_eval)),
                **rep,
                "file_bytes": file_bytes,
            }
        )

    header = (
        f"{'method':<10} {'weight_err':>10} {'out_err_cal':>11} {'out_err_eval':>12}"
        f" {'codes_B':>8} {'scales_B':>9} {'hdr_B':>6} {'bias_B':>7}"
        f" {'total_B':>8} {'bits/w':>6} {'tot_ratio':>9}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['method']:<10} {r['weight_rel_err']:>10.4f} "
            f"{r['out_rel_err_calib']:>11.4f} {r['out_rel_err_eval']:>12.4f} "
            f"{r['codes_bytes']:>8} {r['scales_bytes']:>9} {r['header_bytes']:>6} "
            f"{r['bias_bytes']:>7} {r['total_bytes']:>8} "
            f"{r['effective_bits_per_weight']:>6.2f} {r['total_ratio']:>9.4f}"
        )
    print(f"\nfp32 baseline (weights+bias): {rows[0]['original_fp32_bytes']} bytes")

    # --- inference-path agreement -------------------------------------
    ql = QuantizedLinear.from_float(W, bias, GROUP_SIZE, method="clip-calib",
                                    calib_X=X_cal)
    y_fused = ql.forward_fused(X_eval)
    y_ref = ql.forward_dequant(X_eval)
    d = float(np.abs(y_fused - y_ref).max())
    ok = np.allclose(y_fused, y_ref, rtol=1e-4, atol=1e-5)
    print(f"\nmax |fused - dequant| on eval batch: {d:.3e}  "
          f"(declared tolerance rtol=1e-4, atol=1e-5) -> {'OK' if ok else 'FAIL'}")
    assert ok, "inference paths disagree beyond declared tolerance"

    # --- timing (unpack cost INCLUDED in both quantized paths) --------
    W_hat_cached = ql.weight.dequantize()
    t_fused = time_median(lambda: ql.forward_fused(X_eval))
    t_dequant = time_median(lambda: ql.forward_dequant(X_eval))  # re-dequantizes
    t_cached = time_median(lambda: X_eval @ W_hat_cached.T + bias)  # no unpack
    t_fp32 = time_median(lambda: X_eval @ W.T + bias)
    print(f"\ntiming on eval batch {X_eval.shape} (median of {TIMING_REPEATS}, ms):")
    print(f"  fused unpack-accumulate (unpack included) : {t_fused:8.3f}")
    print(f"  dequantize + matmul     (unpack included) : {t_dequant:8.3f}")
    print(f"  cached dequant matmul   (no unpack, amortized reference): {t_cached:8.3f}")
    print(f"  original fp32 matmul    (no quantization at all)      : {t_fp32:8.3f}")
    print("  note: pure-NumPy unpack, no dedicated kernel; quantized paths are")
    print("        expected to be SLOWER than cached BLAS matmul. No speedup claimed.")

    results = {
        "seed": SEED,
        "config": {
            "out_features": OUT_FEATURES,
            "in_features": IN_FEATURES,
            "group_size": GROUP_SIZE,
            "n_calib": N_CAL,
            "n_eval": N_EVAL,
            "clip_ratios": list(CLIP_RATIOS),
        },
        "metrics": rows,
        "path_agreement_max_abs_diff": float(d),
        "timing_ms": {
            "fused_unpack_accumulate": t_fused,
            "dequant_plus_matmul": t_dequant,
            "cached_dequant_matmul_no_unpack": t_cached,
            "fp32_matmul": t_fp32,
        },
    }
    out = os.path.join(os.path.dirname(__file__), "results.json")
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nresults written to {out}")


if __name__ == "__main__":
    main()
