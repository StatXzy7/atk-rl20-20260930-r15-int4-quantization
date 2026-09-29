"""Grouped symmetric 4-bit quantization and calibration-driven clipping.

Weights are grouped *along the input dimension per output channel*: for a
weight matrix W of shape (out_features, in_features), row ``o`` is split into
contiguous groups of ``group_size`` inputs. The level range is the symmetric
signed set [-7, 7]; each group stores one positive scale.

A group whose values are all zero gets scale exactly 0 (codes all 0) and
reconstructs exactly, including after the float32 scale round-trip.
"""

from __future__ import annotations

import numpy as np

from .packing import INT4_MIN, INT4_MAX

# Candidate clipping thresholds are FIXED before any calibration data is seen.
# A candidate threshold equals fraction * (per-group max-abs). The evaluation
# distribution is never used to choose a threshold; it is reporting only.
CANDIDATE_CLIP_FRACTIONS: tuple[float, ...] = (
    1.0,
    0.9375,
    0.875,
    0.8125,
    0.75,
    0.6875,
    0.625,
    0.5625,
    0.5,
)

LEVELS = float(-INT4_MIN)  # 7
_SCALE_DTYPE = np.float32


def num_groups(in_features: int, group_size: int) -> int:
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    return (in_features + group_size - 1) // group_size


def pad_groups(weight: np.ndarray, group_size: int) -> tuple[np.ndarray, int]:
    """Zero-pad columns so in_features is a multiple of group_size."""
    weight = np.asarray(weight)
    if weight.ndim != 2:
        raise ValueError("weight must be 2-D (out_features, in_features)")
    in_features = weight.shape[1]
    k_padded = num_groups(in_features, group_size) * group_size
    pad = k_padded - in_features
    if pad:
        weight = np.pad(weight, ((0, 0), (0, pad)), mode="constant")
    return np.ascontiguousarray(weight, dtype=np.float64), pad


def pad_inputs(x: np.ndarray, k_padded: int) -> np.ndarray:
    """Zero-pad input columns to the quantized (padded) width."""
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError("x must be 2-D (batch, in_features)")
    if x.shape[-1] > k_padded:
        raise ValueError("input last dimension exceeds padded weight width")
    if x.shape[-1] == k_padded:
        return np.ascontiguousarray(x, dtype=np.float64)
    pad = k_padded - x.shape[-1]
    return np.pad(x, ((0, 0), (0, pad)), mode="constant")


def group_maxabs(weight_padded: np.ndarray, group_size: int) -> np.ndarray:
    """Per-(row, group) maximum absolute value, shape (O, G)."""
    weight_padded = np.asarray(weight_padded, dtype=np.float64)
    out_features, k_padded = weight_padded.shape
    if k_padded % group_size != 0:
        raise ValueError("weight must be group-padded before grouping")
    grouped = weight_padded.reshape(out_features, k_padded // group_size, group_size)
    return np.max(np.abs(grouped), axis=2)


def make_scales(
    maxabs: np.ndarray,
    scale_dtype: np.dtype = _SCALE_DTYPE,
) -> np.ndarray:
    """Positive scale = maxabs / 7, cast to the stored scale dtype.

    Zero groups get scale exactly 0. The cast happens before quantization so
    that calibration optimizes the same rounding inference performs, keeping
    coordinate descent monotone.
    """
    maxabs = np.asarray(maxabs, dtype=np.float64)
    scales = (maxabs / LEVELS).astype(scale_dtype)
    zero = np.zeros((), dtype=scale_dtype)
    return np.where(maxabs == 0.0, zero, scales).astype(scale_dtype)


def quantize_codes(
    weight_padded: np.ndarray,
    scales: np.ndarray,
    group_size: int,
    thresholds: np.ndarray | None = None,
) -> np.ndarray:
    """Quantize padded weights to integer codes in [-7, 7], shape of W."""
    weight_padded = np.asarray(weight_padded, dtype=np.float64)
    out_features, k_padded = weight_padded.shape
    scales64 = np.asarray(scales, dtype=np.float64)
    if scales64.shape != (out_features, k_padded // group_size):
        raise ValueError("scales shape must match (out_features, num_groups)")
    scales_b = np.broadcast_to(
        scales64[:, :, None],
        (out_features, k_padded // group_size, group_size),
    ).reshape(out_features, k_padded)
    if thresholds is None:
        thresholds = LEVELS * scales64
    thresholds_b = np.broadcast_to(
        np.asarray(thresholds, dtype=np.float64)[:, :, None],
        (out_features, k_padded // group_size, group_size),
    ).reshape(out_features, k_padded)
    clipped = np.clip(weight_padded, -thresholds_b, thresholds_b)
    safe_scale = np.where(scales_b > 0, scales_b, 1.0)
    q = np.rint(clipped / safe_scale)
    q = np.where(scales_b > 0, q, 0.0)
    q = np.clip(q, INT4_MIN, INT4_MAX)
    return q.astype(np.int8)


def dequantize(
    codes: np.ndarray,
    scales: np.ndarray,
    group_size: int,
) -> np.ndarray:
    """Reference dequantization: q * scale per group (float64 output)."""
    codes = np.asarray(codes, dtype=np.float64)
    out_features, k_padded = codes.shape
    scales64 = np.asarray(scales, dtype=np.float64)
    scales_b = np.broadcast_to(
        scales64[:, :, None],
        (out_features, k_padded // group_size, group_size),
    ).reshape(out_features, k_padded)
    return codes * scales_b


def quantize_weight(
    weight: np.ndarray,
    group_size: int,
    scale_dtype: np.dtype = _SCALE_DTYPE,
):
    """Max-abs scale path: pad, scale = maxabs/7, quantize.

    Returns (codes, scales, input_padding).
    """
    weight_padded, in_pad = pad_groups(weight, group_size)
    maxabs = group_maxabs(weight_padded, group_size)
    scales = make_scales(maxabs, scale_dtype)
    codes = quantize_codes(weight_padded, scales, group_size)
    return codes, scales, in_pad


def _candidate_group(
    segment: np.ndarray,
    maxabs_val: float,
    fraction: float,
    scale_dtype: np.dtype,
):
    """Quantize one group at threshold = fraction * maxabs.

    Uses exactly the arithmetic inference uses: threshold in float64, scale
    rounded to the stored scale dtype, rounding-to-nearest, clip to [-7, 7].
    Returns (codes int8, dequantized float64, scale).
    """
    if maxabs_val == 0.0:
        return (
            np.zeros(segment.shape[0], dtype=np.int8),
            np.zeros(segment.shape[0], dtype=np.float64),
            np.zeros((), dtype=scale_dtype),
        )
    threshold = fraction * maxabs_val
    scale = np.array(threshold / LEVELS, dtype=scale_dtype)
    clipped = np.clip(segment, -threshold, threshold)
    q = np.clip(np.rint(clipped / scale.astype(np.float64)), INT4_MIN, INT4_MAX)
    q = q.astype(np.int8)
    deq = q.astype(np.float64) * scale.astype(np.float64)
    return q, deq, scale


def fit_weight_clipping(
    weight: np.ndarray,
    group_size: int,
    fractions: tuple[float, ...] = CANDIDATE_CLIP_FRACTIONS,
    scale_dtype: np.dtype = _SCALE_DTYPE,
):
    """Choose each group's fixed candidate minimizing *weight* SSE.

    Groups are minimized independently against the weights they contain.
    Returns (codes, scales, thresholds, input_padding).
    """
    weight_padded, in_pad = pad_groups(weight, group_size)
    out_features, k_padded = weight_padded.shape
    groups = k_padded // group_size
    maxabs = group_maxabs(weight_padded, group_size)
    codes = np.zeros((out_features, k_padded), dtype=np.int8)
    scales = np.zeros((out_features, groups), dtype=scale_dtype)
    thresholds = np.zeros((out_features, groups), dtype=np.float64)
    for o in range(out_features):
        for g in range(groups):
            lo, hi = g * group_size, (g + 1) * group_size
            segment = weight_padded[o, lo:hi]
            m = maxabs[o, g]
            best = None
            for f in fractions:
                q, deq, s = _candidate_group(segment, m, f, scale_dtype)
                sse = float(np.sum((deq - segment) ** 2))
                # Strict comparison: ties keep the earlier (less clipped)
                # candidate, matching the fractions ordering (1.0 first).
                if best is None or sse < best[0]:
                    best = (sse, f, q, deq, s)
            _, f, q, _, s = best
            codes[o, lo:hi] = q
            scales[o, g] = s
            thresholds[o, g] = f * m
    return codes, scales, thresholds, in_pad


def coordinate_clip_search(
    weight: np.ndarray,
    x_cal: np.ndarray,
    group_size: int,
    fractions: tuple[float, ...] = CANDIDATE_CLIP_FRACTIONS,
    max_sweeps: int = 24,
    scale_dtype: np.dtype = _SCALE_DTYPE,
):
    """Grouped coordinate descent on calibration-layer-output MSE.

    Objective: mean over calibration samples and output channels of
        || Q(W) @ x - W @ x ||^2,
    where Q(W) is the dequantized (padded) weight. Bias cancels from the
    difference and therefore does not enter the objective.

    Coordinates are the per-(row, group) threshold candidates. Each coordinate
    move is accepted only if it strictly reduces total MSE, so the objective
    is monotonically non-increasing and the deterministic sweep converges to a
    coordinate-wise optimum over the fixed candidate grid (independent group
    weight-SSE minimization corresponds instead to dropping all cross-group
    terms induced by x; see README).

    Returns (codes, scales, thresholds, input_padding, mse_history).
    """
    weight_padded, in_pad = pad_groups(weight, group_size)
    out_features, k_padded = weight_padded.shape
    groups = k_padded // group_size
    x_padded = pad_inputs(x_cal, k_padded)
    if x_padded.shape[0] == 0:
        raise ValueError("need at least one calibration sample")

    maxabs = group_maxabs(weight_padded, group_size)
    scales = make_scales(maxabs, scale_dtype)
    codes = quantize_codes(weight_padded, scales, group_size)
    deq = dequantize(codes, scales, group_size)

    active_f = np.ones((out_features, groups), dtype=np.float64)
    # Current output error E (N, O) = X @ (Q(W) - W)^T.
    error = x_padded @ (deq - weight_padded).T

    def total_mse(err: np.ndarray) -> float:
        return float(np.mean(err * err))

    history = [total_mse(error)]
    for _ in range(max_sweeps):
        changed = False
        for o in range(out_features):
            x_wt = x_padded  # (N, Kpad)
            for g in range(groups):
                lo, hi = g * group_size, (g + 1) * group_size
                segment = weight_padded[o, lo:hi]
                x_seg = x_wt[:, lo:hi]                      # (N, H)
                cur_delta = deq[o, lo:hi] - segment         # (H,)
                err_other = error[:, o] - x_seg @ cur_delta # (N,)
                m = maxabs[o, g]

                best_mse = total_mse(error[:, o])
                best = None
                for f in fractions:
                    q, cand_deq, s = _candidate_group(segment, m, f, scale_dtype)
                    cand_err = err_other + x_seg @ (cand_deq - segment)
                    mse = float(np.mean(cand_err * cand_err))
                    if mse < best_mse:
                        best_mse = mse
                        best = (f, q, cand_deq, s)
                if best is not None:
                    f, q, cand_deq, s = best
                    error[:, o] = err_other + x_seg @ (cand_deq - segment)
                    codes[o, lo:hi] = q
                    deq[o, lo:hi] = cand_deq
                    scales[o, g] = s
                    active_f[o, g] = f
                    changed = True
        history.append(total_mse(error))
        if not changed:
            break

    thresholds = active_f * maxabs
    return codes, scales, thresholds, in_pad, history
