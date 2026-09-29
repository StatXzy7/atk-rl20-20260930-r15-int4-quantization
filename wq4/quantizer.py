"""Weight-only 4-bit group quantization of a linear layer's weight matrix.

Layout
------
Weight ``W`` has shape ``(out_features, in_features)``.  The input
dimension of every output channel is split into consecutive groups of
``group_size``.  If ``in_features % group_size != 0`` the last group is
padded with zero weights (the padding width is recorded in the file
format and padded columns always quantize to code 0).

Symmetric scheme
----------------
Each group ``g`` gets one positive scale ``s`` and stores integer codes
``q = clip(round(w / s), -7, 7)``; dequantization is ``w_hat = s * q``.
The integer range is the symmetric 15-level range [-7, 7] (not [-8, 7]).

Zero groups
-----------
A group whose weights are all zero has no meaningful positive scale.
It is stored with the reserved sentinel ``scale == 0.0`` and all-zero
codes; dequantization yields exact zeros.  Normal groups always have a
strictly positive scale.  Loaders reject negative scales and reject a
zero scale paired with non-zero codes.

Scale dtypes
------------
Scales may be stored as ``<f4`` (default) or ``<f2``.  Quantization is
performed with the *stored* (dtype-rounded) scale so that dequantization
reproduces exactly what the quantizer saw.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .packing import QMIN, QMAX, pack_codes, unpack_codes

# Candidate clipping ratios, fixed BEFORE any experiment is run.  Each
# candidate threshold is ``ratio * max|w_group|``; ratio 1.0 is plain
# max-abs (no clipping).  Ordered descending: ties keep less clipping.
CLIP_RATIOS: tuple[float, ...] = tuple(round(1.00 - 0.05 * i, 2) for i in range(11))

SCALE_DTYPES = ("<f4", "<f2")


@dataclass
class QuantizedWeight:
    """Packed 4-bit weight: codes (O, G, B) uint8, scales (O, G)."""

    codes: np.ndarray  # (O, G, ceil(group_size/2)) uint8, packed nibbles
    scales: np.ndarray  # (O, G) scale_dtype, >= 0; 0.0 == zero-group sentinel
    out_features: int
    in_features: int
    group_size: int
    padding: int
    scale_dtype: str

    @property
    def num_groups(self) -> int:
        return self.scales.shape[1]

    @property
    def padded_in_features(self) -> int:
        return self.in_features + self.padding

    def dequantize(self) -> np.ndarray:
        """Reference dequantization to a dense float32 (O, in_features)."""
        q = unpack_codes(self.codes, self.group_size).astype(np.float32)
        s = self.scales.astype(np.float32)
        w_hat = (q * s[:, :, None]).reshape(self.out_features, -1)
        return w_hat[:, : self.in_features].copy()


def _check_scale_dtype(scale_dtype: str) -> np.dtype:
    if scale_dtype not in SCALE_DTYPES:
        raise ValueError(f"unsupported scale_dtype {scale_dtype!r}")
    return np.dtype(scale_dtype)


def _pad_weight(W: np.ndarray, group_size: int) -> tuple[np.ndarray, int]:
    if W.ndim != 2:
        raise ValueError("weight must be 2-D (out_features, in_features)")
    if group_size < 1:
        raise ValueError("group_size must be >= 1")
    out_features, in_features = W.shape
    padding = (-in_features) % group_size
    if padding:
        W = np.concatenate(
            [W, np.zeros((out_features, padding), dtype=W.dtype)], axis=1
        )
    return W, padding


def _group_maxabs(Wg: np.ndarray) -> np.ndarray:
    return np.abs(Wg).max(axis=-1)  # (O, G)


def _codes_for_scale(Wg: np.ndarray, s_use: np.ndarray) -> np.ndarray:
    """Integer codes for grouped weights Wg (O, G, g) given scales (O, G)."""
    q = np.round(Wg / s_use[:, :, None])
    return np.clip(q, QMIN, QMAX)


def quantize_maxabs(
    W: np.ndarray, group_size: int = 32, scale_dtype: str = "<f4"
) -> QuantizedWeight:
    """Plain max-abs scaling: s = max|w_group| / 7 (no clipping)."""
    sdt = _check_scale_dtype(scale_dtype)
    W64, padding = _pad_weight(np.asarray(W, dtype=np.float64), group_size)
    out_features, in_features = np.asarray(W).shape
    O, Kp = W64.shape
    Wg = W64.reshape(O, Kp // group_size, group_size)

    m = _group_maxabs(Wg)
    zero = m == 0.0
    scales = (m / 7.0).astype(sdt)
    s_use = scales.astype(np.float64)
    s_use[zero] = 1.0  # avoid 0/0; codes are 0 regardless
    q = _codes_for_scale(Wg, s_use)
    scales[zero] = 0.0  # zero-group sentinel
    return QuantizedWeight(
        codes=pack_codes(q),
        scales=scales,
        out_features=out_features,
        in_features=in_features,
        group_size=group_size,
        padding=padding,
        scale_dtype=scale_dtype,
    )


def quantize_clip_weight_mse(
    W: np.ndarray,
    group_size: int = 32,
    ratios: tuple[float, ...] = CLIP_RATIOS,
    scale_dtype: str = "<f4",
) -> QuantizedWeight:
    """Per-group clipping search minimizing *weight* MSE, independently.

    For each group and each candidate ratio r (fixed set), the scale is
    ``r * max|w| / 7`` and the group error is ``sum((w - s*q)^2)``.
    Candidates are tried in the fixed tuple order and a candidate is
    accepted only on strict improvement, so the result is deterministic
    and never worse than max-abs on the weight-MSE objective.

    NOTE: minimizing per-group weight MSE is generally NOT equivalent to
    minimizing the layer's output error on real inputs -- see
    wq4.calibration for the output-error objective.
    """
    sdt = _check_scale_dtype(scale_dtype)
    W64, padding = _pad_weight(np.asarray(W, dtype=np.float64), group_size)
    out_features, in_features = np.asarray(W).shape
    O, Kp = W64.shape
    Wg = W64.reshape(O, Kp // group_size, group_size)

    m = _group_maxabs(Wg)
    zero = m == 0.0
    best_err = np.full((O, Wg.shape[1]), np.inf)
    best_q = np.zeros_like(Wg, dtype=np.int8)
    best_s = np.zeros((O, Wg.shape[1]), dtype=np.float64)
    for r in ratios:
        if not (0.0 < r <= 1.0):
            raise ValueError("clip ratios must be in (0, 1]")
        s = (r * m / 7.0).astype(sdt).astype(np.float64)
        s_use = np.where(zero, 1.0, s)
        q = _codes_for_scale(Wg, s_use)
        err = np.sum((Wg - q * s_use[:, :, None]) ** 2, axis=-1)
        improved = err < best_err  # strict: ties keep earlier (larger) ratio
        best_err = np.where(improved, err, best_err)
        best_q = np.where(improved[:, :, None], q, best_q)
        best_s = np.where(improved, s, best_s)
    scales = best_s.astype(sdt)
    scales[zero] = 0.0
    return QuantizedWeight(
        codes=pack_codes(best_q),
        scales=scales,
        out_features=out_features,
        in_features=in_features,
        group_size=group_size,
        padding=padding,
        scale_dtype=scale_dtype,
    )
