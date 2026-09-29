"""Calibration-driven clipping: group-wise coordinate search on output error.

Objective
-----------
For one output row ``w`` (length K) and a fixed calibration matrix
``X`` of shape ``(N, K)``, the objective is the squared output error

    f(w_hat) = || X (w - w_hat) ||^2 = (w - w_hat)^T H (w - w_hat),
    H = X^T X.

Why per-group weight-MSE is not equivalent
------------------------------------------
Expanding the quadratic form:

    f = sum_i H_ii e_i^2  +  2 * sum_{i<j} H_ij e_i e_j,   e = w - w_hat.

Per-group (or per-weight) weight-MSE minimization is only exact when
``H`` is a multiple of the identity, i.e. white, uncorrelated,
equal-variance inputs.  With correlated features ``H`` has large
off-diagonal entries, so

* errors on high-energy / highly correlated coordinates are weighted
  more heavily than others -- uniform weight-MSE treats them equally;
* cross terms ``H_ij e_i e_j`` couple the quantization error of
  different coordinates *across groups*, so the best clip threshold of
  one group depends on how other groups are quantized.

Algorithm (deterministic coordinate search)
-------------------------------------------
We minimize ``f`` directly by coordinate descent over groups: start from
the max-abs solution, then repeatedly visit each group and choose, from
the FIXED candidate set ``CLIP_RATIOS``, the scale that minimizes the
total objective with all other groups held fixed.  For group j with
residual ``r = X (w - w_hat)`` and candidate dequantized values
``w_hat_c``, the new objective is

    || r + X_j (w_hat_j^cur - w_hat_j^c) ||^2
        = ||r||^2 - 2 u^T d + d^T (X_j^T X_j) d,
    u = X_j^T r,  d = w_hat_j^c - w_hat_j^cur,

which is evaluated exactly for every candidate.  A candidate is accepted
only on strict improvement (beyond a tiny relative epsilon), sweeps and
groups are visited in a fixed order, and ties resolve to the earlier
candidate -- so the whole procedure is deterministic, the objective is
monotone non-increasing, and the result is never worse than max-abs on
the calibration objective (ratio 1.0 is always in the candidate set).

The candidate thresholds are fixed before experiments; the calibration
set -- never the evaluation set -- drives the choice.
"""
from __future__ import annotations

import numpy as np

from .packing import pack_codes, unpack_codes
from .quantizer import (
    CLIP_RATIOS,
    QuantizedWeight,
    _check_scale_dtype,
    _group_maxabs,
    _pad_weight,
    quantize_maxabs,
)

_ACCEPT_EPS = 1e-10  # relative improvement threshold for accepting a move


def output_error_sq(W: np.ndarray, W_hat: np.ndarray, X: np.ndarray) -> float:
    """|| X (W - W_hat)^T ||_F^2 -- the calibration objective (linear part)."""
    diff = np.asarray(W, dtype=np.float64) - np.asarray(W_hat, dtype=np.float64)
    proj = np.asarray(X, dtype=np.float64) @ diff.T
    return float(np.sum(proj * proj))


def quantize_clip_calibrated(
    W: np.ndarray,
    X_cal: np.ndarray,
    group_size: int = 32,
    ratios: tuple[float, ...] = CLIP_RATIOS,
    scale_dtype: str = "<f4",
    sweeps: int = 3,
) -> QuantizedWeight:
    """Coordinate-search clipping driven by calibration output error.

    Parameters
    ----------
    W : (out_features, in_features) float weights.
    X_cal : (n_calib, in_features) fixed calibration inputs.
    sweeps : fixed number of full passes over groups (early-stops only
        when a full sweep makes no accepted move; still deterministic).
    """
    sdt = _check_scale_dtype(scale_dtype)
    W = np.asarray(W, dtype=np.float64)
    X = np.asarray(X_cal, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != W.shape[1]:
        raise ValueError("X_cal must be (n_calib, in_features)")
    if len(ratios) < 1 or any(not (0.0 < r <= 1.0) for r in ratios):
        raise ValueError("clip ratios must be in (0, 1]")

    Wp, padding = _pad_weight(W, group_size)
    out_features, in_features = W.shape
    O, Kp = Wp.shape
    G = Kp // group_size
    Wg = Wp.reshape(O, G, group_size)

    # Pad calibration inputs the same way; padded coords contribute 0.
    if padding:
        Xp = np.concatenate([X, np.zeros((X.shape[0], padding))], axis=1)
    else:
        Xp = X
    Xg = Xp.reshape(X.shape[0], G, group_size)
    gram = [Xg[:, j, :].T @ Xg[:, j, :] for j in range(G)]  # shared across rows

    base = quantize_maxabs(W, group_size=group_size, scale_dtype=scale_dtype)
    q_all = unpack_codes(base.codes, group_size).astype(np.int8)  # (O, G, g)
    s_all = base.scales.astype(np.float64)  # (O, G)
    zero = _group_maxabs(Wg) == 0.0

    ratios_arr = np.asarray(ratios, dtype=np.float64)
    for o in range(O):
        w_row = Wg[o]  # (G, g)
        w_hat = (q_all[o].astype(np.float64) * s_all[o][:, None]).copy()
        resid = Xp @ (w_row.reshape(-1) - w_hat.reshape(-1))  # (N,)
        for _ in range(sweeps):
            changed = False
            for j in range(G):
                if zero[o, j]:
                    continue
                u = Xg[:, j, :].T @ resid  # (g,)
                m = np.abs(w_row[j]).max()
                s_cand = (ratios_arr * m / 7.0).astype(sdt).astype(np.float64)
                q_cand = np.clip(
                    np.round(w_row[j][None, :] / s_cand[:, None]), -7, 7
                )
                wq_cand = q_cand * s_cand[:, None]  # (R, g)
                delta = wq_cand - w_hat[j][None, :]  # (R, g)
                change = -2.0 * (delta @ u) + np.einsum(
                    "rg,gh,rh->r", delta, gram[j], delta
                )
                c = int(np.argmin(change))
                if change[c] < -_ACCEPT_EPS * (resid @ resid + 1e-30):
                    resid -= Xg[:, j, :] @ delta[c]
                    w_hat[j] = wq_cand[c]
                    q_all[o, j] = q_cand[c].astype(np.int8)
                    s_all[o, j] = s_cand[c]
                    changed = True
            if not changed:
                break

    scales = s_all.astype(sdt)
    scales[zero] = 0.0
    return QuantizedWeight(
        codes=pack_codes(q_all),
        scales=scales,
        out_features=out_features,
        in_features=in_features,
        group_size=group_size,
        padding=padding,
        scale_dtype=scale_dtype,
    )
