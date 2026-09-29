"""Quantized linear layer with two inference paths.

* ``forward_dequant`` -- reference path: fully dequantize the weight to
  float32, then a dense matmul, then add the bias.
* ``forward_fused`` -- per-group unpack-accumulate: for each group,
  unpack that group's nibbles to signed codes, scale them, accumulate
  the partial dot product.  Only a transient ``(out_features,
  group_size)`` buffer is materialized; the packed uint8 codes are the
  storage format.  The bias is added at the end.

Both paths compute identical math in different summation orders, so they
agree up to float32 rounding; the declared agreement tolerance is
``AGREE_RTOL = 1e-4`` / ``AGREE_ATOL = 1e-5`` (verified in tests).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .packing import unpack_codes
from .quantizer import (
    CLIP_RATIOS,
    QuantizedWeight,
    quantize_clip_weight_mse,
    quantize_maxabs,
)

AGREE_RTOL = 1e-4
AGREE_ATOL = 1e-5

METHODS = ("maxabs", "clip-wmse", "clip-calib")


@dataclass
class QuantizedLinear:
    weight: QuantizedWeight
    bias: np.ndarray | None  # (out_features,) float32, exact (unquantized)

    @classmethod
    def from_float(
        cls,
        weight: np.ndarray,
        bias: np.ndarray | None = None,
        group_size: int = 32,
        method: str = "maxabs",
        calib_X: np.ndarray | None = None,
        ratios: tuple[float, ...] = CLIP_RATIOS,
        scale_dtype: str = "<f4",
        sweeps: int = 3,
    ) -> "QuantizedLinear":
        weight = np.asarray(weight)
        if method == "maxabs":
            qw = quantize_maxabs(weight, group_size, scale_dtype)
        elif method == "clip-wmse":
            qw = quantize_clip_weight_mse(weight, group_size, ratios, scale_dtype)
        elif method == "clip-calib":
            if calib_X is None:
                raise ValueError("clip-calib requires calib_X")
            from .calibration import quantize_clip_calibrated

            qw = quantize_clip_calibrated(
                weight, calib_X, group_size, ratios, scale_dtype, sweeps
            )
        else:
            raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
        b = None if bias is None else np.asarray(bias, dtype=np.float32)
        if b is not None and b.shape != (qw.out_features,):
            raise ValueError("bias must have shape (out_features,)")
        return cls(weight=qw, bias=b)

    # ---- inference -------------------------------------------------
    def _prepare_x(self, x: np.ndarray) -> tuple[np.ndarray, tuple]:
        x = np.asarray(x, dtype=np.float32)
        if x.shape[-1] != self.weight.in_features:
            raise ValueError(
                f"expected last dim {self.weight.in_features}, got {x.shape[-1]}"
            )
        lead = x.shape[:-1]
        x2 = x.reshape(-1, self.weight.in_features)
        qw = self.weight
        if qw.padding:
            x2 = np.concatenate(
                [x2, np.zeros((x2.shape[0], qw.padding), dtype=np.float32)], axis=1
            )
        return x2, lead

    def _finish(self, y: np.ndarray, lead: tuple) -> np.ndarray:
        if self.bias is not None:
            y = y + self.bias
        return y.reshape(lead + (self.weight.out_features,))

    def forward_dequant(self, x: np.ndarray) -> np.ndarray:
        """Reference: dequantize to dense float32, matmul, add bias."""
        x2, lead = self._prepare_x(x)
        w_hat = self.weight.dequantize()  # (O, K) float32
        if self.weight.padding:
            w_hat = np.concatenate(
                [
                    w_hat,
                    np.zeros(
                        (self.weight.out_features, self.weight.padding),
                        dtype=np.float32,
                    ),
                ],
                axis=1,
            )
        return self._finish(x2 @ w_hat.T, lead)

    def forward_fused(self, x: np.ndarray) -> np.ndarray:
        """Per-group unpack-accumulate; no dense dequantized weight."""
        qw = self.weight
        x2, lead = self._prepare_x(x)
        n = x2.shape[0]
        xg = x2.reshape(n, qw.num_groups, qw.group_size)
        y = np.zeros((n, qw.out_features), dtype=np.float32)
        for j in range(qw.num_groups):
            q = unpack_codes(qw.codes[:, j, :], qw.group_size)  # (O, g) int8
            wj = q.astype(np.float32) * qw.scales[:, j, None].astype(np.float32)
            y += xg[:, j, :] @ wj.T
        return self._finish(y, lead)

    __call__ = forward_fused

    # ---- serialization / accounting --------------------------------
    def to_bytes(self) -> bytes:
        from .serialization import dumps

        return dumps(self)

    @classmethod
    def from_bytes(cls, data: bytes) -> "QuantizedLinear":
        from .serialization import loads

        return loads(cls, data)

    def save(self, path) -> None:
        from .serialization import save

        save(self, path)

    @classmethod
    def load(cls, path) -> "QuantizedLinear":
        from .serialization import load

        return load(cls, path)

    def storage_report(self) -> dict:
        """Byte accounting: header + scales + codes + bias, vs fp32 baseline."""
        from .serialization import header_nbytes

        qw = self.weight
        header = header_nbytes(self)
        scales = qw.scales.size * qw.scales.dtype.itemsize
        codes = qw.codes.size  # uint8 packed nibbles: the true 4-bit payload
        bias = 0 if self.bias is None else self.bias.size * 4
        total = header + scales + codes + bias
        original = qw.out_features * qw.in_features * 4 + (
            0 if self.bias is None else qw.out_features * 4
        )
        weights_only = scales + codes
        return {
            "header_bytes": header,
            "scales_bytes": scales,
            "codes_bytes": codes,
            "bias_bytes": bias,
            "total_bytes": total,
            "original_fp32_bytes": original,
            "total_ratio": total / original,
            "effective_bits_per_weight": 8.0 * total / (qw.out_features * qw.in_features),
            "weights_only_bytes": weights_only,
            "weights_only_ratio": weights_only / (qw.out_features * qw.in_features * 4),
        }
