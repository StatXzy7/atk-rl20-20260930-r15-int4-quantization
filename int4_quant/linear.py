"""A quantized linear layer with two inference paths.

1. ``dequantize_weights`` + normal matmul (the dequantization reference).
2. ``forward_packed``: unpacks each group's nibbles directly and accumulates
   scale * (codes @ x_group); there is no materialized int8/float weight
   matrix. Timings of this path therefore include the unpacking cost.

Both paths accumulate in float64 and add bias, which isolates any difference
to the nibble addressing/packing itself rather than GEMM precision.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .packing import pack_nibbles, unpack_nibbles, unpack_nibble_range
from .quantize import (
    CANDIDATE_CLIP_FRACTIONS,
    _SCALE_DTYPE,
    coordinate_clip_search,
    dequantize,
    fit_weight_clipping,
    num_groups,
    pad_inputs,
    quantize_codes,
    quantize_weight,
)


@dataclass
class FitResult:
    method: str
    thresholds: np.ndarray | None
    mse_history: list | None = None


class QuantizedLinear:
    """Y = X @ W^T + b with W stored as packed 4-bit groups.

    Parameters
    ----------
    packed:
        Flat uint8 buffer, row-major: output channel ``o`` occupies
        ``ceil(k_padded / 2)`` bytes; within each byte the first nibble
        (low bits 0..3) is the earlier input coordinate.
    scales:
        Per-(output, group) scales in the stored ``scale_dtype``.
    """

    def __init__(
        self,
        packed: np.ndarray,
        scales: np.ndarray,
        in_features: int,
        out_features: int,
        group_size: int,
        input_padding: int = 0,
        bias: np.ndarray | None = None,
        scale_dtype: np.dtype = _SCALE_DTYPE,
        fit: FitResult | None = None,
    ):
        packed = np.asarray(packed, dtype=np.uint8)
        scales = np.asarray(scales)
        if packed.ndim != 1:
            raise ValueError("packed must be a flat uint8 array")
        if in_features <= 0 or out_features <= 0:
            raise ValueError("feature dimensions must be positive")
        groups = num_groups(in_features + input_padding, group_size)
        k_padded = groups * group_size
        if groups * group_size != in_features + input_padding:
            raise ValueError("inconsistent input_padding/group_size")
        row_bytes = (k_padded + 1) // 2
        if packed.shape[0] != out_features * row_bytes:
            raise ValueError(
                f"packed has {packed.shape[0]} bytes, expected "
                f"{out_features * row_bytes} ({out_features} rows x {row_bytes})"
            )
        if scales.shape != (out_features, groups):
            raise ValueError("scales must have shape (out_features, num_groups)")
        if scales.dtype != np.dtype(scale_dtype):
            raise ValueError(
                f"scales dtype {scales.dtype} != declared {np.dtype(scale_dtype)}"
            )
        if bias is not None:
            bias = np.asarray(bias, dtype=np.float32)
            if bias.shape != (out_features,):
                raise ValueError("bias must have shape (out_features,)")
        self.packed = packed
        self.scales = scales
        self.bias = bias
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.num_groups = groups
        self.k_padded = k_padded
        self.input_padding = input_padding
        self.scale_dtype = np.dtype(scale_dtype)
        self.row_bytes = row_bytes
        self.fit = fit

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @classmethod
    def from_weights(
        cls,
        weight: np.ndarray,
        group_size: int,
        bias: np.ndarray | None = None,
        method: str = "maxabs",
        x_cal: np.ndarray | None = None,
        fractions: tuple[float, ...] = CANDIDATE_CLIP_FRACTIONS,
        scale_dtype: np.dtype = _SCALE_DTYPE,
    ) -> "QuantizedLinear":
        """Build a layer. ``method`` is one of maxabs/weight_clip/calibrated.

        The calibrated method requires ``x_cal`` and only looks at that fixed
        calibration split; it never sees evaluation inputs.
        """
        weight = np.asarray(weight, dtype=np.float64)
        if weight.ndim != 2:
            raise ValueError("weight must be 2-D")
        out_features, in_features = weight.shape

        if method == "maxabs":
            codes, scales, in_pad = quantize_weight(weight, group_size, scale_dtype)
            fit = FitResult("maxabs", None)
        elif method == "weight_clip":
            codes, scales, thresholds, in_pad = fit_weight_clipping(
                weight, group_size, fractions, scale_dtype
            )
            fit = FitResult("weight_clip", thresholds)
        elif method == "calibrated":
            if x_cal is None:
                raise ValueError("calibrated method requires x_cal")
            codes, scales, thresholds, in_pad, history = coordinate_clip_search(
                weight, x_cal, group_size, fractions, scale_dtype=scale_dtype
            )
            fit = FitResult("calibrated", thresholds, history)
        else:
            raise ValueError(f"unknown method {method!r}")

        packed_rows = [pack_nibbles(codes[o]) for o in range(out_features)]
        packed = np.concatenate(packed_rows) if packed_rows else np.empty(0, np.uint8)
        return cls(
            packed,
            scales,
            in_features,
            out_features,
            group_size,
            in_pad,
            bias,
            scale_dtype,
            fit,
        )

    # ------------------------------------------------------------------ #
    # Inference paths
    # ------------------------------------------------------------------ #
    def unpack_all_codes(self) -> np.ndarray:
        """Unpack every nibble into an int8 (O, K_padded) matrix.

        Each output row occupies its own byte-aligned window of
        ``row_bytes`` bytes; when k_padded is odd, the window's final high
        nibble is an unused alignment pad and is discarded here.
        """
        codes = np.empty((self.out_features, self.k_padded), dtype=np.int8)
        for o in range(self.out_features):
            start = o * self.row_bytes
            codes[o] = unpack_nibbles(
                self.packed[start:start + self.row_bytes], self.k_padded
            )
        return codes

    def dequantize_weights(self) -> np.ndarray:
        """Reference path: full dequantized (O, K_padded) float64 matrix."""
        return dequantize(self.unpack_all_codes(), self.scales, self.group_size)

    def forward_dequant(self, x: np.ndarray) -> np.ndarray:
        """Reference inference: dequantize W, then X @ W^T + bias."""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.in_features:
            raise ValueError(
                f"x must be (N, {self.in_features}), got {x.shape}"
            )
        w_deq = self.dequantize_weights()[:, : self.in_features]
        out = x @ w_deq.T
        if self.bias is not None:
            out = out + self.bias.astype(np.float64)
        return out

    def forward_packed(self, x: np.ndarray) -> np.ndarray:
        """Direct path: unpack nibbles per group and accumulate, no int8 W.

        Each group is unpacked once across all output rows (the nibble slice
        for a group is contiguous because every row has the same width), then
        its contribution is accumulated as
            out += X_group @ (scale_group * codes_group)^T.
        The half-byte unpack is performed explicitly on every call, so the
        measured cost includes it. ``forward_packed_scalar`` is the strictly
        per-(row, group) variant and is kept as a cross-check.
        """
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.in_features:
            raise ValueError(
                f"x must be (N, {self.in_features}), got {x.shape}"
            )
        x_padded = pad_inputs(x, self.k_padded)
        n = x_padded.shape[0]
        out = np.zeros((n, self.out_features), dtype=np.float64)
        # Explicit half-byte unpack of the whole packed buffer on every call;
        # rows are contiguous K_padded-nibble runs. The unpacked codes are
        # then accumulated group by group, so there is never a full dequantized
        # weight matrix multiplied as a single GEMM.
        codes = self.unpack_all_codes()  # int8 (O, K_padded)
        gsz = self.group_size
        for g in range(self.num_groups):
            lo, hi = g * gsz, (g + 1) * gsz
            weighted = codes[:, lo:hi].astype(np.float64) * self.scales[:, g:g + 1]
            out += x_padded[:, lo:hi] @ weighted.T
        if self.bias is not None:
            out = out + self.bias.astype(np.float64)
        return out

    def forward_packed_scalar(self, x: np.ndarray) -> np.ndarray:
        """Reference direct path: one nibble-range read per (row, group)."""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != self.in_features:
            raise ValueError(
                f"x must be (N, {self.in_features}), got {x.shape}"
            )
        x_padded = pad_inputs(x, self.k_padded)
        n = x_padded.shape[0]
        out = np.zeros((n, self.out_features), dtype=np.float64)
        for o in range(self.out_features):
            # Rows are byte aligned: group nibble offsets are relative to this
            # row's own byte window (the row may start mid-byte globally when
            # k_padded is odd, so flat nibble indexing would be wrong).
            row_start = o * self.row_bytes
            row_end = (o + 1) * self.row_bytes
            for g in range(self.num_groups):
                lo = g * self.group_size
                hi = lo + self.group_size
                codes = unpack_nibble_range(
                    self.packed[row_start:row_end], lo, hi
                ).astype(np.float64)
                out[:, o] += float(self.scales[o, g]) * (x_padded[:, lo:hi] @ codes)
        if self.bias is not None:
            out = out + self.bias.astype(np.float64)
        return out

    # Default inference is the direct packed path.
    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.forward_packed(x)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)
