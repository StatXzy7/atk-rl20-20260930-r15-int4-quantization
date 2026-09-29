"""wq4: weight-only 4-bit group quantization for linear layers."""
from .calibration import output_error_sq, quantize_clip_calibrated
from .linear import AGREE_ATOL, AGREE_RTOL, METHODS, QuantizedLinear
from .packing import QMAX, QMIN, pack_codes, unpack_codes
from .quantizer import (
    CLIP_RATIOS,
    QuantizedWeight,
    quantize_clip_weight_mse,
    quantize_maxabs,
)
from .serialization import FORMAT_VERSION, FormatError

__all__ = [
    "AGREE_ATOL",
    "AGREE_RTOL",
    "CLIP_RATIOS",
    "FORMAT_VERSION",
    "METHODS",
    "QMAX",
    "QMIN",
    "FormatError",
    "QuantizedLinear",
    "QuantizedWeight",
    "output_error_sq",
    "pack_codes",
    "quantize_clip_calibrated",
    "quantize_clip_weight_mse",
    "quantize_maxabs",
    "unpack_codes",
]
