"""Weight-only 4-bit grouped quantization for linear layers.

No third-party quantizer is used anywhere in the core quantization or packing
path; everything is built directly on NumPy integer array operations.
"""

from .packing import (
    INT4_MIN,
    INT4_MAX,
    pack_nibbles,
    unpack_nibbles,
    unpack_nibble_range,
)
from .quantize import (
    CANDIDATE_CLIP_FRACTIONS,
    num_groups,
    pad_groups,
    pad_inputs,
    group_maxabs,
    make_scales,
    quantize_codes,
    dequantize,
    quantize_weight,
    fit_weight_clipping,
    coordinate_clip_search,
)
from .linear import QuantizedLinear
from . import format as format_mod
from .format import save_linear, load_linear, storage_bytes, StorageStats

__all__ = [
    "INT4_MIN",
    "INT4_MAX",
    "CANDIDATE_CLIP_FRACTIONS",
    "num_groups",
    "pad_groups",
    "pad_inputs",
    "group_maxabs",
    "make_scales",
    "quantize_codes",
    "dequantize",
    "quantize_weight",
    "fit_weight_clipping",
    "coordinate_clip_search",
    "pack_nibbles",
    "unpack_nibbles",
    "unpack_nibble_range",
    "QuantizedLinear",
    "format_mod",
    "save_linear",
    "load_linear",
    "storage_bytes",
    "StorageStats",
]
