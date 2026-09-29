"""Nibble packing for signed 4-bit quantization codes.

Encoding
--------
Each signed integer code ``q`` in ``[-7, 7]`` is stored as a 4-bit
two's-complement nibble ``n = q & 0xF``:

    0 -> 0x0   1 -> 0x1  ...   7 -> 0x7
   -1 -> 0xF  -2 -> 0xE  ...  -7 -> 0x9

The decoder is a total function on nibbles: ``n >= 8`` decodes to
``n - 16``.  The encoder never emits ``0x8`` (the range is symmetric
``[-7, 7]``, not ``[-8, 7]``); if a decoder meets ``0x8`` it maps to -8.

Byte order (within one group)
-----------------------------
Codes are packed two per byte in input-dimension order: the code at even
position ``2k`` goes into the LOW nibble (bits 3..0) of byte ``k``, the
code at odd position ``2k+1`` goes into the HIGH nibble (bits 7..4)::

    byte[k] = nibble(code[2k]) | (nibble(code[2k+1]) << 4)

Odd tail rule
-------------
If ``group_size`` is odd, the last byte of the group holds the final code
in its low nibble and ``0x0`` in its high nibble.  Decoders must ignore
that high nibble.  Even group sizes are recommended; odd sizes work.

Groups are laid out row-major as ``(out_features, num_groups,
bytes_per_group)`` uint8.  Only this packed uint8 array is persisted --
no int8 tensor is ever stored, so the on-disk cost is a true 4 bits per
weight (plus scales / header / padding).
"""
from __future__ import annotations

import numpy as np

QMIN = -7
QMAX = 7


def bytes_per_group(group_size: int) -> int:
    if group_size < 1:
        raise ValueError("group_size must be >= 1")
    return (group_size + 1) // 2


def pack_codes(codes: np.ndarray) -> np.ndarray:
    """Pack signed codes ``(..., g)`` in [-7, 7] into uint8 ``(..., ceil(g/2))``."""
    codes = np.asarray(codes)
    if codes.shape[-1] < 1:
        raise ValueError("empty group")
    c = codes.astype(np.int16)
    if c.min() < QMIN or c.max() > QMAX:
        raise ValueError(f"codes out of range [{QMIN}, {QMAX}]")
    nib = (c & 0xF).astype(np.uint8)
    lo = nib[..., 0::2]
    hi = nib[..., 1::2]
    if hi.shape[-1] < lo.shape[-1]:  # odd group size: zero high nibble tail
        pad = np.zeros(hi.shape[:-1] + (1,), dtype=np.uint8)
        hi = np.concatenate([hi, pad], axis=-1)
    return (lo | (hi << np.uint8(4))).astype(np.uint8)


def unpack_codes(packed: np.ndarray, group_size: int) -> np.ndarray:
    """Unpack uint8 ``(..., ceil(g/2))`` back to signed int8 codes ``(..., g)``."""
    packed = np.asarray(packed)
    if packed.dtype != np.uint8:
        raise ValueError("packed codes must be uint8")
    if packed.shape[-1] != bytes_per_group(group_size):
        raise ValueError("packed last dimension does not match group_size")
    lo = packed & np.uint8(0xF)
    hi = packed >> np.uint8(4)
    out = np.empty(packed.shape[:-1] + (2 * packed.shape[-1],), dtype=np.uint8)
    out[..., 0::2] = lo
    out[..., 1::2] = hi
    n = out[..., :group_size].astype(np.int16)
    return np.where(n >= 8, n - 16, n).astype(np.int8)
