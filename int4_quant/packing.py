"""True half-byte (nibble) packing for signed 4-bit integers.

Signed integers are stored in sign-magnitude form, which maps the symmetric
quantization range [-7, 7] bijectively onto nibble codes:

    code(q) = (1 << 3) | |q|        when q < 0
              |q|                   when q >= 0
    decode(c) = (c & 0x7) * (-1 if c & 0x8 else +1)

Nibble order inside every uint8 byte is little-endian: the *first* array
element occupies the low nibble (bits 0..3) and the second element the high
nibble (bits 4..7). A trailing odd element is padded with a zero nibble
(0x0), whose value decodes to 0. This module never uses an int8 tensor as
stand-in for 4-bit storage; the on-disk/in-memory packed representation is
genuinely ceil(n/2) uint8 bytes.
"""

from __future__ import annotations

import numpy as np

INT4_MIN = -7
INT4_MAX = 7

SIGN_BIT = np.uint8(0x8)
NIBBLE_MASK = np.uint8(0xF)


def to_nibble_codes(q: np.ndarray) -> np.ndarray:
    """Map signed integers in [-7, 7] to sign-magnitude nibble codes [0, 15]."""
    q = np.asarray(q)
    if not np.issubdtype(q.dtype, np.integer):
        raise TypeError(f"codes must be integer-valued, got dtype {q.dtype}")
    if np.any(q < INT4_MIN) or np.any(q > INT4_MAX):
        raise ValueError(f"codes must lie in [{INT4_MIN}, {INT4_MAX}]")
    q64 = q.astype(np.int64)
    neg = q64 < 0
    codes = np.abs(q64).astype(np.uint8)
    codes[neg] = codes[neg] | SIGN_BIT
    return codes.astype(np.uint8, copy=False)


def from_nibble_codes(codes: np.ndarray) -> np.ndarray:
    """Map sign-magnitude nibble codes back to signed integers in [-7, 7]."""
    codes = np.asarray(codes, dtype=np.uint8)
    if np.any(codes > 15):
        raise ValueError("nibble codes must lie in [0, 15]")
    mag = codes & np.uint8(0x7)
    neg = (codes & SIGN_BIT) != 0
    q = mag.astype(np.int8)
    q[neg] = -mag[neg].astype(np.int8)
    return q


def pack_nibbles(q: np.ndarray) -> np.ndarray:
    """Pack a flat integer array [-7, 7] into ceil(n/2) uint8 bytes.

    Element order is low-nibble-first. If ``q`` has odd length the final byte
    is completed with a 0x0 high nibble.
    """
    q = np.asarray(q)
    if q.ndim != 1:
        raise ValueError("pack_nibbles expects a flat 1-D array")
    codes = to_nibble_codes(q)
    n = codes.shape[0]
    packed_len = (n + 1) // 2
    padded = np.zeros(packed_len * 2, dtype=np.uint8)
    padded[:n] = codes
    low = padded[0::2]
    high = padded[1::2].astype(np.uint8)
    packed = (low | (high << np.uint8(4))).astype(np.uint8)
    return packed


def unpack_nibbles(packed: np.ndarray, count: int | None = None) -> np.ndarray:
    """Unpack a uint8 buffer into ``count`` signed integers (default: 2 * bytes)."""
    packed = np.asarray(packed, dtype=np.uint8)
    if packed.ndim != 1:
        raise ValueError("unpack_nibbles expects a flat 1-D uint8 buffer")
    low = packed & NIBBLE_MASK
    high = (packed >> np.uint8(4)) & NIBBLE_MASK
    codes = np.empty(packed.shape[0] * 2, dtype=np.uint8)
    codes[0::2] = low
    codes[1::2] = high
    if count is None:
        count = codes.shape[0]
    if count < 0 or count > codes.shape[0]:
        raise ValueError(f"count {count} out of range for {packed.shape[0]} packed bytes")
    return from_nibble_codes(codes[:count])


def _read_nibbles(packed: np.ndarray, start: int, stop: int) -> np.ndarray:
    """Read signed nibbles from a flat uint8 buffer over half-open range [start, stop).

    Works for arbitrary (including odd) starts and stops, so it directly
    implements the odd-tail rule: a group that begins on a high nibble or
    ends on a low nibble costs no wasted bytes.
    """
    nbytes = packed.shape[0]
    if start < 0 or stop < start or stop > 2 * nbytes:
        raise ValueError(f"nibble range [{start}, {stop}) out of bounds ({2 * nbytes})")
    if start == stop:
        return np.empty(0, dtype=np.int8)
    first_byte = start // 2
    last_byte = (stop - 1) // 2
    window = packed[first_byte:last_byte + 1]
    lows = (window & NIBBLE_MASK).astype(np.uint8)
    highs = ((window >> np.uint8(4)) & NIBBLE_MASK).astype(np.uint8)
    inter = np.empty(window.shape[0] * 2, dtype=np.uint8)
    inter[0::2] = lows
    inter[1::2] = highs
    lo = start - 2 * first_byte
    hi = stop - 2 * first_byte
    return from_nibble_codes(inter[lo:hi])


def unpack_nibble_range(packed: np.ndarray, start: int, stop: int) -> np.ndarray:
    """Public alias for the arbitrary-range nibble reader used by packed matmul."""
    return _read_nibbles(np.asarray(packed, dtype=np.uint8), start, stop)
