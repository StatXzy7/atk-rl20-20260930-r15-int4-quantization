"""Versioned, checksummed on-disk format for a QuantizedLinear layer.

File layout
-----------
Header (little-endian, fixed size, struct "<8sHHHBiiiiiiII"):

    magic        8s   b"INT4QNT\0"
    version      H     format version (1)
    header_size  H     header length in bytes (explicit forward-compat field)
    flags        H     bit0: bias present
    scale_id     B     1=float16 2=float32 3=float64
    group_size   i     group size along input dimension
    in_features  i     unpadded width
    out_features i
    k_padded     i     padded width used for nibble rows
    input_pad    i     zero-padded input columns (k_padded - in_features)
    num_groups   i     groups per output channel
    bias         I     reserved (0); payload flag drives bias presence
    crc32        I     CRC-32 over (header with crc zeroed) + payload

Rows of packed nibbles are BYTE ALIGNED: each output channel occupies
ceil(k_padded/2) bytes. When k_padded is odd the final high nibble of the row
is an unused alignment pad (0x0) so the next row starts on a byte boundary;
group-boundary zero columns are tracked separately by input_pad.

Payload (little-endian raw arrays, concatenated):
    scales      out_features * num_groups * itemsize(scale dtype)
    packed      out_features * ceil(k_padded/2) uint8 bytes
    bias        out_features float32 (only if flags bit0)

The odd-tail rule is explicit: the last nibble of every row is a zero pad
when k_padded is odd; ``input_pad`` additionally records column padding to
the group boundary. Storage accounting includes header, scales, packed
nibbles, and bias.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass

import numpy as np

from .linear import QuantizedLinear

MAGIC = b"INT4QNT\x00"
FORMAT_VERSION = 1
HEADER_FORMAT = "<8sHHHBiiiiiiII"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

SCALE_ID_TO_DTYPE = {1: np.dtype(np.float16), 2: np.dtype(np.float32), 3: np.dtype(np.float64)}
DTYPE_TO_SCALE_ID = {d: i for i, d in SCALE_ID_TO_DTYPE.items()}

FLAG_BIAS = 1 << 0


class QuantFormatError(ValueError):
    """Raised on corrupt or incompatible serialized metadata."""


def _header(*, flags, scale_id, group_size, in_features, out_features,
            k_padded, input_pad, num_groups, reserved=0, crc=0):
    return struct.pack(
        HEADER_FORMAT,
        MAGIC,
        FORMAT_VERSION,
        HEADER_SIZE,
        flags,
        scale_id,
        group_size,
        in_features,
        out_features,
        k_padded,
        input_pad,
        num_groups,
        reserved,
        crc & 0xFFFFFFFF,
    )


def save_linear(layer: QuantizedLinear) -> bytes:
    """Serialize a layer (metadata + scales + packed nibbles [+ bias])."""
    if layer.scale_dtype not in DTYPE_TO_SCALE_ID:
        raise QuantFormatError(f"unsupported scale dtype {layer.scale_dtype}")
    flags = FLAG_BIAS if layer.bias is not None else 0
    scale_id = DTYPE_TO_SCALE_ID[layer.scale_dtype]

    scales = np.ascontiguousarray(layer.scales, dtype=layer.scale_dtype)
    packed = np.ascontiguousarray(layer.packed, dtype=np.uint8)
    expected_packed = layer.out_features * layer.row_bytes
    if packed.shape[0] != expected_packed:
        raise QuantFormatError("packed buffer size inconsistent with metadata")
    if layer.input_padding != layer.k_padded - layer.in_features:
        raise QuantFormatError("input padding disagrees with padded width")

    payload_parts = [scales.tobytes(), packed.tobytes()]
    if layer.bias is not None:
        payload_parts.append(
            np.ascontiguousarray(layer.bias, dtype=np.float32).tobytes()
        )
    payload = b"".join(payload_parts)

    zero_header = _header(
        flags=flags,
        scale_id=scale_id,
        group_size=layer.group_size,
        in_features=layer.in_features,
        out_features=layer.out_features,
        k_padded=layer.k_padded,
        input_pad=layer.input_padding,
        num_groups=layer.num_groups,
        crc=0,
    )
    crc = zlib.crc32(zero_header)
    crc = zlib.crc32(payload, crc) & 0xFFFFFFFF
    header = _header(
        flags=flags,
        scale_id=scale_id,
        group_size=layer.group_size,
        in_features=layer.in_features,
        out_features=layer.out_features,
        k_padded=layer.k_padded,
        input_pad=layer.input_padding,
        num_groups=layer.num_groups,
        crc=crc,
    )
    return header + payload


def load_linear(data: bytes) -> QuantizedLinear:
    """Validate metadata/checksum and reconstruct a QuantizedLinear layer."""
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise QuantFormatError("serialized data must be bytes-like")
    data = bytes(data)
    if len(data) < HEADER_SIZE:
        raise QuantFormatError(
            f"file too short: {len(data)} < {HEADER_SIZE} header bytes"
        )
    try:
        (
            magic,
            version,
            header_size,
            flags,
            scale_id,
            group_size,
            in_features,
            out_features,
            k_padded,
            input_pad,
            groups_field,
            reserved,
            crc,
        ) = struct.unpack(HEADER_FORMAT, data[:HEADER_SIZE])
    except struct.error as exc:  # pragma: no cover - length already checked
        raise QuantFormatError(f"malformed header: {exc}") from exc

    if magic != MAGIC:
        raise QuantFormatError("bad magic: not an INT4QNT file")
    if version != FORMAT_VERSION:
        raise QuantFormatError(
            f"incompatible version {version}; supported {FORMAT_VERSION}"
        )
    if header_size != HEADER_SIZE:
        raise QuantFormatError(f"incompatible header size {header_size}")
    if scale_id not in SCALE_ID_TO_DTYPE:
        raise QuantFormatError(f"unknown scale dtype id {scale_id}")
    if group_size <= 0 or in_features <= 0 or out_features <= 0:
        raise QuantFormatError("corrupt metadata: non-positive dimensions")
    if k_padded != in_features + input_pad:
        raise QuantFormatError("corrupt metadata: k_padded != in_features + pad")
    if input_pad < 0 or input_pad >= group_size:
        raise QuantFormatError("corrupt metadata: invalid input padding")
    if k_padded % group_size != 0:
        raise QuantFormatError("corrupt metadata: padded width not group-aligned")
    expected_groups = k_padded // group_size
    if groups_field != expected_groups:
        raise QuantFormatError("corrupt metadata: bad group count")
    if flags & ~FLAG_BIAS:
        raise QuantFormatError(f"unsupported flags 0x{flags:x}")
    if reserved != 0:
        raise QuantFormatError("corrupt metadata: reserved field nonzero")

    scale_dtype = SCALE_ID_TO_DTYPE[scale_id]
    row_bytes = (k_padded + 1) // 2
    scales_bytes = out_features * groups_field * scale_dtype.itemsize
    packed_bytes = out_features * row_bytes
    bias_bytes = out_features * np.dtype(np.float32).itemsize if flags & FLAG_BIAS else 0
    expected_payload = scales_bytes + packed_bytes + bias_bytes
    if len(data) - HEADER_SIZE != expected_payload:
        raise QuantFormatError(
            f"corrupt length: payload {len(data) - HEADER_SIZE} bytes, "
            f"expected {expected_payload}"
        )

    # Verify checksum with the stored crc field zeroed before trusting data.
    zero_header = data[: HEADER_SIZE - 4] + struct.pack("<I", 0)
    actual_crc = zlib.crc32(zero_header)
    actual_crc = zlib.crc32(data[HEADER_SIZE:], actual_crc) & 0xFFFFFFFF
    if actual_crc != crc:
        raise QuantFormatError("checksum mismatch: payload or header is corrupt")

    offset = HEADER_SIZE
    scales = np.frombuffer(
        data, dtype=scale_dtype, count=out_features * groups_field, offset=offset
    ).reshape(out_features, groups_field).copy()
    offset += scales_bytes
    packed = np.frombuffer(data, dtype=np.uint8, count=packed_bytes, offset=offset).copy()
    offset += packed_bytes
    bias = None
    if flags & FLAG_BIAS:
        bias = np.frombuffer(
            data, dtype=np.float32, count=out_features, offset=offset
        ).copy()

    return QuantizedLinear(
        packed,
        scales,
        in_features,
        out_features,
        group_size,
        input_padding=input_pad,
        bias=bias,
        scale_dtype=scale_dtype,
    )


@dataclass(frozen=True)
class StorageStats:
    header_bytes: int
    scales_bytes: int
    packed_weight_bytes: int
    tail_pad_nibbles: int
    input_padding_nibbles: int
    bias_bytes: int
    total_bytes: int
    dense_fp32_bytes: int
    effective_bits_per_weight: float

    def as_dict(self) -> dict:
        d = {
            "header_bytes": self.header_bytes,
            "scales_bytes": self.scales_bytes,
            "packed_weight_bytes": self.packed_weight_bytes,
            "tail_pad_nibbles": self.tail_pad_nibbles,
            "input_padding_nibbles": self.input_padding_nibbles,
            "bias_bytes": self.bias_bytes,
            "total_bytes": self.total_bytes,
            "dense_fp32_bytes": self.dense_fp32_bytes,
            "effective_bits_per_weight": self.effective_bits_per_weight,
            "total_over_fp32_ratio": self.total_bytes / self.dense_fp32_bytes,
        }
        return d


def storage_bytes(layer: QuantizedLinear) -> StorageStats:
    """Full accounting: header + scales + packed nibbles + bias.

    Every real weight occupies one nibble. Padding is tracked separately:
    ``input_padding_nibbles`` are zero-padded columns to the group boundary,
    ``tail_pad_nibbles`` is one zero high nibble per row when k_padded is
    odd (these share a uint8 byte with a real nibble). The dense baseline is
    float32 weights plus float32 bias.
    """
    scales_bytes = int(layer.scales.size * layer.scale_dtype.itemsize)
    packed_bytes = int(layer.packed.size)
    real_weights = layer.out_features * layer.in_features
    # Rows are byte aligned: an odd k_padded adds one alignment nibble per
    # row so the next row starts on a byte boundary. Group-padding columns
    # are tracked separately as input_padding_nibbles.
    row_nibble_capacity = layer.row_bytes * 2
    tail_pad_nibbles = layer.out_features * (row_nibble_capacity - layer.k_padded)
    input_padding_nibbles = layer.out_features * layer.input_padding
    accounted_nibbles = (
        real_weights + tail_pad_nibbles + input_padding_nibbles
    )
    if accounted_nibbles != packed_bytes * 2:
        raise QuantFormatError(
            f"internal accounting error: {accounted_nibbles} nibbles vs "
            f"{packed_bytes * 2} packed"
        )
    bias_bytes = (
        layer.out_features * np.dtype(np.float32).itemsize
        if layer.bias is not None
        else 0
    )
    total = HEADER_SIZE + scales_bytes + packed_bytes + bias_bytes
    dense = real_weights * np.dtype(np.float32).itemsize + bias_bytes
    eff_bits = 8.0 * total / real_weights
    return StorageStats(
        header_bytes=HEADER_SIZE,
        scales_bytes=scales_bytes,
        packed_weight_bytes=packed_bytes,
        tail_pad_nibbles=tail_pad_nibbles,
        input_padding_nibbles=input_padding_nibbles,
        bias_bytes=bias_bytes,
        total_bytes=total,
        dense_fp32_bytes=dense,
        effective_bits_per_weight=eff_bits,
    )
