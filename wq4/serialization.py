"""Binary container for QuantizedLinear ("WQ4" format, version 1).

Layout
------
    [0:4]                    magic: b"WQ4" + version byte (0x01)
    [4:8]                    uint32 LE: header JSON length H
    [8:8+H]                  header JSON (UTF-8, sorted keys)
    [8+H : ...]              payload = scales | codes | bias

Header JSON fields
------------------
    format, version, out_features, in_features, group_size, padding,
    scale_dtype ("<f4" | "<f2"), bias_dtype ("<f4"), has_bias,
    scales_nbytes, codes_nbytes, bias_nbytes, payload_sha256

Validation on load (anything suspicious -> FormatError)
-------------------------------------------------------
* bad magic, unsupported version, truncated header/payload, trailing
  garbage, unparsable or inconsistent header (sizes, padding, dtype);
* payload SHA-256 mismatch (corruption);
* non-finite or negative scales; scale == 0.0 (zero-group sentinel)
  paired with any non-zero code byte.
"""
from __future__ import annotations

import hashlib
import json
import struct

import numpy as np

from .packing import bytes_per_group
from .quantizer import SCALE_DTYPES, QuantizedWeight

MAGIC = b"WQ4"
FORMAT_VERSION = 1
_HEADER_STRUCT = struct.Struct("<I")  # header JSON length
BIAS_DTYPE = "<f4"


class FormatError(ValueError):
    """Raised when a serialized blob is corrupt or incompatible."""


def _header_dict(ql) -> dict:
    qw = ql.weight
    bias_nbytes = 0 if ql.bias is None else ql.bias.size * 4
    payload = (
        qw.scales.astype(qw.scale_dtype).tobytes()
        + qw.codes.tobytes()
        + (b"" if ql.bias is None else ql.bias.astype(BIAS_DTYPE).tobytes())
    )
    return {
        "format": "wq4-linear",
        "version": FORMAT_VERSION,
        "out_features": qw.out_features,
        "in_features": qw.in_features,
        "group_size": qw.group_size,
        "padding": qw.padding,
        "scale_dtype": qw.scale_dtype,
        "bias_dtype": BIAS_DTYPE,
        "has_bias": ql.bias is not None,
        "scales_nbytes": qw.scales.size * np.dtype(qw.scale_dtype).itemsize,
        "codes_nbytes": qw.codes.size,
        "bias_nbytes": bias_nbytes,
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
    }


def _header_json(ql) -> bytes:
    return json.dumps(_header_dict(ql), sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def header_nbytes(ql) -> int:
    """Total header overhead: magic + version + length + JSON."""
    return len(MAGIC) + 1 + _HEADER_STRUCT.size + len(_header_json(ql))


def dumps(ql) -> bytes:
    qw = ql.weight
    payload = (
        qw.scales.astype(qw.scale_dtype).tobytes()
        + qw.codes.tobytes()
        + (b"" if ql.bias is None else ql.bias.astype(BIAS_DTYPE).tobytes())
    )
    header = _header_json(ql)
    return (
        MAGIC
        + bytes([FORMAT_VERSION])
        + _HEADER_STRUCT.pack(len(header))
        + header
        + payload
    )


def _expect(cond: bool, msg: str) -> None:
    if not cond:
        raise FormatError(msg)


def loads(cls, data: bytes) -> "QuantizedLinear":
    data = bytes(data)
    _expect(len(data) >= 8, "file too short for header")
    _expect(data[:3] == MAGIC, "bad magic (not a WQ4 file)")
    version = data[3]
    _expect(version == FORMAT_VERSION, f"unsupported format version {version}")
    (hlen,) = _HEADER_STRUCT.unpack_from(data, 4)
    _expect(hlen > 0 and 8 + hlen <= len(data), "truncated or invalid header")
    try:
        hdr = json.loads(data[8 : 8 + hlen].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FormatError(f"invalid header JSON: {exc}") from None

    required = {
        "format",
        "version",
        "out_features",
        "in_features",
        "group_size",
        "padding",
        "scale_dtype",
        "bias_dtype",
        "has_bias",
        "scales_nbytes",
        "codes_nbytes",
        "bias_nbytes",
        "payload_sha256",
    }
    _expect(required <= set(hdr), "header missing required fields")
    _expect(hdr["format"] == "wq4-linear", "unexpected format tag")
    _expect(hdr["version"] == FORMAT_VERSION, "header version mismatch")

    O = hdr["out_features"]
    K = hdr["in_features"]
    g = hdr["group_size"]
    pad = hdr["padding"]
    _expect(isinstance(O, int) and isinstance(K, int) and O > 0 and K > 0, "bad shape")
    _expect(isinstance(g, int) and g >= 1, "bad group_size")
    _expect(isinstance(pad, int) and 0 <= pad < g, "bad padding")
    _expect((K + pad) % g == 0, "padding inconsistent with group_size")
    _expect(hdr["scale_dtype"] in SCALE_DTYPES, "unsupported scale_dtype")
    _expect(hdr["bias_dtype"] == BIAS_DTYPE, "unsupported bias_dtype")
    _expect(isinstance(hdr["has_bias"], bool), "bad has_bias")

    G = (K + pad) // g
    sdt = np.dtype(hdr["scale_dtype"])
    _expect(hdr["scales_nbytes"] == O * G * sdt.itemsize, "scales_nbytes mismatch")
    _expect(hdr["codes_nbytes"] == O * G * bytes_per_group(g), "codes_nbytes mismatch")
    _expect(
        hdr["bias_nbytes"] == (O * 4 if hdr["has_bias"] else 0), "bias_nbytes mismatch"
    )

    payload = data[8 + hlen :]
    expect_len = hdr["scales_nbytes"] + hdr["codes_nbytes"] + hdr["bias_nbytes"]
    _expect(len(payload) == expect_len, "payload length mismatch (truncated/garbage)")
    digest = hashlib.sha256(payload).hexdigest()
    _expect(digest == hdr["payload_sha256"], "payload checksum mismatch (corrupt)")

    off = 0
    scales = np.frombuffer(
        payload[off : off + hdr["scales_nbytes"]], dtype=sdt
    ).reshape(O, G).copy()
    off += hdr["scales_nbytes"]
    codes = np.frombuffer(payload[off : off + hdr["codes_nbytes"]], dtype=np.uint8)
    codes = codes.reshape(O, G, bytes_per_group(g)).copy()
    off += hdr["codes_nbytes"]
    bias = None
    if hdr["has_bias"]:
        bias = np.frombuffer(payload[off : off + hdr["bias_nbytes"]], dtype=BIAS_DTYPE)
        bias = bias.reshape(O).astype(np.float32).copy()

    _expect(np.all(np.isfinite(scales)), "non-finite scale")
    _expect(np.all(scales >= 0), "negative scale (scales must be positive)")
    zero_groups = scales == 0.0
    if zero_groups.any():
        _expect(
            not codes[zero_groups].any(),
            "zero-scale group with non-zero codes",
        )

    qw = QuantizedWeight(
        codes=codes,
        scales=scales,
        out_features=O,
        in_features=K,
        group_size=g,
        padding=pad,
        scale_dtype=hdr["scale_dtype"],
    )
    return cls(weight=qw, bias=bias)


def save(ql, path) -> None:
    with open(path, "wb") as fh:
        fh.write(dumps(ql))


def load(cls, path) -> "QuantizedLinear":
    with open(path, "rb") as fh:
        return loads(cls, fh.read())
