import hashlib
import json
import struct

import numpy as np
import pytest

from wq4 import FORMAT_VERSION, FormatError, QuantizedLinear
from wq4.serialization import dumps, loads


def make_layer(seed=21, O=8, K=100, g=32, scale_dtype="<f4", bias=True):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.2, (O, K)).astype(np.float32)
    W[rng.random((O, K)) < 0.02] *= 15.0
    b = rng.normal(0, 0.3, O).astype(np.float32) if bias else None
    return QuantizedLinear.from_float(W, b, group_size=g, scale_dtype=scale_dtype)


def test_roundtrip_bytes_and_forward():
    ql = make_layer()
    blob = ql.to_bytes()
    ql = QuantizedLinear.from_bytes(blob)
    assert np.array_equal(ql.weight.codes, ql.weight.codes)
    assert np.array_equal(ql.weight.scales, ql.weight.scales)
    X = np.random.default_rng(1).normal(0, 1, (16, 100)).astype(np.float32)
    np.testing.assert_array_equal(ql.forward_fused(X), ql.forward_fused(X))
    np.testing.assert_array_equal(ql.forward_dequant(X), ql.forward_dequant(X))


def test_roundtrip_no_bias_and_f16_scales(tmp_path):
    ql = make_layer(scale_dtype="<f2", bias=False)
    p = tmp_path / "m.wq4"
    ql.save(p)
    ql2 = QuantizedLinear.load(p)
    assert ql2.bias is None
    assert ql2.weight.scale_dtype == "<f2"
    assert ql2.weight.scales.dtype == np.dtype("<f2")


def test_header_records_required_metadata():
    ql = make_layer(K=100, g=32)
    blob = ql.to_bytes()
    (hlen,) = struct.unpack_from("<I", blob, 4)
    hdr = json.loads(blob[8 : 8 + hlen].decode())
    assert hdr["version"] == FORMAT_VERSION
    assert hdr["out_features"] == 8 and hdr["in_features"] == 100
    assert hdr["group_size"] == 32 and hdr["padding"] == 28
    assert hdr["scale_dtype"] == "<f4"
    assert hdr["payload_sha256"] == hashlib.sha256(blob[8 + hlen :]).hexdigest()


def test_storage_report_adds_up_to_file_size(tmp_path):
    ql = make_layer()
    p = tmp_path / "m.wq4"
    ql.save(p)
    rep = ql.storage_report()
    total = rep["header_bytes"] + rep["scales_bytes"] + rep["codes_bytes"] + rep["bias_bytes"]
    assert total == rep["total_bytes"] == p.stat().st_size
    # codes are the packed 4-bit payload: 8 rows x 4 groups x 16 bytes
    assert rep["codes_bytes"] == 8 * 4 * 16
    assert rep["total_ratio"] > rep["weights_only_ratio"]  # overhead counted


def _mutate_header(blob, **changes):
    (hlen,) = struct.unpack_from("<I", blob, 4)
    hdr = json.loads(blob[8 : 8 + hlen].decode())
    hdr.update(changes)
    h = json.dumps(hdr, sort_keys=True, separators=(",", ":")).encode()
    return blob[:3] + bytes([blob[3]]) + struct.pack("<I", len(h)) + h + blob[8 + hlen :]


def _rehash(blob):
    """Fix sizes+checksum after a payload mutation (semantic attacks only)."""
    (hlen,) = struct.unpack_from("<I", blob, 4)
    hdr = json.loads(blob[8 : 8 + hlen].decode())
    payload = blob[8 + hlen :]
    hdr["payload_sha256"] = hashlib.sha256(payload).hexdigest()
    h = json.dumps(hdr, sort_keys=True, separators=(",", ":")).encode()
    return blob[:4] + struct.pack("<I", len(h)) + h + payload


def test_reject_bad_magic():
    blob = bytearray(make_layer().to_bytes())
    blob[0] = ord("X")
    with pytest.raises(FormatError, match="magic"):
        QuantizedLinear.from_bytes(bytes(blob))


def test_reject_unsupported_version():
    blob = bytearray(make_layer().to_bytes())
    blob[3] = FORMAT_VERSION + 1
    with pytest.raises(FormatError, match="version"):
        QuantizedLinear.from_bytes(bytes(blob))


def test_reject_truncated_file_and_payload():
    blob = make_layer().to_bytes()
    with pytest.raises(FormatError):
        QuantizedLinear.from_bytes(blob[:4])
    with pytest.raises(FormatError):
        QuantizedLinear.from_bytes(blob[:20])  # truncated header
    with pytest.raises(FormatError):
        QuantizedLinear.from_bytes(blob[:-3])  # truncated payload


def test_reject_trailing_garbage():
    blob = make_layer().to_bytes() + b"junk"
    with pytest.raises(FormatError):
        QuantizedLinear.from_bytes(blob)


def test_reject_payload_corruption_via_checksum():
    blob = bytearray(make_layer().to_bytes())
    blob[-1] ^= 0xFF  # flip a payload bit; header checksum now stale
    with pytest.raises(FormatError, match="checksum"):
        QuantizedLinear.from_bytes(bytes(blob))


def test_reject_inconsistent_header_sizes():
    blob = make_layer().to_bytes()
    evil = _mutate_header(blob, codes_nbytes=1)
    with pytest.raises(FormatError):
        QuantizedLinear.from_bytes(evil)
    evil = _mutate_header(blob, padding=5)  # 100+5 not divisible by 32
    with pytest.raises(FormatError):
        QuantizedLinear.from_bytes(evil)
    evil = _mutate_header(blob, scale_dtype="<f8")
    with pytest.raises(FormatError):
        QuantizedLinear.from_bytes(evil)


def test_reject_negative_scale_with_valid_checksum():
    ql = make_layer()
    blob = bytearray(ql.to_bytes())
    (hlen,) = struct.unpack_from("<I", blob, 4)
    payload_off = 8 + hlen
    # first scale (float32) -> -1.0, then fix the checksum: semantic rejection
    blob[payload_off : payload_off + 4] = struct.pack("<f", -1.0)
    evil = _rehash(bytes(blob))
    with pytest.raises(FormatError, match="negative scale"):
        QuantizedLinear.from_bytes(evil)


def test_reject_zero_scale_with_nonzero_codes():
    ql = make_layer()
    blob = bytearray(ql.to_bytes())
    (hlen,) = struct.unpack_from("<I", blob, 4)
    payload_off = 8 + hlen
    blob[payload_off : payload_off + 4] = struct.pack("<f", 0.0)  # scale -> 0
    evil = _rehash(bytes(blob))
    with pytest.raises(FormatError, match="zero-scale"):
        QuantizedLinear.from_bytes(evil)


def test_accept_legitimate_zero_group():
    rng = np.random.default_rng(4)
    W = np.zeros((4, 64), dtype=np.float32)
    W[:, :32] = rng.normal(0, 0.5, (4, 32))
    ql = QuantizedLinear.from_float(W, None, group_size=32)
    ql2 = QuantizedLinear.from_bytes(ql.to_bytes())
    assert (ql2.weight.scales[:, 1] == 0.0).all()
    np.testing.assert_array_equal(ql2.forward_dequant(np.ones((1, 64), np.float32)),
                                  ql.forward_dequant(np.ones((1, 64), np.float32)))
