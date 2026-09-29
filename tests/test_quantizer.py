import numpy as np
import pytest

from int4_quant import (
    INT4_MIN,
    INT4_MAX,
    CANDIDATE_CLIP_FRACTIONS,
    pack_nibbles,
    unpack_nibbles,
    unpack_nibble_range,
    QuantizedLinear,
    save_linear,
    load_linear,
    storage_bytes,
)
from int4_quant import quantize as qz
from int4_quant.format import (
    QuantFormatError,
    FORMAT_VERSION,
    HEADER_SIZE,
    MAGIC,
    HEADER_FORMAT,
)
import struct


def test_pack_roundtrip_all_levels():
    q = np.arange(INT4_MIN, INT4_MAX + 1, dtype=np.int8)
    packed = pack_nibbles(q)
    assert packed.shape[0] == (q.size + 1) // 2
    back = unpack_nibbles(packed, q.size)
    np.testing.assert_array_equal(back, q)


def test_pack_byte_layout_low_nibble_first():
    q = np.array([1, -2, 7, -7, 0, -1], dtype=np.int8)
    packed = pack_nibbles(q)
    # sign-magnitude: 1 -> 0x1, -2 -> 0xA, 7 -> 0x7, -7 -> 0xF, 0, -1 -> 0x9
    expected = np.array([0xA1, 0xF7, 0x90], dtype=np.uint8)
    np.testing.assert_array_equal(packed, expected)
    np.testing.assert_array_equal(unpack_nibbles(packed, 6), q)


def test_pack_odd_tail_zero_high_nibble():
    q = np.array([3, -4, 5], dtype=np.int8)
    packed = pack_nibbles(q)
    assert packed.shape[0] == 2
    # last byte: low nibble = code(5)=0x5, high nibble is explicit zero pad
    assert packed[1] == 0x05
    back = unpack_nibbles(packed, 3)
    np.testing.assert_array_equal(back, q)


def test_pack_rejects_out_of_range():
    with pytest.raises(ValueError):
        pack_nibbles(np.array([8], dtype=np.int8))
    with pytest.raises(ValueError):
        pack_nibbles(np.array([-8], dtype=np.int8))


def test_unpack_range_odd_offsets():
    q = np.arange(-7, 8, dtype=np.int8)  # 15 values, odd
    packed = pack_nibbles(q)
    # range that starts on a high nibble and ends on a low nibble
    np.testing.assert_array_equal(unpack_nibble_range(packed, 3, 11), q[3:11])
    np.testing.assert_array_equal(unpack_nibble_range(packed, 0, 15), q)
    np.testing.assert_array_equal(unpack_nibble_range(packed, 14, 15), q[14:15])
    np.testing.assert_array_equal(unpack_nibble_range(packed, 1, 2), q[1:2])


def test_zero_group_scale_zero_exact():
    W = np.zeros((2, 8))
    W[0, :4] = np.array([1.0, -2.0, 0.5, -0.25])
    # second group of row 0 and all row 1 are zero
    codes, scales, in_pad = qz.quantize_weight(W, 4)
    assert in_pad == 0
    np.testing.assert_array_equal(scales[0, 1], np.float32(0.0))
    np.testing.assert_array_equal(scales[1], np.zeros(2, dtype=np.float32))
    deq = qz.dequantize(codes, scales, 4)
    np.testing.assert_array_equal(deq[0, 4:], np.zeros(4))
    np.testing.assert_array_equal(deq[1], np.zeros(8))


def test_grouping_along_input_dimension_and_padding():
    W = np.arange(2 * 10, dtype=np.float64).reshape(2, 10) - 5
    codes, scales, in_pad = qz.quantize_weight(W, 4)
    assert in_pad == 2
    assert codes.shape == (2, 12)
    assert scales.shape == (2, 3)
    # padding columns reconstruct to zero
    deq = qz.dequantize(codes, scales, 4)
    np.testing.assert_array_equal(deq[:, 10:], np.zeros((2, 2)))
    # scales positive for nonzero groups
    assert np.all(np.asarray(scales[:, :2], dtype=np.float64) > 0)


def test_codes_stay_in_symmetric_range_under_outlier():
    rng = np.random.default_rng(0)
    W = rng.standard_normal((4, 32))
    W[0, 7] = 50.0  # huge outlier in one group
    codes, scales, _ = qz.quantize_weight(W, 8)
    assert codes.min() >= INT4_MIN and codes.max() <= INT4_MAX
    # max-abs scale saturates the outlier to +-7
    assert codes[0, 7] == INT4_MAX


def test_calibrated_monotone_and_beats_maxabs_on_calibration():
    rng = np.random.default_rng(2)
    W = rng.standard_normal((6, 40))
    W[rng.uniform(size=W.shape) > 0.97] *= 8
    X = rng.standard_normal((128, 40))
    ref = X @ W.T
    layer0 = QuantizedLinear.from_weights(W, 8, method="maxabs")
    layer1 = QuantizedLinear.from_weights(W, 8, method="calibrated", x_cal=X)
    hist = layer1.fit.mse_history
    diffs = np.diff(hist)
    assert np.all(diffs <= 1e-10), hist
    e0 = np.mean((layer0.forward_packed(X) - ref) ** 2)
    e1 = np.mean((layer1.forward_packed(X) - ref) ** 2)
    assert e1 <= e0 + 1e-12


def test_weight_clipping_not_optimal_for_output_with_input_variance():
    # One group, two weights, inputs with very different variances. Weight-SSE
    # treats both coordinates equally; the output objective weights them by
    # input variance, so the per-group optimum can differ.
    a, b = 0.8, 0.55
    rng = np.random.default_rng(3)
    # x0 nearly constant (small variance), x1 high variance
    # coordinate 0 has tiny variance; coordinate 1 has unit variance
    X = np.stack(
        [rng.standard_normal(2048) * 0.02,
         rng.standard_normal(2048)],
        axis=1,
    )
    W = np.array([[a, b]])
    ref = X @ W.T
    wc = QuantizedLinear.from_weights(W, 2, method="weight_clip")
    cal = QuantizedLinear.from_weights(W, 2, method="calibrated", x_cal=X)
    e_wc = np.mean((wc.forward_packed(X) - ref) ** 2)
    e_cal = np.mean((cal.forward_packed(X) - ref) ** 2)
    assert e_cal < e_wc


def test_candidate_grid_fixed_in_module():
    assert 1.0 in CANDIDATE_CLIP_FRACTIONS
    assert CANDIDATE_CLIP_FRACTIONS == tuple(sorted(CANDIDATE_CLIP_FRACTIONS, reverse=True))


def test_packed_is_real_4bit_storage_not_int8():
    layer = QuantizedLinear.from_weights(np.zeros((3, 16)), 8)
    # 3 rows * ceil(16/2) = 24 bytes for 48 weights => 0.5 byte/weight
    assert layer.packed.dtype == np.uint8
    assert layer.packed.size == 24
    assert layer.packed.size * 2 == 3 * 16


def test_forward_paths_equivalent_and_bias():
    rng = np.random.default_rng(4)
    W = rng.standard_normal((5, 33))
    W[rng.uniform(size=W.shape) > 0.98] *= 6
    bias = rng.standard_normal(5)
    X = rng.standard_normal((20, 33))
    for method in ("maxabs", "weight_clip", "calibrated"):
        layer = QuantizedLinear.from_weights(
            W, 8, bias=bias, method=method, x_cal=X
        )
        yd = layer.forward_dequant(X)
        yp = layer.forward_packed(X)
        np.testing.assert_allclose(yd, yp, rtol=1e-9, atol=1e-9)
        ynb = QuantizedLinear.from_weights(
            W, 8, bias=None, method=method, x_cal=X
        ).forward_packed(X)
        # bias is stored/added in float32 on both paths
        np.testing.assert_allclose(yd, ynb + bias.astype(np.float64),
                                   rtol=1e-6, atol=1e-6)


def test_save_load_roundtrip_and_storage_accounting():
    rng = np.random.default_rng(5)
    W = rng.standard_normal((4, 17))
    bias = rng.standard_normal(4)
    layer = QuantizedLinear.from_weights(W, 8, bias=bias)
    blob = save_linear(layer)
    restored = load_linear(blob)
    X = rng.standard_normal((7, 17))
    np.testing.assert_array_equal(restored.forward_packed(X), layer.forward_packed(X))
    assert restored.scale_dtype == np.float32
    assert restored.group_size == 8
    assert restored.input_padding == 7
    stats = storage_bytes(layer)
    # 4 rows x ceil(24/2)=12 bytes = 48; k_padded=24 even so no odd tail;
    # 7 pad columns x 4 rows = 28 padding nibbles
    assert stats.packed_weight_bytes == 48
    assert stats.tail_pad_nibbles == 0
    assert stats.input_padding_nibbles == 28
    assert stats.scales_bytes == 4 * 3 * 4
    assert stats.header_bytes == HEADER_SIZE
    assert (
        stats.total_bytes
        == HEADER_SIZE + stats.scales_bytes + 48 + 4 * 4
    )
    assert len(blob) == stats.total_bytes
    # header itself carries shape, group size, padding and version
    unpacked = struct.unpack(HEADER_FORMAT, blob[:HEADER_SIZE])
    assert unpacked[0] == MAGIC and unpacked[1] == FORMAT_VERSION


def test_load_rejects_corrupt_and_incompatible():
    rng = np.random.default_rng(6)
    layer = QuantizedLinear.from_weights(rng.standard_normal((3, 16)), 8)
    blob = bytearray(save_linear(layer))
    with pytest.raises(QuantFormatError):
        load_linear(bytes(blob[:-1]))                 # truncated
    with pytest.raises(QuantFormatError):
        bad = bytearray(blob); bad[0] ^= 0xFF; load_linear(bytes(bad))  # bad magic
    with pytest.raises(QuantFormatError):
        bad = bytearray(blob); bad[HEADER_SIZE + 3] ^= 0x01; load_linear(bytes(bad))
    with pytest.raises(QuantFormatError):
        bad = bytearray(blob); bad[10] = 99; load_linear(bytes(bad))    # bad version
    with pytest.raises(QuantFormatError):
        bad = bytearray(blob); bad[-1] ^= 0xFF; load_linear(bytes(bad))  # payload CRC
    with pytest.raises(QuantFormatError):
        load_linear(b"")
    with pytest.raises(QuantFormatError):
        load_linear(b"x" * (HEADER_SIZE + 5))


def test_scalar_direct_path_matches_vectorized_and_reference():
    rng = np.random.default_rng(9)
    for in_f, gsz in [(128, 16), (33, 8), (11, 3), (16, 16)]:
        W = rng.standard_normal(size=(6, in_f))
        X = rng.standard_normal(size=(13, in_f))
        bias = rng.standard_normal(size=(6,))
        layer = QuantizedLinear.from_weights(W, gsz, bias=bias,
                                             method="calibrated", x_cal=X)
        ref = X @ W.T + bias
        yv = layer.forward_packed(X)
        ys = layer.forward_packed_scalar(X)
        yd = layer.forward_dequant(X)
        np.testing.assert_allclose(yv, ys, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(yv, yd, rtol=1e-9, atol=1e-9)
        # actual functional error is finite and identical regardless of path
        np.testing.assert_allclose(
            yv, ref, rtol=0, atol=np.abs(yd - ref).max() * 1.000001 + 1e-12
        )


def test_odd_padded_width_rows_are_byte_aligned():
    # K=7, G=3 -> k_padded=9 (odd): each row is 5 bytes with one alignment
    # nibble; the next row must still start on a byte boundary.
    rng = np.random.default_rng(12)
    W = rng.standard_normal(size=(3, 7))
    bias = rng.standard_normal(size=(3,))
    X = rng.standard_normal(size=(5, 7))
    layer = QuantizedLinear.from_weights(W, 3, bias=bias)
    assert layer.k_padded == 9
    assert layer.row_bytes == 5
    assert layer.packed.shape[0] == 3 * 5
    # final high nibble of every row is the alignment pad (0x0)
    for o in range(3):
        assert layer.packed[(o + 1) * layer.row_bytes - 1] & 0xF0 == 0
    stats = storage_bytes(layer)
    assert stats.tail_pad_nibbles == 3
    assert stats.input_padding_nibbles == 3 * 2
    yd = layer.forward_dequant(X)
    np.testing.assert_allclose(layer.forward_packed(X), yd, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(
        layer.forward_packed_scalar(X), yd, rtol=1e-9, atol=1e-9
    )
    restored = load_linear(save_linear(layer))
    np.testing.assert_array_equal(restored.forward_packed(X), layer.forward_packed(X))


def test_odd_group_size_packing_and_direct_path():
    # Odd group size forces a group to begin on a high nibble each row.
    rng = np.random.default_rng(7)
    W = rng.standard_normal((2, 11))
    layer = QuantizedLinear.from_weights(W, 3)
    assert layer.k_padded == 12 and layer.input_padding == 1
    X = rng.standard_normal((9, 11))
    np.testing.assert_allclose(
        layer.forward_dequant(X), layer.forward_packed(X), rtol=1e-9, atol=1e-9
    )
    restored = load_linear(save_linear(layer))
    np.testing.assert_allclose(
        restored.forward_packed(X), layer.forward_packed(X), rtol=1e-6
    )
