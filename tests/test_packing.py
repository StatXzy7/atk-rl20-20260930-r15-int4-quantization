import numpy as np
import pytest

from wq4.packing import QMAX, QMIN, bytes_per_group, pack_codes, unpack_codes


def test_known_byte_layout_low_nibble_first():
    # code[0] -> low nibble, code[1] -> high nibble; two's complement nibbles
    q = np.array([[1, -1], [-7, 7], [0, -7], [7, 0]], dtype=np.int8)
    p = pack_codes(q)
    assert p.dtype == np.uint8
    assert p[:, 0].tolist() == [0xF1, 0x79, 0x90, 0x07]


def test_odd_tail_high_nibble_zero():
    q = np.array([[1, 2, 3]], dtype=np.int8)
    p = pack_codes(q)
    assert p.tolist() == [[0x21, 0x03]]  # last byte: low=3, high=0
    assert (unpack_codes(p, 3) == q).all()


@pytest.mark.parametrize("g", [1, 2, 3, 5, 16, 31, 32, 33, 64, 128])
def test_roundtrip_random(g):
    rng = np.random.default_rng(g)
    q = rng.integers(QMIN, QMAX + 1, size=(4, 3, g), dtype=np.int8)
    p = pack_codes(q)
    assert p.shape == (4, 3, bytes_per_group(g))
    assert p.dtype == np.uint8
    assert np.array_equal(unpack_codes(p, g), q)


def test_extremes_roundtrip():
    q = np.array([[QMIN, QMAX, 0, -1, 1]], dtype=np.int8)
    assert np.array_equal(unpack_codes(pack_codes(q), 5), q)


def test_decoder_maps_0x8_to_minus_8():
    # encoder never emits 0x8 (range is [-7, 7]) but the decoder is total
    p = np.array([[0x08]], dtype=np.uint8)
    assert unpack_codes(p, 1)[0, 0] == -8


def test_pack_rejects_out_of_range():
    with pytest.raises(ValueError):
        pack_codes(np.array([[8]], dtype=np.int8))
    with pytest.raises(ValueError):
        pack_codes(np.array([[-9]], dtype=np.int8))


def test_unpack_rejects_wrong_size():
    with pytest.raises(ValueError):
        unpack_codes(np.zeros((2, 2), dtype=np.uint8), 5)  # g=5 needs 3 bytes
    with pytest.raises(ValueError):
        unpack_codes(np.zeros((2, 3), dtype=np.int8), 4)  # not uint8
