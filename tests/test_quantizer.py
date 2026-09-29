import numpy as np
import pytest

from wq4 import CLIP_RATIOS
from wq4.packing import unpack_codes
from wq4.quantizer import (
    quantize_clip_weight_mse,
    quantize_maxabs,
)


def make_W(seed=0, O=8, K=100):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, (O, K)).astype(np.float32)
    W[rng.random((O, K)) < 0.02] *= 20.0  # outliers
    return W


def test_maxabs_scale_and_error_bound():
    W = make_W()
    g = 32
    qw = quantize_maxabs(W, group_size=g)
    assert qw.scales.dtype == np.dtype("<f4")
    assert (qw.scales >= 0).all()
    Wg = W.reshape(W.shape[0], -1)  # unpadded view for maxabs check
    K, pad = W.shape[1], qw.padding
    Wp = np.concatenate([W, np.zeros((W.shape[0], pad), np.float32)], axis=1)
    m = np.abs(Wp.reshape(W.shape[0], -1, g)).max(axis=-1)
    nonzero = m > 0
    np.testing.assert_allclose(
        qw.scales[nonzero], (m[nonzero] / 7).astype(np.float32), rtol=1e-6
    )
    # no clipping at ratio 1.0: |w - w_hat| <= s/2 per element
    W_hat = qw.dequantize()
    s_per_col = np.repeat(qw.scales.astype(np.float32), g, axis=1)[:, :K]
    err = np.abs(W - W_hat)
    assert (err <= s_per_col / 2 + 1e-6).all()


def test_zero_group_sentinel():
    W = np.zeros((4, 64), dtype=np.float32)
    W[:, :32] = 0.5
    qw = quantize_maxabs(W, group_size=32)
    assert (qw.scales[:, 1] == 0.0).all()  # zero group -> sentinel
    assert (qw.scales[:, 0] > 0).all()
    assert not qw.codes[:, 1, :].any()  # all-zero codes
    np.testing.assert_array_equal(qw.dequantize()[:, 32:], 0.0)


def test_padding_recorded_and_exact():
    W = make_W(K=100)
    qw = quantize_maxabs(W, group_size=32)
    assert qw.padding == 28
    assert qw.padded_in_features == 128
    assert qw.codes.shape == (8, 4, 16)
    assert qw.dequantize().shape == (8, 100)


def test_clip_wmse_never_worse_than_maxabs_on_weight_mse():
    W = make_W()
    g = 32
    e_max = np.sum((W - quantize_maxabs(W, g).dequantize()) ** 2)
    e_clip = np.sum((W - quantize_clip_weight_mse(W, g).dequantize()) ** 2)
    assert e_clip <= e_max + 1e-6


def test_clip_uses_fixed_candidate_set():
    assert CLIP_RATIOS[0] == 1.0 and CLIP_RATIOS[-1] == 0.5
    assert len(CLIP_RATIOS) == 11
    with pytest.raises(ValueError):
        quantize_clip_weight_mse(make_W(), 32, ratios=(1.0, 1.5))


def test_determinism():
    W = make_W()
    a = quantize_clip_weight_mse(W, 32)
    b = quantize_clip_weight_mse(W, 32)
    assert np.array_equal(a.codes, b.codes) and np.array_equal(a.scales, b.scales)


def test_scale_dtype_f16():
    W = make_W()
    qw = quantize_maxabs(W, 32, scale_dtype="<f2")
    assert qw.scales.dtype == np.dtype("<f2")
    assert qw.dequantize().shape == W.shape
    with pytest.raises(ValueError):
        quantize_maxabs(W, 32, scale_dtype="<f8")


def test_storage_is_packed_nibbles_not_int8():
    W = make_W(K=64)
    qw = quantize_maxabs(W, 32)
    # 8 rows x 2 groups x 32 weights = 512 weights -> 256 bytes, not 512
    assert qw.codes.dtype == np.uint8
    assert qw.codes.size == 8 * 2 * 16 == 256
