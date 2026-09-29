import numpy as np
import pytest

from wq4 import AGREE_ATOL, AGREE_RTOL, QuantizedLinear


def make_layer(seed=5, O=16, K=100, g=32, method="clip-wmse", **kw):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, (O, K)).astype(np.float32)
    W[rng.random((O, K)) < 0.02] *= 20.0
    b = rng.normal(0, 0.1, O).astype(np.float32)
    return QuantizedLinear.from_float(W, b, group_size=g, method=method, **kw)


def test_fused_matches_dequant_reference():
    ql = make_layer()
    rng = np.random.default_rng(9)
    X = rng.normal(0, 1, (64, ql.weight.in_features)).astype(np.float32)
    y_ref = ql.forward_dequant(X)
    y_fused = ql.forward_fused(X)
    np.testing.assert_allclose(y_fused, y_ref, rtol=AGREE_RTOL, atol=AGREE_ATOL)


def test_bias_included_in_output():
    rng = np.random.default_rng(11)
    O, K = 8, 64
    W = rng.normal(0, 0.2, (O, K)).astype(np.float32)
    b = rng.normal(0, 1.0, O).astype(np.float32)
    x = rng.normal(0, 1, (4, K)).astype(np.float32)
    ql = QuantizedLinear.from_float(W, b, group_size=32)
    y = ql.forward_fused(x)
    y_nobias = QuantizedLinear.from_float(W, None, group_size=32).forward_fused(x)
    np.testing.assert_allclose(y - y_nobias, np.broadcast_to(b, y.shape), rtol=1e-5, atol=1e-6)
    # and both paths equal dequant-matmul + bias explicitly
    np.testing.assert_allclose(
        ql.forward_dequant(x), x @ ql.weight.dequantize().T + b,
        rtol=1e-5, atol=1e-6,
    )


def test_input_shapes_1d_and_3d():
    ql = make_layer()
    K, O = ql.weight.in_features, ql.weight.out_features
    rng = np.random.default_rng(13)
    x1 = rng.normal(0, 1, (K,)).astype(np.float32)
    x3 = rng.normal(0, 1, (2, 3, K)).astype(np.float32)
    assert ql.forward_fused(x1).shape == (O,)
    assert ql.forward_dequant(x3).shape == (2, 3, O)
    np.testing.assert_allclose(
        ql.forward_fused(x3), ql.forward_dequant(x3), rtol=AGREE_RTOL, atol=AGREE_ATOL
    )


def test_wrong_input_dim_rejected():
    ql = make_layer()
    with pytest.raises(ValueError):
        ql.forward_fused(np.zeros((2, ql.weight.in_features + 1), np.float32))


def test_all_methods_agree_between_paths():
    rng = np.random.default_rng(17)
    O, K = 8, 64
    W = rng.normal(0, 0.2, (O, K)).astype(np.float32)
    X = rng.normal(0, 1, (32, K)).astype(np.float32)
    for method in ("maxabs", "clip-wmse", "clip-calib"):
        ql = QuantizedLinear.from_float(W, None, 32, method=method, calib_X=X)
        np.testing.assert_allclose(
            ql.forward_fused(X), ql.forward_dequant(X),
            rtol=AGREE_RTOL, atol=AGREE_ATOL, err_msg=method,
        )
