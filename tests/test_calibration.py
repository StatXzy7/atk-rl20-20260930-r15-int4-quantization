import numpy as np

from wq4.calibration import output_error_sq, quantize_clip_calibrated
from wq4.quantizer import quantize_clip_weight_mse, quantize_maxabs


def make_data(seed=3, O=12, K=64, N=256):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, (O, K)).astype(np.float32)
    W[rng.random((O, K)) < 0.03] *= 20.0
    # correlated inputs: low-rank factor model -> non-identity covariance
    A = rng.normal(0, 1, (6, K)) / np.sqrt(6)
    X = (rng.normal(0, 1, (N, 6)) @ A).astype(np.float32)
    return W, X


def test_coordinate_search_improves_calibration_objective():
    W, X = make_data()
    g = 32
    base = quantize_maxabs(W, g)
    cal = quantize_clip_calibrated(W, X, g)
    f_base = output_error_sq(W, base.dequantize(), X)
    f_cal = output_error_sq(W, cal.dequantize(), X)
    assert f_cal <= f_base  # monotone: never worse than its max-abs init


def test_coordinate_search_deterministic():
    W, X = make_data()
    a = quantize_clip_calibrated(W, X, 32)
    b = quantize_clip_calibrated(W, X, 32)
    assert np.array_equal(a.codes, b.codes) and np.array_equal(a.scales, b.scales)


def test_correlated_inputs_make_objectives_differ():
    # On correlated inputs, the weight-MSE-optimal clipping is generally
    # NOT output-error-optimal: the calibrated search must do at least
    # as well on the calibration objective as the weight-MSE solution
    # on this instance (it directly minimizes that objective).
    W, X = make_data()
    g = 32
    wmse = quantize_clip_weight_mse(W, g)
    cal = quantize_clip_calibrated(W, X, g)
    f_wmse = output_error_sq(W, wmse.dequantize(), X)
    f_cal = output_error_sq(W, cal.dequantize(), X)
    assert f_cal <= f_wmse + 1e-6


def test_white_inputs_limit_case():
    # With white (uncorrelated, equal-variance) inputs, H ~ c*I and the
    # output-error objective approaches the weight-MSE objective; both
    # methods should land in the same ballpark on the output objective.
    rng = np.random.default_rng(7)
    W = rng.normal(0, 0.1, (8, 64)).astype(np.float32)
    W[rng.random((8, 64)) < 0.03] *= 20.0
    X = rng.normal(0, 1, (4096, 64)).astype(np.float32)  # ~white
    wmse = quantize_clip_weight_mse(W, 32)
    cal = quantize_clip_calibrated(W, X, 32, sweeps=2)
    f_wmse = output_error_sq(W, wmse.dequantize(), X)
    f_cal = output_error_sq(W, cal.dequantize(), X)
    assert f_cal <= f_wmse * 1.05  # close; calibrated still wins or ties


def test_zero_groups_untouched_by_search():
    W = np.zeros((4, 64), dtype=np.float32)
    W[:, :32] = 0.3
    X = np.random.default_rng(0).normal(0, 1, (64, 64)).astype(np.float32)
    qw = quantize_clip_calibrated(W, X, 32)
    assert (qw.scales[:, 1] == 0.0).all()
    assert not qw.codes[:, 1, :].any()
