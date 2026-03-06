"""Tests for energykit.benchmark.metrics."""

import numpy as np
import pytest

from energykit.benchmark import (
    mape,
    smape,
    mae,
    rmse,
    cvrmse,
    r2,
    peak_coincidence,
    load_factor_error,
    EnergyForecastBenchmark,
)


@pytest.fixture
def perfect():
    a = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    return a, a.copy()


@pytest.fixture
def typical():
    rng = np.random.default_rng(0)
    a = rng.uniform(100, 500, 500)
    f = a + rng.normal(0, 20, 500)
    return a, f


def test_mape_perfect(perfect):
    a, f = perfect
    assert mape(a, f) == pytest.approx(0.0, abs=1e-8)


def test_mae_perfect(perfect):
    a, f = perfect
    assert mae(a, f) == pytest.approx(0.0)


def test_rmse_perfect(perfect):
    a, f = perfect
    assert rmse(a, f) == pytest.approx(0.0)


def test_r2_perfect(perfect):
    a, f = perfect
    assert r2(a, f) == pytest.approx(1.0)


def test_mape_known():
    a = np.array([100.0, 100.0, 100.0])
    f = np.array([110.0, 90.0, 100.0])
    # Errors: 10%, 10%, 0% → MAPE = (0.1 + 0.1 + 0) / 3 * 100 ≈ 6.667%
    assert mape(a, f) == pytest.approx(20.0 / 3.0, rel=0.01)


def test_cvrmse_known():
    a = np.array([100.0, 100.0, 100.0, 100.0])
    f = np.array([110.0, 90.0, 110.0, 90.0])
    # RMSE = 10, mean = 100 → CVRMSE = 10%
    assert cvrmse(a, f) == pytest.approx(10.0, rel=0.01)


def test_cvrmse_zero_mean_raises():
    a = np.zeros(5)
    f = np.ones(5)
    with pytest.raises(ValueError, match="zero"):
        cvrmse(a, f)


def test_r2_negative():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    f = np.array([4.0, 3.0, 2.0, 1.0])
    assert r2(a, f) < 0


def test_smape_bounded(typical):
    a, f = typical
    s = smape(a, f)
    assert 0 <= s <= 200


def test_peak_coincidence_perfect(perfect):
    a, f = perfect
    assert peak_coincidence(a, f) == pytest.approx(1.0)


def test_peak_coincidence_range(typical):
    a, f = typical
    pc = peak_coincidence(a, f)
    assert 0.0 <= pc <= 1.0


def test_load_factor_error_perfect(perfect):
    a, f = perfect
    assert load_factor_error(a, f) == pytest.approx(0.0)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="Shape mismatch"):
        mape(np.ones(5), np.ones(6))


def test_benchmark_summary(typical):
    a, f = typical
    bench = EnergyForecastBenchmark(a, f, label="test_model")
    summary = bench.summary()
    assert "test_model" in summary.index
    for col in ["MAE", "RMSE", "MAPE (%)", "R²"]:
        assert col in summary.columns


def test_ashrae_check_pass():
    rng = np.random.default_rng(1)
    a = rng.uniform(100, 200, 1000)
    f = a + rng.normal(0, 3, 1000)  # ~2% CVRMSE → should pass
    bench = EnergyForecastBenchmark(a, f)
    result = bench.ashrae_check()
    assert result["hourly_pass"] is True


def test_ashrae_check_fail():
    rng = np.random.default_rng(2)
    a = rng.uniform(10, 20, 100)
    f = a + rng.normal(10, 5, 100)  # high bias → should fail
    bench = EnergyForecastBenchmark(a, f)
    result = bench.ashrae_check()
    assert result["hourly_pass"] is False
