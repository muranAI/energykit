"""Tests for energykit.forecast.load."""

import numpy as np
import pandas as pd
import pytest

from energykit.forecast import LoadForecaster
from energykit.datasets import load_synthetic_load


@pytest.fixture
def synthetic_load():
    return load_synthetic_load(periods=500, seed=42)


def test_fit_returns_self(synthetic_load):
    model = LoadForecaster(horizon=24)
    result = model.fit(synthetic_load)
    assert result is model


def test_is_fitted_flag(synthetic_load):
    model = LoadForecaster()
    assert not getattr(model, "is_fitted_", False)
    model.fit(synthetic_load)
    assert model.is_fitted_


def test_predict_length(synthetic_load):
    model = LoadForecaster(horizon=24)
    model.fit(synthetic_load)
    fc = model.predict()
    assert len(fc) == 24


def test_predict_custom_horizon(synthetic_load):
    model = LoadForecaster(horizon=24)
    model.fit(synthetic_load)
    fc = model.predict(horizon=48)
    assert len(fc) == 48


def test_predict_index_is_future(synthetic_load):
    model = LoadForecaster(horizon=6)
    model.fit(synthetic_load)
    fc = model.predict()
    assert fc.index[0] > synthetic_load.index[-1]


def test_predict_returns_series(synthetic_load):
    model = LoadForecaster(horizon=12)
    model.fit(synthetic_load)
    fc = model.predict()
    assert isinstance(fc, pd.Series)


def test_predict_values_positive(synthetic_load):
    model = LoadForecaster(horizon=24)
    model.fit(synthetic_load)
    fc = model.predict()
    # Load should be positive (synthetic data is positive)
    assert (fc.values > 0).all()


def test_feature_importance(synthetic_load):
    model = LoadForecaster(horizon=24)
    model.fit(synthetic_load)
    imp = model.feature_importance()
    assert isinstance(imp, pd.Series)
    assert len(imp) > 0
    # Should be sorted descending
    assert (imp.values[:-1] >= imp.values[1:]).all()


def test_raises_without_fit():
    model = LoadForecaster()
    with pytest.raises(RuntimeError, match="fit"):
        model.predict()


def test_raises_non_series():
    model = LoadForecaster()
    with pytest.raises(TypeError):
        model.fit([[1, 2, 3]])


def test_raises_non_datetime_index():
    model = LoadForecaster()
    bad = pd.Series([1.0, 2.0, 3.0], index=[0, 1, 2])
    with pytest.raises(ValueError):
        model.fit(bad)


def test_short_series_raises():
    model = LoadForecaster()
    idx = pd.date_range("2025-01-01", periods=5, freq="h")
    short = pd.Series([1.0] * 5, index=idx)
    with pytest.raises(ValueError, match="too short"):
        model.fit(short)


def test_custom_lags_and_windows(synthetic_load):
    model = LoadForecaster(horizon=6, lags=[1, 24], rolling_windows=[24])
    model.fit(synthetic_load)
    fc = model.predict()
    assert len(fc) == 6
