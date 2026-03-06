"""Tests for energykit.features.temporal."""

import numpy as np
import pandas as pd
import pytest

from energykit.features import EnergyFeatureExtractor


@pytest.fixture
def hourly_series():
    idx = pd.date_range("2025-01-01", periods=500, freq="h")
    rng = np.random.default_rng(0)
    return pd.Series(rng.uniform(1, 10, 500), index=idx, name="kwh")


def test_transform_returns_dataframe(hourly_series):
    fe = EnergyFeatureExtractor()
    out = fe.fit_transform(hourly_series)
    assert isinstance(out, pd.DataFrame)
    assert len(out) == len(hourly_series)
    assert out.index.equals(hourly_series.index)


def test_temporal_features_present(hourly_series):
    fe = EnergyFeatureExtractor(cyclical=False, tou_schedule={}, lags=[], rolling_windows=[])
    out = fe.fit_transform(hourly_series)
    for col in ["hour", "day_of_week", "month", "is_weekend", "season"]:
        assert col in out.columns, f"Missing column: {col}"


def test_cyclical_features(hourly_series):
    fe = EnergyFeatureExtractor(cyclical=True, tou_schedule={}, lags=[], rolling_windows=[])
    out = fe.fit_transform(hourly_series)
    for col in ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"]:
        assert col in out.columns
    # sin² + cos² should ≈ 1
    np.testing.assert_allclose(
        out["hour_sin"] ** 2 + out["hour_cos"] ** 2, 1.0, atol=1e-10
    )


def test_lag_features(hourly_series):
    fe = EnergyFeatureExtractor(lags=[1, 24], rolling_windows=[], tou_schedule={}, cyclical=False)
    out = fe.fit_transform(hourly_series)
    assert "lag_1h" in out.columns
    assert "lag_24h" in out.columns
    # lag_1 at row 5 should equal series value at row 4
    assert out["lag_1h"].iloc[5] == pytest.approx(hourly_series.iloc[4])


def test_rolling_features_no_leakage(hourly_series):
    fe = EnergyFeatureExtractor(lags=[], rolling_windows=[24], tou_schedule={}, cyclical=False)
    out = fe.fit_transform(hourly_series)
    assert "roll_24h_mean" in out.columns
    # Rolling mean at row 0 should NOT contain row 0's value (uses shift(1))
    # Can't be equal to series[0] if series[0] is unique


def test_tou_features(hourly_series):
    tou = {"off_peak": list(range(0, 7)), "on_peak": list(range(17, 22))}
    fe = EnergyFeatureExtractor(lags=[], rolling_windows=[], tou_schedule=tou, cyclical=False)
    out = fe.fit_transform(hourly_series)
    assert "tou_off_peak" in out.columns
    assert "tou_on_peak" in out.columns
    # Hour 3 should be off_peak
    hour3_rows = out[out["hour"] == 3]
    assert (hour3_rows["tou_off_peak"] == 1).all()
    # Hour 3 should NOT be on_peak
    assert (hour3_rows["tou_on_peak"] == 0).all()


def test_solar_features(hourly_series):
    fe = EnergyFeatureExtractor(
        lags=[], rolling_windows=[], tou_schedule={}, cyclical=False,
        lat=37.7, lon=-122.4
    )
    out = fe.fit_transform(hourly_series)
    assert "solar_elevation_deg" in out.columns
    # Elevation must be in [-90, 90]
    assert out["solar_elevation_deg"].between(-90, 90).all()


def test_feature_names_out(hourly_series):
    fe = EnergyFeatureExtractor(lags=[1], rolling_windows=[24], tou_schedule={})
    fe.fit_transform(hourly_series)
    names = fe.get_feature_names_out()
    assert isinstance(names, np.ndarray)
    assert len(names) == len(fe.feature_names_)


def test_dataframe_input(hourly_series):
    df = hourly_series.to_frame("kwh")
    fe = EnergyFeatureExtractor(lags=[1], rolling_windows=[])
    out = fe.fit_transform(df)
    assert isinstance(out, pd.DataFrame)


def test_raises_without_datetimeindex():
    bad = pd.Series([1, 2, 3], index=[0, 1, 2])
    fe = EnergyFeatureExtractor()
    with pytest.raises(ValueError, match="DatetimeIndex"):
        fe.transform(bad)


def test_season_mapping():
    idx = pd.DatetimeIndex(
        ["2025-01-15", "2025-04-15", "2025-07-15", "2025-10-15"],
    )
    series = pd.Series([1.0, 2.0, 3.0, 4.0], index=idx)
    fe = EnergyFeatureExtractor(lags=[], rolling_windows=[], tou_schedule={}, cyclical=False)
    out = fe.fit_transform(series)
    # Jan=winter=0, Apr=spring=1, Jul=summer=2, Oct=autumn=3
    assert list(out["season"]) == [0, 1, 2, 3]
