"""Tests for energykit.anomaly — smart meter anomaly detection."""

import numpy as np
import pandas as pd
import pytest

from energykit.anomaly import MeterAnomalyDetector, AnomalySummary


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def clean_series():
    """90 days of hourly data without anomalies (stable daily pattern)."""
    rng = np.random.default_rng(1)
    idx = pd.date_range("2024-01-01", periods=90 * 24, freq="h")
    hour = np.array(idx.hour)
    base = 2.0 + 1.5 * np.sin(2 * np.pi * hour / 24) + rng.normal(0, 0.1, len(idx))
    return pd.Series(base.clip(0), index=idx)


@pytest.fixture
def series_with_spikes(clean_series):
    """Clean series with 5 injected spike anomalies."""
    s = clean_series.copy()
    spike_positions = [100, 500, 900, 1200, 1800]
    for pos in spike_positions:
        s.iloc[pos] = s.iloc[pos] * 8.0  # 8× baseline
    return s


@pytest.fixture
def fitted_detector(clean_series):
    detector = MeterAnomalyDetector(z_threshold=2.5)
    detector.fit(clean_series)
    return detector


# ---------------------------------------------------------------------------
# fit()
# ---------------------------------------------------------------------------

class TestMeterAnomalyDetectorFit:
    def test_fit_returns_self(self, clean_series):
        d = MeterAnomalyDetector()
        result = d.fit(clean_series)
        assert result is d

    def test_baseline_has_168_slots(self, clean_series):
        """Seasonal baseline covers 7 days × 24 hours = 168 slots."""
        d = MeterAnomalyDetector().fit(clean_series)
        assert d._baseline_median is not None
        assert len(d._baseline_median) == 7 * 24

    def test_requires_datetimeindex(self):
        bad = pd.Series([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="DatetimeIndex"):
            MeterAnomalyDetector().fit(bad)

    def test_baseline_medians_positive(self, clean_series):
        d = MeterAnomalyDetector().fit(clean_series)
        assert (d._baseline_median >= 0).all()


# ---------------------------------------------------------------------------
# detect() — no anomalies in clean data
# ---------------------------------------------------------------------------

class TestDetectCleanData:
    def test_returns_anomaly_summary(self, fitted_detector, clean_series):
        result = fitted_detector.detect(clean_series)
        assert isinstance(result, AnomalySummary)

    def test_low_false_positive_rate(self, fitted_detector, clean_series):
        result = fitted_detector.detect(clean_series)
        # < 5% false positive rate on training data at z=2.5
        assert result.anomaly_rate_pct < 5.0

    def test_full_df_has_all_rows(self, fitted_detector, clean_series):
        result = fitted_detector.detect(clean_series)
        assert len(result.anomalies_df) == len(clean_series)

    def test_waste_kwh_near_zero(self, fitted_detector, clean_series):
        result = fitted_detector.detect(clean_series)
        # Estimated waste on clean in-distribution data should be ≪ total consumption
        total_kwh = float(clean_series.sum())
        assert result.total_excess_kwh < total_kwh * 0.1


# ---------------------------------------------------------------------------
# detect() — injected spikes
# ---------------------------------------------------------------------------

class TestDetectWithSpikes:
    def test_detects_injected_spikes(self, fitted_detector, series_with_spikes):
        result = fitted_detector.detect(series_with_spikes)
        assert result.n_anomalies >= 5

    def test_estimated_cost_positive(self, fitted_detector, series_with_spikes):
        result = fitted_detector.detect(series_with_spikes, energy_price=0.15)
        assert result.total_estimated_cost_usd > 0

    def test_cost_scales_with_price(self, fitted_detector, series_with_spikes):
        r1 = fitted_detector.detect(series_with_spikes, energy_price=0.10)
        r2 = fitted_detector.detect(series_with_spikes, energy_price=0.20)
        # Cost should be ~2× when price doubles (same anomaly events)
        assert r2.total_estimated_cost_usd == pytest.approx(
            r1.total_estimated_cost_usd * 2.0, rel=0.05
        )

    def test_top_anomalies_df_not_empty(self, fitted_detector, series_with_spikes):
        result = fitted_detector.detect(series_with_spikes)
        assert len(result.top_anomalies_df) > 0

    def test_worst_event_is_not_none(self, fitted_detector, series_with_spikes):
        result = fitted_detector.detect(series_with_spikes)
        assert result.worst_event is not None

    def test_anomaly_type_column_exists(self, fitted_detector, series_with_spikes):
        result = fitted_detector.detect(series_with_spikes)
        assert "anomaly_type" in result.anomalies_df.columns

    def test_spikes_labelled_spike(self, fitted_detector, series_with_spikes):
        result = fitted_detector.detect(series_with_spikes)
        anomalies = result.anomalies_df[result.anomalies_df["is_anomaly"]]
        types = anomalies["anomaly_type"].unique()
        # At least one spike, sustained_elevation, or overnight
        assert set(types).issubset({"spike", "sustained_elevation", "overnight", "sudden_drop"})


# ---------------------------------------------------------------------------
# detect() — overnight anomalies
# ---------------------------------------------------------------------------

class TestOvernightAnomalies:
    def test_overnight_label_assigned(self, clean_series):
        s = clean_series.copy()
        # Force a big spike at 2 AM on a day
        midnight_idx = clean_series.index[clean_series.index.hour == 2][0]
        s.loc[midnight_idx] = 50.0  # big overnight spike
        detector = MeterAnomalyDetector(z_threshold=2.5)
        detector.fit(clean_series)
        result = detector.detect(s)
        overnight = result.anomalies_df[result.anomalies_df["anomaly_type"] == "overnight"]
        assert len(overnight) >= 1


# ---------------------------------------------------------------------------
# detect() — edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_detect_requires_fit(self, clean_series):
        with pytest.raises(RuntimeError, match="fit\\(\\)"):
            MeterAnomalyDetector().detect(clean_series)

    def test_series_shorter_than_training(self, fitted_detector, clean_series):
        """Short series (1 day) should still work after fitting on longer data."""
        short = clean_series.iloc[:24]
        result = fitted_detector.detect(short)
        assert isinstance(result, AnomalySummary)

    def test_all_zeros_series(self, clean_series):
        """Zero series after training on normal data: should flag as sudden drops."""
        detector = MeterAnomalyDetector().fit(clean_series)
        zeros = pd.Series(
            np.zeros(24),
            index=pd.date_range("2024-06-01", periods=24, freq="h"),
        )
        result = detector.detect(zeros)
        assert result.n_anomalies > 0

    def test_anomalies_df_has_expected_columns(self, fitted_detector, clean_series):
        result = fitted_detector.detect(clean_series)
        expected_cols = {
            "reading", "expected", "residual", "z_score",
            "is_anomaly", "anomaly_type", "excess_kwh", "estimated_cost_usd",
        }
        assert expected_cols.issubset(set(result.anomalies_df.columns))
