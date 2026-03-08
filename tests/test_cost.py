"""Tests for energykit.cost — demand charge and imbalance settlement."""

import numpy as np
import pandas as pd
import pytest

from energykit.cost import DemandChargeAnalyzer, DemandChargeResult
from energykit.cost.imbalance import ImbalanceCostCalculator, forecast_value_of_accuracy


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def hourly_power():
    """One year of hourly synthetic power data (kW)."""
    rng = np.random.default_rng(42)
    idx = pd.date_range("2024-01-01", periods=8760, freq="h")
    # Base load ~3 kW with daily / weekly pattern + noise
    hour = np.array(idx.hour)
    base = 2.0 + 1.5 * np.sin(2 * np.pi * hour / 24) + rng.normal(0, 0.3, 8760)
    # Inject a clear peak event in May
    may_peak_loc = np.where((np.array(idx.month) == 5) & (np.array(idx.day) == 14) & (hour == 14))[0]
    if len(may_peak_loc):
        base[may_peak_loc[0]] = 9.0
    return pd.Series(base.clip(0), index=idx, name="load_kw")


@pytest.fixture
def forecast_actual_pair():
    """Fake forecast vs actual (30 days, hourly) with known MAPE ~10%."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-03-01", periods=720, freq="h")
    actual = pd.Series(
        3.0 + 1.0 * np.sin(2 * np.pi * idx.hour / 24) + rng.normal(0, 0.1, 720),
        index=idx,
    ).clip(lower=0.01)
    # Introduce ~10% MAPE via correlated noise
    noise = rng.normal(0, 0.1, 720) * actual.values
    forecast = (actual + noise).clip(lower=0)
    return forecast, actual


# ---------------------------------------------------------------------------
# DemandChargeAnalyzer
# ---------------------------------------------------------------------------

class TestDemandChargeAnalyzer:
    def test_returns_result_type(self, hourly_power):
        result = DemandChargeAnalyzer().analyze(hourly_power)
        assert isinstance(result, DemandChargeResult)

    def test_positive_annual_charge(self, hourly_power):
        result = DemandChargeAnalyzer(demand_rate=12.50).analyze(hourly_power)
        assert result.total_annual_charge_usd > 0

    def test_demand_rate_scaling(self, hourly_power):
        r1 = DemandChargeAnalyzer(demand_rate=10.0).analyze(hourly_power)
        r2 = DemandChargeAnalyzer(demand_rate=20.0).analyze(hourly_power)
        assert pytest.approx(r2.total_annual_charge_usd, rel=1e-3) == 2 * r1.total_annual_charge_usd

    def test_twelve_monthly_periods(self, hourly_power):
        result = DemandChargeAnalyzer().analyze(hourly_power)
        assert result.n_periods == 12

    def test_worst_event_is_may_14(self, hourly_power):
        result = DemandChargeAnalyzer().analyze(hourly_power)
        we = result.worst_event
        assert we.timestamp.month == 5
        assert we.timestamp.day == 14
        assert we.peak_kw == pytest.approx(9.0, abs=0.01)

    def test_battery_savings_df_structure(self, hourly_power):
        result = DemandChargeAnalyzer().analyze(hourly_power)
        df = result.battery_savings_df
        assert "battery_kwh" in df.columns
        assert "annual_savings_usd" in df.columns
        assert "pct_reduction" in df.columns

    def test_larger_battery_more_savings(self, hourly_power):
        result = DemandChargeAnalyzer().analyze(hourly_power)
        savings = result.battery_savings_df["annual_savings_usd"].values
        # Larger battery should give equal or greater savings
        for i in range(len(savings) - 1):
            assert savings[i + 1] >= savings[i] - 1e-6

    def test_peak_hours_filter_reduces_charge(self, hourly_power):
        r_all = DemandChargeAnalyzer(demand_rate=12.50).analyze(hourly_power)
        r_peak = DemandChargeAnalyzer(
            demand_rate=12.50, peak_hours=list(range(9, 18))
        ).analyze(hourly_power)
        # On-peak only ≤ all-hours charge
        assert r_peak.total_annual_charge_usd <= r_all.total_annual_charge_usd

    def test_events_df_has_expected_columns(self, hourly_power):
        result = DemandChargeAnalyzer().analyze(hourly_power)
        df = result.peak_events_df
        assert set(["period", "peak_kw", "peak_timestamp", "demand_charge_usd"]).issubset(df.columns)

    def test_requires_datetimeindex(self):
        bad = pd.Series([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="DatetimeIndex"):
            DemandChargeAnalyzer().analyze(bad)

    def test_zero_demand_rate(self, hourly_power):
        result = DemandChargeAnalyzer(demand_rate=0.0).analyze(hourly_power)
        assert result.total_annual_charge_usd == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# ImbalanceCostCalculator
# ---------------------------------------------------------------------------

class TestImbalanceCostCalculator:
    def test_zero_error_zero_cost(self):
        idx = pd.date_range("2024-01-01", periods=100, freq="h")
        a = pd.Series(np.ones(100), index=idx)
        calc = ImbalanceCostCalculator(imbalance_price=0.10)
        result = calc.compute(a, a)
        assert result.total_cost_usd == pytest.approx(0.0, abs=1e-9)

    def test_perfect_step_error(self):
        """Constant 1 kW over-forecast for 10 hours: cost = 10 kWh × price."""
        idx = pd.date_range("2024-01-01", periods=10, freq="h")
        actual = pd.Series(np.zeros(10), index=idx)
        forecast = pd.Series(np.ones(10), index=idx)
        price = 0.05
        calc = ImbalanceCostCalculator(imbalance_price=price)
        result = calc.compute(forecast, actual)
        expected_cost = 10 * 1.0 * price  # 10 h × 1 kW × $0.05/kWh = $0.50
        assert result.total_cost_usd == pytest.approx(expected_cost, rel=1e-6)

    def test_asymmetric_pricing(self):
        """Over- and under-forecast with different prices."""
        idx = pd.date_range("2024-01-01", periods=4, freq="h")
        actual = pd.Series([1.0, 1.0, 1.0, 1.0], index=idx)
        forecast = pd.Series([2.0, 0.5, 2.0, 0.5], index=idx)
        # over: [1, 0, 1, 0] × price_up × dt
        # under: [0, 0.5, 0, 0.5] × price_down × dt
        price_up = 0.10
        price_down = 0.05
        calc = ImbalanceCostCalculator(
            imbalance_price_up=price_up,
            imbalance_price_down=price_down,
        )
        result = calc.compute(forecast, actual)
        expected = (1.0 + 1.0) * price_up + (0.5 + 0.5) * price_down
        assert result.total_cost_usd == pytest.approx(expected, rel=1e-6)

    def test_returns_cost_breakdowns(self, forecast_actual_pair):
        forecast, actual = forecast_actual_pair
        calc = ImbalanceCostCalculator(imbalance_price=0.08)
        result = calc.compute(forecast, actual)
        assert len(result.cost_by_hour_df) == 24
        assert len(result.cost_by_dow_df) == 7

    def test_positive_annual_estimate(self, forecast_actual_pair):
        forecast, actual = forecast_actual_pair
        calc = ImbalanceCostCalculator(imbalance_price=0.08)
        result = calc.compute(forecast, actual)
        assert result.annual_cost_estimate_usd > 0

    def test_mape_reported(self, forecast_actual_pair):
        forecast, actual = forecast_actual_pair
        result = ImbalanceCostCalculator().compute(forecast, actual)
        assert result.current_mape_pct > 0

    def test_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="same length"):
            ImbalanceCostCalculator().compute([1, 2, 3], [1, 2])

    def test_top_errors_limit(self, forecast_actual_pair):
        forecast, actual = forecast_actual_pair
        result = ImbalanceCostCalculator().compute(forecast, actual)
        assert len(result.top_errors_df) <= 20


# ---------------------------------------------------------------------------
# forecast_value_of_accuracy
# ---------------------------------------------------------------------------

class TestForecastValueOfAccuracy:
    def test_returns_report_with_table(self, forecast_actual_pair):
        forecast, actual = forecast_actual_pair
        report = forecast_value_of_accuracy(actual, forecast, imbalance_price=0.08)
        assert report.improvement_table.shape[0] == 8  # 10–80% reduction steps

    def test_potential_savings_positive(self, forecast_actual_pair):
        forecast, actual = forecast_actual_pair
        report = forecast_value_of_accuracy(actual, forecast, imbalance_price=0.08)
        assert report.potential_annual_savings_usd > 0

    def test_savings_less_than_current_cost(self, forecast_actual_pair):
        forecast, actual = forecast_actual_pair
        report = forecast_value_of_accuracy(actual, forecast, imbalance_price=0.08)
        assert report.potential_annual_savings_usd < report.current_annual_cost_usd

    def test_str_output(self, forecast_actual_pair):
        forecast, actual = forecast_actual_pair
        report = forecast_value_of_accuracy(actual, forecast, imbalance_price=0.08)
        s = str(report)
        assert "FORECAST VALUE ANALYSIS" in s
        assert "MAPE" in s

    def test_custom_target_mape(self, forecast_actual_pair):
        forecast, actual = forecast_actual_pair
        report = forecast_value_of_accuracy(
            actual, forecast, imbalance_price=0.08, target_mape_pct=1.0
        )
        assert report.target_mape_pct == pytest.approx(1.0, abs=0.01)
