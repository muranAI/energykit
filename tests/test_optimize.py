"""Tests for energykit.optimize.der."""

import numpy as np
import pytest

from energykit.optimize import BatteryScheduler, EVScheduler


@pytest.fixture
def tou_prices():
    """24-hour TOU prices: cheap off-peak (0-7, 22-23), expensive on-peak (17-21)."""
    return np.array(
        [0.09] * 7 + [0.14] * 10 + [0.28] * 5 + [0.09] * 2,
        dtype=float,
    )


class TestBatteryScheduler:
    def test_returns_schedule_result(self, tou_prices):
        batt = BatteryScheduler(capacity_kwh=10, max_power_kw=5)
        result = batt.optimize(tou_prices)
        assert result.status == "optimal"
        assert len(result.schedule_df) == 24

    def test_savings_positive_with_tou(self, tou_prices):
        load = np.full(24, 3.0)  # constant 3 kW load
        batt = BatteryScheduler(capacity_kwh=13.5, max_power_kw=5.0)
        result = batt.optimize(tou_prices, load_kw=load)
        # Battery should reduce cost vs baseline
        assert result.savings_usd >= 0

    def test_soc_within_bounds(self, tou_prices):
        batt = BatteryScheduler(
            capacity_kwh=10, max_power_kw=3,
            min_soc=0.10, max_soc=0.95, initial_soc=0.5
        )
        result = batt.optimize(tou_prices)
        soc = result.schedule_df["soc_kwh"]
        assert (soc >= batt.min_soc * batt.capacity_kwh - 1e-6).all()
        assert (soc <= batt.max_soc * batt.capacity_kwh + 1e-6).all()

    def test_charge_nonnegative(self, tou_prices):
        batt = BatteryScheduler(capacity_kwh=10, max_power_kw=5)
        result = batt.optimize(tou_prices)
        assert (result.schedule_df["charge_kw"] >= -1e-9).all()
        assert (result.schedule_df["discharge_kw"] >= -1e-9).all()

    def test_invalid_efficiency_raises(self):
        with pytest.raises(ValueError):
            BatteryScheduler(capacity_kwh=10, max_power_kw=5, efficiency=1.5)

    def test_invalid_soc_bounds_raises(self):
        with pytest.raises(ValueError):
            BatteryScheduler(capacity_kwh=10, max_power_kw=5, min_soc=0.9, max_soc=0.5)

    def test_flat_prices_near_zero_savings(self):
        flat = np.full(24, 0.15)
        load = np.full(24, 2.0)
        # initial_soc=0 so battery has no stored energy to arbitrage
        batt = BatteryScheduler(capacity_kwh=10, max_power_kw=5, initial_soc=0.0, min_soc=0.0)
        result = batt.optimize(flat, load_kw=load)
        # With flat prices and empty battery, no cost arbitrage is possible
        assert result.savings_usd <= 0.01


class TestEVScheduler:
    def test_returns_schedule_result(self, tou_prices):
        ev = EVScheduler(battery_kwh=60, max_charge_kw=7.4)
        result = ev.optimize(tou_prices, initial_soc=0.20, target_soc=0.80)
        assert result.status == "optimal"

    def test_reaches_target_soc(self, tou_prices):
        ev = EVScheduler(battery_kwh=60, max_charge_kw=11.0, efficiency=1.0)
        target = 0.80
        result = ev.optimize(tou_prices, initial_soc=0.20, target_soc=target, departure_step=23)
        needed = (target - 0.20) * 60
        delivered = result.schedule_df["charge_kw"].sum()
        assert delivered == pytest.approx(needed, rel=0.05)

    def test_no_charging_after_departure(self, tou_prices):
        ev = EVScheduler(battery_kwh=70, max_charge_kw=7.4)
        dep_step = 16
        result = ev.optimize(tou_prices, initial_soc=0.5, target_soc=0.8, departure_step=dep_step)
        after_departure = result.schedule_df["charge_kw"].iloc[dep_step:]
        assert (after_departure == 0).all()

    def test_soc_monotone_increasing(self, tou_prices):
        ev = EVScheduler(battery_kwh=60, max_charge_kw=7.4)
        result = ev.optimize(tou_prices, initial_soc=0.20, target_soc=0.80)
        soc = result.schedule_df["soc_kwh"]
        assert (soc.diff().dropna() >= -1e-9).all()
