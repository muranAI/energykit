"""
energykit quickstart — end-to-end demo in ~50 lines.

Demonstrates:
  1. Synthetic load data generation
  2. Energy feature engineering
  3. Load forecasting (24-hour ahead)
  4. Battery optimization on real TOU prices
  5. Forecast accuracy benchmarking

Run: python examples/quickstart.py
"""

import numpy as np
import pandas as pd

import energykit as ek
from energykit.datasets import load_synthetic_load, load_sample_tou_prices
from energykit.optimize import BatteryScheduler
from energykit.benchmark import EnergyForecastBenchmark

print("=" * 60)
print("  energykit — Energy AI Toolkit  |  Quick Start Demo")
print("=" * 60)

# ── 1. Load Data ──────────────────────────────────────────────
print("\n[1/4] Generating synthetic load profile (1 year, hourly)...")
load = load_synthetic_load(periods=8760, seed=42)
print(f"      {len(load)} hourly observations — {load.index[0].date()} to {load.index[-1].date()}")
print(f"      Mean: {load.mean():.2f} kW  |  Peak: {load.max():.2f} kW  |  Min: {load.min():.2f} kW")

# ── 2. Feature Engineering ───────────────────────────────────
print("\n[2/4] Extracting energy features...")
fe = ek.EnergyFeatureExtractor(
    lags=[1, 2, 3, 24, 48, 168],
    rolling_windows=[24, 168],
    cyclical=True,
    lat=40.71,   # New York
    lon=-74.00,
)
features = fe.fit_transform(load)
print(f"      Generated {features.shape[1]} features for {features.shape[0]} samples.")
print(f"      Features: {', '.join(features.columns[:8].tolist())} ...")

# ── 3. Load Forecasting ───────────────────────────────────────
print("\n[3/4] Training 24-hour load forecaster...")
train = load.iloc[:-48]
test  = load.iloc[-48:-24]   # hold-out: next 24h

model = ek.LoadForecaster(horizon=24, lags=[1, 2, 3, 24, 48, 168])
model.fit(train)
forecast = model.predict()

bench = EnergyForecastBenchmark(test.values, forecast.values, label="LightGBM 24h")
summary = bench.summary()
print(f"\n      Forecast accuracy:")
print(f"        MAPE     : {summary['MAPE (%)'].iloc[0]:.2f} %")
print(f"        CVRMSE   : {summary['CVRMSE (%)'].iloc[0]:.2f} %")
print(f"        R²       : {summary['R²'].iloc[0]:.4f}")

ashrae = bench.ashrae_check()
print(f"        ASHRAE-14: {ashrae['message']}")

top_features = model.feature_importance().head(5)
print(f"\n      Top 5 most important features:")
for feat, imp in top_features.items():
    print(f"        {feat:<25} {imp:.0f}")

# ── 4. Battery Optimization ───────────────────────────────────
print("\n[4/4] Optimizing 13.5 kWh battery on TOU pricing...")
prices = load_sample_tou_prices("residential_us", periods=24)
daily_load = load.iloc[:24].values  # first day's load

battery = BatteryScheduler(
    capacity_kwh=13.5,  # Tesla Powerwall 2
    max_power_kw=5.0,
    efficiency=0.90,
    initial_soc=0.20,
)
result = battery.optimize(prices, load_kw=daily_load)

print(f"      Status         : {result.status}")
print(f"      Baseline cost  : ${result.baseline_cost_usd:.4f}")
print(f"      Optimized cost : ${result.total_cost_usd:.4f}")
print(f"      Daily savings  : ${result.savings_usd:.4f}")
print(f"      Annual savings : ~${result.savings_usd * 365:.0f}/year")

print("\n" + "=" * 60)
print("  Done. See energykit docs at https://energykit.readthedocs.io")
print("=" * 60)
