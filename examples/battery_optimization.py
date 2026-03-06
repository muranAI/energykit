"""
Battery optimization example — showing real savings from smart TOU scheduling.

Shows:
- Baseline (uncontrolled) vs optimized battery dispatch
- Hourly SoC curve
- Per-hour cost breakdown

Run: python examples/battery_optimization.py
"""

import numpy as np
import pandas as pd

from energykit.optimize import BatteryScheduler, EVScheduler
from energykit.datasets import load_sample_tou_prices, load_synthetic_load

print("Battery & EV Scheduling Demo")
print("-" * 40)

# 24-hour TOU prices
prices = load_sample_tou_prices("residential_us", periods=24)
load_kw = load_synthetic_load(periods=24).values

print("\n--- Residential Battery (Tesla Powerwall 2) ---")
battery = BatteryScheduler(
    capacity_kwh=13.5,
    max_power_kw=5.0,
    efficiency=0.90,
    initial_soc=0.10,
    min_soc=0.10,
    max_soc=0.95,
)
result = battery.optimize(prices, load_kw=load_kw)
print(f"Baseline cost : ${result.baseline_cost_usd:.3f}")
print(f"Optimized cost: ${result.total_cost_usd:.3f}")
print(f"Daily savings : ${result.savings_usd:.3f}  (annual ~${result.savings_usd*365:.0f})")
print(f"\nHourly schedule:")
print(result.schedule_df[["price_per_kwh", "charge_kw", "discharge_kw", "soc_pct"]].round(2).to_string())

print("\n--- Smart EV Charging (75 kWh, Level 2 EVSE) ---")
ev = EVScheduler(
    battery_kwh=75.0,
    max_charge_kw=11.0,
    efficiency=0.92,
)
ev_result = ev.optimize(
    prices=prices,
    initial_soc=0.15,
    target_soc=0.85,
    departure_step=18,  # must be ready by 18:00
)
print(f"Smart charging cost    : ${ev_result.total_cost_usd:.3f}")
print(f"Dumb charging baseline : ${ev_result.baseline_cost_usd:.3f}")
print(f"Savings                : ${ev_result.savings_usd:.3f}")
print(f"\nCharging profile (kW by hour, departure at step 18):")
print(ev_result.schedule_df[["price_per_kwh", "charge_kw", "soc_pct"]].round(2).to_string())
