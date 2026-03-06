# energykit

[![CI](https://github.com/muranai/energykit/actions/workflows/ci.yml/badge.svg)](https://github.com/muranai/energykit/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/energykit.svg)](https://badge.fury.io/py/energykit)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Downloads](https://static.pepy.tech/badge/energykit)](https://pepy.tech/project/energykit)

**The Python toolkit for energy AI.** Load forecasting, NILM disaggregation, DER optimization, and energy feature engineering — in one package.

```python
from energykit import LoadForecaster

model = LoadForecaster(horizon=24, country="US")
model.fit(meter_data)          # your smart-meter Series
forecast = model.predict()     # next 24 hours, done
```

---

## Why energykit?

Every energy data scientist writes the same 500 lines of boilerplate:
- Custom time-of-day / holiday / lag features that pandas doesn't handle
- Energy-specific accuracy metrics (MAPE, CVRMSE) that scikit-learn doesn't have  
- Battery scheduling code that's just LP but buried in proprietary tools
- NILM baselines that NILMTK makes needlessly complex to run

`energykit` solves all of this with a clean, sklearn-compatible API you can use in one line. No boilerplate. No PhD required.

|                        | energykit | statsforecast | NILMTK | custom code |
|------------------------|:---------:|:-------------:|:------:|:-----------:|
| Energy-specific features | ✅       | ❌           | ❌    | 500 lines   |
| Load forecasting        | ✅       | ✅ (generic) | ❌    | 200 lines   |
| NILM disaggregation     | ✅       | ❌           | ✅ (complex) | 300 lines |
| Battery optimization    | ✅       | ❌           | ❌    | 400 lines   |
| ASHRAE-14 benchmarks    | ✅       | ❌           | ❌    | 100 lines   |
| sklearn compatible      | ✅       | ✅           | ❌    | —           |

---

## Installation

```bash
# Core (feature engineering + benchmarking)
pip install energykit

# With gradient boosting forecasting (recommended)
pip install "energykit[forecast]"

# With DER optimization only
pip install "energykit[optimize]"

# Everything
pip install "energykit[all]"
```

---

## Modules

### `energykit.features` — Energy feature engineering

Stop manually engineering time, lag, and rolling features. `EnergyFeatureExtractor` is a drop-in sklearn transformer that generates **40+ energy-specific features** from any hourly time series:

```python
from energykit.features import EnergyFeatureExtractor

fe = EnergyFeatureExtractor(
    lags=[1, 2, 3, 24, 48, 168],     # 1h, 2h, 3h, 1-day, 2-day, 1-week ago
    rolling_windows=[24, 168],         # 24h and 1-week rolling stats
    cyclical=True,                     # sin/cos encoding (no boundary artifacts)
    country="US",                      # automatic holiday detection
    lat=40.71, lon=-74.00,            # solar elevation angle features
)

X = fe.fit_transform(meter_series)    # pd.Series with DatetimeIndex → pd.DataFrame
```

**Generated features include:**
- Temporal: `hour`, `day_of_week`, `month`, `season`, `week_of_year`, `is_weekend`, `is_holiday`
- Cyclical sin/cos: `hour_sin/cos`, `dow_sin/cos`, `month_sin/cos`, `doy_sin/cos`
- Time-of-use blocks: `tou_off_peak`, `tou_mid_peak`, `tou_on_peak`
- Autoregressive lags: `lag_1h`, `lag_24h`, `lag_168h`, ...
- Rolling statistics: `roll_24h_mean`, `roll_24h_std`, `roll_168h_max`, ...
- Solar position: `solar_elevation_deg`

---

### `energykit.forecast` — Load & energy forecasting

```python
from energykit.forecast import LoadForecaster

# Train on 1 year of hourly data
model = LoadForecaster(
    horizon=24,          # next 24 hours
    country="DE",        # holiday calendar for Germany
    lags=[1, 24, 168],
)
model.fit(load_series)

# Forecast
forecast = model.predict()          # pd.Series, 24 rows
forecast_48h = model.predict(horizon=48)

# Understand the model
top_features = model.feature_importance().head(10)
```

Works out-of-the-box with **LightGBM** (if installed) or falls back to scikit-learn's `HistGradientBoostingRegressor`. Handles missing values automatically.

---

### `energykit.optimize` — DER scheduling

Minimize electricity bills with provably-optimal battery and EV schedules — no PhD, no commercial solver required.

```python
import numpy as np
from energykit.optimize import BatteryScheduler, EVScheduler

# TOU prices ($/kWh), 24 hours
prices = np.array([0.09]*8 + [0.22]*9 + [0.28]*5 + [0.09]*2)

# Battery: charge overnight, discharge at peak
battery = BatteryScheduler(
    capacity_kwh=13.5,   # Tesla Powerwall 2
    max_power_kw=5.0,
    efficiency=0.90,
    initial_soc=0.20,
)
result = battery.optimize(prices, load_kw=baseline_load)
print(f"Daily savings: ${result.savings_usd:.2f}")

# EV: smart charging — reach 80% SoC by 8am at minimum cost
ev = EVScheduler(battery_kwh=75, max_charge_kw=11.0)
ev_result = ev.optimize(
    prices=prices,
    initial_soc=0.15,
    target_soc=0.80,
    departure_step=8,
)
print(f"Smart vs dumb charging savings: ${ev_result.savings_usd:.2f}")
```

---

### `energykit.disaggregate` — NILM (device-level disaggregation)

Break aggregate smart-meter data into per-appliance consumption without sub-meters:

```python
from energykit.disaggregate import ApplianceDisaggregator, EdgeDetector
from energykit.disaggregate.nilm import Appliance

# Unsupervised: detect switching events from step changes
detector = EdgeDetector(step_threshold_w=100)
events = detector.fit_transform(aggregate_watts_series)
print(detector.detected_appliances_)

# Supervised: disaggregate with known appliance signatures
appliances = [
    Appliance("HVAC",          rated_w=3500, duty_cycle=0.55),
    Appliance("Water Heater",  rated_w=4500, duty_cycle=0.25),
    Appliance("Refrigerator",  rated_w=150,  duty_cycle=0.35),
    Appliance("EV Charger",    rated_w=7400, duty_cycle=0.15),
]
disag = ApplianceDisaggregator(appliances)
disag.fit(aggregate_series)
loads = disag.transform(aggregate_series)
print(disag.energy_summary(loads))
```

---

### `energykit.benchmark` — Accuracy metrics

```python
from energykit.benchmark import mape, cvrmse, EnergyForecastBenchmark

# Quick metrics
print(mape(actual, forecast))       # 4.2  (percent)
print(cvrmse(actual, forecast))     # 8.7  (percent, ASHRAE-14)

# Full benchmark report
bench = EnergyForecastBenchmark(actual, forecast, label="LightGBM 24h")
print(bench.summary())

# ASHRAE Guideline 14 compliance check
print(bench.ashrae_check())
# {'cvrmse_pct': 8.7, 'nmbe_pct': 0.3, 'hourly_pass': True, 'message': 'PASS ...'}
```

**Metrics included:** MAPE, sMAPE, MAE, RMSE, CVRMSE (ASHRAE-14), R², peak coincidence, load factor error.

---

### `energykit.datasets` — Public dataset loaders

```python
from energykit.datasets import load_uci_household, load_synthetic_load, load_sample_tou_prices

# UCI Household Power (~2M samples, auto-download & cache)
df = load_uci_household(resample="h")

# Synthetic 1-year load for testing (no download needed)
load = load_synthetic_load(periods=8760, base_kw=2.5, peak_kw=5.0)

# Sample TOU prices
prices = load_sample_tou_prices("residential_us", periods=24)
```

---

## Quick example — full pipeline

```python
import energykit as ek
from energykit.datasets import load_synthetic_load
from energykit.benchmark import EnergyForecastBenchmark

# 1. Load one year of hourly data
load = load_synthetic_load(periods=8760)
train, test = load.iloc[:-168], load.iloc[-168:-144]

# 2. Fit a 24h forecaster
model = ek.LoadForecaster(horizon=24, country="US")
model.fit(train)
forecast = model.predict()

# 3. Benchmark
bench = EnergyForecastBenchmark(test.values, forecast.values)
print(bench.summary())
```

---

## Roadmap

| Version | Features |
|---------|----------|
| v0.1 | Feature engineering, LightGBM forecaster, battery/EV optimizer, NILM baseline, benchmark metrics |
| v0.2 | Price forecasting (`PriceForecaster`), solar/wind generation forecasting, V2G optimization |
| v0.3 | Virtual Power Plant (VPP) aggregation, fleet scheduling, ENTSO-E live data integration |
| v0.4 | Neural forecasters (N-BEATS, PatchTST), probabilistic forecasting (prediction intervals) |
| v1.0 | Stable API, documentation site, benchmark suite vs NILMTK / statsforecast |

---

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md).

Key areas where help is needed:
- More public dataset loaders (ENTSO-E, OpenEI, EPEX)
- Neural network based load forecasters (LSTM, N-HiTS)
- Probabilistic forecasting (quantile regression, conformal prediction)
- V2G (vehicle-to-grid) bidirectional scheduler
- Documentation and tutorials

---

## Citation

If you use energykit in research, please cite:

```bibtex
@software{energykit2026,
  author       = {Muranai},
  title        = {energykit: Python toolkit for energy AI},
  year         = 2026,
  url          = {https://github.com/muranai/energykit},
  version      = {0.1.0}
}
```

---

## License

MIT — see [LICENSE](LICENSE).

Built by [Muranai](https://muranai.com) — enterprise AI for the energy sector.
