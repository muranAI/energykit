<p align="center">
  <a href="https://muranai.com">
    <img src="https://muranai.com/images/logo_white.png" alt="Muranai" width="220"/>
  </a>
</p>

<h1 align="center">energykit</h1>

<p align="center">
  <a href="https://github.com/muranai/energykit/actions/workflows/ci.yml"><img src="https://github.com/muranai/energykit/actions/workflows/ci.yml/badge.svg" alt="CI"/></a>
  <a href="https://badge.fury.io/py/energykit"><img src="https://badge.fury.io/py/energykit.svg" alt="PyPI version"/></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.9+-blue.svg" alt="Python 3.9+"/></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License: MIT"/></a>
  <a href="https://pepy.tech/project/energykit"><img src="https://img.shields.io/pypi/dm/energykit?label=downloads&color=blue" alt="Downloads"/></a>
</p>

<p align="center">
  <strong>The Python toolkit that turns energy data into dollars.</strong><br/>
  Most energy tools stop at the metric. energykit goes all the way to the money.
</p>

<p align="center">
  <a href="https://pypi.org/project/energykit/">📦 PyPI</a> &nbsp;·&nbsp;
  <a href="https://github.com/muranAI/energykit/issues">🐛 Issues</a> &nbsp;·&nbsp;
  <a href="https://muranai.com">🌐 Muranai.com</a>
</p>

```python
import energykit as ek

report = ek.diagnose(your_meter_data)
```

```
╔════════════════════════════════════════════════════════════════════╗
║           ⚡  ENERGYKIT  |  ENERGY FINANCIAL DIAGNOSIS  ⚡        ║
╠════════════════════════════════════════════════════════════════════╣
║  Period  : Jan 2025 → Dec 2025   (8,760 readings)                  ║
║  Total   : 26,461 kWh   Avg: 3.02 kW   Peak: 5.86 kW               ║
╠════════════════════════════════════════════════════════════════════╣
║  💡 DEMAND CHARGE RISK                                             ║
║  Peak event  : May 14 @ 14:00  →  5.86 kW                          ║
║  Est. annual demand charge : $702  (@$12.50/kW)                    ║
║  Battery [10 kWh / 5 kW]   : save $677/yr  (96%)                   ║
╠════════════════════════════════════════════════════════════════════╣
║  🔍 ANOMALY DETECTION                                              ║
║  Anomalies : 23 events  (0.26% of readings)                        ║
║  Est. waste : 312 kWh  →  $47  over the period                     ║
║  Top anomaly : Mar 12 @ 02:00 - overnight  +87 kWh  ($13)          ║
╠════════════════════════════════════════════════════════════════════╣
║  🔋 DER OPPORTUNITY  (battery dispatch optimisation)               ║ 
║  Battery [13.5 kWh / 5 kW] annual savings : $729                   ║
║  Estimated payback (@$8,000 install)       : 11.0 yr               ║
╠════════════════════════════════════════════════════════════════════╣
║  📊 TOTAL ADDRESSABLE SAVINGS                                      ║
║  Anomaly correction   :      $47/yr                                ║
║  Demand charge opt.   :     $677/yr  [10 kWh battery]              ║
║  DER dispatch         :     $729/yr  [13.5 kWh]                    ║
║  ────────────────────────────────────────────────────────────────  ║
║  TOTAL POTENTIAL      :   $1,453/yr  (37% of annual spend)         ║
╚════════════════════════════════════════════════════════════════════╝
```

---

## The before / after

**Before energykit:**  
*"I have a 7% MAPE forecast."*

**After energykit:**  
*"My 7% MAPE costs $234,000/year in imbalance settlement - and 80% of that comes from 50 peak hours. My worst demand charge event was May 14 at 2pm ($879 that month). A 10 kWh battery pays back in 3.8 years. Total addressable savings: $1,453/year."*

That's the difference between a technical metric and a business case.

---

## Installation

> **Requires Python 3.9 or higher.** Core dependencies: `numpy`, `pandas 2.x`, `scikit-learn`, `scipy`.

### Quick install (all platforms)

```bash
pip install energykit
```

```bash
# + LightGBM forecasting (recommended for best accuracy)
pip install "energykit[forecast]"

# + Everything (forecast, DER optimizer, dataset downloaders)
pip install "energykit[all]"
```

---

### Windows

**Using Command Prompt or PowerShell:**

```powershell
# 1. Check your Python version (must be 3.9+)
python --version

# 2. Install energykit
pip install energykit

# 3. Verify the install
python -c "import energykit; print(energykit.__version__)"
```

**Using Anaconda / Miniconda (recommended on Windows):**

```powershell
# Create a dedicated environment
conda create -n energykit-env python=3.11
conda activate energykit-env

# Install energykit with all extras
pip install "energykit[all]"
```

> **No Python yet?** Download from [python.org/downloads](https://www.python.org/downloads/) or install [Anaconda](https://www.anaconda.com/download). Make sure to check **"Add Python to PATH"** during installation.

---

### macOS

**Using the built-in Terminal:**

```bash
# 1. Check your Python version
python3 --version

# 2. Install energykit (use pip3 on macOS)
pip3 install energykit

# 3. Verify
python3 -c "import energykit; print(energykit.__version__)"
```

**Using Homebrew + pyenv (recommended):**

```bash
# Install pyenv to manage Python versions
brew install pyenv
pyenv install 3.11
pyenv global 3.11

# Install energykit
pip install "energykit[all]"
```

**Using Anaconda on macOS:**

```bash
conda create -n energykit-env python=3.11
conda activate energykit-env
pip install "energykit[all]"
```

> **No Python yet?** `brew install python@3.11` or download from [python.org](https://www.python.org/downloads/macos/).

---

### Linux

**Ubuntu / Debian:**

```bash
# Install Python 3.11 if not already present
sudo apt update
sudo apt install python3.11 python3.11-venv python3-pip -y

# Create a virtual environment (best practice)
python3.11 -m venv .venv
source .venv/bin/activate

# Install energykit
pip install "energykit[all]"

# Verify
python -c "import energykit; print(energykit.__version__)"
```

**RHEL / Fedora / CentOS:**

```bash
sudo dnf install python3.11 python3-pip -y
python3.11 -m venv .venv
source .venv/bin/activate
pip install "energykit[all]"
```

**Using Conda on Linux:**

```bash
conda create -n energykit-env python=3.11
conda activate energykit-env
pip install "energykit[all]"
```

---

### Install extras explained

| Extra | What it adds | When to use |
|-------|-------------|-------------|
| `energykit[forecast]` | LightGBM, statsmodels | Better load forecasting accuracy |
| `energykit[optimize]` | PuLP solver | Advanced DER dispatch (optional) |
| `energykit[datasets]` | requests, tqdm | Auto-download public datasets |
| `energykit[all]` | Everything above | Development / full feature set |

---

## Core modules

### `energykit.diagnose` - One-call financial audit

The entry point. Feed it any smart-meter Series and get a complete financial audit - demand charges, anomaly waste, battery ROI - as a terminal dashboard and a structured object.

```python
import energykit as ek
from energykit.datasets import load_synthetic_load

data = load_synthetic_load(periods=8760, freq="h")   # or your own pd.Series
report = ek.diagnose(data, energy_price=0.15, demand_rate=12.50)

# All numbers are also in the return value:
print(report.total_addressable_savings_usd)   # 1453.21
print(report.demand_charge_annual_usd)        # 701.55
print(report.anomaly_count)                   # 23
print(report.der_annual_savings_usd)          # 729.00
```

---

### `energykit.cost` - Translate data into dollars

#### Demand charge analysis

Most commercial bills have a demand charge: a fee based on the *single highest kW reading in the month*. One HVAC unit switching on at the wrong time can cost thousands.

```python
from energykit.cost import DemandChargeAnalyzer

analyzer = DemandChargeAnalyzer(demand_rate=12.50)   # $/kW/month
result = analyzer.analyze(power_kw_series)

# Which events cost the most?
print(result.peak_events_df)
#    period  peak_kw         peak_timestamp  demand_charge_usd
# 0  2025-01    4.81  2025-01-15 17:00:00              60.13
# 1  2025-02    4.23  2025-02-08 18:30:00              52.88
# ...

# What would a battery have saved?
print(result.battery_savings_df)
#    battery_kwh  max_power_kw  annual_savings_usd  pct_reduction
#            5.0           2.5              375.10           53.5
#           10.0           5.0              677.25           96.5
#           13.5           5.0              677.25           96.5
#           20.0          10.0              701.55          100.0
```

#### Imbalance settlement cost

For generators, aggregators, and portfolios - forecast errors create imbalance charges that can dwarf the headline MAPE number.

```python
from energykit.cost import ImbalanceCostCalculator, forecast_value_of_accuracy

# How much do our forecast errors cost right now?
calc = ImbalanceCostCalculator(imbalance_price=0.08)   # $/kWh penalty
result = calc.compute(forecast, actual)

print(f"Annual imbalance cost : ${result.annual_cost_estimate_usd:,.0f}")
print(f"Current MAPE          : {result.current_mape_pct:.1f}%")
print(f"Cost per 1% MAPE      : ${result.cost_per_mape_pct_usd:,.0f}/yr")
# Annual imbalance cost : $234,000
# Current MAPE          : 7.2%
# Cost per 1% MAPE      : $32,500/yr

# What is it worth to improve our forecaster?
report = forecast_value_of_accuracy(actual, forecast, imbalance_price=0.08)
print(report)
# ──────────────────────────────────────────────────────────
#   FORECAST VALUE ANALYSIS
# ──────────────────────────────────────────────────────────
#   Current MAPE             : 7.2%
#   Current annual cost      : $234,000/yr
#   Value per 1% MAPE gain   : $32,500/yr
# ──────────────────────────────────────────────────────────
#   Target MAPE              : 3.6%  (50% improvement)
#   Potential annual savings : $117,000/yr
#   1-Year break-even invest : $117,000
# ──────────────────────────────────────────────────────────
```

---

### `energykit.anomaly` - Smart meter anomaly detection with financial impact

Not just *"you have an anomaly"* - but *"this event wasted 450 kWh and cost you $67"*.

```python
from energykit.anomaly import MeterAnomalyDetector

detector = MeterAnomalyDetector(z_threshold=2.5)
detector.fit(historical_series)                          # learns seasonal baseline
result = detector.detect(new_series, energy_price=0.15)

print(result)
# AnomalySummary(n=23, rate=0.26%, waste=312.4 kWh, cost=$46.86)

# What are the most expensive anomaly events?
print(result.top_anomalies_df[["anomaly_type", "excess_kwh", "estimated_cost_usd"]])
#           anomaly_type  excess_kwh  estimated_cost_usd
# 2025-03-12 02:00  overnight        87.4               13.11
# 2025-01-22 14:00      spike        62.1                9.32
# 2025-05-07 01:30  overnight        45.0                6.75
```

**Anomaly types detected:**
| Type | Meaning |
|------|---------|
| `spike` | Instantaneous outlier - equipment fault, data error |
| `sustained_elevation` | ≥3 consecutive readings above threshold - HVAC fault, equipment left on |
| `overnight` | Anomaly between midnight–5am - after-hours waste or energy theft risk |
| `sudden_drop` | Far below baseline - meter fault or curtailment event |

---

### `energykit.forecast` - Load forecasting

```python
from energykit.forecast import LoadForecaster

model = LoadForecaster(horizon=24, country="US", lags=[1, 24, 168])
model.fit(load_series)
forecast = model.predict()             # next 24 hours as pd.Series
top_features = model.feature_importance().head(10)
```

Works with **LightGBM** (if installed) or scikit-learn's `HistGradientBoostingRegressor`. Auto-handles missing values.

---

### `energykit.optimize` - DER scheduling

Provably-optimal battery and EV dispatch - no commercial solver required.

```python
from energykit.optimize import BatteryScheduler, EVScheduler
import numpy as np

prices = np.array([0.09]*8 + [0.22]*9 + [0.28]*5 + [0.09]*2)

battery = BatteryScheduler(capacity_kwh=13.5, max_power_kw=5.0, efficiency=0.90)
result = battery.optimize(prices, load_kw=baseline_load)
print(f"Daily savings: ${result.savings_usd:.2f}")

ev = EVScheduler(battery_kwh=75, max_charge_kw=11.0)
ev_result = ev.optimize(prices=prices, initial_soc=0.15, target_soc=0.80, departure_step=8)
print(f"Smart vs dumb charging savings: ${ev_result.savings_usd:.2f}")
```

---

### `energykit.features` - Energy feature engineering

40+ energy-specific features from any hourly time series - in one sklearn-compatible transformer.

```python
from energykit.features import EnergyFeatureExtractor

fe = EnergyFeatureExtractor(
    lags=[1, 2, 3, 24, 48, 168],
    rolling_windows=[24, 168],
    cyclical=True,          # sin/cos encoding, no boundary artifacts
    country="US",           # automatic holiday detection
    lat=40.71, lon=-74.00,  # solar elevation angle
)
X = fe.fit_transform(meter_series)   # pd.Series → pd.DataFrame
```

Features: temporal (`hour`, `is_holiday`, `season`), cyclical sin/cos, TOU blocks, lags, rolling stats, solar position.

---

### `energykit.benchmark` - ASHRAE-14 compliant metrics

```python
from energykit.benchmark import mape, cvrmse, EnergyForecastBenchmark

bench = EnergyForecastBenchmark(actual, forecast)
print(bench.summary())
print(bench.ashrae_check())
# {'cvrmse_pct': 8.7, 'nmbe_pct': 0.3, 'hourly_pass': True, 'message': 'PASS'}
```

Metrics: MAPE, sMAPE, MAE, RMSE, CVRMSE, R², peak coincidence, load factor error.

---

### `energykit.datasets` - Dataset loaders

```python
from energykit.datasets import load_uci_household, load_synthetic_load, load_sample_tou_prices

df    = load_uci_household(resample="h")      # UCI Household, auto-download
load  = load_synthetic_load(periods=8760)     # 1-year synthetic, no download
prices = load_sample_tou_prices("residential_us", periods=24)
```

---

## Why energykit?

|                               | energykit | statsforecast | NILMTK | pandas / custom |
|-------------------------------|:---------:|:-------------:|:------:|:---------------:|
| Dollar translation (demand charges, imbalance) | ✅ | ❌ | ❌ | 600 lines |
| Anomaly cost quantification   | ✅ | ❌ | ❌ | 400 lines |
| One-call financial audit      | ✅ | ❌ | ❌ | ❌ |
| Energy-specific features      | ✅ | ❌ | ❌ | 500 lines |
| Load forecasting              | ✅ | ✅ (generic) | ❌ | 200 lines |
| Battery / EV optimization     | ✅ | ❌ | ❌ | 400 lines |
| NILM disaggregation           | ✅ | ❌ | ✅ (complex) | 300 lines |
| ASHRAE-14 benchmarks          | ✅ | ❌ | ❌ | 100 lines |
| sklearn compatible            | ✅ | ✅ | ❌ | - |

---

## Roadmap

| Version | Features |
|---------|----------|
| v0.1 | Feature engineering, LightGBM forecaster, battery/EV optimizer, NILM baseline, benchmark metrics |
| **v0.2** ← current | **Financial translation layer: demand charges, imbalance cost, anomaly detection, `diagnose()`** |
| v0.3 | Price forecasting, solar/wind generation, ENTSO-E live data integration |
| v0.4 | Virtual Power Plant (VPP) aggregation, fleet scheduling, V2G optimization |
| v0.5 | Neural forecasters (N-BEATS, PatchTST), probabilistic prediction intervals |
| v1.0 | Stable API, documentation site, full benchmark suite |

---

## Contributing

⭐ **If energykit saves you time or money, please [star the repo](https://github.com/muranAI/energykit) — it helps others find it.**

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md).

Key areas where help is needed:
- More public dataset loaders (ENTSO-E, OpenEI, EPEX)
- Neural network load forecasters (LSTM, N-HiTS)
- Probabilistic forecasting (quantile regression, conformal prediction)
- V2G bidirectional scheduler
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
  version      = {0.2.0}
}
```

---

## License

MIT - see [LICENSE](LICENSE).

Built by [Muranai](https://muranai.com) - enterprise AI for the energy sector.
