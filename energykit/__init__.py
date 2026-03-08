"""
energykit — Python toolkit for energy AI.

Translate your energy data into dollars.

Modules
-------
features      Energy-specific feature engineering (time-of-day, lags, solar position)
forecast      Load, price, solar and wind forecasting
disaggregate  NILM — device-level consumption disaggregation
optimize      DER scheduling — battery, EV, solar dispatch
cost          Demand charge analysis and imbalance settlement cost quantification
anomaly       Smart meter anomaly detection with financial impact estimation
datasets      Loaders for public energy datasets (ENTSO-E, UCI, OpenEI)
benchmark     Standard metrics and benchmarks for energy AI tasks

Quick start — one call financial diagnosis
------------------------------------------
>>> import energykit as ek
>>> from energykit.datasets import load_synthetic_load
>>>
>>> data = load_synthetic_load(periods=8_760, freq="h")["load_kw"]
>>> report = ek.diagnose(data)     # prints full ASCII dashboard + returns report
>>> print(report.total_addressable_savings_usd)

Deeper analysis
---------------
>>> from energykit.cost import DemandChargeAnalyzer, ImbalanceCostCalculator
>>> from energykit.anomaly import MeterAnomalyDetector
>>>
>>> # What are my peak demand events costing me?
>>> dc = DemandChargeAnalyzer(demand_rate=12.50).analyze(meter)
>>> print(dc.total_annual_charge_usd)
>>>
>>> # How much do forecast errors cost in imbalance settlement?
>>> calc = ImbalanceCostCalculator(imbalance_price=0.08)
>>> result = calc.compute(forecast, actual)
>>> print(f"Imbalance cost: ${result.annual_cost_estimate_usd:,.0f}/yr")
>>> print(f"Cost per 1% MAPE: ${result.cost_per_mape_pct_usd:,.0f}/yr")
>>>
>>> # Find anomalies with financial impact
>>> detector = MeterAnomalyDetector()
>>> detector.fit(historical_data)
>>> anomalies = detector.detect(recent_data, energy_price=0.15)
>>> print(f"Waste: {anomalies.total_excess_kwh:.1f} kWh = ${anomalies.total_estimated_cost_usd:.2f}")
"""

__version__ = "0.2.0"
__author__ = "Muranai"

from energykit.features import EnergyFeatureExtractor
from energykit.forecast import LoadForecaster
from energykit.benchmark import mape, rmse, mae, cvrmse
from energykit.diagnose import diagnose, DiagnosisReport

__all__ = [
    "__version__",
    # one-call analysis
    "diagnose",
    "DiagnosisReport",
    # core modules
    "EnergyFeatureExtractor",
    "LoadForecaster",
    # metrics
    "mape",
    "rmse",
    "mae",
    "cvrmse",
]
