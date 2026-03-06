"""
energykit — Python toolkit for energy AI.

Modules
-------
features      Energy-specific feature engineering (time-of-day, lags, solar position)
forecast      Load, price, solar and wind forecasting
disaggregate  NILM — device-level consumption disaggregation
optimize      DER scheduling — battery, EV, solar dispatch
datasets      Loaders for public energy datasets (ENTSO-E, UCI, OpenEI)
benchmark     Standard metrics and benchmarks for energy AI tasks

Quick start
-----------
>>> import pandas as pd
>>> import energykit as ek
>>>
>>> # Load your smart-meter data (datetime index, hourly kWh readings)
>>> meter = pd.read_csv("meter.csv", index_col=0, parse_dates=True)["kwh"]
>>>
>>> # Extract energy features
>>> features = ek.EnergyFeatureExtractor(country="US").fit_transform(meter)
>>>
>>> # Forecast next 24 hours
>>> model = ek.LoadForecaster(horizon=24, country="US")
>>> model.fit(meter)
>>> forecast = model.predict()
>>> print(forecast)
"""

__version__ = "0.1.0"
__author__ = "Muranai"

from energykit.features import EnergyFeatureExtractor
from energykit.forecast import LoadForecaster
from energykit.benchmark import mape, rmse, mae, cvrmse

__all__ = [
    "__version__",
    "EnergyFeatureExtractor",
    "LoadForecaster",
    "mape",
    "rmse",
    "mae",
    "cvrmse",
]
