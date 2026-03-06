"""energykit.benchmark — Standard metrics and benchmarks for energy AI tasks."""

from energykit.benchmark.metrics import (
    mape,
    smape,
    mae,
    rmse,
    cvrmse,
    r2,
    peak_coincidence,
    load_factor_error,
    EnergyForecastBenchmark,
)

__all__ = [
    "mape",
    "smape",
    "mae",
    "rmse",
    "cvrmse",
    "r2",
    "peak_coincidence",
    "load_factor_error",
    "EnergyForecastBenchmark",
]
