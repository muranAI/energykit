"""
energykit.benchmark.metrics
============================
Standard accuracy metrics for energy forecasting evaluation.

Metric summary
--------------
===================  =============================================  ==========
Function             Full name                                      Range
===================  =============================================  ==========
``mape``             Mean Absolute Percentage Error                 [0, ∞)  %
``smape``            Symmetric MAPE                                 [0, 200]%
``mae``              Mean Absolute Error                            [0, ∞)
``rmse``             Root Mean Squared Error                        [0, ∞)
``cvrmse``           Coefficient of Variation RMSE (ASHRAE 14)     [0, ∞)  %
``r2``               Coefficient of determination                   (-∞, 1]
``peak_coincidence`` % of peak hours correctly identified           [0, 1]
``load_factor_error``Δ load factor (forecast vs actual)             [−1, 1]
===================  =============================================  ==========

ASHRAE note
-----------
CVRMSE is the standard metric for building energy model calibration under
ASHRAE Guideline 14. Thresholds: hourly ≤ 30%, monthly ≤ 15%.

Usage
-----
>>> from energykit.benchmark import mape, cvrmse, EnergyForecastBenchmark
>>> print(mape(actual, forecast))
>>> bench = EnergyForecastBenchmark(actual, forecast)
>>> bench.summary()
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def _validate(actual: np.ndarray, forecast: np.ndarray) -> tuple:
    """Convert and validate inputs, returning cleaned numpy arrays."""
    a = np.asarray(actual, dtype=float).ravel()
    f = np.asarray(forecast, dtype=float).ravel()
    if a.shape != f.shape:
        raise ValueError(
            f"Shape mismatch: actual {a.shape} vs forecast {f.shape}."
        )
    return a, f


def mape(actual, forecast, epsilon: float = 1e-8) -> float:
    """Mean Absolute Percentage Error (%).

    Parameters
    ----------
    actual, forecast : array-like
    epsilon : float
        Small value added to denominator to avoid division by zero.

    Returns
    -------
    float
        MAPE in percent (e.g. 5.2 means 5.2%).
    """
    a, f = _validate(actual, forecast)
    return float(np.mean(np.abs((a - f) / (np.abs(a) + epsilon))) * 100)


def smape(actual, forecast) -> float:
    """Symmetric Mean Absolute Percentage Error (%).

    Better behaved than MAPE when actuals are near zero.

    Returns
    -------
    float
        sMAPE in percent (0–200).
    """
    a, f = _validate(actual, forecast)
    denom = (np.abs(a) + np.abs(f)) / 2 + 1e-8
    return float(np.mean(np.abs(a - f) / denom) * 100)


def mae(actual, forecast) -> float:
    """Mean Absolute Error (same units as input)."""
    a, f = _validate(actual, forecast)
    return float(np.mean(np.abs(a - f)))


def rmse(actual, forecast) -> float:
    """Root Mean Squared Error (same units as input)."""
    a, f = _validate(actual, forecast)
    return float(np.sqrt(np.mean((a - f) ** 2)))


def cvrmse(actual, forecast) -> float:
    """Coefficient of Variation of RMSE (%) — ASHRAE Guideline 14.

    ``CVRMSE = RMSE / mean(actual) × 100``

    Returns
    -------
    float
        CV(RMSE) in percent.
    """
    a, f = _validate(actual, forecast)
    mean_a = np.mean(a)
    if mean_a == 0:
        raise ValueError("Mean of actual values is zero; CVRMSE is undefined.")
    return float(rmse(a, f) / mean_a * 100)


def r2(actual, forecast) -> float:
    """Coefficient of determination R².

    Returns
    -------
    float
        R² ∈ (−∞, 1]. 1.0 = perfect fit.
    """
    a, f = _validate(actual, forecast)
    ss_res = np.sum((a - f) ** 2)
    ss_tot = np.sum((a - np.mean(a)) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return float(1.0 - ss_res / ss_tot)


def peak_coincidence(
    actual,
    forecast,
    top_pct: float = 0.10,
) -> float:
    """Fraction of actual peak hours that are also forecast as peaks.

    Relevant for demand charge management: missing a peak is expensive.

    Parameters
    ----------
    actual, forecast : array-like
    top_pct : float, default 0.10
        Top percentile to define "peak" hours (0.10 = top 10%).

    Returns
    -------
    float
        Coincidence rate ∈ [0, 1]. 1.0 = all peaks correctly identified.
    """
    a, f = _validate(actual, forecast)
    k = max(1, int(len(a) * top_pct))
    actual_peak_idx = set(np.argpartition(a, -k)[-k:])
    forecast_peak_idx = set(np.argpartition(f, -k)[-k:])
    return float(len(actual_peak_idx & forecast_peak_idx) / k)


def load_factor_error(actual, forecast) -> float:
    """Difference in load factor between forecast and actual.

    Load factor = mean / peak. Higher = more uniform load, lower = peakier.
    A negative error means the forecast predicts a peakier load than reality.

    Returns
    -------
    float
        Δ load factor = LF(forecast) − LF(actual).
    """
    a, f = _validate(actual, forecast)
    lf_actual = np.mean(a) / np.max(a) if np.max(a) > 0 else 0.0
    lf_forecast = np.mean(f) / np.max(f) if np.max(f) > 0 else 0.0
    return float(lf_forecast - lf_actual)


# ---------------------------------------------------------------------------
# Composite benchmark report
# ---------------------------------------------------------------------------

class EnergyForecastBenchmark:
    """Compute and display a comprehensive forecast accuracy report.

    Parameters
    ----------
    actual : array-like
        Observed values.
    forecast : array-like
        Forecasted values.
    label : str
        Optional label for the model (shown in report).

    Examples
    --------
    >>> bench = EnergyForecastBenchmark(actual, forecast, label="LightGBM 24h")
    >>> report = bench.summary()
    >>> print(report)
    """

    def __init__(self, actual, forecast, label: str = "model") -> None:
        self.actual = np.asarray(actual, dtype=float).ravel()
        self.forecast = np.asarray(forecast, dtype=float).ravel()
        self.label = label

    def summary(self) -> pd.DataFrame:
        """Return all metrics as a one-row DataFrame.

        Returns
        -------
        pd.DataFrame
            Columns: MAE, RMSE, MAPE (%), sMAPE (%), CVRMSE (%), R²,
            Peak Coincidence, Load Factor Error.
        """
        a, f = self.actual, self.forecast
        metrics = {
            "MAE": mae(a, f),
            "RMSE": rmse(a, f),
            "MAPE (%)": mape(a, f),
            "sMAPE (%)": smape(a, f),
            "CVRMSE (%)": cvrmse(a, f),
            "R²": r2(a, f),
            "Peak Coincidence": peak_coincidence(a, f),
            "Load Factor Error": load_factor_error(a, f),
        }
        return pd.DataFrame(metrics, index=[self.label]).round(4)

    def ashrae_check(self) -> dict:
        """Check whether the forecast meets ASHRAE Guideline 14 thresholds.

        Thresholds
        ----------
        Hourly:  CVRMSE ≤ 30 %, NMBE ≤ ±5 %
        Monthly: CVRMSE ≤ 15 %, NMBE ≤ ±5 %

        Returns
        -------
        dict with keys ``"hourly_pass"`` and diagnostic message.
        """
        cv = cvrmse(self.actual, self.forecast)
        nmbe = float(
            np.mean(self.forecast - self.actual) / (np.mean(self.actual) + 1e-8) * 100
        )
        hourly_pass = cv <= 30.0 and abs(nmbe) <= 5.0
        return {
            "cvrmse_pct": round(cv, 2),
            "nmbe_pct": round(nmbe, 2),
            "hourly_pass": hourly_pass,
            "message": (
                "PASS — meets ASHRAE-14 hourly calibration threshold."
                if hourly_pass
                else f"FAIL — CVRMSE={cv:.1f}% (threshold 30%), NMBE={nmbe:.1f}% (threshold ±5%)."
            ),
        }
