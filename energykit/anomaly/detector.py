"""
energykit.anomaly.detector
===========================
Smart meter anomaly detection with financial impact quantification.

The key insight missing from every other anomaly detection library in the energy
space: anomaly detection is only useful if it tells you **what it's costing you**.
This module detects anomalies *and* estimates the dollar value of each one.

Detection strategy
------------------
Uses a **seasonal baseline** approach: for each (hour-of-day, day-of-week) slot,
compute the median historical reading.  An anomaly is a reading that deviates
beyond a z-score threshold from that baseline.  This approach:

  - Requires no heavy dependencies (pure numpy/pandas/sklearn)
  - Handles daily and weekly seasonality
  - Is robust to missing data (median is robust to outliers)
  - Works equally well for 15-minute, hourly, and daily data

Anomaly types
-------------
``spike``
    Single-timestep outlier: instantaneous spike (equipment fault, data error).
``sustained_elevation``
    ≥3 consecutive readings above threshold: equipment left running, HVAC fault.
``overnight``
    Anomaly during hours typically considered off (0:00–5:00).  High risk for
    energy theft or after-hours equipment left on.
``sudden_drop``
    Reading far below baseline: meter malfunction, curtailment event, or grid
    interruption.

Usage
-----
>>> from energykit.anomaly import MeterAnomalyDetector
>>>
>>> detector = MeterAnomalyDetector(z_threshold=2.5)
>>> detector.fit(historical_series)          # learns seasonal baseline
>>> anomalies = detector.detect(series, energy_price=0.15)
>>>
>>> # DataFrame: is_anomaly, score, anomaly_type, excess_kwh, estimated_cost_usd
>>> wasteful = anomalies[anomalies.estimated_cost_usd > 5].sort_values(
...     "estimated_cost_usd", ascending=False
... )
>>> print(wasteful.head(10))
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# AnomalyResult — returned by detect(); also exposed at package level
# ---------------------------------------------------------------------------

@dataclass
class AnomalySummary:
    """High-level summary of a :meth:`MeterAnomalyDetector.detect` run.

    Attributes
    ----------
    n_anomalies : int
    anomaly_rate_pct : float
    total_excess_kwh : float
        Sum of excess energy across all anomalies (above baseline).
    total_estimated_cost_usd : float
        Estimated dollar impact of all anomalies.
    worst_event : pd.Series
        Row from the anomaly DataFrame with the highest ``estimated_cost_usd``.
    top_anomalies_df : pd.DataFrame
        Top 10 most expensive anomaly events.
    anomalies_df : pd.DataFrame
        Full detection result.
    """

    n_anomalies: int
    anomaly_rate_pct: float
    total_excess_kwh: float
    total_estimated_cost_usd: float
    worst_event: Optional[pd.Series]
    top_anomalies_df: pd.DataFrame
    anomalies_df: pd.DataFrame

    def __repr__(self) -> str:
        return (
            f"AnomalySummary("
            f"n={self.n_anomalies}, "
            f"rate={self.anomaly_rate_pct:.2f}%, "
            f"waste={self.total_excess_kwh:.1f} kWh, "
            f"cost=${self.total_estimated_cost_usd:.2f})"
        )


class MeterAnomalyDetector:
    """Detect energy meter anomalies with financial impact estimation.

    Parameters
    ----------
    z_threshold : float, default 2.5
        Number of standard deviations above/below the seasonal baseline that
        classify a reading as an anomaly.  Lower = more sensitive.
    overnight_hours : tuple of int, default (0, 5)
        Half-open interval ``[start, stop)`` of hours considered "off-peak
        overnight".  Anomalies in this window get labelled ``overnight``.
    sustained_window : int, default 3
        Minimum number of consecutive anomalous readings to be labelled
        ``sustained_elevation`` (instead of individual ``spike``).
    use_isolation_forest : bool, default False
        If True and scikit-learn is available, adds a second layer of detection
        using IsolationForest on the residuals.  Useful for multivariate or
        pattern-based anomalies.

    Examples
    --------
    >>> detector = MeterAnomalyDetector(z_threshold=2.5)
    >>> detector.fit(training_series)      # learn seasonal baseline
    >>> result = detector.detect(new_series, energy_price=0.15)
    >>> print(result)
    AnomalySummary(n=23, rate=0.26%, waste=312.4 kWh, cost=$46.86)
    """

    def __init__(
        self,
        z_threshold: float = 2.5,
        overnight_hours: Tuple[int, int] = (0, 5),
        sustained_window: int = 3,
        use_isolation_forest: bool = False,
    ) -> None:
        self.z_threshold = float(z_threshold)
        self.overnight_hours = overnight_hours
        self.sustained_window = int(sustained_window)
        self.use_isolation_forest = use_isolation_forest

        self._baseline_median: Optional[pd.Series] = None  # (dow, hour) → median
        self._baseline_std: Optional[pd.Series] = None     # (dow, hour) → std
        self._iso_forest = None

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, series: pd.Series) -> "MeterAnomalyDetector":
        """Learn baseline patterns from historical meter data.

        Parameters
        ----------
        series : pd.Series
            Meter readings (kW or kWh) with a ``pd.DatetimeIndex``.
            More historical data (≥60 days) produces more reliable baselines.

        Returns
        -------
        self
        """
        if not isinstance(series.index, pd.DatetimeIndex):
            raise ValueError("series must have a pd.DatetimeIndex")

        s = series.dropna().clip(lower=0)
        df = pd.DataFrame({
            "value": s.values,
            "dow": s.index.dayofweek,
            "hour": s.index.hour,
        })

        grouped = df.groupby(["dow", "hour"])["value"]
        self._baseline_median = grouped.median()
        # Use IQR-based std for robustness: std ≈ IQR / 1.349
        q75 = grouped.quantile(0.75)
        q25 = grouped.quantile(0.25)
        self._baseline_std = (q75 - q25) / 1.349
        # Avoid division by zero for near-constant slots
        self._baseline_std = self._baseline_std.replace(0, self._baseline_std.median())

        if self.use_isolation_forest:
            try:
                from sklearn.ensemble import IsolationForest

                residuals = self._compute_residuals(s).values.reshape(-1, 1)
                self._iso_forest = IsolationForest(
                    contamination=0.05, random_state=42, n_jobs=-1
                )
                self._iso_forest.fit(residuals)
            except ImportError:
                pass  # silently degrade

        return self

    # ------------------------------------------------------------------
    # detect
    # ------------------------------------------------------------------

    def detect(
        self,
        series: pd.Series,
        energy_price: float = 0.15,
    ) -> AnomalySummary:
        """Detect anomalies in meter readings and estimate financial impact.

        Parameters
        ----------
        series : pd.Series
            Meter readings with a ``pd.DatetimeIndex``.
        energy_price : float, default 0.15
            Energy price in $/kWh.  Used to estimate the cost of excess energy.

        Returns
        -------
        AnomalySummary
        """
        if self._baseline_median is None:
            raise RuntimeError(
                "Call fit() before detect(). "
                "Example: detector.fit(historical_data)"
            )
        if not isinstance(series.index, pd.DatetimeIndex):
            raise ValueError("series must have a pd.DatetimeIndex")

        s = series.clip(lower=0)
        expected = self._compute_expected(s)
        residuals = self._compute_residuals(s)
        std = self._lookup_std(s)

        epsilon = 1e-6
        z_scores = residuals / (std + epsilon)

        is_anomaly = z_scores.abs() > self.z_threshold

        # Isolation forest overlay
        if self._iso_forest is not None:
            iso_labels = self._iso_forest.predict(residuals.values.reshape(-1, 1))
            iso_anomaly = pd.Series(iso_labels == -1, index=s.index)
            is_anomaly = is_anomaly | iso_anomaly

        # Classify anomaly types
        anomaly_type = self._classify(s, is_anomaly, residuals)

        # Excess kWh (above baseline, for positive anomalies)
        excess_kwh = residuals.clip(lower=0)
        excess_kwh[~is_anomaly] = 0.0
        estimated_cost = excess_kwh * energy_price

        result_df = pd.DataFrame(
            {
                "reading": s.values,
                "expected": expected.values,
                "residual": residuals.values,
                "z_score": z_scores.values,
                "is_anomaly": is_anomaly.values,
                "anomaly_type": anomaly_type.values,
                "excess_kwh": excess_kwh.values,
                "estimated_cost_usd": estimated_cost.values,
            },
            index=s.index,
        )

        anomaly_only = result_df[result_df["is_anomaly"]]
        n = int(anomaly_only.shape[0])
        rate = n / len(result_df) * 100 if len(result_df) > 0 else 0.0
        total_excess = float(anomaly_only["excess_kwh"].sum())
        total_cost = float(anomaly_only["estimated_cost_usd"].sum())
        top10 = anomaly_only.nlargest(10, "estimated_cost_usd")
        worst = top10.iloc[0] if len(top10) > 0 else None

        return AnomalySummary(
            n_anomalies=n,
            anomaly_rate_pct=round(rate, 4),
            total_excess_kwh=round(total_excess, 2),
            total_estimated_cost_usd=round(total_cost, 2),
            worst_event=worst,
            top_anomalies_df=top10,
            anomalies_df=result_df,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_expected(self, s: pd.Series) -> pd.Series:
        """Look up the seasonal median for each timestamp in s."""
        keys = list(zip(s.index.dayofweek, s.index.hour))
        expected = pd.Series(
            [self._baseline_median.get(k, float("nan")) for k in keys],
            index=s.index,
        )
        # Fill any unseen (dow, hour) combinations with global median
        global_median = float(self._baseline_median.median())
        return expected.fillna(global_median)

    def _compute_residuals(self, s: pd.Series) -> pd.Series:
        expected = self._compute_expected(s)
        return s - expected

    def _lookup_std(self, s: pd.Series) -> pd.Series:
        assert self._baseline_std is not None
        keys = list(zip(s.index.dayofweek, s.index.hour))
        std = pd.Series(
            [self._baseline_std.get(k, float("nan")) for k in keys],
            index=s.index,
        )
        global_std = float(self._baseline_std.median())
        return std.fillna(global_std).replace(0, global_std)

    def _classify(
        self,
        s: pd.Series,
        is_anomaly: pd.Series,
        residuals: pd.Series,
    ) -> pd.Series:
        """Classify anomaly type for each flagged reading."""
        atype = pd.Series("none", index=s.index)

        # Mark sudden drops first
        atype[is_anomaly & (residuals < 0)] = "sudden_drop"

        # Positive anomalies: spike by default
        atype[is_anomaly & (residuals >= 0)] = "spike"

        # Override spikes in overnight window to "overnight"
        h = s.index.hour
        overnight_mask = (h >= self.overnight_hours[0]) & (h < self.overnight_hours[1])
        atype[is_anomaly & overnight_mask & (residuals >= 0)] = "overnight"

        # Relabel consecutive spikes as "sustained_elevation"
        anomaly_arr = is_anomaly.values.astype(int)
        window = self.sustained_window
        for i in range(window - 1, len(anomaly_arr)):
            if anomaly_arr[i - window + 1 : i + 1].sum() >= window:
                span = slice(i - window + 1, i + 1)
                mask = is_anomaly.index[span]
                atype[mask] = "sustained_elevation"

        return atype
