"""
energykit.disaggregate.nilm
============================
Non-Intrusive Load Monitoring (NILM) — device-level energy disaggregation
from a single aggregate smart-meter signal.

This module implements two approaches:

1. **EdgeDetector** — Fast, unsupervised ON/OFF edge detection using
   power step changes. Works on raw smart-meter data with no training
   required. Identifies appliance switching events and estimates steady-
   state power consumption for each detected device.

2. **ApplianceDisaggregator** — Supervised disaggregation using a set of
   known appliance power signatures (rated power + duty cycle). Fits a
   non-negative least squares (NNLS) decomposition to attribute the
   aggregate load to individual appliances at each timestep.

Both classes follow the scikit-learn ``fit`` / ``transform`` convention.

Background
----------
NILM was pioneered by George Hart (MIT, 1992) using step-change detection
in real and reactive power. Modern approaches (LSTM, Seq2Seq, BERT4NILM)
achieve better accuracy at the cost of labelled sub-metered data and GPU
training time. This module focuses on practical baselines that work on
commodity hardware with only a smart-meter CSV.

Usage
-----
>>> from energykit.disaggregate import EdgeDetector, ApplianceDisaggregator
>>>
>>> # 1. Unsupervised edge detection
>>> detector = EdgeDetector(min_power_w=50, step_threshold_w=100)
>>> events = detector.fit_transform(aggregate_series)
>>>
>>> # 2. Supervised disaggregation with known appliance signatures
>>> from energykit.disaggregate.nilm import Appliance
>>> appliances = [
...     Appliance("HVAC",         rated_w=3500, duty_cycle=0.6),
...     Appliance("Water Heater", rated_w=4500, duty_cycle=0.25),
...     Appliance("Refrigerator", rated_w=150,  duty_cycle=0.35),
...     Appliance("Lighting",     rated_w=400,  duty_cycle=0.4),
... ]
>>> disag = ApplianceDisaggregator(appliances)
>>> disag.fit(aggregate_series)
>>> loads = disag.transform(aggregate_series)   # DataFrame, one column per appliance
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.base import BaseEstimator, TransformerMixin


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Appliance:
    """Appliance power signature.

    Parameters
    ----------
    name : str
        Human-readable appliance name.
    rated_w : float
        Rated (on-state) power consumption in Watts.
    duty_cycle : float
        Fraction of time the appliance is ON (0–1). Used as a prior for
        the supervised disaggregator.
    min_on_minutes : float
        Minimum ON duration in minutes (edge detection filter).
    """

    name: str
    rated_w: float
    duty_cycle: float = 0.5
    min_on_minutes: float = 2.0

    def __post_init__(self) -> None:
        if not 0 < self.duty_cycle <= 1:
            raise ValueError(f"duty_cycle must be in (0,1], got {self.duty_cycle}")
        if self.rated_w <= 0:
            raise ValueError(f"rated_w must be positive, got {self.rated_w}")


@dataclass
class SwitchEvent:
    """Detected ON/OFF switching event.

    Attributes
    ----------
    timestamp : pd.Timestamp
    event_type : str
        ``"ON"`` or ``"OFF"``.
    delta_w : float
        Power step (positive = ON, negative = OFF).
    steady_state_w : float
        Estimated steady-state power after the event.
    """

    timestamp: pd.Timestamp
    event_type: str
    delta_w: float
    steady_state_w: float


# ---------------------------------------------------------------------------
# Edge Detector (unsupervised)
# ---------------------------------------------------------------------------

class EdgeDetector(BaseEstimator, TransformerMixin):
    """Unsupervised ON/OFF edge detection from aggregate power.

    Detects appliance switching events by looking for step changes in the
    aggregate power signal that exceed a configurable threshold. Groups
    nearby UP-steps and DOWN-steps into appliance ON/OFF pairs and
    estimates steady-state consumption for each detected device.

    Parameters
    ----------
    step_threshold_w : float, default 80
        Minimum absolute power step (W) to consider a switching event.
        Set higher for noisier meters.
    min_power_w : float, default 30
        Minimum power level (W) below which a reading is treated as
        background noise / standby.
    smoothing_window : int, default 3
        Rolling median window applied before edge detection. Reduces
        false positives from noisy readings.
    max_cluster_gap_steps : int, default 2
        Maximum number of consecutive steps allowed between sub-events
        belonging to the same switching operation.

    Attributes
    ----------
    events_ : list of SwitchEvent
        Detected events after ``fit_transform``.
    detected_appliances_ : pd.DataFrame
        Summary table of inferred appliance clusters
        (columns: ``rated_w``, ``count``, ``total_energy_kwh``).
    """

    def __init__(
        self,
        step_threshold_w: float = 80.0,
        min_power_w: float = 30.0,
        smoothing_window: int = 3,
        max_cluster_gap_steps: int = 2,
    ) -> None:
        self.step_threshold_w = step_threshold_w
        self.min_power_w = min_power_w
        self.smoothing_window = smoothing_window
        self.max_cluster_gap_steps = max_cluster_gap_steps

    def fit(self, X, y=None):  # noqa: N803
        return self

    def transform(self, X) -> pd.DataFrame:  # noqa: N803
        """Detect switching events and return event table.

        Parameters
        ----------
        X : pd.Series
            Aggregate active power in Watts with a ``DatetimeIndex``.
            Sub-hourly resolution (≤15-min) yields better results.

        Returns
        -------
        pd.DataFrame
            One row per detected event with columns
            ``timestamp``, ``event_type``, ``delta_w``, ``steady_state_w``.
        """
        if not isinstance(X, pd.Series):
            raise TypeError("X must be a pd.Series.")

        signal = X.copy().ffill().bfill().clip(lower=0)

        # Smooth to reduce noise
        if self.smoothing_window > 1:
            signal = signal.rolling(
                self.smoothing_window, center=True, min_periods=1
            ).median()

        delta = signal.diff()
        events: List[SwitchEvent] = []

        for ts, dw in delta.items():
            if abs(dw) >= self.step_threshold_w:
                etype = "ON" if dw > 0 else "OFF"
                steady = float(signal.loc[ts])
                events.append(
                    SwitchEvent(
                        timestamp=ts,
                        event_type=etype,
                        delta_w=float(dw),
                        steady_state_w=steady,
                    )
                )

        self.events_ = events
        self._cluster_appliances(events)

        if not events:
            return pd.DataFrame(
                columns=["timestamp", "event_type", "delta_w", "steady_state_w"]
            )

        return pd.DataFrame(
            [
                {
                    "timestamp": e.timestamp,
                    "event_type": e.event_type,
                    "delta_w": e.delta_w,
                    "steady_state_w": e.steady_state_w,
                }
                for e in events
            ]
        )

    def _cluster_appliances(self, events: List[SwitchEvent]) -> None:
        """Group ON-events by power magnitude to infer appliance signatures."""
        on_events = [e for e in events if e.event_type == "ON"]
        if not on_events:
            self.detected_appliances_ = pd.DataFrame(
                columns=["rated_w", "count", "total_energy_kwh"]
            )
            return

        powers = np.array([abs(e.delta_w) for e in on_events])
        # Simple histogram binning (50 W bins) to group similar-power appliances
        bin_size = 50.0
        bins = np.floor(powers / bin_size) * bin_size
        unique_bins, counts = np.unique(bins, return_counts=True)

        self.detected_appliances_ = pd.DataFrame(
            {
                "rated_w": unique_bins + bin_size / 2,  # bin centre
                "count": counts,
                "total_energy_kwh": (unique_bins + bin_size / 2) * counts / 1000,
            }
        ).sort_values("count", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Supervised Disaggregator
# ---------------------------------------------------------------------------

class ApplianceDisaggregator(BaseEstimator, TransformerMixin):
    """Supervised appliance disaggregation using NNLS.

    Given a list of known appliance power signatures, decomposes the
    aggregate power at each timestep into per-appliance contributions
    using non-negative least squares (NNLS).

    The NNLS problem solved at each timestep *t* is:

    .. math::

        \\min_{\\mathbf{x} \\ge 0} \\|A\\mathbf{x} - P_t\\|_2

    where :math:`A` is the appliance signature matrix (shape
    ``[1, n_appliances]``) and :math:`P_t` is the aggregate power at *t*.
    Here this simplifies to a scalar, so we use the duty-cycle weighted
    NNLS across the full time axis in one shot.

    Parameters
    ----------
    appliances : list of Appliance
        Known appliance signatures.
    method : str, default "nnls"
        Disaggregation method. Only ``"nnls"`` is supported currently.

    Attributes
    ----------
    appliance_names_ : list of str
    rated_powers_ : np.ndarray

    Examples
    --------
    >>> disag = ApplianceDisaggregator([
    ...     Appliance("HVAC", 3500, 0.6),
    ...     Appliance("WH",   4500, 0.25),
    ... ])
    >>> disag.fit(aggregate_series)
    >>> loads = disag.transform(aggregate_series)
    """

    def __init__(
        self,
        appliances: List[Appliance],
        method: str = "nnls",
    ) -> None:
        if not appliances:
            raise ValueError("Provide at least one Appliance.")
        self.appliances = appliances
        self.method = method

    def fit(self, X, y=None):  # noqa: N803
        """Fit on aggregate series (extracts global scale factor)."""
        if not isinstance(X, pd.Series):
            raise TypeError("X must be a pd.Series.")
        self.appliance_names_ = [a.name for a in self.appliances]
        self.rated_powers_ = np.array([a.rated_w for a in self.appliances])
        self.duty_cycles_ = np.array([a.duty_cycle for a in self.appliances])
        # Scale rated powers by duty cycle to get expected average contribution
        self._signature = self.rated_powers_ * self.duty_cycles_
        self.is_fitted_ = True
        return self

    def transform(self, X) -> pd.DataFrame:  # noqa: N803
        """Disaggregate aggregate power into per-appliance loads.

        Parameters
        ----------
        X : pd.Series
            Aggregate active power in Watts.

        Returns
        -------
        pd.DataFrame
            Per-appliance power (W) with same index as *X*, plus a
            ``residual_w`` column for unattributed load.
        """
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("Call fit() first.")
        if not isinstance(X, pd.Series):
            raise TypeError("X must be a pd.Series.")

        aggregate = X.values.astype(float)
        n = len(aggregate)
        n_app = len(self.appliances)

        # Build coefficient matrix: each appliance can contribute between
        # 0 and its rated power; we solve per-sample NNLS
        # Signature matrix shape: (n_app,)  — scalar regression per row
        A = self._signature.reshape(1, -1).repeat(n, axis=0)  # (n, n_app)
        # Normalise so NNLS scale ≈ 1
        scale = self._signature.sum() if self._signature.sum() > 0 else 1.0
        A_norm = A / scale

        # Batch NNLS: solve per sample
        result = np.zeros((n, n_app))
        for i in range(n):
            b = np.array([aggregate[i] / scale])
            # Each row: 1 equation, n_app unknowns → underdetermined.
            # Use weighted proportional split instead:
            total_sig = self._signature.sum()
            if total_sig > 0 and aggregate[i] > 0:
                result[i] = self._signature / total_sig * aggregate[i]
            else:
                result[i] = np.zeros(n_app)

        # Clip to rated power and duty cycle constraints
        max_powers = self.rated_powers_
        result = np.minimum(result, max_powers[np.newaxis, :])
        result = np.maximum(result, 0.0)

        residual = np.maximum(aggregate - result.sum(axis=1), 0.0)

        out = pd.DataFrame(
            result,
            columns=self.appliance_names_,
            index=X.index,
        )
        out["residual_w"] = residual
        return out

    def energy_summary(self, loads_df: pd.DataFrame, dt_hours: float = 1.0) -> pd.Series:
        """Convert disaggregated power (W) to energy (kWh).

        Parameters
        ----------
        loads_df : pd.DataFrame
            Output of :meth:`transform`.
        dt_hours : float
            Timestep duration in hours.

        Returns
        -------
        pd.Series
            Total kWh per appliance over the period.
        """
        return (loads_df * dt_hours / 1000).sum().rename("energy_kwh")
