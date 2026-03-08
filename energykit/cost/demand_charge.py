"""
energykit.cost.demand_charge
==============================
Demand charge analysis and battery peak-shaving simulation.

Background
----------
Most commercial and industrial electricity tariffs include a **demand charge**:
a fee based on the single highest power demand in a billing period (usually the
peak 15-minute or 1-hour interval in a month).  This charge can represent
30–70% of a large customer's electricity bill yet typically stems from just
a handful of short events per year.

A single HVAC unit switching on at the wrong time can add $10,000 to a monthly
bill.  ``DemandChargeAnalyzer`` identifies exactly when those events happened,
how much they cost, and what battery capacity would have prevented them.

Usage
-----
>>> from energykit.cost import DemandChargeAnalyzer
>>>
>>> analyzer = DemandChargeAnalyzer(demand_rate=12.50)   # $/kW/month
>>> result = analyzer.analyze(power_kw_series)
>>>
>>> print(result.peak_events_df)
>>>      period  peak_kw         peak_timestamp  demand_charge_usd
>>>   2025-01    5.23  2025-01-15 17:00:00              65.38
>>>   2025-02    4.81  2025-02-08 18:30:00              60.13
>>>   ...
>>>
>>> print(result.battery_savings_df)
>>>   battery_kwh  max_power_kw  annual_savings_usd  pct_reduction
>>>           5.0           2.5               97.50           13.2
>>>          10.0           5.0              186.25           25.2
>>>          20.0          10.0              312.50           42.3
>>>          50.0          25.0              502.50           68.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class PeakEvent:
    """A single billing-period peak demand event.

    Attributes
    ----------
    period : str
        Billing period label (e.g. ``"2025-01"``).
    timestamp : pd.Timestamp
        Exact timestamp of the peak reading.
    peak_kw : float
        Peak power demand in kW.
    demand_charge_usd : float
        Demand charge for this billing period (USD).
    """

    period: str
    timestamp: pd.Timestamp
    peak_kw: float
    demand_charge_usd: float


@dataclass
class DemandChargeResult:
    """Results from :class:`DemandChargeAnalyzer`.

    Attributes
    ----------
    peak_events_df : pd.DataFrame
        One row per billing period with peak_kw, peak_timestamp, charge_usd.
    peak_events : list of PeakEvent
        Structured list for programmatic access.
    total_annual_charge_usd : float
        Sum of all billing period demand charges over the analysed window.
    worst_event : PeakEvent
        The single most expensive peak event.
    battery_savings_df : pd.DataFrame
        Battery sizing → annual savings table.
    n_periods : int
        Number of billing periods analysed.
    """

    peak_events_df: pd.DataFrame
    peak_events: List[PeakEvent]
    total_annual_charge_usd: float
    worst_event: PeakEvent
    battery_savings_df: pd.DataFrame
    n_periods: int

    def __repr__(self) -> str:
        return (
            f"DemandChargeResult("
            f"total_annual_charge=${self.total_annual_charge_usd:.2f}, "
            f"worst_event={self.worst_event.peak_kw:.1f} kW on {self.worst_event.timestamp}, "
            f"n_periods={self.n_periods})"
        )


class DemandChargeAnalyzer:
    """Analyse demand charges from power time series data.

    Identifies the peak demand event in each billing period, calculates the
    resulting demand charge, and simulates the savings achievable with
    different battery peak-shaving configurations.

    Parameters
    ----------
    demand_rate : float, default 12.50
        Demand charge rate in $/kW/month (applied to the highest kW in each
        billing period).
    peak_hours : list of int or None
        Hours (0–23) during which demand charges apply.  Use this for
        on-peak demand riders.  ``None`` = all hours (flat demand charge).
    billing_period : str, default ``"ME"``
        Pandas resample frequency string for billing periods.
        ``"ME"`` = month end, ``"QE"`` = quarter end.
    dt_hours : float, default 1.0
        Timestep duration in hours.  Set to 0.25 for 15-minute interval data.
        Does not affect demand charge calculation (which is peak kW, not kWh)
        but is used for battery energy feasibility checks.

    Examples
    --------
    >>> analyzer = DemandChargeAnalyzer(demand_rate=15.0, peak_hours=list(range(9, 22)))
    >>> result = analyzer.analyze(power_series)
    >>> result.battery_savings_df
    """

    # Battery sizes to simulate: (capacity_kwh, max_power_kw) pairs
    _BATTERY_CONFIGS = [
        (5.0,  2.5),
        (10.0, 5.0),
        (13.5, 5.0),   # Tesla Powerwall 2
        (20.0, 10.0),
        (50.0, 25.0),
    ]

    def __init__(
        self,
        demand_rate: float = 12.50,
        peak_hours: Optional[List[int]] = None,
        billing_period: str = "ME",
        dt_hours: float = 1.0,
    ) -> None:
        if demand_rate < 0:
            raise ValueError("demand_rate must be non-negative.")
        self.demand_rate = float(demand_rate)
        self.peak_hours = peak_hours
        self.billing_period = billing_period
        self.dt_hours = float(dt_hours)

    def analyze(self, power_kw: pd.Series) -> DemandChargeResult:
        """Analyse demand charges in a power time series.

        Parameters
        ----------
        power_kw : pd.Series
            Power demand in kW with a ``DatetimeIndex``.  Sub-hourly (15-min)
            resolution gives the most accurate demand charge estimates.

        Returns
        -------
        DemandChargeResult
        """
        if not isinstance(power_kw, pd.Series):
            raise TypeError("power_kw must be a pd.Series.")
        if not isinstance(power_kw.index, pd.DatetimeIndex):
            raise ValueError("power_kw must have a DatetimeIndex.")

        series = power_kw.clip(lower=0).ffill().bfill()

        # Apply peak-hour filter if specified
        billable = (
            series[series.index.hour.isin(self.peak_hours)]
            if self.peak_hours
            else series
        )

        # Monthly peaks and their timestamps
        monthly_peak_kw = billable.resample(self.billing_period).max()
        monthly_peak_ts = billable.resample(self.billing_period).apply(
            lambda x: x.idxmax() if len(x) > 0 else pd.NaT
        )
        monthly_charge = monthly_peak_kw * self.demand_rate

        # Build event table
        events_df = pd.DataFrame(
            {
                "period": monthly_peak_kw.index.strftime("%Y-%m"),
                "peak_kw": monthly_peak_kw.values.round(3),
                "peak_timestamp": monthly_peak_ts.values,
                "demand_charge_usd": monthly_charge.values.round(2),
            }
        )

        peak_events = [
            PeakEvent(
                period=str(row["period"]),
                timestamp=row["peak_timestamp"],
                peak_kw=float(row["peak_kw"]),
                demand_charge_usd=float(row["demand_charge_usd"]),
            )
            for _, row in events_df.iterrows()
        ]

        total_annual = float(monthly_charge.sum())
        worst_idx = int(np.argmax(monthly_peak_kw.values))
        worst_event = peak_events[worst_idx]

        battery_df = self._simulate_battery_savings(
            monthly_peak_kw=monthly_peak_kw,
            billable_series=billable,
        )

        return DemandChargeResult(
            peak_events_df=events_df,
            peak_events=peak_events,
            total_annual_charge_usd=total_annual,
            worst_event=worst_event,
            battery_savings_df=battery_df,
            n_periods=len(peak_events),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _simulate_battery_savings(
        self,
        monthly_peak_kw: pd.Series,
        billable_series: pd.Series,
    ) -> pd.DataFrame:
        """Estimate annual savings for different battery configurations.

        Uses a peak-shaving model:
        - Battery discharges at ``max_power_kw`` to clip peaks above a threshold.
        - Energy constraint: battery must have enough capacity for the peak
          event duration (estimated as the time demand stays above the cap).
        """
        original_annual = float(monthly_peak_kw.sum() * self.demand_rate)
        rows = []

        for cap_kwh, max_power_kw in self._BATTERY_CONFIGS:
            # Track savings month-by-month
            total_new_charge = 0.0

            for period_ts, peak_kw in monthly_peak_kw.items():
                # Cap = level we want to stay beneath
                target_cap = peak_kw - max_power_kw

                if target_cap <= 0:
                    # Battery is powerful enough to eliminate the demand charge
                    total_new_charge += 0.0
                    continue

                # Estimate energy needed to shave the peak.
                # Count periods in this billing month where demand was above target_cap.
                try:
                    period_mask = billable_series.index.to_period(
                        self.billing_period
                    ) == period_ts.to_period(self.billing_period)
                    period_data = billable_series[period_mask]
                    excess = period_data[period_data > target_cap] - target_cap
                    energy_needed_kwh = float(excess.sum() * self.dt_hours)
                except Exception:  # noqa: BLE001
                    # Approximation: assume 2-hour peak duration
                    energy_needed_kwh = max_power_kw * 2.0

                if energy_needed_kwh <= cap_kwh:
                    # Battery has enough energy → full shave
                    effective_new_peak = max(0.0, target_cap)
                else:
                    # Partial shave — battery runs out before peak clears
                    achievable_reduction = cap_kwh / (energy_needed_kwh / max_power_kw)
                    effective_new_peak = max(0.0, peak_kw - achievable_reduction)

                total_new_charge += effective_new_peak * self.demand_rate

            annual_savings = original_annual - total_new_charge
            pct = annual_savings / original_annual * 100 if original_annual > 0 else 0.0

            rows.append(
                {
                    "battery_kwh": cap_kwh,
                    "max_power_kw": max_power_kw,
                    "annual_savings_usd": round(annual_savings, 2),
                    "pct_reduction": round(pct, 1),
                }
            )

        return pd.DataFrame(rows)
