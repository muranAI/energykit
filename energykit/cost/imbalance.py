"""
energykit.cost.imbalance
=========================
Imbalance settlement cost calculation and forecast value quantification.

Background
----------
In deregulated energy markets (EPEX-SPOT, Nord Pool, PJM, CAISO, etc.), market
participants submit day-ahead schedules / bids.  Deviations from the scheduled
quantity in real time are settled at the **imbalance price** — which is almost
always more expensive than the day-ahead price, creating a direct financial
penalty for forecast errors.

For a portfolio manager:
  - Positive imbalance (produced/consumed more than forecast): settled at a
    typically unfavourable price (curtailment or spill cost).
  - Negative imbalance (underproduced / under-consumed): settled at the
    imbalance price which may be 2–5× the day-ahead price during scarcity.

``ImbalanceCostCalculator`` maps a forecast error time series directly to a
settlement cost.  ``forecast_value_of_accuracy`` answers the question every
energy manager asks: *"How much is it worth to improve our MAPE from 8% to 4%?"*

Usage
-----
>>> from energykit.cost import ImbalanceCostCalculator, forecast_value_of_accuracy
>>>
>>> calc = ImbalanceCostCalculator(imbalance_price=0.08)  # $80/MWh → $0.08/kWh
>>> result = calc.compute(forecast=my_forecast, actual=actual_load)
>>>
>>> print(f"Imbalance cost this month: ${result.total_cost_usd:.2f}")
>>> print(result.cost_by_hour_df)    # which hours of day are most expensive
>>> print(result.top_errors_df)      # the 10 most costly individual errors
>>>
>>> # Value of accuracy improvement
>>> report = forecast_value_of_accuracy(actual, forecast, imbalance_price=0.08)
>>> print(report)
>>> # Current MAPE 7.2% costs $234,000/yr in imbalance settlement.
>>> # Improving to 3.6% MAPE saves $117,000/yr.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from energykit.benchmark.metrics import mape as _mape


@dataclass
class ImbalanceResult:
    """Output of :class:`ImbalanceCostCalculator`.

    Attributes
    ----------
    total_cost_usd : float
        Total imbalance settlement cost over the input period.
    annual_cost_estimate_usd : float
        Annualised cost (extrapolated from the input period length).
    cost_by_hour_df : pd.DataFrame
        Mean hourly imbalance cost.  Reveals which times of day drive losses.
    cost_by_dow_df : pd.DataFrame
        Mean daily imbalance cost by day of week.
    cost_by_month_df : pd.DataFrame
        Monthly cost totals.
    top_errors_df : pd.DataFrame
        The 20 individual timesteps with the highest imbalance cost.
    current_mape_pct : float
        MAPE between forecast and actual (%).
    cost_per_mape_pct_usd : float
        Annual cost per 1% MAPE.  Use this as the "price tag" of forecast error.
    """

    total_cost_usd: float
    annual_cost_estimate_usd: float
    cost_by_hour_df: pd.DataFrame
    cost_by_dow_df: pd.DataFrame
    cost_by_month_df: pd.DataFrame
    top_errors_df: pd.DataFrame
    current_mape_pct: float
    cost_per_mape_pct_usd: float

    def __repr__(self) -> str:
        return (
            f"ImbalanceResult("
            f"total_cost=${self.total_cost_usd:.2f}, "
            f"annualised=${self.annual_cost_estimate_usd:.0f}, "
            f"MAPE={self.current_mape_pct:.2f}%, "
            f"cost_per_1pct_MAPE=${self.cost_per_mape_pct_usd:.0f}/yr)"
        )


@dataclass
class ForecastValueReport:
    """Output of :func:`forecast_value_of_accuracy`.

    Attributes
    ----------
    current_mape_pct : float
    current_annual_cost_usd : float
        Estimated annual imbalance cost at current MAPE.
    target_mape_pct : float
        The improvement target (default: half of current MAPE).
    target_annual_cost_usd : float
        Estimated annual cost at the target MAPE.
    potential_annual_savings_usd : float
        Savings from reaching the target MAPE.
    cost_per_mape_pct_usd : float
        Dollar value of each 1% point of MAPE improvement per year.
    breakeven_investment_usd : float
        How much a forecasting system could cost and break even in 1 year.
    improvement_table : pd.DataFrame
        Columns: target_mape_pct, annual_cost_usd, annual_savings_usd.
    """

    current_mape_pct: float
    current_annual_cost_usd: float
    target_mape_pct: float
    target_annual_cost_usd: float
    potential_annual_savings_usd: float
    cost_per_mape_pct_usd: float
    breakeven_investment_usd: float
    improvement_table: pd.DataFrame

    def __str__(self) -> str:
        lines = [
            "─" * 58,
            "  FORECAST VALUE ANALYSIS",
            "─" * 58,
            f"  Current MAPE             : {self.current_mape_pct:.1f}%",
            f"  Current annual cost      : ${self.current_annual_cost_usd:,.0f}/yr",
            f"  Value per 1% MAPE gain   : ${self.cost_per_mape_pct_usd:,.0f}/yr",
            "─" * 58,
            f"  Target MAPE              : {self.target_mape_pct:.1f}%  (50% improvement)",
            f"  Target annual cost       : ${self.target_annual_cost_usd:,.0f}/yr",
            f"  Potential annual savings : ${self.potential_annual_savings_usd:,.0f}/yr",
            f"  1-Year break-even invest : ${self.breakeven_investment_usd:,.0f}",
            "─" * 58,
        ]
        return "\n".join(lines)


class ImbalanceCostCalculator:
    """Calculate imbalance settlement costs from forecast vs actual series.

    Models the financial cost of forecast errors in energy markets.  Uses a
    symmetric imbalance price by default (any deviation is penalised equally),
    or an asymmetric model where over- and under-forecasting have different prices.

    Parameters
    ----------
    imbalance_price : float
        Symmetric imbalance price in $/kWh (applied to absolute error).
        Typical values: $0.05–$0.15/kWh for retail; $0.02–$0.08/kWh wholesale.
    imbalance_price_up : float or None
        Price for positive imbalance (you consumed/produced MORE than forecast).
        If ``None``, uses ``imbalance_price`` for both directions.
    imbalance_price_down : float or None
        Price for negative imbalance (you consumed/produced LESS than forecast).
        If ``None``, uses ``imbalance_price`` for both directions.
    dt_hours : float, default 1.0
        Timestep duration in hours.  Multiply energy error (kW × dt) → kWh.

    Examples
    --------
    >>> calc = ImbalanceCostCalculator(imbalance_price=0.08)
    >>> result = calc.compute(forecast, actual)
    >>> print(result)
    """

    def __init__(
        self,
        imbalance_price: float = 0.08,
        imbalance_price_up: Optional[float] = None,
        imbalance_price_down: Optional[float] = None,
        dt_hours: float = 1.0,
    ) -> None:
        self.imbalance_price = float(imbalance_price)
        self.price_up = float(imbalance_price_up) if imbalance_price_up is not None else self.imbalance_price
        self.price_down = float(imbalance_price_down) if imbalance_price_down is not None else self.imbalance_price
        self.dt_hours = float(dt_hours)

    def compute(
        self,
        forecast,
        actual,
        index: Optional[pd.DatetimeIndex] = None,
    ) -> ImbalanceResult:
        """Compute imbalance costs from forecast and actual series.

        Parameters
        ----------
        forecast : array-like or pd.Series
            Forecasted energy / power values (kW or kWh).
        actual : array-like or pd.Series
            Observed values (same units as forecast).
        index : pd.DatetimeIndex or None
            Datetime index for the series.  Required for temporal breakdowns.
            Inferred from ``actual`` if it is a ``pd.Series``.

        Returns
        -------
        ImbalanceResult
        """
        if isinstance(actual, pd.Series) and index is None:
            index = actual.index
        if isinstance(forecast, pd.Series) and index is None:
            index = forecast.index

        f = np.asarray(forecast, dtype=float).ravel()
        a = np.asarray(actual, dtype=float).ravel()

        if f.shape != a.shape:
            raise ValueError("forecast and actual must have the same length.")

        # Per-step costs
        over = np.maximum(f - a, 0.0)   # over-forecast
        under = np.maximum(a - f, 0.0)  # under-forecast
        step_cost = (over * self.price_up + under * self.price_down) * self.dt_hours

        total_cost = float(step_cost.sum())

        # Annualise based on input period length
        if index is not None and len(index) >= 2:
            total_hours = (index[-1] - index[0]).total_seconds() / 3600
            annual_cost = total_cost * (8760 / max(total_hours, 1))
        else:
            annual_cost = total_cost

        current_mape = _mape(a, f)
        cost_per_mape = annual_cost / current_mape if current_mape > 0 else 0.0

        if index is not None:
            idx = pd.DatetimeIndex(index)
            df = pd.DataFrame(
                {"forecast": f, "actual": a, "error": f - a, "cost_usd": step_cost},
                index=idx,
            )
            cost_by_hour = (
                df.groupby(df.index.hour)["cost_usd"].mean()
                .reset_index()
                .rename(columns={"index": "hour"})
            )
            cost_by_hour.columns = ["hour", "mean_cost_usd"]

            cost_by_dow = (
                df.groupby(df.index.dayofweek)["cost_usd"].mean()
                .reset_index()
            )
            cost_by_dow.columns = ["day_of_week", "mean_cost_usd"]
            cost_by_dow["day_name"] = pd.to_datetime("2024-01-01") + pd.to_timedelta(
                cost_by_dow["day_of_week"], unit="D"
            )
            cost_by_dow["day_name"] = cost_by_dow["day_name"].dt.strftime("%A")

            cost_by_month = (
                df.resample("ME")["cost_usd"].sum()
                .reset_index()
                .rename(columns={"index": "month", "cost_usd": "total_cost_usd"})
            )
            cost_by_month["period"] = (
                pd.to_datetime(cost_by_month.iloc[:, 0]).dt.strftime("%Y-%m")
            )

            top20 = df.nlargest(20, "cost_usd").copy()
            top20["abs_error"] = np.abs(top20["error"])
        else:
            empty = pd.DataFrame()
            cost_by_hour = empty
            cost_by_dow = empty
            cost_by_month = empty
            top20 = pd.DataFrame(
                {"forecast": f, "actual": a, "error": f - a, "cost_usd": step_cost}
            ).nlargest(20, "cost_usd")

        return ImbalanceResult(
            total_cost_usd=round(total_cost, 4),
            annual_cost_estimate_usd=round(annual_cost, 2),
            cost_by_hour_df=cost_by_hour,
            cost_by_dow_df=cost_by_dow,
            cost_by_month_df=cost_by_month,
            top_errors_df=top20,
            current_mape_pct=round(current_mape, 4),
            cost_per_mape_pct_usd=round(cost_per_mape, 2),
        )


def forecast_value_of_accuracy(
    actual,
    forecast,
    imbalance_price: float = 0.08,
    dt_hours: float = 1.0,
    target_mape_pct: Optional[float] = None,
    index: Optional[pd.DatetimeIndex] = None,
) -> ForecastValueReport:
    """Quantify the annual financial value of improving forecast accuracy.

    This function answers: *"How much is each percentage point of MAPE
    improvement worth in dollars per year?"*

    The model is linear: imbalance cost ∝ MAPE.  This is an approximation —
    the true relationship depends on the price distribution and error
    distribution — but it is a solid first-order estimate used in practice
    by trading desks for investment decisions.

    Parameters
    ----------
    actual : array-like or pd.Series
        Observed values.
    forecast : array-like or pd.Series
        Forecasted values.
    imbalance_price : float
        Imbalance settlement price in $/kWh (default: $0.08/kWh).
    dt_hours : float
        Timestep duration in hours.
    target_mape_pct : float or None
        The improvement target.  Defaults to half the current MAPE.
    index : pd.DatetimeIndex or None
        Datetime index for annualisation.

    Returns
    -------
    ForecastValueReport
        Includes a full improvement table and the "break-even investment" number.

    Examples
    --------
    >>> report = forecast_value_of_accuracy(actual, forecast, imbalance_price=0.10)
    >>> print(report)
    """
    if isinstance(actual, pd.Series) and index is None:
        index = actual.index

    calc = ImbalanceCostCalculator(imbalance_price=imbalance_price, dt_hours=dt_hours)
    result = calc.compute(forecast, actual, index=index)

    current_mape = result.current_mape_pct
    current_annual = result.annual_cost_estimate_usd
    cost_per_pct = result.cost_per_mape_pct_usd

    if target_mape_pct is None:
        target_mape_pct = max(current_mape / 2.0, 0.5)

    target_annual = target_mape_pct * cost_per_pct
    savings = current_annual - target_annual

    # Build improvement table: 10%, 20%, ..., 80% reduction in MAPE
    rows = []
    for reduction_pct in [10, 20, 30, 40, 50, 60, 70, 80]:
        t_mape = current_mape * (1 - reduction_pct / 100)
        t_cost = t_mape * cost_per_pct
        rows.append(
            {
                "mape_reduction_pct": reduction_pct,
                "target_mape_pct": round(t_mape, 2),
                "annual_cost_usd": round(t_cost, 2),
                "annual_savings_usd": round(current_annual - t_cost, 2),
            }
        )

    return ForecastValueReport(
        current_mape_pct=round(current_mape, 2),
        current_annual_cost_usd=round(current_annual, 2),
        target_mape_pct=round(target_mape_pct, 2),
        target_annual_cost_usd=round(target_annual, 2),
        potential_annual_savings_usd=round(savings, 2),
        cost_per_mape_pct_usd=round(cost_per_pct, 2),
        breakeven_investment_usd=round(savings, 2),
        improvement_table=pd.DataFrame(rows),
    )
