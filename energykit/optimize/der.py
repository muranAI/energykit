"""
energykit.optimize.der
=======================
Optimal scheduling of Distributed Energy Resources (DER) using Linear
Programming via ``scipy.optimize.linprog`` (no additional dependencies).

Classes
-------
BatteryScheduler
    Minimize electricity cost by optimally scheduling battery charge /
    discharge given time-varying retail prices and technical constraints
    (capacity, max power, round-trip efficiency, SoC limits).

EVScheduler
    Smart EV charging scheduler. Charges the vehicle to a target state of
    charge by a departure time while minimizing electricity cost.

Both classes follow a simple ``optimize(prices)`` API that returns a
:class:`~energykit.optimize.der.ScheduleResult` dataclass.

Usage
-----
>>> import numpy as np
>>> from energykit.optimize import BatteryScheduler
>>>
>>> # 24-hour time-of-use prices ($/kWh), hourly resolution
>>> prices = np.array([0.08]*8 + [0.22]*4 + [0.12]*4 + [0.28]*4 + [0.10]*4)
>>>
>>> battery = BatteryScheduler(
...     capacity_kwh=13.5,     # Tesla Powerwall 2
...     max_power_kw=5.0,
...     efficiency=0.90,
...     initial_soc=0.20,
... )
>>> result = battery.optimize(prices)
>>> print(result.savings_usd)
1.84
>>> print(result.schedule_df)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import linprog


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class ScheduleResult:
    """Optimization result from :class:`BatteryScheduler` or :class:`EVScheduler`.

    Attributes
    ----------
    schedule_df : pd.DataFrame
        Hourly schedule with columns depending on optimizer type.
    total_cost_usd : float
        Total electricity cost after optimization.
    baseline_cost_usd : float
        Electricity cost without any DER dispatch (flat import).
    savings_usd : float
        Cost savings vs. baseline.
    status : str
        Solver status (``"optimal"``, ``"infeasible"``, etc.)
    """

    schedule_df: pd.DataFrame
    total_cost_usd: float
    baseline_cost_usd: float
    savings_usd: float
    status: str

    def __repr__(self) -> str:
        return (
            f"ScheduleResult(status={self.status!r}, "
            f"savings_usd={self.savings_usd:.4f}, "
            f"rows={len(self.schedule_df)})"
        )


# ---------------------------------------------------------------------------
# Battery Scheduler
# ---------------------------------------------------------------------------

class BatteryScheduler:
    """Optimal battery charge/discharge scheduler.

    Formulates a linear program to minimize total electricity cost over a
    planning horizon by choosing when to charge (buy cheap) and discharge
    (avoid buying expensive).

    The LP is:

    .. math::

        \\min_{c_t, d_t} \\sum_t p_t (c_t - d_t) \\cdot \\Delta t

    subject to:

    - SoC dynamics: :math:`SoC_{t+1} = SoC_t + \\eta_c c_t \\Delta t - (1/\\eta_d) d_t \\Delta t`
    - Capacity:     :math:`SoC_{\\min} \\le SoC_t \\le SoC_{\\max}`
    - Power limits: :math:`0 \\le c_t, d_t \\le P_{\\max}`
    - No simultaneous charge/discharge (relaxed LP — typically satisfied at optimum)

    Parameters
    ----------
    capacity_kwh : float
        Usable battery capacity in kWh.
    max_power_kw : float
        Maximum charge and discharge power in kW.
    efficiency : float, default 0.90
        Round-trip efficiency (0–1). Applied symmetrically to charge and discharge.
    initial_soc : float, default 0.50
        Initial state of charge as a fraction of capacity (0–1).
    min_soc : float, default 0.10
        Minimum allowed SoC (depth-of-discharge protection).
    max_soc : float, default 0.95
        Maximum allowed SoC (overcharge protection).
    dt_hours : float, default 1.0
        Timestep duration in hours. Use 0.5 for 30-minute data.
    index : pd.DatetimeIndex or None
        Optional datetime index for the output schedule.

    Examples
    --------
    >>> batt = BatteryScheduler(capacity_kwh=10, max_power_kw=5)
    >>> result = batt.optimize(prices_array)
    >>> result.schedule_df[["charge_kw", "discharge_kw", "soc_kwh"]].plot()
    """

    def __init__(
        self,
        capacity_kwh: float,
        max_power_kw: float,
        efficiency: float = 0.90,
        initial_soc: float = 0.50,
        min_soc: float = 0.10,
        max_soc: float = 0.95,
        dt_hours: float = 1.0,
        index: Optional[pd.DatetimeIndex] = None,
    ) -> None:
        if not 0 < efficiency <= 1:
            raise ValueError("efficiency must be in (0, 1].")
        if not 0 <= min_soc < max_soc <= 1:
            raise ValueError("Require 0 <= min_soc < max_soc <= 1.")

        self.capacity_kwh = float(capacity_kwh)
        self.max_power_kw = float(max_power_kw)
        self.efficiency = float(efficiency)
        self.initial_soc = float(initial_soc)
        self.min_soc = float(min_soc)
        self.max_soc = float(max_soc)
        self.dt_hours = float(dt_hours)
        self.index = index

    def optimize(
        self,
        prices: np.ndarray,
        load_kw: Optional[np.ndarray] = None,
    ) -> ScheduleResult:
        """Run the battery optimization.

        Parameters
        ----------
        prices : array-like of shape (T,)
            Electricity import price at each timestep ($/kWh or any currency/kWh).
        load_kw : array-like of shape (T,) or None
            Baseline load at each timestep (kW). Used only for baseline cost
            calculation. Does not affect the battery dispatch.

        Returns
        -------
        ScheduleResult
        """
        prices = np.asarray(prices, dtype=float)
        T = len(prices)

        if load_kw is not None:
            load_kw = np.asarray(load_kw, dtype=float)
            baseline_cost = float(np.sum(load_kw * prices * self.dt_hours))
        else:
            baseline_cost = 0.0

        # Decision variables: [c_0, ..., c_{T-1}, d_0, ..., d_{T-1}]
        # c_t = charge power (kW), d_t = discharge power (kW)
        n_vars = 2 * T
        sqrt_eff = np.sqrt(self.efficiency)  # symmetric split

        # Objective: minimize sum_t price_t * (c_t - d_t) * dt
        c_obj = np.concatenate([prices * self.dt_hours, -prices * self.dt_hours])

        # Inequality constraints: A_ub @ x <= b_ub
        # SoC upper bound:  SoC_t = SoC_0 + sum_{k<t}(sqrt_eff*c_k - (1/sqrt_eff)*d_k)*dt <= SoC_max*cap
        # SoC lower bound: -SoC_t <= -SoC_min*cap
        A_ub_rows = []
        b_ub_rows = []

        soc_init = self.initial_soc * self.capacity_kwh
        soc_max = self.max_soc * self.capacity_kwh
        soc_min = self.min_soc * self.capacity_kwh

        for t in range(1, T + 1):
            # Upper bound row: SoC_t <= soc_max
            row_c = np.zeros(n_vars)
            row_c[:t] = sqrt_eff * self.dt_hours
            row_c[T : T + t] = -(1.0 / sqrt_eff) * self.dt_hours
            A_ub_rows.append(row_c)
            b_ub_rows.append(soc_max - soc_init)

            # Lower bound row: -SoC_t <= -soc_min
            A_ub_rows.append(-row_c)
            b_ub_rows.append(soc_init - soc_min)

        A_ub = np.array(A_ub_rows)
        b_ub = np.array(b_ub_rows)

        # Variable bounds: 0 <= c_t, d_t <= max_power
        bounds = [(0, self.max_power_kw)] * n_vars

        result = linprog(
            c_obj,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=bounds,
            method="highs",
        )

        if result.status == 0:
            charge = result.x[:T]
            discharge = result.x[T:]
            status = "optimal"
        else:
            charge = np.zeros(T)
            discharge = np.zeros(T)
            status = result.message

        soc = np.zeros(T + 1)
        soc[0] = soc_init
        for t in range(T):
            soc[t + 1] = (
                soc[t]
                + sqrt_eff * charge[t] * self.dt_hours
                - (1.0 / sqrt_eff) * discharge[t] * self.dt_hours
            )

        net_import = (load_kw if load_kw is not None else np.zeros(T)) + charge - discharge
        total_cost = float(np.sum(np.maximum(net_import, 0) * prices * self.dt_hours))

        idx = self.index if self.index is not None else pd.RangeIndex(T)
        schedule_df = pd.DataFrame(
            {
                "price_per_kwh": prices,
                "charge_kw": charge,
                "discharge_kw": discharge,
                "net_import_kw": net_import,
                "soc_kwh": soc[:T],
                "soc_pct": soc[:T] / self.capacity_kwh * 100,
            },
            index=idx,
        )

        return ScheduleResult(
            schedule_df=schedule_df,
            total_cost_usd=total_cost,
            baseline_cost_usd=baseline_cost,
            savings_usd=baseline_cost - total_cost,
            status=status,
        )


# ---------------------------------------------------------------------------
# EV Scheduler
# ---------------------------------------------------------------------------

class EVScheduler:
    """Smart EV charging scheduler.

    Charges an EV battery to a target SoC by a departure time while
    minimizing electricity cost. Models unidirectional (G1V) charging only.

    Parameters
    ----------
    battery_kwh : float
        EV battery capacity (usable, kWh).
    max_charge_kw : float
        Maximum AC charging power (e.g. 7.4 kW for Level 2 EVSE).
    efficiency : float, default 0.92
        Charging efficiency (AC-to-DC).
    dt_hours : float, default 1.0
        Timestep duration in hours.
    index : pd.DatetimeIndex or None
        Optional datetime index for the output schedule.

    Examples
    --------
    >>> ev = EVScheduler(battery_kwh=75, max_charge_kw=11.0)
    >>> prices = np.tile([0.08, 0.10, 0.12, 0.25, 0.30], 5)  # TOU prices
    >>> result = ev.optimize(
    ...     prices=prices,
    ...     initial_soc=0.20,
    ...     target_soc=0.80,
    ...     departure_step=22,    # must be fully charged by step 22
    ... )
    >>> print(result.savings_usd)
    """

    def __init__(
        self,
        battery_kwh: float,
        max_charge_kw: float,
        efficiency: float = 0.92,
        dt_hours: float = 1.0,
        index: Optional[pd.DatetimeIndex] = None,
    ) -> None:
        self.battery_kwh = float(battery_kwh)
        self.max_charge_kw = float(max_charge_kw)
        self.efficiency = float(efficiency)
        self.dt_hours = float(dt_hours)
        self.index = index

    def optimize(
        self,
        prices: np.ndarray,
        initial_soc: float,
        target_soc: float,
        departure_step: Optional[int] = None,
    ) -> ScheduleResult:
        """Compute the cost-optimal EV charging schedule.

        Parameters
        ----------
        prices : array-like, shape (T,)
            Electricity price over the planning horizon.
        initial_soc : float
            Current SoC as fraction of capacity (0–1).
        target_soc : float
            Required SoC at departure (0–1).
        departure_step : int or None
            Timestep index by which ``target_soc`` must be reached.
            Defaults to the last step.

        Returns
        -------
        ScheduleResult
        """
        prices = np.asarray(prices, dtype=float)
        T = len(prices)
        dep = T if departure_step is None else int(departure_step)

        needed_kwh = (target_soc - initial_soc) * self.battery_kwh
        if needed_kwh < 0:
            needed_kwh = 0.0

        # Objective: minimize total charging cost
        c_obj = prices * self.dt_hours

        # Equality constraint: total energy delivered >= needed_kwh
        A_eq = np.ones((1, T))
        A_eq[0, dep:] = 0  # no charging after departure
        b_eq = np.array([needed_kwh / self.efficiency])

        # Per-step SoC cap: cumulative charge <= (1-initial_soc)*cap
        soc_headroom = (1.0 - initial_soc) * self.battery_kwh / self.efficiency
        A_ub_rows = []
        b_ub_rows = []
        for t in range(1, T + 1):
            row = np.zeros(T)
            row[:t] = 1.0
            A_ub_rows.append(row)
            b_ub_rows.append(soc_headroom)

        A_ub = np.array(A_ub_rows)
        b_ub = np.array(b_ub_rows)

        bounds = [(0, self.max_charge_kw)] * T
        # Zero out steps after departure
        for t in range(dep, T):
            bounds[t] = (0, 0)

        result = linprog(
            c_obj,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
        )

        if result.status == 0:
            charge = result.x
            status = "optimal"
        else:
            # Fallback: charge as early/cheaply as possible
            charge = np.zeros(T)
            status = result.message

        total_cost = float(np.sum(charge * prices * self.dt_hours))
        # Baseline: charge at max power from first available step
        baseline_charge = np.zeros(T)
        remaining = needed_kwh / self.efficiency
        for t in range(dep):
            c = min(self.max_charge_kw, remaining / self.dt_hours)
            baseline_charge[t] = c
            remaining -= c * self.dt_hours
            if remaining <= 0:
                break
        baseline_cost = float(np.sum(baseline_charge * prices * self.dt_hours))

        soc = np.zeros(T + 1)
        soc[0] = initial_soc * self.battery_kwh
        for t in range(T):
            soc[t + 1] = soc[t] + charge[t] * self.efficiency * self.dt_hours

        idx = self.index if self.index is not None else pd.RangeIndex(T)
        schedule_df = pd.DataFrame(
            {
                "price_per_kwh": prices,
                "charge_kw": charge,
                "soc_kwh": soc[:T],
                "soc_pct": soc[:T] / self.battery_kwh * 100,
            },
            index=idx,
        )

        return ScheduleResult(
            schedule_df=schedule_df,
            total_cost_usd=total_cost,
            baseline_cost_usd=baseline_cost,
            savings_usd=baseline_cost - total_cost,
            status=status,
        )
