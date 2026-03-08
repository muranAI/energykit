"""
energykit.diagnose
===================
One-call energy financial diagnosis.

``diagnose()`` is the **showcase function** of energykit.  Feed it a meter
series and get back a complete financial audit — demand charges, forecast value,
anomaly costs, battery ROI — printed as a terminal dashboard and returned as a
structured ``DiagnosisReport`` object.

The "before / after" story
--------------------------

**Before energykit:**
  "I have a 7% MAPE forecast and I spent $25,000 on electricity last year."

**After energykit (one call):**

.. code-block:: python

    import energykit as ek
    report = ek.diagnose(meter_data)

.. code-block:: text

    ╔══════════════════════════════════════════════════════════════════╗
    ║    ⚡  ENERGYKIT  |  ENERGY FINANCIAL DIAGNOSIS  ⚡             ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║  Period : Jan 2025 → Dec 2025   (8 760 hourly readings)         ║
    ║  Total  : 26 456 kWh   Avg: 3.02 kW   Peak: 5.86 kW            ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║  💡 DEMAND CHARGE RISK                                          ║
    ║  Peak event : May 14 @ 14:00 → 5.86 kW                         ║
    ║  Est. annual demand charge : $879  (@$12.50/kW)                 ║
    ║  Battery [10 kWh / 5 kW]   : save $219/yr  (25%)               ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║  🔍 ANOMALY DETECTION                                           ║
    ║  Anomalies: 23 events  (0.26% of readings)                      ║
    ║  Estimated waste : 312 kWh  →  $47  over the period             ║
    ║  Top anomaly : Mar 12 @ 14:00 — spike  +450 kWh   ($68)         ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║  🔋 DER OPPORTUNITY  (battery dispatch optimisation)            ║
    ║  Battery [13.5 kWh / 5 kW] annual savings : $729                ║
    ║  Estimated payback (@$8 000 install)       : 11.0 yr            ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║  📊 TOTAL ADDRESSABLE SAVINGS                                   ║
    ║  Anomaly correction     :   $47/yr                              ║
    ║  Demand charge opt.     :  $219/yr  [10 kWh battery]            ║
    ║  DER dispatch           :  $729/yr  [13.5 kWh]                  ║
    ║  ──────────────────────────────────────────────────────         ║
    ║  TOTAL POTENTIAL        :  $995/yr  (25% of spend)              ║
    ╚══════════════════════════════════════════════════════════════════╝

Usage
-----
>>> import energykit as ek
>>> report = ek.diagnose(meter_data)              # minimal — uses defaults
>>> report = ek.diagnose(                          # full config
...     meter_data,
...     energy_price=0.15,
...     demand_rate=12.50,
...     battery_cost_usd=8_000,
...     currency="USD",
...     silent=False,
... )
>>> report.total_addressable_savings_usd
994.8
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class DiagnosisReport:
    """Structured output of :func:`diagnose`.

    Attributes
    ----------
    total_kwh : float
        Total consumption over the input period.
    avg_kw : float
        Average power demand.
    peak_kw : float
        Maximum peak power observed.
    peak_timestamp : pd.Timestamp or None

    demand_charge_annual_usd : float
        Estimated annual demand charge at the given rate.
    demand_peak_kw : float
    demand_peak_timestamp : pd.Timestamp or None
    demand_best_battery_kwh : float
        Battery capacity (kWh) that gives best $/year savings.
    demand_best_battery_savings_usd : float

    anomaly_count : int
    anomaly_rate_pct : float
    anomaly_waste_kwh : float
    anomaly_cost_usd : float
        Dollar value of all anomalous energy waste.

    der_battery_kwh : float
        Battery size used in the DER dispatch simulation.
    der_annual_savings_usd : float
    der_payback_years : float

    total_addressable_savings_usd : float
        Sum of all identified savings opportunities.
    pct_of_spend : float
        Savings as a percentage of estimated annual energy spend.

    raw : dict
        Raw outputs from each sub-module for further analysis.
    """

    # --- consumption stats ---
    total_kwh: float = 0.0
    avg_kw: float = 0.0
    peak_kw: float = 0.0
    peak_timestamp: Optional[pd.Timestamp] = None
    n_readings: int = 0
    data_start: Optional[str] = None
    data_end: Optional[str] = None

    # --- demand charge ---
    demand_charge_annual_usd: float = 0.0
    demand_peak_kw: float = 0.0
    demand_peak_timestamp: Optional[pd.Timestamp] = None
    demand_best_battery_kwh: float = 0.0
    demand_best_battery_savings_usd: float = 0.0

    # --- anomaly ---
    anomaly_count: int = 0
    anomaly_rate_pct: float = 0.0
    anomaly_waste_kwh: float = 0.0
    anomaly_cost_usd: float = 0.0

    # --- DER optimisation ---
    der_battery_kwh: float = 13.5
    der_annual_savings_usd: float = 0.0
    der_payback_years: float = float("inf")

    # --- summary ---
    total_addressable_savings_usd: float = 0.0
    pct_of_spend: float = 0.0

    # --- raw module outputs ---
    raw: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"DiagnosisReport("
            f"total_kwh={self.total_kwh:.0f}, "
            f"savings=${self.total_addressable_savings_usd:.0f}/yr, "
            f"anomalies={self.anomaly_count})"
        )


# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------

def diagnose(
    meter_data: pd.Series,
    energy_price: float = 0.15,
    demand_rate: float = 12.50,
    peak_hours: Optional[list] = None,
    battery_cost_usd: float = 8_000.0,
    currency: str = "$",
    silent: bool = False,
) -> DiagnosisReport:
    """Run a complete energy financial diagnosis on a meter series.

    Parameters
    ----------
    meter_data : pd.Series
        Electricity meter readings (kW or kWh) with a ``pd.DatetimeIndex``.
        Hourly or 15-minute resolution recommended.
    energy_price : float, default 0.15
        Retail energy price in $/kWh.  Used for anomaly cost estimation and
        DER savings valuation.
    demand_rate : float, default 12.50
        Demand charge rate in $/kW/month.  A common commercial rate in the US.
        Set to 0 to skip demand charge analysis.
    peak_hours : list of int or None
        Hours (0–23) when demand charges apply.  ``None`` = all hours.
    battery_cost_usd : float, default 8_000
        All-in installed cost of a residential/small-commercial battery (USD).
        Used only to compute the payback period estimate.
    currency : str, default "USD"
        Currency symbol for the report.  Does not perform conversion.
    silent : bool, default False
        If True, suppress the printed ASCII dashboard.

    Returns
    -------
    DiagnosisReport
        Structured report with all financials.  Access ``report.raw`` for the
        full outputs of each sub-module.

    Examples
    --------
    >>> from energykit.datasets import load_synthetic_load
    >>> data = load_synthetic_load(periods=8_760, freq="h")["load_kw"]
    >>> report = diagnose(data)
    """
    from energykit.cost.demand_charge import DemandChargeAnalyzer
    from energykit.anomaly.detector import MeterAnomalyDetector
    from energykit.optimize.der import BatteryScheduler
    from energykit.datasets.loaders import load_sample_tou_prices

    report = DiagnosisReport()

    if not isinstance(meter_data.index, pd.DatetimeIndex):
        raise ValueError(
            "meter_data must have a pd.DatetimeIndex.  "
            "Example: pd.Series(values, index=pd.date_range(...))"
        )

    series = meter_data.dropna().clip(lower=0)
    dt_hours = _infer_dt_hours(series)

    # ------------------------------------------------------------------
    # 1. Basic stats
    # ------------------------------------------------------------------
    report.total_kwh = float(series.sum() * dt_hours)
    report.avg_kw = float(series.mean())
    report.peak_kw = float(series.max())
    report.peak_timestamp = series.idxmax()
    report.n_readings = int(len(series))
    report.data_start = series.index[0].strftime("%b %Y")
    report.data_end = series.index[-1].strftime("%b %Y")

    # ------------------------------------------------------------------
    # 2. Demand charge analysis
    # ------------------------------------------------------------------
    demand_result = None
    if demand_rate > 0:
        try:
            analyzer = DemandChargeAnalyzer(
                demand_rate=demand_rate,
                peak_hours=peak_hours,
                dt_hours=dt_hours,
            )
            demand_result = analyzer.analyze(series)
            report.demand_charge_annual_usd = demand_result.total_annual_charge_usd
            report.demand_peak_kw = demand_result.worst_event.peak_kw
            report.demand_peak_timestamp = demand_result.worst_event.timestamp
            report.raw["demand_charge"] = demand_result

            # Best battery recommendation: highest savings-to-capacity ratio
            bc = demand_result.battery_savings_df
            if not bc.empty:
                best = bc.loc[bc["annual_savings_usd"].idxmax()]
                report.demand_best_battery_kwh = float(best["battery_kwh"])
                report.demand_best_battery_savings_usd = float(best["annual_savings_usd"])
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 3. Anomaly detection
    # ------------------------------------------------------------------
    anomaly_result = None
    try:
        detector = MeterAnomalyDetector(z_threshold=2.5)
        detector.fit(series)
        anomaly_result = detector.detect(series, energy_price=energy_price)
        report.anomaly_count = anomaly_result.n_anomalies
        report.anomaly_rate_pct = anomaly_result.anomaly_rate_pct
        report.anomaly_waste_kwh = anomaly_result.total_excess_kwh
        report.anomaly_cost_usd = anomaly_result.total_estimated_cost_usd
        report.raw["anomaly"] = anomaly_result
    except Exception:
        pass

    # ------------------------------------------------------------------
    # 4. DER optimisation (battery dispatch)
    # ------------------------------------------------------------------
    der_result = None
    try:
        prices = load_sample_tou_prices(tariff="residential_us", periods=len(series))
        scheduler = BatteryScheduler(
            capacity_kwh=13.5,
            max_power_kw=5.0,
            efficiency=0.95,
        )
        der_result = scheduler.optimize(prices=prices, load_kw=series)
        raw_savings = float(der_result.savings_usd)
        data_hours = report.total_kwh / max(report.avg_kw, 1e-6)
        scale = 8760.0 / max(data_hours, 1.0)
        annual_der_savings = raw_savings * scale
        report.der_battery_kwh = 13.5
        report.der_annual_savings_usd = round(annual_der_savings, 2)
        if annual_der_savings > 0:
            report.der_payback_years = round(battery_cost_usd / annual_der_savings, 1)
        report.raw["der"] = der_result
    except Exception:
        pass

    # ------------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------------
    total_savings = (
        report.anomaly_cost_usd
        + report.demand_best_battery_savings_usd
        + report.der_annual_savings_usd
    )
    report.total_addressable_savings_usd = round(total_savings, 2)

    annual_spend = report.total_kwh * energy_price * (8760.0 / max(report.n_readings * dt_hours, 1))
    if annual_spend > 0:
        report.pct_of_spend = round(total_savings / annual_spend * 100, 1)

    # ------------------------------------------------------------------
    # 6. Print ASCII dashboard
    # ------------------------------------------------------------------
    if not silent:
        _print_report(report, demand_result, anomaly_result, currency, energy_price)

    return report


# ---------------------------------------------------------------------------
# ASCII terminal report
# ---------------------------------------------------------------------------

_W = 68  # total box width (inner)


def _box_line(content: str = "", fill: str = " ") -> str:
    """Left-pad content inside the box."""
    inner = f"  {content}"
    return "║" + inner.ljust(_W) + "║"


def _separator(char: str = "═") -> str:
    return "╠" + char * _W + "╣"


def _top() -> str:
    return "╔" + "═" * _W + "╗"


def _bottom() -> str:
    return "╚" + "═" * _W + "╝"


def _divider() -> str:
    return _box_line("─" * (_W - 4))


def _fmt(value: float, currency: str = "$") -> str:
    return f"{currency}{value:,.0f}"


def _print_report(
    r: DiagnosisReport,
    demand_result,
    anomaly_result,
    currency: str,
    energy_price: float,
) -> None:
    C = currency

    header_text = "⚡  ENERGYKIT  |  ENERGY FINANCIAL DIAGNOSIS  ⚡"
    header_inner = header_text.center(_W - 2)
    header_line = "║ " + header_inner + " ║"

    lines = [
        _top(),
        header_line,
        _separator(),
    ]

    # --- Data summary ---
    start = r.data_start or "?"
    end = r.data_end or "?"
    period_str = f"{start} → {end}"
    n_str = f"{r.n_readings:,} readings"
    lines.append(_box_line(f"Period  : {period_str}   ({n_str})"))
    lines.append(
        _box_line(
            f"Total   : {r.total_kwh:,.0f} kWh   "
            f"Avg: {r.avg_kw:.2f} kW   "
            f"Peak: {r.peak_kw:.2f} kW"
        )
    )
    lines.append(_separator())

    # --- Demand charge ---
    lines.append(_box_line("💡 DEMAND CHARGE RISK"))
    lines.append(_divider())
    if r.demand_charge_annual_usd > 0:
        ts_str = (
            r.demand_peak_timestamp.strftime("%b %d @ %H:%M")
            if r.demand_peak_timestamp is not None
            else "N/A"
        )
        lines.append(_box_line(f"Peak event  : {ts_str}  →  {r.demand_peak_kw:.2f} kW"))
        lines.append(
            _box_line(
                f"Est. annual demand charge : {_fmt(r.demand_charge_annual_usd, C)}"
                f"  (@{C}{12.50:.2f}/kW)"
            )
        )
        if demand_result is not None:
            bc = demand_result.battery_savings_df
            for _, row in bc.iterrows():
                savings = row["annual_savings_usd"]
                pct = row["pct_reduction"]
                label = f"Battery [{row['battery_kwh']:.0f} kWh / {row['max_power_kw']:.0f} kW]"
                lines.append(
                    _box_line(
                        f"{label:<28}: save {_fmt(savings, C)}/yr  ({pct:.0f}%)"
                    )
                )
    else:
        lines.append(_box_line("Demand rate not configured — skipped."))
    lines.append(_separator())

    # --- Anomaly ---
    lines.append(_box_line("🔍 ANOMALY DETECTION"))
    lines.append(_divider())
    if r.anomaly_count >= 0:
        lines.append(
            _box_line(
                f"Anomalies : {r.anomaly_count} events  ({r.anomaly_rate_pct:.2f}% of readings)"
            )
        )
        lines.append(
            _box_line(
                f"Est. waste : {r.anomaly_waste_kwh:.1f} kWh  →  "
                f"{_fmt(r.anomaly_cost_usd, C)}  over the period"
            )
        )
        if anomaly_result is not None and anomaly_result.worst_event is not None:
            w = anomaly_result.worst_event
            ts_str = w.name.strftime("%b %d @ %H:%M") if hasattr(w.name, "strftime") else str(w.name)
            lines.append(
                _box_line(
                    f"Top anomaly : {ts_str}"
                    f" — {w['anomaly_type']}"
                    f"  +{w['excess_kwh']:.0f} kWh"
                    f"  ({_fmt(w['estimated_cost_usd'], C)})"
                )
            )
    else:
        lines.append(_box_line("No anomaly data available."))
    lines.append(_separator())

    # --- DER ---
    lines.append(_box_line("🔋 DER OPPORTUNITY  (battery dispatch optimisation)"))
    lines.append(_divider())
    if r.der_annual_savings_usd > 0:
        lines.append(
            _box_line(
                f"Battery [{r.der_battery_kwh:.1f} kWh / 5 kW] annual savings"
                f" : {_fmt(r.der_annual_savings_usd, C)}"
            )
        )
        pb = (
            f"{r.der_payback_years:.1f} yr"
            if r.der_payback_years < 100
            else "N/A"
        )
        lines.append(
            _box_line(
                f"Estimated payback (@{_fmt(8_000, C)} install)"
                f"       : {pb}"
            )
        )
    else:
        lines.append(_box_line("No DER opportunity computed."))
    lines.append(_separator())

    # --- Summary ---
    lines.append(_box_line("📊 TOTAL ADDRESSABLE SAVINGS"))
    lines.append(_divider())
    lines.append(
        _box_line(
            f"Anomaly correction   : {_fmt(r.anomaly_cost_usd, C):>8}/yr"
        )
    )
    bk = r.demand_best_battery_kwh
    lines.append(
        _box_line(
            f"Demand charge opt.   : {_fmt(r.demand_best_battery_savings_usd, C):>8}/yr"
            + (f"  [{bk:.0f} kWh battery]" if bk > 0 else "")
        )
    )
    lines.append(
        _box_line(
            f"DER dispatch         : {_fmt(r.der_annual_savings_usd, C):>8}/yr"
            f"  [{r.der_battery_kwh:.1f} kWh]"
        )
    )
    lines.append(_divider())
    pct_str = f"  ({r.pct_of_spend:.0f}% of annual spend)" if r.pct_of_spend > 0 else ""
    lines.append(
        _box_line(
            f"TOTAL POTENTIAL      : {_fmt(r.total_addressable_savings_usd, C):>8}/yr"
            + pct_str
        )
    )
    lines.append(_bottom())

    print("\n".join(lines))


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _infer_dt_hours(series: pd.Series) -> float:
    """Infer the timestep duration in hours from the DatetimeIndex."""
    if len(series) < 2:
        return 1.0
    diffs = pd.Series(series.index).diff().dropna()
    median_seconds = float(diffs.median().total_seconds())
    return median_seconds / 3600.0
