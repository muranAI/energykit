Quickstart
==========

One-call financial audit
-------------------------

Feed any smart-meter ``pd.Series`` to ``ek.diagnose()`` and get a complete financial audit:

.. code-block:: python

   import energykit as ek
   from energykit.datasets import load_synthetic_load

   # Load one year of hourly meter data (or bring your own pd.Series)
   data = load_synthetic_load(periods=8760, freq="h")

   # Run the full financial audit
   report = ek.diagnose(data, energy_price=0.15, demand_rate=12.50)

   # All numbers are in the return value:
   print(report.total_addressable_savings_usd)   # 1453.21
   print(report.demand_charge_annual_usd)        # 701.55
   print(report.anomaly_count)                   # 23
   print(report.der_annual_savings_usd)          # 729.00

Demand charge analysis
-----------------------

.. code-block:: python

   from energykit.cost import DemandChargeAnalyzer

   analyzer = DemandChargeAnalyzer(demand_rate=12.50)
   result = analyzer.analyze(power_kw_series)

   print(result.peak_events_df)         # which events cost the most
   print(result.battery_savings_df)     # what a battery would have saved

Anomaly detection with financial impact
----------------------------------------

.. code-block:: python

   from energykit.anomaly import MeterAnomalyDetector

   detector = MeterAnomalyDetector(z_threshold=2.5)
   detector.fit(historical_series)
   result = detector.detect(new_series, energy_price=0.15)

   print(result)
   # AnomalySummary(n=23, rate=0.26%, waste=312.4 kWh, cost=$46.86)

Imbalance cost
--------------

.. code-block:: python

   from energykit.cost import ImbalanceCostCalculator

   calc = ImbalanceCostCalculator(imbalance_price=0.08)
   result = calc.compute(forecast, actual)

   print(f"Annual imbalance cost: ${result.annual_cost_estimate_usd:,.0f}")
   print(f"Cost per 1% MAPE:      ${result.cost_per_mape_pct_usd:,.0f}/yr")

Battery scheduling
------------------

.. code-block:: python

   import numpy as np
   from energykit.optimize import BatteryScheduler

   prices = np.array([0.09]*8 + [0.22]*9 + [0.28]*5 + [0.09]*2)
   battery = BatteryScheduler(capacity_kwh=13.5, max_power_kw=5.0, efficiency=0.90)
   result = battery.optimize(prices, load_kw=baseline_load)
   print(f"Daily savings: ${result.savings_usd:.2f}")

Load forecasting
----------------

.. code-block:: python

   from energykit.forecast import LoadForecaster

   model = LoadForecaster(horizon=24, country="US", lags=[1, 24, 168])
   model.fit(load_series)
   forecast = model.predict()    # next 24 hours as pd.Series
