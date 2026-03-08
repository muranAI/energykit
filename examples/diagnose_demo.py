from energykit.datasets import load_synthetic_load
import energykit as ek

data = load_synthetic_load(periods=8760, freq="h")
report = ek.diagnose(data, energy_price=0.15, demand_rate=12.50)

print()
print("=== DiagnosisReport structured fields ===")
print(f"Total kWh         : {report.total_kwh:,.0f}")
print(f"Peak kW           : {report.peak_kw:.2f}")
print(f"Anomalies found   : {report.anomaly_count}")
print(f"Anomaly cost      : ${report.anomaly_cost_usd:.2f}")
print(f"Annual demand chg : ${report.demand_charge_annual_usd:.2f}")
print(f"DER savings/yr    : ${report.der_annual_savings_usd:.2f}")
print(f"TOTAL savings/yr  : ${report.total_addressable_savings_usd:.2f}")
print(f"Pct of spend      : {report.pct_of_spend:.1f}%")
