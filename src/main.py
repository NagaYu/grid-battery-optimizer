"""
Main script: load the simulation data, optimize battery charge/discharge, and
print a cost-reduction report to the console.

Run:
    python src/main.py
"""

import os
import sys
import pandas as pd

# Add the src directory to the import path (so it runs from anywhere)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from optimizer import optimize_battery, baseline_cost  # noqa: E402

# --- Battery specs ---
CAPACITY = 100.0      # maximum capacity (MWh)
MAX_RATE = 20.0       # maximum charge/discharge rate (MW = MWh/h)
INITIAL_SOC = 0.0     # initial state of charge (MWh)
LOSS_COEFF = 0.0006   # transmission loss coeff a (loss = a * flow^2, I^2R approx)


def load_data():
    data_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "data", "power_data.csv"
    )
    data_path = os.path.normpath(data_path)
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Data not found: {data_path}\n"
            f"Run `python data/generate_data.py` first."
        )
    return pd.read_csv(data_path)


def fmt_money(x):
    return f"${x:,.2f}"


def main():
    print("=" * 70)
    print(" Grid Battery Dispatch Optimizer")
    print(" Battery Charge/Discharge Optimization for Grid Loss & Cost Reduction")
    print("=" * 70)

    df = load_data()
    solar = df["solar_generation"].tolist()
    demand = df["power_demand"].tolist()
    price = df["grid_electricity_price"].tolist()

    print(f"\nBattery specs: capacity {CAPACITY:.0f} MWh / "
          f"max rate {MAX_RATE:.0f} MW / initial SoC {INITIAL_SOC:.0f} MWh\n")

    # --- Optimization ---
    result = optimize_battery(
        solar, demand, price,
        capacity=CAPACITY, max_rate=MAX_RATE, initial_soc=INITIAL_SOC,
        loss_coeff=LOSS_COEFF,
    )
    base = baseline_cost(solar, demand, price, loss_coeff=LOSS_COEFF)

    # --- Status ---
    print("[Optimization status]")
    print(f"  Solver result: {result['status']}")
    if result["status"] != "Optimal":
        print("  ! No optimal solution found. Check the constraints.")
        return
    print("  OK: optimal solution found.\n")

    # --- Hourly table ---
    print("[Hourly optimal schedule]")
    header = (
        f"{'Hr':>3} | {'Solar':>7} | {'Demand':>7} | {'Price':>7} | "
        f"{'Buy':>7} | {'Chg':>6} | {'Dis':>6} | {'Action':>9} | {'SoC':>9}"
    )
    print(header)
    print("-" * len(header))
    for t in range(24):
        ch = result["charge"][t] or 0.0
        dis = result["discharge"][t] or 0.0
        soc = result["soc"][t] or 0.0
        if ch > 1e-4:
            action = "CHARGE ^"
        elif dis > 1e-4:
            action = "DISCHG v"
        else:
            action = "IDLE  -"
        # SoC bar (simple gauge)
        bar_len = int(round(soc / CAPACITY * 10))
        soc_bar = "█" * bar_len + "·" * (10 - bar_len)
        print(
            f"{t:>3} | {solar[t]:>7.2f} | {demand[t]:>7.2f} | {price[t]:>7.2f} | "
            f"{result['grid_buy'][t]:>7.2f} | {ch:>6.2f} | {dis:>6.2f} | "
            f"{action:>9} | {soc:>6.1f} {soc_bar}"
        )

    # --- Cost comparison ---
    base_cost = base["total_cost"]
    opt_cost = result["total_cost"]
    saving = base_cost - opt_cost
    saving_pct = (saving / base_cost * 100.0) if base_cost > 0 else 0.0

    total_curtail_base = sum(base["solar_curtail"])
    total_curtail_opt = sum(result["solar_curtail"])
    total_loss_base = sum(base.get("transmission_loss", []))
    total_loss_opt = sum(result.get("transmission_loss", []))

    print("\n" + "=" * 70)
    print("[Cost & transmission-loss reduction report]")
    print("-" * 70)
    print(f"  Before (no battery: curtail solar surplus, buy all shortfall)")
    print(f"      Total cost        : {fmt_money(base_cost)}")
    print(f"      Transmission loss : {total_loss_base:,.1f} MWh")
    print(f"      Curtailed solar   : {total_curtail_base:,.1f} MWh")
    print(f"  After (smart battery charge/discharge)")
    print(f"      Total cost        : {fmt_money(opt_cost)}")
    print(f"      Transmission loss : {total_loss_opt:,.1f} MWh")
    print(f"      Curtailed solar   : {total_curtail_opt:,.1f} MWh")
    loss_delta = total_loss_base - total_loss_opt  # positive = reduction, negative = increase
    print("-" * 70)
    print(f"  $ Cost saved         : {fmt_money(saving)}")
    print(f"  % Cost reduction     : {saving_pct:.1f} %")
    print(f"  Solar recovered      : {total_curtail_base - total_curtail_opt:,.1f} MWh "
          f"(reused curtailed renewables = less renewable waste)")
    if loss_delta >= 0:
        print(f"  Transmission loss    : {total_loss_base:.1f} -> {total_loss_opt:.1f} MWh "
              f"({loss_delta:.1f} MWh reduced)")
    else:
        print(f"  Transmission loss    : {total_loss_base:.1f} -> {total_loss_opt:.1f} MWh "
              f"({-loss_delta:.1f} MWh increase / trade-off: overnight arbitrage raises throughput)")
    print("-" * 70)
    print("  Note: the objective minimizes total cost with transmission loss priced in.")
    print("        To suppress loss further, raise the optimizer's loss_coeff so the")
    print("        marginal cost of loss is weighted more heavily.")
    print("=" * 70)


if __name__ == "__main__":
    main()
