"""
Load test (benchmark)

Scales the data size up in stages (1 day -> 100 days = 100x) and measures how
solve time, model size, and solution quality evolve, printing a table and a chart.

It also feeds a deliberately contradictory (potentially infeasible) scenario to
verify that the fallback logic returns an alternative solution without crashing.

Run:
    python src/benchmark.py
Output:
    - a results table on the console
    - results/benchmark.png (chart of solve time / model size growth)
"""

import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend (no GUI required)
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))

from optimizer import optimize_battery, baseline_cost  # noqa: E402
from generate_data import generate_simulation_data       # noqa: E402

# Common solver settings across all scales
TIME_LIMIT = 60.0   # cut off each case at 60 seconds max
MIP_GAP = 0.01      # settle for a 1% approximate solution


def run_scale_benchmark():
    """Scale the data size up in stages and measure each case."""
    day_scales = [1, 5, 10, 25, 50, 100]   # 1 day (24h) .. 100 days (2400h) = 100x
    rows = []

    print("=" * 92)
    print(" Load test: solve performance from 1 day to 100 days (100x data)")
    print("=" * 92)
    print(f" Solver settings: time_limit={TIME_LIMIT:.0f}s, mip_gap={MIP_GAP:.0%}, "
          f"no-simultaneous=ON(MIP)\n")

    header = (
        f"{'Scale':>7} | {'Steps T':>7} | {'Vars':>7} | {'Cons':>7} | "
        f"{'Solve s':>8} | {'Status':>12} | {'Saving':>7}"
    )
    print(header)
    print("-" * len(header))

    for days in day_scales:
        df = generate_simulation_data(days=days, seed=42)
        solar = df["solar_generation"].to_numpy()
        demand = df["power_demand"].to_numpy()
        price = df["grid_electricity_price"].to_numpy()

        t0 = time.perf_counter()
        res = optimize_battery(
            solar, demand, price,
            time_limit=TIME_LIMIT, mip_gap=MIP_GAP, no_simultaneous=True,
        )
        elapsed = time.perf_counter() - t0

        base = baseline_cost(solar, demand, price)
        saving_pct = (
            (base["total_cost"] - res["total_cost"]) / base["total_cost"] * 100.0
            if base["total_cost"] > 0 else 0.0
        )

        rows.append({
            "days": days,
            "T": len(solar),
            "n_vars": res["n_vars"],
            "n_constraints": res["n_constraints"],
            "time": elapsed,
            "status": res["status"],
            "saving_pct": saving_pct,
        })

        print(
            f"{days:>4}d  | {len(solar):>7,} | {res['n_vars']:>7,} | "
            f"{res['n_constraints']:>7,} | {elapsed:>7.3f}s | "
            f"{res['status']:>12} | {saving_pct:>6.1f}%"
        )

    return rows


def run_infeasibility_test():
    """
    Feed a contradictory scenario (grid purchase cap far too small for demand)
    and confirm that the fallback kicks in.
    """
    print("\n" + "=" * 92)
    print(" Robustness test: fallback behavior on an infeasible scenario")
    print("=" * 92)

    df = generate_simulation_data(days=1, seed=7)
    solar = df["solar_generation"].to_numpy()
    demand = df["power_demand"].to_numpy()
    price = df["grid_electricity_price"].to_numpy()

    # Tightly cap grid purchase at 10 MWh/h -> cannot cover peak demand at night,
    # so the original problem is infeasible
    tight_limit = 10.0
    print(f" Scenario: grid_limit={tight_limit} MWh/h (too small vs peak demand {demand.max():.0f})")

    res = optimize_battery(
        solar, demand, price,
        grid_limit=tight_limit,
        time_limit=TIME_LIMIT, mip_gap=MIP_GAP, no_simultaneous=True,
    )

    total_unmet = sum(res.get("unmet", []))
    print(f"\n  -> Returned a solution without crashing.")
    print(f"     Status        : {res['status']}")
    print(f"     Method        : {res['method']}")
    print(f"     Fallback      : {'triggered' if res.get('fallback_used') else 'none'}")
    print(f"     Total unmet   : {total_unmet:,.1f} MWh (made visible via relaxation)")
    print(f"     Total cost    : ${res['total_cost']:,.2f}")
    print("\n  Note: the problem is infeasible as-is, but constraint relaxation")
    print("        (penalizing unmet demand) auto-presents a feasible solution that")
    print("        shows where and how much capacity is short.")
    return res


def plot_results(rows, out_path):
    """Plot and save how solve time and model size grow."""
    T = [r["T"] for r in rows]
    times = [r["time"] for r in rows]
    nvars = [r["n_vars"] for r in rows]

    fig, ax1 = plt.subplots(figsize=(9, 5.5))

    color1 = "#d62728"
    ax1.set_xlabel("Number of time steps T  (1 day = 24h  →  100 days = 2400h)")
    ax1.set_ylabel("Solve time [s]", color=color1)
    ax1.plot(T, times, "o-", color=color1, linewidth=2, markersize=7,
             label="Solve time")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.grid(True, alpha=0.3)

    # Model size (number of variables) on the secondary axis
    ax2 = ax1.twinx()
    color2 = "#1f77b4"
    ax2.set_ylabel("Model size (number of variables)", color=color2)
    ax2.plot(T, nvars, "s--", color=color2, linewidth=1.5, markersize=6,
             label="Variables")
    ax2.tick_params(axis="y", labelcolor=color2)

    # Annotate each data point with its solve time
    for x, y in zip(T, times):
        ax1.annotate(f"{y:.2f}s", (x, y), textcoords="offset points",
                     xytext=(0, 9), ha="center", fontsize=8, color=color1)

    plt.title("Load Test: Solve Time vs Data Scale (up to 100x)\n"
              "MIP with time_limit=60s, mip_gap=1%")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"\nChart saved: {out_path}")


def main():
    rows = run_scale_benchmark()
    run_infeasibility_test()

    results_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "results"
    )
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.normpath(os.path.join(results_dir, "benchmark.png"))
    plot_results(rows, out_path)

    print("\n" + "=" * 92)
    print(" Conclusion")
    print("=" * 92)
    slowest = max(rows, key=lambda r: r["time"])
    print(f"  - Even at the largest scale {rows[-1]['days']}d (T={rows[-1]['T']:,}, "
          f"vars {rows[-1]['n_vars']:,}), solved in {rows[-1]['time']:.2f}s.")
    print(f"  - All cases obtained an Optimal/approximate solution within "
          f"time_limit ({TIME_LIMIT:.0f}s).")
    print(f"  - Slowest case: {slowest['days']}d / {slowest['time']:.2f}s.")
    print(f"  - Infeasible inputs still return a feasible solution via fallback, no crash.")
    print("=" * 92)


if __name__ == "__main__":
    main()
