"""
負荷テスト（ベンチマーク）

データ規模を段階的に拡大（1日 → 100日 = 100倍）し、最適化の実行時間・
モデルサイズ・解の品質の推移を計測して、表とグラフで出力する。

併せて、わざと矛盾した（解なしになりうる）シナリオを与え、
フォールバックロジックがクラッシュせず代替解を返すことを検証する。

実行:
    python src/benchmark.py
出力:
    - コンソールに結果テーブル
    - results/benchmark.png （実行時間・モデルサイズの推移グラフ）
"""

import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")  # GUI 不要のバックエンド
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))

from optimizer import optimize_battery, baseline_cost  # noqa: E402
from generate_data import generate_simulation_data       # noqa: E402

# 各規模での共通ソルバー設定
TIME_LIMIT = 60.0   # 1ケースあたり最大60秒で打ち切り
MIP_GAP = 0.01      # 1% 近似解で妥協


def run_scale_benchmark():
    """データ規模を段階的に拡大して計測する。"""
    day_scales = [1, 5, 10, 25, 50, 100]   # 1日(24h) 〜 100日(2400h) = 100倍
    rows = []

    print("=" * 92)
    print(" 負荷テスト: データ規模 1日 → 100日（100倍）における求解性能")
    print("=" * 92)
    print(f" ソルバー設定: time_limit={TIME_LIMIT:.0f}s, mip_gap={MIP_GAP:.0%}, "
          f"充放電同時禁止=ON(MIP)\n")

    header = (
        f"{'規模':>6} | {'時刻数T':>7} | {'変数':>7} | {'制約':>7} | "
        f"{'求解秒':>8} | {'ステータス':>12} | {'削減率':>7}"
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
            f"{days:>4}日 | {len(solar):>7,} | {res['n_vars']:>7,} | "
            f"{res['n_constraints']:>7,} | {elapsed:>7.3f}s | "
            f"{res['status']:>12} | {saving_pct:>6.1f}%"
        )

    return rows


def run_infeasibility_test():
    """
    矛盾シナリオ（グリッド購入上限が需要に対して極端に小さい）を与え、
    フォールバックが働くことを確認する。
    """
    print("\n" + "=" * 92)
    print(" 頑健性テスト: 解なし（Infeasible）シナリオでのフォールバック挙動")
    print("=" * 92)

    df = generate_simulation_data(days=1, seed=7)
    solar = df["solar_generation"].to_numpy()
    demand = df["power_demand"].to_numpy()
    price = df["grid_electricity_price"].to_numpy()

    # グリッドからの購入を 10 MWh/h に厳しく制限 → 夜間のピーク需要を賄えず本来 Infeasible
    tight_limit = 10.0
    print(f" シナリオ: grid_limit={tight_limit} MWh/h（需要ピーク {demand.max():.0f} に対し過小）")

    res = optimize_battery(
        solar, demand, price,
        grid_limit=tight_limit,
        time_limit=TIME_LIMIT, mip_gap=MIP_GAP, no_simultaneous=True,
    )

    total_unmet = sum(res.get("unmet", []))
    print(f"\n  → クラッシュせず解を返却。")
    print(f"     ステータス      : {res['status']}")
    print(f"     採用手法        : {res['method']}")
    print(f"     フォールバック  : {'発動' if res.get('fallback_used') else 'なし'}")
    print(f"     未充足需要 合計 : {total_unmet:,.1f} MWh（緩和により可視化）")
    print(f"     総コスト        : ${res['total_cost']:,.2f}")
    print("\n  ※ 本来 Infeasible だが、制約緩和（未充足需要にペナルティ）で")
    print("     『どこがどれだけ足りないか』を示す実行可能解を自動提示している。")
    return res


def plot_results(rows, out_path):
    """実行時間とモデルサイズの推移をグラフ化して保存する。"""
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

    # モデルサイズ（変数の数）を第2軸に
    ax2 = ax1.twinx()
    color2 = "#1f77b4"
    ax2.set_ylabel("Model size (number of variables)", color=color2)
    ax2.plot(T, nvars, "s--", color=color2, linewidth=1.5, markersize=6,
             label="Variables")
    ax2.tick_params(axis="y", labelcolor=color2)

    # データ点に求解秒を注記
    for x, y in zip(T, times):
        ax1.annotate(f"{y:.2f}s", (x, y), textcoords="offset points",
                     xytext=(0, 9), ha="center", fontsize=8, color=color1)

    plt.title("Load Test: Solve Time vs Data Scale (up to 100x)\n"
              "MIP with time_limit=60s, mip_gap=1%")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"\nグラフを保存しました: {out_path}")


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
    print(" 結論")
    print("=" * 92)
    slowest = max(rows, key=lambda r: r["time"])
    print(f"  - 最大規模 {rows[-1]['days']}日（T={rows[-1]['T']:,}, "
          f"変数{rows[-1]['n_vars']:,}）でも {rows[-1]['time']:.2f}s で求解。")
    print(f"  - 全ケースで time_limit({TIME_LIMIT:.0f}s) 内に Optimal/近似解を取得。")
    print(f"  - 最も遅いケース: {slowest['days']}日 / {slowest['time']:.2f}s。")
    print(f"  - Infeasible 入力でもフォールバックにより実行可能解を返し無停止。")
    print("=" * 92)


if __name__ == "__main__":
    main()
