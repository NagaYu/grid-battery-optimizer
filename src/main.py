"""
メインスクリプト: シミュレーションデータを読み込み、バッテリー充放電を最適化し、
コスト削減効果のレポートをコンソールに出力する。

実行:
    python src/main.py
"""

import os
import sys
import pandas as pd

# src ディレクトリを import パスに追加（どこから実行しても動くように）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from optimizer import optimize_battery, baseline_cost  # noqa: E402

# --- バッテリー諸元 ---
CAPACITY = 100.0      # 最大容量 (MWh)
MAX_RATE = 20.0       # 最大充放電レート (MW = MWh/h)
INITIAL_SOC = 0.0     # 初期蓄電量 (MWh)
LOSS_COEFF = 0.0006   # 送電損失係数 a（loss = a * 流量^2, I^2R 損の近似）


def load_data():
    data_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "data", "power_data.csv"
    )
    data_path = os.path.normpath(data_path)
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"データが見つかりません: {data_path}\n"
            f"先に `python data/generate_data.py` を実行してください。"
        )
    return pd.read_csv(data_path)


def fmt_money(x):
    return f"${x:,.2f}"


def main():
    print("=" * 70)
    print(" BHE 蓄電池 充放電最適化シミュレーター")
    print(" Battery Charge/Discharge Optimization for Grid Loss & Cost Reduction")
    print("=" * 70)

    df = load_data()
    solar = df["solar_generation"].tolist()
    demand = df["power_demand"].tolist()
    price = df["grid_electricity_price"].tolist()

    print(f"\nバッテリー諸元: 最大容量 {CAPACITY:.0f} MWh / "
          f"最大充放電レート {MAX_RATE:.0f} MW / 初期蓄電量 {INITIAL_SOC:.0f} MWh\n")

    # --- 最適化 ---
    result = optimize_battery(
        solar, demand, price,
        capacity=CAPACITY, max_rate=MAX_RATE, initial_soc=INITIAL_SOC,
        loss_coeff=LOSS_COEFF,
    )
    base = baseline_cost(solar, demand, price, loss_coeff=LOSS_COEFF)

    # --- ステータス ---
    print("【最適化ステータス】")
    print(f"  ソルバー結果: {result['status']}")
    if result["status"] != "Optimal":
        print("  ⚠ 最適解が得られませんでした。制約条件を確認してください。")
        return
    print("  ✓ 最適解 (Optimal) が得られました。\n")

    # --- 時間別テーブル ---
    print("【時間別 最適スケジュール】")
    header = (
        f"{'時':>3} | {'太陽光':>7} | {'需要':>7} | {'価格':>7} | "
        f"{'購入':>7} | {'充電':>6} | {'放電':>6} | {'動作':>6} | {'蓄電量SoC':>9}"
    )
    print(header)
    print("-" * len(header))
    for t in range(24):
        ch = result["charge"][t] or 0.0
        dis = result["discharge"][t] or 0.0
        soc = result["soc"][t] or 0.0
        if ch > 1e-4:
            action = "充電 ▲"
        elif dis > 1e-4:
            action = "放電 ▼"
        else:
            action = "待機 ―"
        # SoC バー（簡易ゲージ）
        bar_len = int(round(soc / CAPACITY * 10))
        soc_bar = "█" * bar_len + "·" * (10 - bar_len)
        print(
            f"{t:>3} | {solar[t]:>7.2f} | {demand[t]:>7.2f} | {price[t]:>7.2f} | "
            f"{result['grid_buy'][t]:>7.2f} | {ch:>6.2f} | {dis:>6.2f} | "
            f"{action:>6} | {soc:>6.1f} {soc_bar}"
        )

    # --- コスト比較 ---
    base_cost = base["total_cost"]
    opt_cost = result["total_cost"]
    saving = base_cost - opt_cost
    saving_pct = (saving / base_cost * 100.0) if base_cost > 0 else 0.0

    total_curtail_base = sum(base["solar_curtail"])
    total_curtail_opt = sum(result["solar_curtail"])
    total_loss_base = sum(base.get("transmission_loss", []))
    total_loss_opt = sum(result.get("transmission_loss", []))

    print("\n" + "=" * 70)
    print("【コスト・送電ロス削減レポート】")
    print("-" * 70)
    print(f"  最適化前（バッテリーなし: 余剰太陽光を捨て不足分を全量購入）")
    print(f"      総コスト          : {fmt_money(base_cost)}")
    print(f"      送電ロス          : {total_loss_base:,.1f} MWh")
    print(f"      捨てた太陽光       : {total_curtail_base:,.1f} MWh")
    print(f"  最適化後（バッテリーを賢く充放電）")
    print(f"      総コスト          : {fmt_money(opt_cost)}")
    print(f"      送電ロス          : {total_loss_opt:,.1f} MWh")
    print(f"      捨てた太陽光       : {total_curtail_opt:,.1f} MWh")
    loss_delta = total_loss_base - total_loss_opt  # 正=削減, 負=増加
    print("-" * 70)
    print(f"  💰 コスト削減額      : {fmt_money(saving)}")
    print(f"  📉 コスト削減率      : {saving_pct:.1f} %")
    print(f"  ☀ 救済した太陽光     : {total_curtail_base - total_curtail_opt:,.1f} MWh "
          f"(捨てていた再エネを活用 = 再エネロスを削減)")
    if loss_delta >= 0:
        print(f"  ⚡ 送電ロス          : {total_loss_base:.1f} → {total_loss_opt:.1f} MWh "
              f"（{loss_delta:.1f} MWh 削減）")
    else:
        print(f"  ⚡ 送電ロス          : {total_loss_base:.1f} → {total_loss_opt:.1f} MWh "
              f"（{-loss_delta:.1f} MWh 増 / 夜間アービトラージで送電量が増えるトレードオフ）")
    print("-" * 70)
    print("  ※ 目的関数は『送電損失を価格に織り込んだ総コスト』の最小化。")
    print("     送電ロスを更に抑えたい場合は optimizer の loss_coeff を上げ、")
    print("     損失の限界コストを高く評価させると挙動が変わります。")
    print("=" * 70)


if __name__ == "__main__":
    main()
