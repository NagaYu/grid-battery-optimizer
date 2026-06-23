"""
24時間分（1時間刻み）の電力シミュレーションデータを生成する。

想定シナリオ（BHE: バークシャー・ハサウェイ・エナジー の一地域グリッド）:
  - 太陽光発電(solar_generation): 日中(10〜14時)にピーク、夜間はゼロ。
  - 電力需要(power_demand): 朝(7〜9時)と夕方(18〜21時)に二峰性のピーク。
  - 系統電力価格(grid_electricity_price): 需要に連動し、朝夕のピーク時間帯に高騰。

単位:
  - solar_generation, power_demand : MW（その時間に一定とみなす → MWh）
  - grid_electricity_price         : $/MWh
"""

import numpy as np
import pandas as pd
import os


def generate_simulation_data(days: int = 1, seed: int = 42) -> pd.DataFrame:
    """
    `days` 日分（24*days 時刻）のシミュレーションデータを生成する。
    days=1 で従来の24時間。負荷テストでは days を増やしてデータ規模を拡大する。
    """
    rng = np.random.default_rng(seed)
    T = 24 * days
    t_idx = np.arange(T)
    hod = t_idx % 24  # hour-of-day (0..23)

    # --- 太陽光発電 (MW): 日の出〜日の入りの釣鐘型カーブ ---
    solar_peak = 80.0
    solar = np.zeros(T)
    daylight = (hod >= 6) & (hod <= 18)
    solar[daylight] = solar_peak * np.sin(np.pi * (hod[daylight] - 6) / 12.0)
    solar = np.clip(solar + rng.normal(0, 2.0, T) * (solar > 0), 0, None)

    # --- 電力需要 (MW): 朝・夕の二峰性ピーク + ベース負荷 ---
    base_demand = 50.0
    morning_peak = 35.0 * np.exp(-((hod - 8) ** 2) / (2 * 1.8 ** 2))
    evening_peak = 45.0 * np.exp(-((hod - 19) ** 2) / (2 * 2.0 ** 2))
    demand = base_demand + morning_peak + evening_peak + rng.normal(0, 1.5, T)

    # --- 系統電力価格 ($/MWh): 需要ピーク時間帯に高騰、深夜は安い ---
    base_price = 40.0
    morning_price = 60.0 * np.exp(-((hod - 8) ** 2) / (2 * 2.0 ** 2))
    evening_price = 90.0 * np.exp(-((hod - 19) ** 2) / (2 * 2.2 ** 2))
    midday_discount = -15.0 * np.exp(-((hod - 12.5) ** 2) / (2 * 2.0 ** 2))
    price = base_price + morning_price + evening_price + midday_discount
    price = np.clip(price + rng.normal(0, 1.0, T), 10.0, None)

    df = pd.DataFrame({
        "hour": hod,                       # 0..23（日内時刻）
        "t": t_idx,                        # 0..T-1（通し時刻）
        "solar_generation": np.round(solar, 2),
        "power_demand": np.round(demand, 2),
        "grid_electricity_price": np.round(price, 2),
    })
    return df


def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    df = generate_simulation_data()
    out_path = os.path.join(out_dir, "power_data.csv")
    df.to_csv(out_path, index=False)
    print(f"シミュレーションデータを生成しました: {out_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
