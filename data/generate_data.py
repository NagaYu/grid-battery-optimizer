"""
Generate synthetic hourly electricity simulation data (24 hours by default).

Assumed scenario (a hypothetical regional grid):
  - solar_generation: peaks during the day (10:00-14:00), zero at night.
  - power_demand: bimodal peaks in the morning (07:00-09:00) and evening
    (18:00-21:00).
  - grid_electricity_price: tracks demand, spiking during the morning/evening
    peak hours.

Units:
  - solar_generation, power_demand : MW (assumed constant over the hour -> MWh)
  - grid_electricity_price         : $/MWh
"""

import numpy as np
import pandas as pd
import os


def generate_simulation_data(days: int = 1, seed: int = 42) -> pd.DataFrame:
    """
    Generate `days` days (24*days time steps) of simulation data.
    days=1 gives the classic 24 hours. The load test increases `days` to scale
    the data size up.
    """
    rng = np.random.default_rng(seed)
    T = 24 * days
    t_idx = np.arange(T)
    hod = t_idx % 24  # hour-of-day (0..23)

    # --- Solar generation (MW): bell-shaped curve from sunrise to sunset ---
    solar_peak = 80.0
    solar = np.zeros(T)
    daylight = (hod >= 6) & (hod <= 18)
    solar[daylight] = solar_peak * np.sin(np.pi * (hod[daylight] - 6) / 12.0)
    solar = np.clip(solar + rng.normal(0, 2.0, T) * (solar > 0), 0, None)

    # --- Power demand (MW): bimodal morning/evening peaks + base load ---
    base_demand = 50.0
    morning_peak = 35.0 * np.exp(-((hod - 8) ** 2) / (2 * 1.8 ** 2))
    evening_peak = 45.0 * np.exp(-((hod - 19) ** 2) / (2 * 2.0 ** 2))
    demand = base_demand + morning_peak + evening_peak + rng.normal(0, 1.5, T)

    # --- Grid price ($/MWh): spikes at demand peaks, cheap overnight ---
    base_price = 40.0
    morning_price = 60.0 * np.exp(-((hod - 8) ** 2) / (2 * 2.0 ** 2))
    evening_price = 90.0 * np.exp(-((hod - 19) ** 2) / (2 * 2.2 ** 2))
    midday_discount = -15.0 * np.exp(-((hod - 12.5) ** 2) / (2 * 2.0 ** 2))
    price = base_price + morning_price + evening_price + midday_discount
    price = np.clip(price + rng.normal(0, 1.0, T), 10.0, None)

    df = pd.DataFrame({
        "hour": hod,                       # 0..23 (hour of day)
        "t": t_idx,                        # 0..T-1 (global time index)
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
