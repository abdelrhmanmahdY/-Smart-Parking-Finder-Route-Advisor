"""
generate_data.py
----------------
Generates synthetic parking occupancy data for model training.

Features per record
-------------------
  hour          : 0-23
  weekday       : 0 (Mon) – 6 (Sun)
  lot_id        : integer (encoded)
  class_density : 0-10  (how many classes running nearby at that hour)
  event         : 0/1   (campus event flag)

Target
------
  occupancy_pct : 0.0 – 1.0  (fraction of capacity in use)
"""

import numpy as np
import pandas as pd

from campus_graph import PARKING_LOTS, LOT_ID_MAP

# Deterministic seed for reproducibility
RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Domain knowledge priors
# ---------------------------------------------------------------------------

# Typical class schedule: busy 8-17, light evenings, dead at night
HOUR_WEIGHT = np.array([
    0.05, 0.03, 0.02, 0.02, 0.03, 0.05,   # 00-05
    0.10, 0.30, 0.75, 0.90, 0.95, 0.92,   # 06-11
    0.85, 0.88, 0.87, 0.82, 0.70, 0.50,   # 12-17
    0.35, 0.25, 0.18, 0.12, 0.08, 0.06,   # 18-23
])

# Mon-Thu busier than Fri; weekend much quieter
WEEKDAY_WEIGHT = np.array([1.0, 1.0, 1.0, 0.95, 0.80, 0.30, 0.15])

# Each lot has a base popularity factor
LOT_BASE = {
    "P_NORTH":   0.82,
    "P_EAST":    0.70,
    "P_CENTRAL": 0.95,   # central → very popular
    "P_WEST":    0.60,
    "P_SOUTH":   0.55,
}


def _class_density(hour: int, weekday: int) -> int:
    """Simulate number of concurrent classes (0-10)."""
    if weekday >= 5:
        return 0
    if 8 <= hour <= 17:
        return int(RNG.integers(4, 11))
    if 18 <= hour <= 20:
        return int(RNG.integers(0, 4))
    return 0


def generate_occupancy_data(n_days: int = 180) -> pd.DataFrame:
    """
    Simulate *n_days* of hourly occupancy across all parking lots.

    Returns a DataFrame with columns:
        hour, weekday, lot_id, class_density, event, occupancy_pct
    """
    records = []
    for day in range(n_days):
        weekday = day % 7
        # ~15 % of weekdays have a campus event
        event = int(weekday < 5 and RNG.random() < 0.15)

        for hour in range(24):
            cd = _class_density(hour, weekday)
            hw = HOUR_WEIGHT[hour]
            ww = WEEKDAY_WEIGHT[weekday]
            event_boost = 0.15 if event and 10 <= hour <= 20 else 0.0

            for lot_name, lot_id in LOT_ID_MAP.items():
                base = LOT_BASE[lot_name]
                occ = base * hw * ww + event_boost
                # Add noise
                occ += RNG.normal(0, 0.06)
                occ = float(np.clip(occ, 0.0, 1.0))

                records.append({
                    "hour": hour,
                    "weekday": weekday,
                    "lot_id": lot_id,
                    "class_density": cd,
                    "event": event,
                    "occupancy_pct": round(occ, 4),
                })

    df = pd.DataFrame(records)
    return df


if __name__ == "__main__":
    df = generate_occupancy_data()
    print(df.head(10))
    print(f"\nShape: {df.shape}")
    print(df.groupby("lot_id")["occupancy_pct"].mean())
