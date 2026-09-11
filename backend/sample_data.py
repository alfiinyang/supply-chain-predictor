"""
backend/sample_data.py
----------------------
Provides realistic sample operational inventory data for all 28 SKUs:
- 25 Established SKUs (SKU-1000 to SKU-1024) across Snacks, Beverages, Dairy, Grains
- 3 Newly Launched SKUs (SKU-2000, SKU-2001, SKU-2002)
Contains 45 days of daily history leading up to the latest operational date (2026-06-29),
enabling robust calculation of 7-day and 30-day rolling/lag features.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def get_sample_dataset(days_of_history: int = 45) -> pd.DataFrame:
    """
    Generates a realistic baseline dataframe matching project1_supply_chain_demand.csv.
    End date is 2026-06-29 (matching the assessment notebook snapshot).
    """
    np.random.seed(42)
    end_date = datetime(2026, 6, 29)
    start_date = end_date - timedelta(days=days_of_history - 1)
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")

    categories = ["Snacks", "Beverages", "Dairy", "Grains"]

    # 25 Established SKUs
    established_skus = [f"SKU-{1000 + i}" for i in range(25)]
    # 3 Cold Start SKUs launched ~11 days before end_date
    new_skus = ["SKU-2000", "SKU-2001", "SKU-2002"]

    records = []

    for idx, sku in enumerate(established_skus):
        cat = categories[idx % len(categories)]
        lead_time = int(np.random.choice([3, 5, 7, 10, 14], p=[0.2, 0.3, 0.3, 0.1, 0.1]))
        base_demand = np.random.randint(120, 380)

        # Baseline stock level
        current_stock = np.random.randint(400, 2500)

        for d in date_range:
            day_of_week = d.dayofweek
            # Slight weekly seasonality
            dow_factor = 1.25 if day_of_week in [4, 5] else 0.95
            sold = max(10, int(np.random.normal(base_demand * dow_factor, 25)))

            # Occasional replenishment
            received = int(np.random.choice([0, base_demand * 5, base_demand * 8], p=[0.82, 0.12, 0.06]))
            current_stock = max(0, current_stock + received - sold)

            # Tailor latest day (2026-06-29) for specific SKUs to replicate known alerts
            if d == end_date:
                if sku in ["SKU-1000", "SKU-1004", "SKU-1007", "SKU-1012", "SKU-1018"]:
                    # Trigger Stockout Risk
                    current_stock = int(sold * 0.4)
                elif sku in ["SKU-1001", "SKU-1002", "SKU-1003", "SKU-1005"]:
                    # Trigger Overstock Warning
                    current_stock = int(sold * 5.2)

            records.append({
                "date": d,
                "sku_id": sku,
                "category": cat,
                "units_sold": sold,
                "units_received": received,
                "closing_stock": current_stock,
                "lead_time_days": lead_time
            })

    # Add 3 Cold-Start SKUs (only active in the last 12 days)
    cold_start_dates = [d for d in date_range if (end_date - d).days <= 11]
    for idx, sku in enumerate(new_skus):
        cat = categories[idx % len(categories)]
        lead_time = 7
        base_demand = np.random.randint(80, 200)
        current_stock = np.random.randint(150, 600)

        for d in cold_start_dates:
            sold = max(5, int(np.random.normal(base_demand, 20)))
            received = int(np.random.choice([0, 500], p=[0.85, 0.15]))
            current_stock = max(0, current_stock + received - sold)

            records.append({
                "date": d,
                "sku_id": sku,
                "category": cat,
                "units_sold": sold,
                "units_received": received,
                "closing_stock": current_stock,
                "lead_time_days": lead_time
            })

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df["category"] = df["category"].str.title()
    return df
