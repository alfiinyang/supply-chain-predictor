"""
backend/alert_engine.py
-----------------------
Implements business rules for stockout and overstock detection from Project1_SKUs.ipynb:
- Potential Stockout: Predicted Demand > Current Closing Stock
- Overstocked: Current Closing Stock > 3 * Predicted Demand
- Healthy: In-balance buffer stock
"""

import pandas as pd
import numpy as np

def compute_alerts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes stockout and overstock risk flags, status strings, and inventory metrics.
    Requires 'predicted_demand' and 'closing_stock' columns.
    """
    res = df.copy()

    # Ensure integer/float types
    res["predicted_demand"] = res["predicted_demand"].round().astype(int)
    res["closing_stock"] = res["closing_stock"].round().astype(int)

    # Risk flags
    res["stockout_risk"] = res["predicted_demand"] > res["closing_stock"]
    res["overstock_risk"] = res["closing_stock"] > (3 * res["predicted_demand"])

    # Human-readable status badges
    def get_status(row):
        if row["stockout_risk"]:
            return "🚨 Action Required: Potential Stockout"
        if row["overstock_risk"]:
            return "⚠️ Attention: Overstocked"
        return "✅ Healthy"

    res["alert_status"] = res.apply(get_status, axis=1)

    # Actionable metrics
    res["stock_gap"] = res["closing_stock"] - res["predicted_demand"]
    res["days_of_supply"] = np.where(
        res["predicted_demand"] > 0,
        (res["closing_stock"] / res["predicted_demand"]).round(1),
        99.9
    )

    return res
