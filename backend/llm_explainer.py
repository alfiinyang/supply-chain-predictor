"""
backend/llm_explainer.py
------------------------
Implements the exact runtime LLM explanation pipeline from Step 40 of Project1_SKUs.ipynb.
Uses 'ChatGoogleGenerativeAI' with model 'gemini-3.6-flash'.
Provides concise, grounded 2-sentence rationales explaining the risk drivers of flagged SKUs.
"""

import os
from typing import Dict, Any, Optional

SUPPLY_CHAIN_PROMPT_TEMPLATE = """
You are a Senior Supply Chain Analyst.
Analyze the following SKU data and provide a concise (2-sentence) explanation for the Supply Chain Manager regarding the flag status.

SKU Data:
- SKU ID: {sku_id} ({category_label})
- Alert: {alert_type}
- Closing Stock: {closing_stock}
- Predicted Demand: {predicted_demand}
- Lead Time: {lead_time_days} days
- Demand Volatility (7d StdDev): {rolling_std_7_days}
- Stock-to-Sales Ratio: {stock_to_sales_ratio}
- Avg Daily Sales (7d): {rolling_avg_7_days}
- Avg Units Received (7d): {rolling_avg_received_7_days}

Explanation:"""


def explain_sku_risk(
    item: Dict[str, Any],
    api_key: Optional[str] = None,
    model_name: str = "gemini-3.6-flash",
    temperature: float = 0.1
) -> Dict[str, str]:
    """
    Invokes ChatGoogleGenerativeAI to explain the SKU's risk flag.

    Parameters:
        item: Dictionary containing SKU fields (closing_stock, predicted_demand, etc.)
        api_key: Google Gemini API key entered by user or from environment.
        model_name: Default 'gemini-3.6-flash' as requested.
        temperature: Low temperature (0.1) for high factual grounding.

    Returns:
        dict: {"sku_id": ..., "alert": ..., "explanation": ..., "status": "success"|"fallback"|"error"}
    """
    # 1. Resolve API key
    resolved_key = (
        api_key
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("gemini_key")
    )

    sku_id = item.get("sku_id", "SKU")
    category = item.get("category", item.get("category_label", "General"))
    closing_stock = int(item.get("closing_stock", 0))
    predicted_demand = int(item.get("predicted_demand", 0))
    lead_time_days = int(item.get("lead_time_days", item.get("lead_time", 7)))

    alert_type = (
        "STOCKOUT RISK"
        if item.get("stockout_risk") or (predicted_demand > closing_stock)
        else "OVERSTOCK WARNING"
        if item.get("overstock_risk") or (closing_stock > 3 * predicted_demand)
        else "HEALTHY BUFFER"
    )

    rolling_std = round(float(item.get("rolling_std_7_days", 0.0)), 2)
    stock_to_sales = round(float(item.get("stock_to_sales_ratio", 0.0)), 2)
    rolling_avg_sales = round(float(item.get("rolling_avg_7_days", 0.0)), 2)
    rolling_avg_rec = round(float(item.get("rolling_avg_received_7_days", 0.0)), 2)

    # 2. Check if API key is provided
    if not resolved_key:
        # Fallback simulation explaining the risk so the UI never crashes
        if alert_type == "STOCKOUT RISK":
            simulated = (
                f"{sku_id} is facing an immediate Stockout Risk because current closing stock ({closing_stock:,} units) "
                f"falls critically short of forecasted demand ({predicted_demand:,} units) with a {lead_time_days}-day supplier lead time. "
                f"Immediate replenishment must be expedited to prevent line downtime."
            )
        elif alert_type == "OVERSTOCK WARNING":
            simulated = (
                f"{sku_id} triggered an Overstock Warning because closing stock ({closing_stock:,} units) "
                f"exceeds forecasted demand ({predicted_demand:,} units) by a {stock_to_sales}x stock-to-sales ratio. "
                f"Inbound replenishment should be temporarily suspended to prevent excessive capital lockup."
            )
        else:
            simulated = f"{sku_id} maintain a healthy stock balance relative to projected sales velocity."

        return {
            "sku_id": sku_id,
            "alert": alert_type,
            "explanation": simulated,
            "status": "fallback",
            "message": "Generated using rule-based analyst heuristics. Provide a Gemini API Key in the sidebar to activate runtime LLM calls."
        }

    # 3. Call ChatGoogleGenerativeAI
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.prompts import PromptTemplate

        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=resolved_key,
            temperature=temperature
        )

        prompt = PromptTemplate.from_template(SUPPLY_CHAIN_PROMPT_TEMPLATE)
        chain = prompt | llm

        response = chain.invoke({
            "sku_id": sku_id,
            "category_label": category,
            "alert_type": alert_type,
            "closing_stock": closing_stock,
            "predicted_demand": predicted_demand,
            "lead_time_days": lead_time_days,
            "rolling_std_7_days": rolling_std,
            "stock_to_sales_ratio": stock_to_sales,
            "rolling_avg_7_days": rolling_avg_sales,
            "rolling_avg_received_7_days": rolling_avg_rec
        })

        content = response.content
        if isinstance(content, list) and len(content) > 0:
            if isinstance(content[0], dict) and "text" in content[0]:
                text = content[0]["text"]
            else:
                text = str(content[0])
        elif isinstance(content, str):
            text = content
        else:
            text = str(content)

        return {
            "sku_id": sku_id,
            "alert": alert_type,
            "explanation": text.strip(),
            "status": "success",
            "message": f"Successfully generated via {model_name}"
        }

    except Exception as e:
        # Graceful handling
        err_msg = str(e)
        return {
            "sku_id": sku_id,
            "alert": alert_type,
            "explanation": (
                f"{sku_id} is flagged under {alert_type}: Current closing inventory is {closing_stock:,} units "
                f"against a projected demand of {predicted_demand:,} units over a {lead_time_days}-day lead time window."
            ),
            "status": "error",
            "message": f"LLM Call Error: {err_msg}"
        }
