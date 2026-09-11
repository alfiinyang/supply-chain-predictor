"""
app.py
------
Streamlit Dashboard for Project 1: Supply Chain Demand Forecasting & Inventory Intelligence.
Model: alfiinyang/XGSupply (Hugging Face)
LLM: gemini-3.6-flash via ChatGoogleGenerativeAI (LangChain)
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Local backend imports
from backend.model_loader import load_artifacts, ensure_model_files
from backend.sample_data import get_sample_dataset
from backend.inference_engine import run_pipeline
from backend.alert_engine import compute_alerts
from backend.llm_explainer import explain_sku_risk

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Supply Chain Demand & Inventory Intelligence",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-box {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #E2E8F0;
    }
    .badge-stockout {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
    .badge-overstock {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
    .badge-healthy {
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
    .explanation-box {
        background-color: #F0F9FF;
        border-left: 4px solid #0284C7;
        padding: 14px 18px;
        border-radius: 0 8px 8px 0;
        margin: 12px 0;
        font-size: 1.02rem;
        line-height: 1.55;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Data & Model Caching
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading XGBoost model from Hugging Face...")
def get_cached_artifacts():
    return load_artifacts()

@st.cache_data(show_spinner="Running forecast pipeline...")
def predict_and_score(df_input):
    artifacts = get_cached_artifacts()
    df_enriched = run_pipeline(df_input, artifacts)
    return df_enriched


# -----------------------------------------------------------------------------
# Sidebar: Configuration, API Keys & Filters
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/warehouse.png", width=64)
    st.title("Control Center")

    st.subheader("🔑 Gemini API Settings")
    default_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or ""
    api_key_input = st.text_input(
        "Google Gemini API Key",
        value=default_key,
        type="password",
        help="Input your Gemini API key to activate runtime 'gemini-3.6-flash' alert explanations."
    )

    if api_key_input:
        st.success("API Key Active (gemini-3.6-flash ready)")
    else:
        st.info("No API key provided. Using rule-based analyst preview fallback.")

    st.divider()

    st.subheader("📂 Data Source")
    data_mode = st.radio(
        "Select Inventory Feed:",
        ["Built-in Assessment Snapshot (28 SKUs)", "Upload Custom CSV"]
    )

    raw_df = None
    if data_mode == "Upload Custom CSV":
        uploaded_file = st.file_uploader("Upload CSV (project1_supply_chain_demand.csv)", type=["csv"])
        if uploaded_file is not None:
            raw_df = pd.read_csv(uploaded_file)
            st.success(f"Loaded {len(raw_df):,} rows from file")
        else:
            st.warning("Awaiting CSV upload. Falling back to built-in dataset.")
            raw_df = get_sample_dataset()
    else:
        raw_df = get_sample_dataset()

    st.divider()

    st.subheader("🔍 Filters")
    all_categories = sorted(raw_df["category"].dropna().unique().tolist())
    selected_categories = st.multiselect("Category", options=all_categories, default=all_categories)

    status_filter = st.radio(
        "Risk Status",
        ["All SKUs", "🚨 Stockout Risk Only", "⚠️ Overstock Only", "✅ Healthy Only"]
    )

    st.divider()
    st.caption("Model: `alfiinyang/XGSupply` (XGBoost) | LLM: `gemini-3.6-flash` (LangChain)")


# -----------------------------------------------------------------------------
# Main Application Flow
# -----------------------------------------------------------------------------
st.markdown('<div class="main-title">📦 Supply Chain Demand & Inventory Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Proactive SKU-level demand forecasting, automated stockout/overstock detection, and runtime AI risk reasoning.</div>', unsafe_allow_html=True)

# Run prediction pipeline
try:
    df_predicted = predict_and_score(raw_df)
except Exception as e:
    st.error(f"Error during model inference: {e}")
    st.stop()

# Isolate latest operational date snapshot
latest_date = df_predicted["date"].max()
latest_snapshot = df_predicted[df_predicted["date"] == latest_date].copy()
latest_snapshot = compute_alerts(latest_snapshot)

# Apply Category Filter
if selected_categories:
    filtered_snapshot = latest_snapshot[latest_snapshot["category"].isin(selected_categories)]
else:
    filtered_snapshot = latest_snapshot

# Apply Risk Filter
if status_filter == "🚨 Stockout Risk Only":
    filtered_snapshot = filtered_snapshot[filtered_snapshot["stockout_risk"]]
elif status_filter == "⚠️ Overstock Only":
    filtered_snapshot = filtered_snapshot[filtered_snapshot["overstock_risk"]]
elif status_filter == "✅ Healthy Only":
    filtered_snapshot = filtered_snapshot[~filtered_snapshot["stockout_risk"] & ~filtered_snapshot["overstock_risk"]]

# -----------------------------------------------------------------------------
# Top Metric KPI Cards
# -----------------------------------------------------------------------------
total_skus = len(latest_snapshot)
total_stockouts = int(latest_snapshot["stockout_risk"].sum())
total_overstocks = int(latest_snapshot["overstock_risk"].sum())
total_healthy = total_skus - total_stockouts - total_overstocks
healthy_pct = round((total_healthy / total_skus) * 100, 1) if total_skus > 0 else 0

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.metric("Total Monitored SKUs", f"{total_skus}", delta=f"As of {latest_date.strftime('%Y-%m-%d')}")
with kpi2:
    st.metric("🚨 Stockout Alerts", f"{total_stockouts}", delta="- Production at Risk", delta_color="inverse")
with kpi3:
    st.metric("⚠️ Overstock Warnings", f"{total_overstocks}", delta="Capital Tied Up", delta_color="inverse")
with kpi4:
    st.metric("✅ Healthy Inventory", f"{total_healthy} ({healthy_pct}%)", delta="Balanced Buffer")

st.write("")

# -----------------------------------------------------------------------------
# Tabs Layout
# -----------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "📋 SKU Risk Matrix & AI Insights",
    "📈 Demand & Inventory Visualizer",
    "ℹ️ Strategic Rationale & Model Card"
])

# -----------------------------------------------------------------------------
# TAB 1: SKU Risk Matrix & Click-to-Explain
# -----------------------------------------------------------------------------
with tab1:
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.subheader(f"Operational Risk Matrix ({len(filtered_snapshot)} SKUs)")

        # Format display table
        display_df = filtered_snapshot[[
            "sku_id", "category", "closing_stock", "predicted_demand",
            "stock_gap", "lead_time_days", "days_of_supply", "alert_status"
        ]].copy()

        display_df = display_df.rename(columns={
            "sku_id": "SKU ID",
            "category": "Category",
            "closing_stock": "Closing Stock",
            "predicted_demand": "Forecast Demand",
            "stock_gap": "Stock Gap",
            "lead_time_days": "Lead Time (Days)",
            "days_of_supply": "Days Supply",
            "alert_status": "Risk Status"
        })

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            height=380
        )

    with col_right:
        st.subheader("🔍 AI Risk Explanation")

        # SKU Selector
        sku_list = filtered_snapshot["sku_id"].tolist()
        if not sku_list:
            sku_list = latest_snapshot["sku_id"].tolist()

        # Default to first stockout SKU if present
        default_index = 0
        stockout_candidates = filtered_snapshot[filtered_snapshot["stockout_risk"]]["sku_id"].tolist()
        if stockout_candidates and stockout_candidates[0] in sku_list:
            default_index = sku_list.index(stockout_candidates[0])

        selected_sku = st.selectbox("Select SKU to Inspect:", options=sku_list, index=default_index)

        sku_record = latest_snapshot[latest_snapshot["sku_id"] == selected_sku].iloc[0].to_dict()

        # Mini summary card
        sub_c1, sub_c2, sub_c3 = st.columns(3)
        with sub_c1:
            st.caption("Closing Stock")
            st.write(f"**{int(sku_record['closing_stock']):,}**")
        with sub_c2:
            st.caption("Predicted Demand")
            st.write(f"**{int(sku_record['predicted_demand']):,}**")
        with sub_c3:
            st.caption("Lead Time")
            st.write(f"**{int(sku_record['lead_time_days'])} days**")

        st.write(f"**Status:** {sku_record['alert_status']}")

        # Button to generate/refresh AI explanation
        if st.button("✨ Explain Risk Drivers with AI", type="primary", use_container_width=True):
            with st.spinner("Analyzing risk drivers via Google Gemini..."):
                explanation_data = explain_sku_risk(
                    sku_record,
                    api_key=api_key_input,
                    model_name="gemini-3.6-flash"
                )
                st.session_state[f"expl_{selected_sku}"] = explanation_data

        # Show stored explanation or trigger initially
        if f"expl_{selected_sku}" not in st.session_state:
            with st.spinner("Generating initial insight..."):
                st.session_state[f"expl_{selected_sku}"] = explain_sku_risk(
                    sku_record,
                    api_key=api_key_input,
                    model_name="gemini-3.6-flash"
                )

        expl = st.session_state[f"expl_{selected_sku}"]

        # Render styled explanation card
        st.markdown(f"""
        <div class="explanation-box">
            <b>Analyst Rationale:</b><br>
            {expl['explanation']}
        </div>
        """, unsafe_allow_html=True)

        if expl.get("status") == "success":
            st.caption("⚡ Grounded explanation generated live via `gemini-3.6-flash`.")
        else:
            st.caption(f"ℹ️ {expl.get('message', '')}")


# -----------------------------------------------------------------------------
# TAB 2: Visualizer
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("Inventory vs. Projected Demand Analysis")

    viz_col1, viz_col2 = st.columns([1, 1])

    with viz_col1:
        # Bar Chart of Stock vs Demand for Top 15 SKUs
        plot_data = filtered_snapshot.head(15).copy()
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=plot_data["sku_id"],
            y=plot_data["closing_stock"],
            name="Closing Stock",
            marker_color="#3B82F6"
        ))
        fig_bar.add_trace(go.Bar(
            x=plot_data["sku_id"],
            y=plot_data["predicted_demand"],
            name="Forecast Demand",
            marker_color="#EF4444"
        ))
        fig_bar.update_layout(
            title="Closing Stock vs. Forecasted Demand (Top Filtered SKUs)",
            barmode="group",
            xaxis_title="SKU ID",
            yaxis_title="Units",
            height=380,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with viz_col2:
        # Time-Series historical trend for selected SKU
        history_sku = df_predicted[df_predicted["sku_id"] == selected_sku].sort_values("date")
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(
            x=history_sku["date"],
            y=history_sku["units_sold"],
            mode="lines+markers",
            name="Daily Units Sold",
            line=dict(color="#64748B", width=1.5)
        ))
        if "rolling_avg_7_days" in history_sku.columns:
            fig_line.add_trace(go.Scatter(
                x=history_sku["date"],
                y=history_sku["rolling_avg_7_days"],
                mode="lines",
                name="7-Day Moving Avg",
                line=dict(color="#2563EB", width=2.5)
            ))
        # Add prediction marker on latest date
        fig_line.add_trace(go.Scatter(
            x=[latest_date],
            y=[sku_record["predicted_demand"]],
            mode="markers",
            name="Upcoming Forecast",
            marker=dict(color="#DC2626", size=12, symbol="star")
        ))
        fig_line.update_layout(
            title=f"Historical Demand Trend: {selected_sku}",
            xaxis_title="Date",
            yaxis_title="Units Sold",
            height=380,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_line, use_container_width=True)


with tab3:
    st.subheader("Model Selection & Strategic Rationale")
    st.markdown("""
    #### 1. Why a Global Model Over Local Time-Series (Prophet & SARIMA)
    Although univariate models like Prophet or SARIMA perform well for single established items (e.g., SKU-1000 Prophet MAE: 27.81), the production architecture uses a **Global Model** (`alfiinyang/XGSupply`):
    - **Cold-Start Capability:** Newly launched SKUs (such as `SKU-2000`, `SKU-2001`, `SKU-2002`) have under 2 weeks of history. Local time-series models fail without sufficient history. The global model learns category-wide purchasing dynamics to forecast demand accurately for new items.
    - **Operational Scalability:** Managing and monitoring a single global model for hundreds of SKUs is production-feasible, whereas maintaining separate individual ARIMA pipelines causes high operational overhead.
    - **Multivariate Exogenous Features:** Ingests supplier lead time, rolling volatility, and replenishment velocity directly into the decision boundary.

    #### 2. Why XGBoost Over Other Global Models (MAE & RMSE Evaluation)
    Three global candidate models were trained and benchmarked on the chronological 20% test split:

    | Candidate Model | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) | Status |
    | :--- | :---: | :---: | :--- |
    | **Linear Regression** (Baseline) | 53.30 units | **82.64 units** | Benchmarked |
    | **Random Forest Regressor** | 54.23 units | 84.35 units | Benchmarked |
    | **XGBoost Regressor** (Selected) | **53.18 units** | 82.90 units | **Selected Production Model** |

    **Key Evaluation Insights:**
    - **Lowest Mean Absolute Error (MAE: 53.18):** XGBoost achieved the best MAE across all candidate models. In supply chain operations, MAE measures the average daily unit error. Minimizing MAE directly minimizes the safety buffer misallocations required to prevent stockouts.
    - **RMSE Analysis vs. Linear Regression:** While Linear Regression produced a marginally lower RMSE (82.64 vs. 82.90), linear models assume strictly linear, additive relationships. They fail to capture critical non-linear interactions between supplier lead time constraints, rolling stock-to-sales ratios, and category demand swings.
    - **Outperforming Random Forest:** Random Forest lagged behind on both metrics (MAE 54.23, RMSE 84.35) due to sub-optimal tree averaging on continuous trend features, while requiring significantly higher memory and inference latency.
    - **Conclusion:** XGBoost offers the ideal combination of lowest absolute forecast error, gradient-boosted error correction, and fast, lightweight execution suitable for production inference.

    #### 3. Hugging Face Deployment
    - **Repository:** [`alfiinyang/XGSupply`](https://huggingface.co/alfiinyang/XGSupply)
    - **Serialized Artifacts:** `xgboost_supply_model.joblib`, `scaler.joblib`, `training_feature_names.joblib`, `scaled_column_names.joblib`
    """)
