# Supply Chain Demand Predictor & Inventory Risk Dashboard

An interactive supply chain tool that forecasts upcoming product demand, flags inventory risks before they cause problems, and provides plain-English explanations using AI.

---

## 1. Problem Understanding

The Supply Chain team frequently faces two costly challenges in daily inventory operations:
- **Unexpected Stockouts:** Products run out without warning, resulting in halted factory production lines, delayed customer deliveries, and expensive emergency expediting.
- **Excess Overstocking:** Warehouses hold far too much inventory for certain products, unnecessarily tying up valuable working capital and driving up storage and holding costs.

Previously, no proactive alert system was in place. Operations were purely reactive—stockouts and inventory imbalances were discovered only after they had already disrupted the business.

**Core Objective:** Deliver an intuitive, self-service tool that the Supply Chain team can access independently to:
1. View projected upcoming demand for every product (SKU).
2. Receive automated, clear risk flags identifying items requiring immediate action.
3. Obtain plain-language, AI-generated explanations of *why* an item is flagged, grounded in real inventory metrics rather than raw numbers.

---

## 2. Approach: System Architecture and Methodology

The solution combines machine learning for demand forecasting, inventory heuristics for risk alerts, and Generative AI for plain-English explanations.

### A. Demand Forecasting & Model Selection

#### Global Model vs. Local Time-Series (Prophet & SARIMA)
While individual models like Prophet (MAE: 27.81) and SARIMA (MAE: 29.86) performed well on single established products, a **Global Model** was selected for production because:
- **Cold-Start Handling:** Newly introduced products (`SKU-2000`, `SKU-2001`, `SKU-2002`) had only ~11 days of transaction history. Local time-series models fail when historical data is insufficient. A global model learns category-level purchasing patterns, producing reliable forecasts for new items from day one.
- **Operational Scalability:** Managing and monitoring a single global model for hundreds of SKUs is production-feasible; retraining and monitoring hundreds of individual ARIMA models creates unsustainable operational overhead.
- **Multivariate Feature Ingestion:** The global architecture directly ingests external operational drivers such as supplier lead times, stock-to-sales ratios, and replenishment rates.

#### Why XGBoost Was Preferred Over Other Global Models (MAE & RMSE Focus)
Three global candidate models were benchmarked across all 28 SKUs using a chronological 20% test split:

| Candidate Model | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) | Production Status |
| :--- | :---: | :---: | :--- |
| **Linear Regression** (Baseline) | 53.30 units | **82.64 units** | Benchmarked |
| **Random Forest Regressor** | 54.23 units | 84.35 units | Benchmarked |
| **XGBoost Regressor** (Selected) | **53.18 units** | 82.90 units | **Selected Production Model** |

**Evaluation Insights:**
- **Lowest MAE (53.18 units):** XGBoost achieved the lowest Mean Absolute Error across all candidate models. In supply chain operations, MAE measures the average daily unit error. Minimizing MAE directly minimizes the safety stock buffer needed to prevent stockouts, avoiding wasted working capital.
- **RMSE Trade-off vs. Linear Regression:** While Linear Regression produced a marginally lower RMSE (82.64 vs. 82.90), linear models assume strictly linear, additive relationships. They cannot capture critical non-linear interactions—such as how a 14-day supplier delivery window compounds the risk of sudden sales spikes.
- **Outperforming Random Forest:** Random Forest trailed across both metrics (MAE 54.23, RMSE 84.35) because simple tree averaging struggles with continuous time-series trend features, while also requiring substantially higher memory and inference latency.
- **Decision:** XGBoost provides the optimal combination of lowest absolute forecast error (best MAE), tree-based handling of non-linear supply chain constraints, and fast, lightweight execution in production.

Trained model artifacts are hosted and loaded directly from the Hugging Face Hub: [`alfiinyang/XGSupply`](https://huggingface.co/alfiinyang/XGSupply).

---

### B. Proactive Risk Flags
Forecast outputs are converted into clear business rules:
- 🚨 **Potential Stockout:** `Predicted Demand > Available Closing Stock` (Immediate replenishment required before stock runs out).
- ⚠️ **Overstocked:** `Current Closing Stock > 3 × Predicted Demand` (Excess capital tied up; pause new purchase orders).
- ✅ **Healthy:** Inventory buffer comfortably matches projected demand.

---

### C. Plain-English AI Explanations
A colored badge alone does not explain the root cause. When a manager selects any flagged product in the dashboard:
- The system extracts that product's numerical context (closing stock, forecast demand, supplier lead time, 7-day demand volatility, stock-to-sales ratio, and inbound receipt velocity).
- It prompts **Google Gemini (`gemini-3.6-flash`)** via LangChain with a low temperature (`0.1`) for strict factual grounding.
- The AI generates a concise, 2-sentence executive summary explaining the exact drivers behind the flag in plain English.

---

### D. User Interface (Streamlit)
A clean, intuitive dashboard designed for operational stakeholders:
- **KPI Summary Cards:** Instant count of total monitored items, critical stockout alerts, overstock warnings, and healthy buffer rates.
- **Inventory Risk Table:** Sortable and filterable by product category or risk status to prioritize urgent tasks.
- **Click-to-Explain Panel:** Instant drill-down with runtime AI analysis.
- **Visual Trends:** Interactive Plotly charts comparing stock levels against demand, plus historical sales trajectories with 7-day moving averages.

---

## 3. How to Run and Use

### Live Cloud Deployment
The interactive dashboard is publicly hosted and ready to use without local installation:
- **Live Application URL:** [https://supply-chain-predictor-qcdp.onrender.com/](https://supply-chain-predictor-qcdp.onrender.com/)

---

### Local Installation

#### Prerequisites
- Python 3.10+ installed.

#### Step 1: Clone the Repository
```bash
git clone https://github.com/alfiinyang/supply-chain-predictor.git
cd supply-chain-predictor
```

#### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

#### Step 3: Launch the Dashboard
```bash
streamlit run app.py
```

#### Step 4: Open in Browser
- Navigate to `http://localhost:8501` in any modern web browser.
- In the sidebar, enter a **Google Gemini API Key** to enable real-time `gemini-3.6-flash` risk explanations.
- By default, the dashboard runs on the built-in 28-SKU snapshot, with an option in the sidebar to upload custom operational CSV files.

---

### Custom Input Data Specifications

When uploading custom inventory CSV files to the dashboard, the data must adhere to the following schema and formatting standards:

#### 1. Required Schema & Columns

| Column Name | Data Type | Description | Example Values |
| :--- | :---: | :--- | :--- |
| `date` | String / Date | Daily observation timestamp in `YYYY-MM-DD` format | `2026-06-29` |
| `sku_id` | String | Unique item / stock keeping unit identifier | `SKU-1000`, `SKU-2001` |
| `category` | String | Product department or grouping | `Snacks`, `Beverages`, `Dairy`, `Grains` |
| `units_sold` | Numeric ($\ge 0$) | Daily sales volume (primary target feature) | `145`, `210` |
| `units_received` | Integer ($\ge 0$) | Stock replenishment delivered to warehouse on that date | `0`, `500` |
| `closing_stock` | Numeric ($\ge 0$) | Physical inventory remaining on hand at close of day | `320`, `1450` |
| `lead_time_days` | Integer ($\ge 1$) | Supplier delivery lead time in calendar days | `5`, `7`, `14` |

#### 2. Pre-Addressed Data Quality Requirements
To ensure accurate predictions and prevent processing errors, the uploaded dataset must be pre-cleaned by the user:
- **Zero Missing / Null Values:** Columns `date`, `sku_id`, `units_sold`, `units_received`, and `closing_stock` must not contain blank or `NaN` cells. Missing values must be imputed or resolved before ingestion.
- **Non-Negative Values:** Units sold, received, closing stock balances, and lead times cannot be negative.
- **Continuous Daily Cadence:** Observations should be reported on a daily frequency without skipped calendar dates per SKU. Inactive sales days should have `units_sold = 0` rather than omitted rows.
- **Standardized Identifiers:** Categorical labels (`sku_id`, `category`) should follow uniform naming conventions and casing without trailing whitespace.

#### 3. Length of Historical Data Window
- **Minimum Requirement:** At least **30 consecutive days** of daily operational records per SKU (30–45+ days recommended) leading up to the forecast date.
- **Rationale:** The predictive pipeline automatically derives 7-day and 30-day rolling averages (`rolling_avg_7_days`, `rolling_avg_30_days`), demand volatility measures (`rolling_std_7_days`, `rolling_std_30_days`), and historical lag indicators (`lag_7_days`, `lag_30_days`). Providing 30+ consecutive days provides full numerical coverage for the XGBoost model's feature schema. Cold-start SKUs with shorter histories are supported via backward filling, but established SKUs require 30+ days for optimal forecast fidelity.

---

## 4. Limitations & Next Steps

### Current Limitations
1. **Single-Day Ahead Horizon:** The model forecasts demand for the next operational day. It does not yet generate a dynamic multi-week forecast curve.
2. **Static Lead Times:** Lead times reflect historical supplier averages rather than dynamically predicting transit delays.

### Next Steps
1. **Multi-Week Forecast Horizons:** Expand predictions across a rolling 14-to-30 day horizon matching the exact lead-time window of each supplier.
2. **Automated Purchase Orders:** Connect the alert engine to the company ERP to automatically generate draft purchase orders when stockout flags trigger.
3. **Automated Alerts:** Send daily digest summaries directly to warehouse supervisors via Email or Slack every morning.
