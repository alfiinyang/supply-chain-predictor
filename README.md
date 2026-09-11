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

## 3. How to Run

### Prerequisites
- Python 3.10+ installed.

### Step 1: Clone the Repository
```bash
git clone https://github.com/alfiinyang/supply-chain-predictor.git
cd supply-chain-predictor
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Launch the Dashboard
```bash
streamlit run app.py
```

### Step 4: Open in Browser
- Navigate to `http://localhost:8501` in any browser.
- In the sidebar, enter a **Google Gemini API Key** to activate live AI explanations.
- The app comes pre-loaded with sample operational data for all 28 SKUs, and also supports uploading custom CSV batches.

---

## 4. Limitations & Next Steps

### Current Limitations
1. **Single-Day Ahead Horizon:** The model forecasts demand for the next operational day. It does not yet generate a dynamic multi-week forecast curve.
2. **Static Lead Times:** Lead times reflect historical supplier averages rather than dynamically predicting transit delays.

### Next Steps
1. **Multi-Week Forecast Horizons:** Expand predictions across a rolling 14-to-30 day horizon matching the exact lead-time window of each supplier.
2. **Automated Purchase Orders:** Connect the alert engine to the company ERP to automatically generate draft purchase orders when stockout flags trigger.
3. **Automated Alerts:** Send daily digest summaries directly to warehouse supervisors via Email or Slack every morning.
