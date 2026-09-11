# Supply Chain Demand Predictor & Inventory Risk Dashboard

An interactive supply chain tool that forecasts upcoming product demand, flags inventory risks before they cause problems, and provides plain-English explanations using AI.

---

## 1. Problem Understanding

When I reviewed the sponsor's problem statement, I identified two recurring, expensive bottlenecks in their daily supply chain operations:
- **Unexpected Stockouts:** Products run out without warning. This halts factory production lines, delays customer shipments, and forces costly emergency expediting.
- **Excess Overstocking:** Warehouses hold far too much inventory for certain products, unnecessarily tying up valuable working capital and driving up storage and holding costs.

Prior to this solution, the team had no proactive alert system in place. They were operating in a purely reactive mode—only discovering stockouts or excess inventory after the problems had already disrupted operations.

**My Objective:** Build an intuitive, self-service tool that the Supply Chain team can open daily to:
1. See predicted upcoming demand for every product (SKU).
2. Get clear, automated flags identifying exactly which items need immediate action.
3. Receive a plain-language, AI-generated explanation of *why* each item is flagged, grounded in real inventory metrics rather than raw numbers.

---

## 2. My Approach: What I Built and How It Works

I developed a full-stack solution combining machine learning for demand forecasting, rule-based inventory heuristics for risk alerts, and Generative AI for plain-English explanations.

### A. Demand Forecasting & Model Selection
I evaluated multiple forecasting paradigms to find the best production fit:

#### Why I Chose a Global Model Over Local Time-Series (Prophet & SARIMA)
While individual models like Prophet (MAE: 27.81) and SARIMA (MAE: 29.86) performed well on established products, I chose a **Global Model** for production because:
1. **Cold-Start Capability:** Three newly launched products (`SKU-2000`, `SKU-2001`, `SKU-2002`) had only ~11 days of transaction history. Local time-series models fail completely when there is insufficient historical data. My global model learns category-wide dynamics, enabling reliable forecasts for new products from day one.
2. **Operational Scalability:** Maintaining and monitoring a single global model for hundreds of SKUs is production-feasible; retraining and monitoring hundreds of individual ARIMA models creates unsustainable operational overhead.
3. **Multivariate Feature Ingestion:** The global architecture directly ingests external drivers such as supplier lead times, stock-to-sales ratios, and replenishment rates.

#### Why I Preferred XGBoost Over Other Global Models (MAE & RMSE Focus)
I benchmarked three candidate global regression models on the chronological 20% test split:

| Candidate Model | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) | Production Decision |
| :--- | :---: | :---: | :--- |
| **Linear Regression** (Baseline) | 53.30 units | **82.64 units** | Benchmarked |
| **Random Forest Regressor** | 54.23 units | 84.35 units | Benchmarked |
| **XGBoost Regressor** (My Choice) | **53.18 units** | 82.90 units | **Selected Production Model** |

**Why the Numbers Favored XGBoost:**
- **Lowest MAE (53.18):** XGBoost achieved the lowest Mean Absolute Error of all candidate models. In supply chain planning, MAE measures the average daily unit error. Minimizing MAE directly minimizes the safety stock buffer required to prevent stockouts, avoiding wasted working capital.
- **RMSE Analysis vs. Linear Regression:** While Linear Regression produced a marginally lower RMSE (82.64 vs. 82.90), linear models assume rigid, straight-line relationships. They cannot model non-linear interactions—such as how a 14-day supplier lead time compounds the risk of sudden weekend sales spikes.
- **Outperforming Random Forest:** Random Forest lagged behind on both metrics (MAE 54.23, RMSE 84.35) because simple tree averaging struggles with continuous time-series trend features. Furthermore, Random Forest required significantly higher memory and slower inference latency.
- **My Conclusion:** XGBoost delivered the winning balance: the lowest day-to-day forecast error (best MAE), tree-based handling of non-linear supply chain constraints, and fast, lightweight execution in production.

I serialized and deployed my trained model artifacts to the Hugging Face Hub: [`alfiinyang/XGSupply`](https://huggingface.co/alfiinyang/XGSupply).

---

### B. Proactive Risk Flags
I translated forecast outputs into clear business rules:
- 🚨 **Potential Stockout:** `Predicted Demand > Available Closing Stock` (Immediate replenishment required).
- ⚠️ **Overstocked:** `Current Closing Stock > 3 × Predicted Demand` (Pause upcoming purchase orders).
- ✅ **Healthy:** Inventory buffer comfortably matches forecasted demand.

---

### C. Plain-English AI Explanations
A colored badge alone doesn't explain *why* an item is at risk. When a manager selects any flagged product in the dashboard:
- The app packages that product's real numbers (current stock, forecast demand, supplier lead time, 7-day demand volatility, stock-to-sales ratio, and inbound receipt velocity).
- It prompts **Google Gemini (`gemini-3.6-flash`)** using LangChain with a low temperature (`0.1`) for strict factual grounding.
- The AI returns a concise, 2-sentence executive summary explaining the exact drivers behind the flag.

---

### D. User Interface (Streamlit)
I built a clean, intuitive dashboard that non-technical managers can operate without assistance:
- **KPI Cards:** Instant count of total monitored SKUs, critical stockout alerts, overstock warnings, and healthy buffer rates.
- **Risk Matrix Table:** Filterable by product category and alert status, displaying stock gaps and days of supply.
- **Click-to-Explain Panel:** Instant drill-down with live AI analysis.
- **Visual Trends:** Interactive Plotly charts comparing stock vs. demand, plus historical sales trajectories with 7-day moving averages.

---

## 3. How to Run

### Prerequisites
- Python 3.10+ installed on your system.

### Step 1: Clone the Repository
```bash
git clone https://github.com/alfiinyang/supply-chain-predictor.git
cd supply-chain-predictor
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Launch the App
```bash
streamlit run app.py
```

### Step 4: Open in Browser
- Navigate to `http://localhost:8501` in your browser.
- In the sidebar, paste your **Google Gemini API Key** to activate runtime AI explanations.
- The app comes pre-loaded with sample operational data for all 28 SKUs, or you can upload a new CSV file directly.

---

## 4. Limitations & Next Steps

### Current Limitations
1. **Single-Day Ahead Horizon:** The model currently forecasts demand for the next operational day. It does not yet generate a dynamic multi-week forecast curve.
2. **Static Lead Times:** Lead times are based on historical vendor averages rather than dynamically predicting supplier transit delays.

### What I Would Improve With More Time
1. **Multi-Week Forecast Horizons:** Expand predictions across a rolling 14-to-30 day horizon matching the exact lead-time window of each supplier.
2. **Automated Purchase Orders:** Connect the alert engine to the company ERP to automatically generate draft purchase orders when stockout flags trigger.
3. **Automated Alerts:** Send daily digest summaries directly to warehouse supervisors via Email or Slack every morning.
