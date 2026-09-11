# Supply Chain Demand Predictor & Inventory Risk Dashboard

An interactive supply chain tool that forecasts upcoming product demand, flags inventory risks before they cause problems, and gives plain-English explanations using AI.

---

## 1. Problem Understanding

The Supply Chain team frequently faces two costly challenges:
- **Unexpected Stockouts:** Products run out without warning, causing production delays, halted lines, and unhappy customers.
- **Overstocking:** Warehouses hold too much inventory, tying up company money and increasing storage costs.

Previously, there was no proactive system in place. The team had to react to inventory problems after they had already occurred. 

**The Goal:** Build an easy-to-use tool that the Supply Chain team can open themselves to:
1. See projected upcoming demand for every product (SKU).
2. Get clear, automatic flags showing which items are at risk.
3. Understand *why* an item is flagged through plain-language explanations instead of confusing numbers.

---

## 2. Approach: What We Built and How It Works

We built a lightweight web dashboard connected to a trained forecasting model and an AI explanation engine:

### A. Demand Forecasting (Machine Learning)
- We use an **XGBoost** model trained on historical daily sales, stock levels, delivery lead times, and delivery patterns across 28 products.
- The model handles both **established products** (with 6 months of data) and **newly launched products** (with under 2 weeks of data) using category patterns.
- The trained model is automatically pulled from Hugging Face: [`alfiinyang/XGSupply`](https://huggingface.co/alfiinyang/XGSupply).

### B. Proactive Risk Flags
The system automatically compares forecasted demand against available stock:
- 🚨 **Potential Stockout:** Forecasted demand is greater than current stock. Immediate reorder is needed before stock runs out.
- ⚠️ **Overstocked:** Current stock is more than 3 times the forecasted demand. Excess cash is tied up; pause new orders.
- ✅ **Healthy:** Inventory is balanced and safe.

### C. Plain-English AI Explanations
A colored flag is not enough—managers need to know the reason. When you click on any flagged SKU in the dashboard:
- The app sends that product's real numbers (current stock, forecast demand, supplier lead time, and recent delivery trends) to **Google Gemini (`gemini-3.6-flash`)**.
- The AI generates a concise, 2-sentence explanation explaining the exact reason for the flag in plain English.

### D. User Interface
- **KPI Summary Cards:** See total monitored items, active stockout alerts, and overstock warnings at a glance.
- **Inventory Matrix Table:** Filter by product category or alert status to prioritize urgent tasks.
- **Visual Trends:** Clear bar charts and trendlines comparing stock levels against demand.

---

## 3. How to Run

### Prerequisites
- Python 3.10+ installed on your computer.

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
- The app will automatically open at: `http://localhost:8501`
- In the sidebar, enter your **Google Gemini API Key** to activate live AI explanations.
- The app comes pre-loaded with sample operational data for all 28 SKUs, or you can upload a new CSV file directly through the sidebar.

---

## 4. Limitations & Next Steps

### Current Limitations
1. **Single-Day Ahead Forecast:** The model currently predicts upcoming daily demand. It does not yet show a full 30-day projected curve.
2. **Fixed Lead Times:** Lead times are based on the latest recorded delivery window rather than dynamically predicting supplier shipping delays.

### Next Steps with More Time
1. **Multi-Week Forecasting:** Extend predictions across a rolling 14-to-30 day horizon matching the exact lead time of each supplier.
2. **One-Click Purchase Orders:** Connect the dashboard to the procurement system to automatically draft reorder forms when a stockout alert is triggered.
3. **Automated Email / Slack Alerts:** Send daily digest alerts directly to warehouse supervisors every morning.
