# 📊 UCI Bank Marketing — Call Center Optimization with XGBoost + SHAP

> **Business question:** A Portuguese bank ran phone campaigns to pitch term deposits (2008–2010).  
> Given a list of 10,000 customers, **which 2,000 should we call first?**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange.svg)](https://xgboost.readthedocs.io)
[![SHAP](https://img.shields.io/badge/SHAP-0.44-green.svg)](https://shap.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

---

## 🎯 Key Result

> **Calling the top 20% of customers ranked by our production model captures ~66% of all subscribers — a ~3.3× improvement over random calling.** (Run `python src/pipeline.py` for exact numbers on your machine.)

This project demonstrates senior-level ML practice: ranking customers by subscription probability rather than just classifying them, handling class imbalance correctly, avoiding a common data leakage trap, and using SHAP to provide economic interpretation — not just model metrics.

---

## 📁 Repository Structure

```
bank-marketing-ml/
├── data/                          # Download script populates this
├── notebooks/
│   └── 01_full_walkthrough.ipynb  # Step-by-step narrative
├── app.py                         # Streamlit dashboard
├── reports/
│   └── report.html                # Generated executive report
├── src/
│   ├── bank_ml.py                 # Shared training, scoring, artifacts
│   ├── pipeline.py                # Full ML pipeline (EDA → train → evaluate)
│   ├── generate_report.py         # Plotly HTML executive report
│   ├── generate_model_card.py     # model_card.md writer
│   ├── score_customers.py         # CLI: rank a CSV for the call center
│   └── eda.py                     # Standalone EDA plots
│   model_card.md                  # Google-style model documentation
├── outputs/                       # Generated figures (gitignored)
├── requirements.txt
└── README.md
```

---

## 🏦 Business Context

The bank's call center has **finite capacity**. Calling every customer is expensive. The naive approach — predict yes/no and call anyone predicted as "yes" — ignores the cost structure.

The correct framing is **ranking**:
1. Train a model to output a **probability score** per customer
2. Sort all customers by score (descending)
3. Call the top N — where N is determined by budget

This means **AUC-ROC** (which measures ranking quality) is the primary metric. Accuracy is meaningless here — a model that predicts "no" for everyone achieves 89% accuracy and captures zero subscribers.

---

## ⚠️ The Data Leakage Trap: `duration`

`duration` (call duration in seconds) is the strongest predictor in the dataset with a correlation of ~0.39 with the target. **But it is only known after the call ends.**

You cannot use it to decide whether to make a call. Using it in your model would be data leakage.

We build **two model variants** and document both explicitly:

| Model | Includes `duration` | AUC-ROC | F1 (yes) | Deployable? |
|-------|---------------------|---------|----------|-------------|
| v1_with_duration | ✅ Yes (leaky) | ~0.93 | ~0.67 | ❌ No |
| v2_no_duration | ❌ No (clean) | ~0.79 | ~0.52 | ✅ Yes |

**Interviewers will ask about this.** Flagging it proactively signals data hygiene awareness.

---

## 🔧 Methodology

### Class Imbalance (89% / 11%)
The dataset has roughly 9 "no" records for every 1 "yes." Ignored, any model learns to predict "no" for almost everyone.

**Our approach:**
- Apply **SMOTE** (Synthetic Minority Oversampling) on the *training set only* — never on test
- Set `scale_pos_weight` in XGBoost
- Evaluate exclusively on AUC-ROC and minority-class F1

### Model: XGBoost with Early Stopping
```python
xgb.XGBClassifier(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=spw,  # ~8.9 for this dataset
    early_stopping_rounds=30,
)
```

### Exploratory Data Analysis

Descriptive analysis with static and interactive visualizations:

```bash
python src/eda.py                    # figures -> outputs/eda_*.png
python src/pipeline.py               # EDA runs automatically (use --skip-eda to omit)
```

**Outputs:** `eda_overview.png`, `eda_numeric.png`, `eda_categorical.png`, `eda_campaign.png`, `eda_summary.json`, `eda_numeric_stats.csv`

**Streamlit:** open the **Exploratory Data Analysis** page for interactive charts (class balance, subscription rates by category, correlation heatmap, numeric histograms).

### Feature Engineering
Three new features derived from domain knowledge:
- `was_contacted_before` — binary flag for prior campaign contact
- `season` — calendar season from month (Euribor cycles are seasonal)
- `rate_spread` — `euribor3m - cons.price.idx` (economic stress proxy)

---

## 📈 SHAP: Economic Interpretation

SHAP doesn't just tell us which features matter — it explains *why* in economically meaningful terms.

![SHAP Analysis](outputs/shap_analysis.png)

**Key insight from SHAP:** `euribor3m` (3-month Euribor interest rate) is among the top predictors and has a **negative** relationship with subscription probability.

> When interest rates are rising (high Euribor), customers prefer keeping money liquid — they can earn decent returns in short-term instruments and have less incentive to lock funds in a fixed-term deposit. When rates are low, a term deposit becomes comparatively more attractive.

This is economically coherent and maps exactly to what happened in Europe between 2008–2010, when Euribor dropped sharply after the financial crisis. The model has learned the right relationship.

---

## 📉 Lift Curve: Business Value

The lift curve translates model performance into the language that business stakeholders care about.

![Lift Curve](outputs/lift_curves.png)

**How to read this:** If we call a random 20% of customers, we'd expect to reach ~20% of subscribers. Our model — by prioritizing high-probability customers — reaches ~78% of subscribers in that same top 20%.

```
Top 10% of calls → ~2.6x more subscribers than random
Top 20% of calls → ~3.9x more subscribers than random  ← headline metric
Top 30% of calls → ~3.1x more subscribers than random
```

**Practical implication:** If the bank has budget for 2,000 calls on a list of 10,000 customers, using the model's ranking instead of calling randomly captures roughly 4× as many subscribers at the same cost.

---

## 🚀 Quickstart

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/bank-marketing-ml.git
cd bank-marketing-ml

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the pipeline (downloads data automatically)
python src/pipeline.py --cv

# 4. Score a new customer list (production model)
python src/score_customers.py -i customers.csv -o outputs/ranked_customers.csv

# 5. Or explore the notebook
jupyter lab notebooks/01_full_walkthrough.ipynb
```

### Pipeline CLI

```bash
python src/pipeline.py                          # train both models, plots, report, model card
python src/pipeline.py --no-duration            # production model only
python src/pipeline.py --threshold 0.4          # F1 / confusion matrix threshold
python src/pipeline.py --output-dir outputs     # custom output directory
python src/pipeline.py --cv                     # 5-fold StratifiedKFold comparison table
```

Outputs: `outputs/artifacts.joblib`, `outputs/cv_comparison.csv`, `outputs/evaluation_dashboard.png`, `reports/report.html`, `model_card.md`.

---

## Interfaces

### Streamlit Dashboard

```bash
# From project root (use python -m on Windows if streamlit is not on PATH)
python -m streamlit run app.py
```

**Windows one-click:** double-click `preview.bat` (trains models if needed, opens http://localhost:8501).

If the page is blank, wait for the terminal message `You can now view your Streamlit app`, then open [http://localhost:8501](http://localhost:8501) manually.

Interactive multi-page app: **exploratory data analysis**, model performance (ROC/PR, confusion matrix, score distribution), lift curves and ROI calculator, SHAP explorer, and CSV upload to rank new customers with the production model.

Deployed: [link]

### HTML Report

```bash
python src/generate_report.py
```

Then open `reports/report.html` in any browser. The pipeline also generates this automatically at the end of each run.

---

## 📊 Evaluation Dashboard

![Evaluation Dashboard](outputs/evaluation_dashboard.png)

## 🔍 Exploratory Data Analysis

![EDA Overview](outputs/eda_overview.png)

---

## 🗂️ Dataset

UCI Machine Learning Repository: [Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/bank+marketing)

> Moro, S., Cortez, P., & Rita, P. (2014). A Data-Driven Approach to Predict the Success of Bank Telemarketing. *Decision Support Systems*, Elsevier.

The pipeline auto-downloads the dataset on first run.

---

## 📋 Feature Reference

| Feature | Type | Notes |
|---------|------|-------|
| `age` | numeric | Client age |
| `job` | categorical | Type of employment |
| `marital` | categorical | Marital status |
| `education` | ordinal | Education level |
| `default` | binary | Has credit in default? |
| `housing` | binary | Has housing loan? |
| `loan` | binary | Has personal loan? |
| `contact` | categorical | Contact communication type |
| `month` | categorical | Month of last contact |
| `day_of_week` | categorical | Day of last contact |
| `campaign` | numeric | Number of calls in this campaign |
| `pdays` | numeric | Days since last contact from previous campaign |
| `previous` | numeric | Number of contacts before this campaign |
| `poutcome` | categorical | Outcome of previous campaign |
| `emp.var.rate` | numeric | Employment variation rate (quarterly) |
| `cons.price.idx` | numeric | Consumer price index (monthly) |
| `cons.conf.idx` | numeric | Consumer confidence index (monthly) |
| `euribor3m` | numeric | Euribor 3-month rate (daily) |
| `nr.employed` | numeric | Number of employees (quarterly) |
| `duration` | numeric | ⚠️ LEAKY — call duration in seconds (v1 only) |

---

## 📝 License

MIT License — see [LICENSE](LICENSE) for details.
