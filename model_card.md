# Model Card: UCI Bank Marketing — Term Deposit Ranking

## Model Details

- **Developed by:** Portfolio ML project (UCI Bank Marketing dataset)
- **Model type:** XGBoost binary classifier (probability scores used for ranking)
- **Version:** v2_no_duration (production) / v1_with_duration (benchmark with leakage)
- **Last updated:** Generated from `outputs/artifacts.joblib`

### Production model (v2_no_duration)

| Metric (hold-out test) | Value |
|------------------------|-------|
| AUC-ROC | 0.8102 |
| F1 (minority class, threshold=0.5) | 0.4309 |
| Lift @ top 20% | 3.29x |
| Recall @ top 20% | 65.7% |

### Benchmark model (v1_with_duration) — NOT for deployment

| Metric | Value |
|--------|-------|
| AUC-ROC | 0.9547 |
| F1 (minority) | 0.6530 |
| Lift @ 20% | 4.50x |

AUC gap (v1 - v2): 0.144 — largely explained by leaky `duration` when v1 is trained.


### Cross-validation (5-fold StratifiedKFold)

| Model | AUC-ROC | F1 (minority) | Deployable |
|-------|---------|---------------|------------|
| v1_with_duration | 0.949 +/- 0.002 | 0.626 +/- 0.010 | No (leakage) |
| v2_no_duration | 0.796 +/- 0.006 | 0.423 +/- 0.022 | Yes |


## Intended Use

- **Primary use:** Rank customers by predicted probability of subscribing to a term deposit before outbound calls.
- **Users:** Call-center planners, marketing analytics teams.
- **Out of scope:** Credit risk, loan default prediction, real-time regulatory decisions.

## Factors

- **Relevant:** Macroeconomic indicators (Euribor, employment), prior campaign outcomes, demographics.
- **Top SHAP features (production):** campaign, day_of_week, nr.employed, contact, marital.
- **Excluded from production:** `duration` (call length) — only known after a call ends.

## Metrics

Primary: **AUC-ROC** (ranking quality). Secondary: **minority-class F1**, **lift @ 20%** (business headline).

Accuracy is not reported — the dataset is ~89% negative; a constant "no" model has high accuracy and zero business value.

## Training Data

- **Source:** [UCI Bank Marketing](https://archive.ics.uci.edu/ml/datasets/bank+marketing) — Portuguese bank telemarketing (2008–2010).
- **Rows:** 41,188 | **Positive rate:** 11.3%
- **Train/test split:** 80/20 stratified (random_state=42)
- **Imbalance handling:** SMOTE on training fold only; XGBoost with early stopping

### Engineered features

- `was_contacted_before`, `season`, `rate_spread`

## Evaluation Data

- Hold-out 20% test set (never used for SMOTE or hyperparameter tuning via CV in final metrics table above).
- Cross-validation reported separately when `python src/pipeline.py --cv` is run.

## Ethical Considerations

- Model may reflect historical biases in who was contacted or who could afford deposits.
- Should not be used to exclude protected classes from financial products without human review.
- Campaigns must comply with local telemarketing and privacy regulations.

## Caveats and Recommendations

1. **`duration` leakage:** v1 is for benchmarking only; deploy **v2_no_duration**.
2. **Temporal drift:** Macro features (Euribor era 2008–2010) may not transfer to today's rate environment — retrain periodically.
3. **Threshold:** Default 0.5 for F1 reporting; call-center prioritization should use **ranking**, not a fixed threshold.
4. **Scoring new files:** `python src/score_customers.py -i customers.csv -o ranked.csv`

## How to reproduce

```bash
pip install -r requirements.txt
python src/pipeline.py --cv
streamlit run app.py
```
