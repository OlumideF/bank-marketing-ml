"""
Write model_card.md from saved pipeline artifacts (Google Model Card style).
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bank_ml import ROOT as ML_ROOT, load_artifacts  # noqa: E402


def write_model_card(artifacts: dict, out_path: Path | None = None) -> Path:
    out_path = out_path or ML_ROOT / "model_card.md"
    results = artifacts["results"]
    stats = artifacts["dataset_stats"]
    v1 = results.get("v1_with_duration")
    v2 = results["v2_no_duration"]
    cv = artifacts.get("cv_comparison")
    v1_auc = f"{v1['auc']:.4f}" if v1 else "N/A"
    v1_f1 = f"{v1['f1']:.4f}" if v1 else "N/A"
    v1_lift = f"{v1['lift_20']:.2f}x" if v1 else "N/A"
    auc_gap = f"{v1['auc'] - v2['auc']:.3f}" if v1 else "N/A"

    cv_section = ""
    if cv is not None and len(cv):
        cv_section = "\n### Cross-validation (5-fold StratifiedKFold)\n\n"
        cv_section += "| Model | AUC-ROC | F1 (minority) | Deployable |\n|-------|---------|---------------|------------|\n"
        for _, row in cv.iterrows():
            deploy = "Yes" if row["deployable"] else "No (leakage)"
            cv_section += (
                f"| {row['model']} | {row['auc_mean']:.3f} +/- {row['auc_std']:.3f} | "
                f"{row['f1_mean']:.3f} +/- {row['f1_std']:.3f} | {deploy} |\n"
            )

    top_features = artifacts["shap"]["importance"].head(5)["feature"].tolist()

    md = f"""# Model Card: UCI Bank Marketing — Term Deposit Ranking

## Model Details

- **Developed by:** Portfolio ML project (UCI Bank Marketing dataset)
- **Model type:** XGBoost binary classifier (probability scores used for ranking)
- **Version:** v2_no_duration (production) / v1_with_duration (benchmark with leakage)
- **Last updated:** Generated from `outputs/artifacts.joblib`

### Production model (v2_no_duration)

| Metric (hold-out test) | Value |
|------------------------|-------|
| AUC-ROC | {v2['auc']:.4f} |
| F1 (minority class, threshold={v2.get('threshold', 0.5)}) | {v2['f1']:.4f} |
| Lift @ top 20% | {v2['lift_20']:.2f}x |
| Recall @ top 20% | {v2['recall_20']:.1f}% |

### Benchmark model (v1_with_duration) — NOT for deployment

| Metric | Value |
|--------|-------|
| AUC-ROC | {v1_auc} |
| F1 (minority) | {v1_f1} |
| Lift @ 20% | {v1_lift} |

AUC gap (v1 - v2): {auc_gap} — largely explained by leaky `duration` when v1 is trained.

{cv_section}

## Intended Use

- **Primary use:** Rank customers by predicted probability of subscribing to a term deposit before outbound calls.
- **Users:** Call-center planners, marketing analytics teams.
- **Out of scope:** Credit risk, loan default prediction, real-time regulatory decisions.

## Factors

- **Relevant:** Macroeconomic indicators (Euribor, employment), prior campaign outcomes, demographics.
- **Top SHAP features (production):** {", ".join(top_features)}.
- **Excluded from production:** `duration` (call length) — only known after a call ends.

## Metrics

Primary: **AUC-ROC** (ranking quality). Secondary: **minority-class F1**, **lift @ 20%** (business headline).

Accuracy is not reported — the dataset is ~89% negative; a constant "no" model has high accuracy and zero business value.

## Training Data

- **Source:** [UCI Bank Marketing](https://archive.ics.uci.edu/ml/datasets/bank+marketing) — Portuguese bank telemarketing (2008–2010).
- **Rows:** {stats['n_rows']:,} | **Positive rate:** {stats['positive_rate']*100:.1f}%
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
"""
    out_path.write_text(md, encoding="utf-8")
    return out_path


if __name__ == "__main__":
    artifacts = load_artifacts(retrain_if_missing=True)
    path = write_model_card(artifacts)
    print(f"Model card written to {path}")
