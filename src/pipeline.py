"""
UCI Bank Marketing — Full ML Pipeline
======================================
Goal: Rank customers by subscription probability to maximize
      call center efficiency (not just classify them).

Two model variants:
  - v1_with_duration   : includes 'duration' (strong but leaky)
  - v2_no_duration     : production-safe, deployable before calling

Run:
    python src/pipeline.py
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.metrics import (
    roc_auc_score, f1_score, classification_report,
    roc_curve, precision_recall_curve, confusion_matrix
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

import xgboost as xgb
import shap

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

# ── 1. Load Data ─────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    path = DATA_DIR / "bank-additional-full.csv"
    if not path.exists():
        print("  Downloading dataset from UCI repository...")
        import urllib.request, zipfile, io
        url = ("https://archive.ics.uci.edu/ml/machine-learning-databases/"
               "00222/bank-additional.zip")
        with urllib.request.urlopen(url) as r:
            z = zipfile.ZipFile(io.BytesIO(r.read()))
        z.extractall(DATA_DIR)
        # rename if needed
        extracted = DATA_DIR / "bank-additional" / "bank-additional-full.csv"
        if extracted.exists():
            extracted.rename(path)
    df = pd.read_csv(path, sep=";")
    print(f"  Loaded {len(df):,} rows × {df.shape[1]} cols")
    return df


# ── 2. Feature Engineering ───────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["y"] = (df["y"] == "yes").astype(int)

    # Aggregate previous campaign contact into a single flag
    df["was_contacted_before"] = (df["previous"] > 0).astype(int)

    # Season from month (European rate cycles are seasonal)
    season_map = {
        "jan": "winter", "feb": "winter", "mar": "spring",
        "apr": "spring", "may": "spring", "jun": "summer",
        "jul": "summer", "aug": "summer", "sep": "autumn",
        "oct": "autumn", "nov": "autumn", "dec": "winter",
    }
    df["season"] = df["month"].map(season_map)

    # Economic stress proxy
    df["rate_spread"] = df["euribor3m"] - df["cons.price.idx"]

    return df


# ── 3. Preprocessing ─────────────────────────────────────────────────────────
CAT_COLS = ["job", "marital", "education", "default", "housing",
            "loan", "contact", "month", "day_of_week", "poutcome", "season"]

NUM_COLS_BASE = [
    "age", "campaign", "pdays", "previous",
    "emp.var.rate", "cons.price.idx", "cons.conf.idx",
    "euribor3m", "nr.employed",
    "was_contacted_before", "rate_spread"
]

LEAKY_COL = "duration"

def build_preprocessor(include_duration: bool) -> ColumnTransformer:
    num_cols = NUM_COLS_BASE + ([LEAKY_COL] if include_duration else [])
    return ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]), CAT_COLS),
    ], remainder="drop")


# ── 4. Train & Evaluate ──────────────────────────────────────────────────────
def train_model(X_train, y_train, scale_pos_weight: float):
    clf = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=30,
    )
    # Eval set for early stopping (no SMOTE on val)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.1, stratify=y_train, random_state=42
    )
    clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return clf


def evaluate(clf, preprocessor, X_test, y_test, label: str):
    X_proc = preprocessor.transform(X_test)
    proba = clf.predict_proba(X_proc)[:, 1]
    pred = (proba >= 0.5).astype(int)

    auc = roc_auc_score(y_test, proba)
    f1 = f1_score(y_test, pred)
    print(f"\n{'─'*50}")
    print(f"  {label}")
    print(f"  AUC-ROC : {auc:.4f}")
    print(f"  F1(yes) : {f1:.4f}")
    print(f"\n{classification_report(y_test, pred, target_names=['no','yes'])}")
    return proba, auc, f1


# ── 5. Lift Curve ─────────────────────────────────────────────────────────────
def lift_curve_data(y_true, proba, n_bins=20):
    df = pd.DataFrame({"y": y_true, "p": proba}).sort_values("p", ascending=False)
    chunk = len(df) // n_bins
    lifts, recalls = [], []
    base_rate = y_true.mean()
    for i in range(1, n_bins + 1):
        top = df.iloc[: i * chunk]
        hit_rate = top["y"].mean()
        lifts.append(hit_rate / base_rate)
        recalls.append(top["y"].sum() / y_true.sum())
    pct_called = [(i / n_bins) * 100 for i in range(1, n_bins + 1)]
    return pct_called, lifts, recalls


# ── 6. Plots ─────────────────────────────────────────────────────────────────
def make_plots(results: dict, out_dir: Path):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    colors = {"v1_with_duration": "#2563EB", "v2_no_duration": "#DC2626"}
    labels = {"v1_with_duration": "With duration (leaky)", "v2_no_duration": "No duration (production)"}

    # ── ROC Curves ────────────────────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, 0])
    for key, res in results.items():
        fpr, tpr, _ = roc_curve(res["y_test"], res["proba"])
        ax0.plot(fpr, tpr, color=colors[key],
                 label=f"{labels[key]} (AUC={res['auc']:.3f})", lw=2)
    ax0.plot([0,1],[0,1],"--", color="#9CA3AF", lw=1)
    ax0.set(xlabel="FPR", ylabel="TPR", title="ROC Curves")
    ax0.legend(fontsize=8)

    # ── PR Curves ─────────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 1])
    for key, res in results.items():
        p, r, _ = precision_recall_curve(res["y_test"], res["proba"])
        ax1.plot(r, p, color=colors[key], label=labels[key], lw=2)
    ax1.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curves")
    ax1.legend(fontsize=8)

    # ── Lift Curves ───────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    for key, res in results.items():
        pct, lifts, _ = lift_curve_data(res["y_test"], res["proba"])
        ax2.plot(pct, lifts, color=colors[key], label=labels[key], lw=2)
    ax2.axhline(1.0, linestyle="--", color="#9CA3AF", lw=1, label="Random baseline")
    ax2.set(xlabel="% Customers Called", ylabel="Lift", title="Cumulative Lift Curve")
    ax2.legend(fontsize=8)

    # ── Recall vs % Called ────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    for key, res in results.items():
        pct, _, recalls = lift_curve_data(res["y_test"], res["proba"])
        ax3.plot(pct, [r * 100 for r in recalls], color=colors[key],
                 label=labels[key], lw=2)
    # Random baseline
    ax3.plot([0, 100], [0, 100], "--", color="#9CA3AF", lw=1, label="Random")
    ax3.set(xlabel="% Customers Called", ylabel="% Subscribers Captured",
            title="Recall vs. Effort Curve")
    ax3.legend(fontsize=8)

    # ── Confusion Matrix (no-duration model) ─────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    cm = confusion_matrix(results["v2_no_duration"]["y_test"],
                          (results["v2_no_duration"]["proba"] >= 0.5).astype(int))
    im = ax4.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax4.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                     fontsize=14, color="white" if cm[i,j] > cm.max()/2 else "black")
    ax4.set(xticks=[0,1], yticks=[0,1], xticklabels=["No","Yes"],
            yticklabels=["No","Yes"], xlabel="Predicted", ylabel="Actual",
            title="Confusion Matrix\n(Production Model, threshold=0.5)")
    plt.colorbar(im, ax=ax4)

    # ── Score Distribution ────────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    res = results["v2_no_duration"]
    ax5.hist(res["proba"][res["y_test"] == 0], bins=50, alpha=0.6,
             color="#6B7280", label="No (did not subscribe)", density=True)
    ax5.hist(res["proba"][res["y_test"] == 1], bins=50, alpha=0.7,
             color="#DC2626", label="Yes (subscribed)", density=True)
    ax5.set(xlabel="Predicted Probability", ylabel="Density",
            title="Score Separation\n(Production Model)")
    ax5.legend(fontsize=8)

    fig.suptitle("UCI Bank Marketing — Model Evaluation Dashboard", fontsize=14, y=1.01)
    out = out_dir / "evaluation_dashboard.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: {out}")


def make_shap_plots(clf, X_proc, feature_names: list, out_dir: Path):
    print("\n  Computing SHAP values...")
    explainer = shap.TreeExplainer(clf)
    shap_vals = explainer.shap_values(X_proc[:2000])  # sample for speed

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    plt.sca(axes[0])
    shap.summary_plot(shap_vals, X_proc[:2000], feature_names=feature_names,
                      show=False, plot_type="dot", max_display=15)
    axes[0].set_title("SHAP Summary — Feature Impact", fontsize=12)

    plt.sca(axes[1])
    shap.summary_plot(shap_vals, X_proc[:2000], feature_names=feature_names,
                      show=False, plot_type="bar", max_display=15)
    axes[1].set_title("SHAP Bar — Mean |SHAP| (Global Importance)", fontsize=12)

    plt.tight_layout()
    out = out_dir / "shap_analysis.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  UCI Bank Marketing — ML Pipeline")
    print("=" * 60)

    # 1. Data
    print("\n[1/5] Loading & engineering features...")
    DATA_DIR.mkdir(exist_ok=True)
    df = load_data()
    df = engineer_features(df)

    X = df.drop(columns=["y"])
    y = df["y"]

    spw = (y == 0).sum() / (y == 1).sum()
    print(f"  Class ratio (neg/pos): {spw:.1f}x  →  scale_pos_weight={spw:.1f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    results = {}

    for variant, include_dur in [("v1_with_duration", True), ("v2_no_duration", False)]:
        print(f"\n[{'2' if include_dur else '3'}/5] Training: {variant}...")

        preprocessor = build_preprocessor(include_duration=include_dur)
        X_tr_proc = preprocessor.fit_transform(X_train)
        X_te_proc = preprocessor.transform(X_test)

        # SMOTE on training set only
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X_tr_proc, y_train)
        print(f"  After SMOTE: {y_res.value_counts().to_dict()}")

        clf = train_model(X_res, y_res, scale_pos_weight=1.0)  # balanced after SMOTE

        proba, auc, f1 = evaluate(clf, preprocessor, X_test, y_test, variant)

        # Lift @ 20%
        pct, lifts, recalls = lift_curve_data(y_test.values, proba)
        idx_20 = next(i for i, p in enumerate(pct) if p >= 20)
        print(f"  Lift @ top 20%: {lifts[idx_20]:.2f}x  |  "
              f"Recall @ top 20%: {recalls[idx_20]*100:.1f}%")

        results[variant] = {
            "clf": clf,
            "preprocessor": preprocessor,
            "proba": proba,
            "y_test": y_test.values,
            "auc": auc,
            "f1": f1,
            "feature_names": (
                preprocessor.transformers_[0][2] + preprocessor.transformers_[1][2]
            ),
        }

    print("\n[4/5] Generating evaluation plots...")
    make_plots(results, OUT_DIR)

    print("\n[5/5] Generating SHAP analysis (production model)...")
    res = results["v2_no_duration"]
    X_te_proc = res["preprocessor"].transform(X_test)
    make_shap_plots(res["clf"], X_te_proc, res["feature_names"], OUT_DIR)

    # Save lift data to CSV for README badge
    pct, lifts, recalls = lift_curve_data(
        results["v2_no_duration"]["y_test"],
        results["v2_no_duration"]["proba"]
    )
    lift_df = pd.DataFrame({"pct_called": pct, "lift": lifts, "recall_pct": [r*100 for r in recalls]})
    lift_df.to_csv(OUT_DIR / "lift_curve_data.csv", index=False)

    print("\n" + "=" * 60)
    print("  Pipeline complete. Outputs saved to /outputs/")
    print("=" * 60)


if __name__ == "__main__":
    main()
