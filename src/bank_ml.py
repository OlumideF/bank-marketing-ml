"""
Shared ML logic for pipeline, Streamlit app, and HTML report.
"""

from __future__ import annotations

import io
import urllib.request
import warnings
import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import xgboost as xgb

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
REPORTS_DIR = ROOT / "reports"
ARTIFACTS_PATH = OUT_DIR / "artifacts.joblib"

COLOR_PRODUCTION = "#2563EB"
COLOR_LEAKY = "#DC2626"

VARIANTS = {
    "v1_with_duration": {"include_duration": True, "label": "With duration (leaky)"},
    "v2_no_duration": {"include_duration": False, "label": "No duration (production)"},
}

CAT_COLS = [
    "job", "marital", "education", "default", "housing",
    "loan", "contact", "month", "day_of_week", "poutcome", "season",
]

NUM_COLS_BASE = [
    "age", "campaign", "pdays", "previous",
    "emp.var.rate", "cons.price.idx", "cons.conf.idx",
    "euribor3m", "nr.employed",
    "was_contacted_before", "rate_spread",
]

LEAKY_COL = "duration"

SHAP_EXPLANATIONS = {
    "euribor3m": (
        "High Euribor rates suppress subscription likelihood. When short-term rates "
        "are elevated, customers prefer liquid instruments and have less incentive "
        "to lock funds in a term deposit."
    ),
    "emp.var.rate": (
        "Employment variation rate reflects labor-market stress. Higher unemployment "
        "signals reduce willingness to commit to long-term savings products."
    ),
    "nr.employed": (
        "Fewer employed workers in the economy correlate with weaker deposit appetite "
        "as households preserve cash buffers."
    ),
    "cons.conf.idx": (
        "Consumer confidence shapes savings behavior — pessimistic consumers defer "
        "locking money into term deposits."
    ),
    "cons.price.idx": (
        "Inflation dynamics affect real returns on deposits; price index shifts "
        "change the perceived attractiveness of fixed-term products."
    ),
    "age": (
        "Age segments differ in liquidity needs and risk tolerance, shifting "
        "responsiveness to telemarketing for term deposits."
    ),
    "campaign": (
        "Repeated contacts in the same campaign can fatigue customers, lowering "
        "conversion on additional calls."
    ),
    "pdays": (
        "Recent prior contact affects receptiveness — very recent outreach may "
        "help or hurt depending on prior experience."
    ),
    "poutcome": (
        "Outcome of the previous campaign is a strong behavioral signal — past "
        "success or failure predicts current responsiveness."
    ),
    "was_contacted_before": (
        "Customers contacted in earlier campaigns may be warmer leads or already "
        "exhausted, depending on history."
    ),
    "rate_spread": (
        "Spread between Euribor and consumer price index proxies economic stress; "
        "wider spreads often coincide with weaker deposit uptake."
    ),
    "housing": (
        "Existing housing loan obligations reduce free cash flow available for "
        "new term deposits."
    ),
    "loan": (
        "Personal loan holders may prioritize debt service over new savings "
        "commitments."
    ),
    "marital": (
        "Household structure influences financial planning and deposit decisions."
    ),
    "job": (
        "Occupation proxies income stability and product fit for term deposits."
    ),
}


def load_data() -> pd.DataFrame:
    path = DATA_DIR / "bank-additional-full.csv"
    if not path.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        url = (
            "https://archive.ics.uci.edu/ml/machine-learning-databases/"
            "00222/bank-additional.zip"
        )
        with urllib.request.urlopen(url) as r:
            z = zipfile.ZipFile(io.BytesIO(r.read()))
        z.extractall(DATA_DIR)
        extracted = DATA_DIR / "bank-additional" / "bank-additional-full.csv"
        if extracted.exists():
            extracted.rename(path)
    df = pd.read_csv(path, sep=";")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "y" in df.columns:
        y = df["y"]
        if pd.api.types.is_numeric_dtype(y):
            df["y"] = y.astype(int)
        else:
            df["y"] = (y.astype(str).str.lower() == "yes").astype(int)

    df["was_contacted_before"] = (df["previous"] > 0).astype(int)
    season_map = {
        "jan": "winter", "feb": "winter", "mar": "spring",
        "apr": "spring", "may": "spring", "jun": "summer",
        "jul": "summer", "aug": "summer", "sep": "autumn",
        "oct": "autumn", "nov": "autumn", "dec": "winter",
    }
    df["season"] = df["month"].map(season_map)
    df["rate_spread"] = df["euribor3m"] - df["cons.price.idx"]
    return df


def build_preprocessor(include_duration: bool) -> ColumnTransformer:
    num_cols = NUM_COLS_BASE + ([LEAKY_COL] if include_duration else [])
    return ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("enc", OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1
            )),
        ]), CAT_COLS),
    ], remainder="drop")


def train_model(X_train, y_train):
    clf = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=1.0,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=30,
    )
    y_arr = np.asarray(y_train).ravel()
    if y_arr.dtype == object or np.issubdtype(y_arr.dtype, np.str_):
        y_arr = (y_arr == "yes").astype(int)
    else:
        y_arr = y_arr.astype(int)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_arr, test_size=0.1, stratify=y_arr, random_state=42
    )
    clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return clf


def evaluate(clf, preprocessor, X_test, y_test, threshold: float = 0.5):
    X_proc = preprocessor.transform(X_test)
    proba = clf.predict_proba(X_proc)[:, 1]
    pred = (proba >= threshold).astype(int)
    auc = roc_auc_score(y_test, proba)
    f1 = f1_score(y_test, pred)
    return proba, auc, f1


def _fit_variant(X_train, y_train, include_duration: bool):
    preprocessor = build_preprocessor(include_duration)
    X_tr_proc = preprocessor.fit_transform(X_train)
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_tr_proc, y_train)
    y_res = np.asarray(y_res).astype(int)
    clf = train_model(X_res, y_res)
    return clf, preprocessor


def cross_validate_models(n_splits: int = 5, verbose: bool = True) -> pd.DataFrame:
    """StratifiedKFold CV for v1 vs v2; returns comparison table with mean +/- std."""
    df = load_data()
    df = engineer_features(df)
    X = df.drop(columns=["y"])
    y = df["y"]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    scores = {k: {"auc": [], "f1": []} for k in VARIANTS}

    for variant, cfg in VARIANTS.items():
        if verbose:
            print(f"  CV: {variant} ({n_splits} folds)...")
        for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
            clf, preprocessor = _fit_variant(X_tr, y_tr, cfg["include_duration"])
            _, auc, f1 = evaluate(clf, preprocessor, X_va, y_va)
            scores[variant]["auc"].append(auc)
            scores[variant]["f1"].append(f1)
            if verbose:
                print(f"    fold {fold}: AUC={auc:.4f}  F1={f1:.4f}")

    rows = []
    for variant in VARIANTS:
        aucs = scores[variant]["auc"]
        f1s = scores[variant]["f1"]
        rows.append({
            "model": variant,
            "auc_mean": np.mean(aucs),
            "auc_std": np.std(aucs),
            "f1_mean": np.mean(f1s),
            "f1_std": np.std(f1s),
            "deployable": variant == "v2_no_duration",
        })
    return pd.DataFrame(rows)


def lift_curve_data(y_true, proba, n_bins=20):
    df = pd.DataFrame({"y": y_true, "p": proba}).sort_values("p", ascending=False)
    chunk = max(len(df) // n_bins, 1)
    lifts, recalls = [], []
    base_rate = float(np.mean(y_true))
    pct_called = []
    for i in range(1, n_bins + 1):
        top = df.iloc[: i * chunk]
        hit_rate = top["y"].mean() if len(top) else 0.0
        lifts.append(hit_rate / base_rate if base_rate else 0.0)
        recalls.append(top["y"].sum() / max(y_true.sum(), 1))
        pct_called.append((i / n_bins) * 100)
    return pct_called, lifts, recalls


def metrics_at_pct(y_true, proba, pct=20.0):
    pct_list, lifts, recalls = lift_curve_data(y_true, proba)
    idx = next(i for i, p in enumerate(pct_list) if p >= pct)
    return {
        "lift": lifts[idx],
        "recall_pct": recalls[idx] * 100,
        "pct_called": pct_list[idx],
    }


def confusion_at_threshold(y_true, proba, threshold=0.5):
    pred = (proba >= threshold).astype(int)
    return confusion_matrix(y_true, pred)


def compute_shap(clf, X_proc, feature_names, sample_size=2000):
    n = min(sample_size, len(X_proc))
    X_sample = X_proc[:n]
    explainer = shap.TreeExplainer(clf)
    shap_vals = explainer.shap_values(X_sample)
    mean_abs = np.abs(shap_vals).mean(axis=0)
    mean_shap = shap_vals.mean(axis=0)
    importance = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs,
        "mean_shap": mean_shap,
    }).sort_values("mean_abs_shap", ascending=False)
    return shap_vals, X_sample, importance


def shap_feature_narratives(importance_df, top_n=5):
    narratives = []
    for _, row in importance_df.head(top_n).iterrows():
        feat = row["feature"]
        direction = "increases" if row["mean_shap"] > 0 else "decreases"
        text = SHAP_EXPLANATIONS.get(
            feat,
            f"Higher values of {feat} tend to {direction} subscription probability "
            f"based on SHAP impact across the scored population.",
        )
        narratives.append({
            "feature": feat,
            "direction": direction,
            "mean_abs_shap": float(row["mean_abs_shap"]),
            "text": text,
        })
    return narratives


def score_customers(df: pd.DataFrame, clf, preprocessor) -> pd.DataFrame:
    df = engineer_features(df)
    if "y" in df.columns:
        df = df.drop(columns=["y"])
    X_proc = preprocessor.transform(df)
    proba = clf.predict_proba(X_proc)[:, 1]
    out = df.copy()
    out["subscription_probability"] = proba
    return out.sort_values("subscription_probability", ascending=False)


def roi_estimates(n_customers, pct_call, y_true, proba):
    """Estimate subscribers captured vs random and calls saved (production curve)."""
    pct_list, lifts, recalls = lift_curve_data(y_true, proba)
    base_rate = float(np.mean(y_true))
    total_subs = max(float(y_true.sum()), 1.0)
    idx = min(range(len(pct_list)), key=lambda i: abs(pct_list[i] - pct_call))
    lift = lifts[idx]
    recall_frac = recalls[idx]

    n_call = int(n_customers * pct_call / 100)
    random_subs = n_call * base_rate
    model_subs = recall_frac * total_subs * (n_customers / len(y_true))

    # Calls needed with random dialing to capture same subscribers as model top pct_call
    subs_target = recall_frac * total_subs * (n_customers / len(y_true))
    random_pct_needed = min(100.0, (subs_target / (n_customers * base_rate)) * pct_call) if base_rate else pct_call
    calls_saved = max(0, int(n_customers * (random_pct_needed - pct_call) / 100))

    return {
        "n_call": n_call,
        "random_subscribers": random_subs,
        "model_subscribers": model_subs,
        "lift": lift,
        "recall_pct": recall_frac * 100,
        "calls_saved": calls_saved,
    }


def train_all_variants(
    verbose: bool = True,
    variant_keys: list[str] | None = None,
    threshold: float = 0.5,
    run_cv: bool = False,
):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(exist_ok=True)

    df = load_data()
    if verbose:
        print(f"  Loaded {len(df):,} rows x {df.shape[1]} cols")
    df = engineer_features(df)

    X = df.drop(columns=["y"])
    y = df["y"]
    spw = (y == 0).sum() / (y == 1).sum()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    keys = variant_keys or list(VARIANTS.keys())
    results = {}
    for variant in keys:
        cfg = VARIANTS[variant]
        if verbose:
            print(f"\n  Training: {variant}...")
        clf, preprocessor = _fit_variant(X_train, y_train, cfg["include_duration"])
        proba, auc, f1 = evaluate(
            clf, preprocessor, X_test, y_test, threshold=threshold,
        )
        m20 = metrics_at_pct(y_test.values, proba, pct=20.0)

        fpr, tpr, _ = roc_curve(y_test, proba)
        prec, rec, _ = precision_recall_curve(y_test, proba)
        pct, lifts, recalls = lift_curve_data(y_test.values, proba)
        feature_names = (
            list(preprocessor.transformers_[0][2])
            + list(preprocessor.transformers_[1][2])
        )

        results[variant] = {
            "clf": clf,
            "preprocessor": preprocessor,
            "proba": proba,
            "y_test": y_test.values,
            "auc": auc,
            "f1": f1,
            "threshold": threshold,
            "lift_20": m20["lift"],
            "recall_20": m20["recall_pct"],
            "feature_names": feature_names,
            "roc": {"fpr": fpr, "tpr": tpr},
            "pr": {"precision": prec, "recall": rec},
            "lift_curve": {"pct": pct, "lift": lifts, "recall": recalls},
        }

    prod_key = "v2_no_duration" if "v2_no_duration" in results else keys[-1]
    prod = results[prod_key]
    X_te_proc = prod["preprocessor"].transform(X_test)
    shap_vals, X_shap, importance = compute_shap(
        prod["clf"], X_te_proc, prod["feature_names"]
    )

    cv_table = cross_validate_models(verbose=verbose) if run_cv and len(keys) == len(VARIANTS) else None

    artifacts = {
        "results": results,
        "cv_comparison": cv_table,
        "dataset_stats": {
            "n_rows": len(df),
            "positive_rate": float(y.mean()),
            "train_size": len(X_train),
            "test_size": len(X_test),
            "scale_pos_weight": float(spw),
        },
        "shap": {
            "values": shap_vals,
            "X_sample": X_shap,
            "feature_names": prod["feature_names"],
            "importance": importance,
            "narratives": shap_feature_narratives(importance),
        },
        "X_test": X_test,
    }
    return artifacts


def save_artifacts(artifacts: dict):
    OUT_DIR.mkdir(exist_ok=True)
    joblib.dump(artifacts, ARTIFACTS_PATH)


def load_artifacts(retrain_if_missing: bool = True):
    if ARTIFACTS_PATH.exists():
        return joblib.load(ARTIFACTS_PATH)
    if retrain_if_missing:
        artifacts = train_all_variants(verbose=False)
        save_artifacts(artifacts)
        return artifacts
    raise FileNotFoundError(
        f"No artifacts at {ARTIFACTS_PATH}. Run: python src/pipeline.py"
    )
