"""
Exploratory Data Analysis — UCI Bank Marketing Dataset
========================================================
Descriptive statistics + publication-ready figures.

Run standalone:
    python src/eda.py
    python src/eda.py --output-dir outputs
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bank_ml import DATA_DIR, OUT_DIR, COLOR_PRODUCTION, load_data  # noqa: E402

PLOTLY_TEMPLATE = "plotly_dark"
CAT_FEATURES = [
    "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "day_of_week", "poutcome",
]
NUM_FEATURES = [
    "age", "duration", "campaign", "pdays", "previous",
    "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed",
]
MACRO_FEATURES = ["emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]


def _target_series(df: pd.DataFrame) -> pd.Series:
    y = df["y"]
    if pd.api.types.is_numeric_dtype(y):
        return y.astype(int)
    return (y.astype(str).str.lower() == "yes").astype(int)


def _subscription_rate(df: pd.DataFrame, column: str) -> pd.DataFrame:
    t = _target_series(df)
    tmp = df.copy()
    tmp["_y"] = t
    out = tmp.groupby(column, observed=True)["_y"].agg(["mean", "count"])
    out.columns = ["subscription_rate", "count"]
    out["subscription_rate"] *= 100
    return out.sort_values("subscription_rate", ascending=False)


def summary_statistics(df: pd.DataFrame) -> dict:
    """Descriptive stats for reports and Streamlit."""
    t = _target_series(df)
    numeric = df[NUM_FEATURES].describe().round(3)
    missing = df.isna().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    corr_target = (
        df[NUM_FEATURES].assign(subscribed=t).corr(numeric_only=True)["subscribed"]
        .drop("subscribed")
        .sort_values(key=abs, ascending=False)
        .round(3)
    )
    cat_cardinality = {c: int(df[c].nunique()) for c in CAT_FEATURES if c in df.columns}

    return {
        "n_rows": len(df),
        "n_columns": df.shape[1],
        "positive_rate_pct": round(float(t.mean() * 100), 2),
        "negative_count": int((t == 0).sum()),
        "positive_count": int((t == 1).sum()),
        "class_ratio": round(float((t == 0).sum() / max((t == 1).sum(), 1)), 1),
        "missing_cells": int(missing.sum()),
        "duration_corr_target": round(float(corr_target.get("duration", 0)), 3),
        "numeric_summary": numeric,
        "missing": missing[missing > 0] if missing.any() else pd.Series(dtype=int),
        "missing_pct": missing_pct[missing_pct > 0] if missing_pct.any() else pd.Series(dtype=float),
        "corr_with_target": corr_target,
        "categorical_cardinality": cat_cardinality,
    }


def plot_eda_overview(df: pd.DataFrame, out_dir: Path) -> Path:
    """Six-panel overview (class balance, age, euribor, month, duration, poutcome)."""
    t = _target_series(df)
    work = df.copy()
    work["_y"] = t
    work["_ylabel"] = np.where(t == 1, "yes", "no")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    counts = work["_ylabel"].value_counts().reindex(["no", "yes"])
    axes[0].bar(counts.index, counts.values, color=["#6B7280", COLOR_PRODUCTION], width=0.5)
    axes[0].set(title="Target Class Imbalance", xlabel="Subscribed?", ylabel="Count")
    for bar, val in zip(axes[0].patches, counts.values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 100,
            f"{val:,}\n({val / len(work) * 100:.1f}%)", ha="center", fontsize=10,
        )

    for label, color in [("no", "#6B7280"), ("yes", COLOR_PRODUCTION)]:
        subset = work.loc[work["_ylabel"] == label, "age"]
        subset.plot(kind="kde", ax=axes[1], color=color, label=label)
    axes[1].set(title="Age Distribution by Outcome", xlabel="Age")
    axes[1].legend(title="Subscribed")

    work["euribor_bin"] = pd.cut(work["euribor3m"], bins=6)
    rate = work.groupby("euribor_bin", observed=True)["_y"].mean() * 100
    axes[2].bar(range(len(rate)), rate.values, color=COLOR_PRODUCTION, alpha=0.85)
    axes[2].set_xticks(range(len(rate)))
    axes[2].set_xticklabels([str(b) for b in rate.index], rotation=30, ha="right", fontsize=7)
    axes[2].set(title="Subscription Rate by Euribor3m", ylabel="Rate (%)")

    month_order = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
    pivot = work.groupby("month")["_y"].mean().reindex(month_order)
    axes[3].bar(pivot.index, pivot.values * 100, color="#DC2626", alpha=0.8)
    axes[3].set(title="Subscription Rate by Month", ylabel="Rate (%)")
    plt.setp(axes[3].xaxis.get_majorticklabels(), rotation=45)

    yes_dur = work.loc[work["_y"] == 1, "duration"]
    no_dur = work.loc[work["_y"] == 0, "duration"].sample(min(len(yes_dur) * 2, len(work)), random_state=42)
    axes[4].hist(no_dur.clip(0, 1500), bins=40, alpha=0.6, color="#6B7280", label="No (sample)", density=True)
    axes[4].hist(yes_dur.clip(0, 1500), bins=40, alpha=0.7, color="#F59E0B", label="Yes", density=True)
    axes[4].set(title="Duration (LEAKY — not for production)", xlabel="Seconds")
    axes[4].legend(fontsize=9)
    axes[4].set_facecolor("#FFF7ED")

    pout = work.groupby("poutcome")["_y"].mean() * 100
    axes[5].bar(pout.index, pout.values, color="#10B981", alpha=0.85)
    axes[5].set(title="Subscription Rate by Previous Outcome", ylabel="Rate (%)")

    fig.suptitle("UCI Bank Marketing — EDA Overview", fontsize=14, y=1.01)
    plt.tight_layout()
    out = out_dir / "eda_overview.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return out


def plot_eda_numeric(df: pd.DataFrame, out_dir: Path) -> Path:
    """Numeric distributions + correlation heatmap."""
    t = _target_series(df)
    num_df = df[NUM_FEATURES].assign(subscribed=t)

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.45, wspace=0.35)

    hist_cols = ["age", "campaign", "pdays", "euribor3m", "cons.conf.idx", "nr.employed"]
    for i, col in enumerate(hist_cols):
        ax = fig.add_subplot(gs[i // 4, i % 4])
        ax.hist(df[col], bins=35, color=COLOR_PRODUCTION, alpha=0.75, edgecolor="white")
        ax.set_title(col, fontsize=10)
        ax.set_ylabel("Count")

    ax_corr = fig.add_subplot(gs[2, 2:])
    corr = num_df.corr()
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
        ax=ax_corr, square=True, cbar_kws={"shrink": 0.8},
        annot_kws={"size": 7},
    )
    ax_corr.set_title("Numeric Feature Correlation Matrix")

    fig.suptitle("Numeric Features — Distributions & Correlations", fontsize=14, y=1.01)
    out = out_dir / "eda_numeric.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return out


def plot_eda_categorical(df: pd.DataFrame, out_dir: Path) -> Path:
    """Subscription rates across key categoricals."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    panels = [
        ("job", 12, "Subscription Rate by Job (top 12)"),
        ("marital", None, "Subscription Rate by Marital Status"),
        ("education", None, "Subscription Rate by Education"),
        ("contact", None, "Subscription Rate by Contact Type"),
    ]
    for ax, (col, top_n, title) in zip(axes.flatten(), panels):
        rates = _subscription_rate(df, col)
        if top_n:
            rates = rates.head(top_n)
        ax.barh(rates.index.astype(str), rates["subscription_rate"], color=COLOR_PRODUCTION, alpha=0.85)
        ax.set_xlabel("Subscription rate (%)")
        ax.set_title(title)
        ax.invert_yaxis()

    fig.suptitle("Categorical Features vs Subscription", fontsize=14, y=1.01)
    plt.tight_layout()
    out = out_dir / "eda_categorical.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return out


def plot_eda_campaign(df: pd.DataFrame, out_dir: Path) -> Path:
    """Campaign intensity and prior contact patterns."""
    t = _target_series(df)
    work = df.copy()
    work["_y"] = t

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for label, color in [(0, "#6B7280"), (1, COLOR_PRODUCTION)]:
        axes[0].hist(
            work.loc[work["_y"] == label, "campaign"], bins=range(1, 36),
            alpha=0.6, label="No" if label == 0 else "Yes", color=color, density=True,
        )
    axes[0].set(title="Campaign Contacts This Run", xlabel="# contacts", ylabel="Density")
    axes[0].legend()

    prev_rate = work.groupby("previous")["_y"].mean()
    prev_rate = prev_rate[prev_rate.index <= 10] * 100
    axes[1].plot(prev_rate.index, prev_rate.values, marker="o", color=COLOR_PRODUCTION, lw=2)
    axes[1].set(title="Subscription Rate vs Prior Contacts", xlabel="previous", ylabel="Rate (%)")

    pdays_ok = work[work["pdays"] != 999]
    if len(pdays_ok) > 100:
        pdays_ok = pdays_ok.copy()
        pdays_ok["pdays_bin"] = pd.cut(pdays_ok["pdays"], bins=8)
        pr = pdays_ok.groupby("pdays_bin", observed=True)["_y"].mean() * 100
        axes[2].bar(range(len(pr)), pr.values, color="#10B981", alpha=0.85)
        axes[2].set_xticks(range(len(pr)))
        axes[2].set_xticklabels([str(x) for x in pr.index], rotation=35, ha="right", fontsize=7)
    axes[2].set(title="Subscription Rate by Days Since Last Contact", ylabel="Rate (%)")

    fig.suptitle("Campaign & Contact History", fontsize=14, y=1.02)
    plt.tight_layout()
    out = out_dir / "eda_campaign.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return out


def save_summary_json(summary: dict, out_dir: Path) -> Path:
    """Persist scalar stats (not full DataFrames) for dashboards."""
    payload = {
        k: v for k, v in summary.items()
        if k not in ("numeric_summary", "missing", "missing_pct", "corr_with_target")
    }
    payload["top_correlations"] = summary["corr_with_target"].head(8).to_dict()
    path = out_dir / "eda_summary.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    summary["numeric_summary"].to_csv(out_dir / "eda_numeric_stats.csv")
    summary["corr_with_target"].to_csv(out_dir / "eda_target_correlations.csv", header=["correlation"])
    return path


def run_eda(df: pd.DataFrame | None = None, out_dir: Path | None = None, verbose: bool = True) -> dict:
    """Full EDA: stats + all figures."""
    out_dir = out_dir or OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if df is None:
        if verbose:
            print("  Loading data...")
        df = load_data()

    summary = summary_statistics(df)
    if verbose:
        print(f"  Rows: {summary['n_rows']:,}  |  Positive rate: {summary['positive_rate_pct']}%")
        print(f"  Class ratio (neg:pos): {summary['class_ratio']}:1")
        print(f"  Duration corr w/ target: {summary['duration_corr_target']} (leakage signal)")

    paths = [
        plot_eda_overview(df, out_dir),
        plot_eda_numeric(df, out_dir),
        plot_eda_categorical(df, out_dir),
        plot_eda_campaign(df, out_dir),
    ]
    json_path = save_summary_json(summary, out_dir)

    if verbose:
        for p in paths:
            print(f"  Saved: {p}")
        print(f"  Saved: {json_path}")

    return {"summary": summary, "figure_paths": paths}


# ── Plotly helpers for Streamlit ─────────────────────────────────────────────

def plotly_class_balance(df: pd.DataFrame) -> go.Figure:
    t = _target_series(df)
    labels = ["No", "Yes"]
    values = [(t == 0).sum(), (t == 1).sum()]
    fig = go.Figure(go.Pie(labels=labels, values=values, hole=0.45, marker_colors=["#6B7280", COLOR_PRODUCTION]))
    fig.update_layout(template=PLOTLY_TEMPLATE, title="Target Class Balance", height=380)
    return fig


def plotly_subscription_by_binned(df: pd.DataFrame, column: str, bins: int = 10) -> go.Figure:
    work = df.copy()
    work["_bin"] = pd.cut(work[column], bins=bins)
    return plotly_subscription_by(work, "_bin", top_n=None)


def plotly_subscription_by(df: pd.DataFrame, column: str, top_n: int | None = 15) -> go.Figure:
    rates = _subscription_rate(df, column)
    if top_n:
        rates = rates.head(top_n)
    fig = go.Figure(go.Bar(
        x=rates["subscription_rate"], y=rates.index.astype(str),
        orientation="h", marker_color=COLOR_PRODUCTION,
        text=[f"{v:.1f}%" for v in rates["subscription_rate"]], textposition="outside",
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=f"Subscription Rate by {column}",
        xaxis_title="Rate (%)", height=max(400, 28 * len(rates)),
        margin=dict(l=120),
    )
    return fig


def plotly_numeric_hist(df: pd.DataFrame, column: str) -> go.Figure:
    fig = px.histogram(df, x=column, nbins=40, color_discrete_sequence=[COLOR_PRODUCTION])
    fig.update_layout(template=PLOTLY_TEMPLATE, title=f"Distribution: {column}", height=360)
    return fig


def plotly_corr_heatmap(df: pd.DataFrame) -> go.Figure:
    t = _target_series(df)
    corr = df[NUM_FEATURES].assign(subscribed=t).corr()
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns,
        colorscale="RdBu", zmid=0, text=np.round(corr.values, 2),
        texttemplate="%{text}", textfont={"size": 8},
    ))
    fig.update_layout(template=PLOTLY_TEMPLATE, title="Numeric Correlations", height=520)
    return fig


def plotly_scatter_macro(df: pd.DataFrame, x_col: str, y_col: str = "euribor3m") -> go.Figure:
    t = _target_series(df)
    sample = df.sample(min(5000, len(df)), random_state=42).copy()
    sample["_y"] = t.loc[sample.index]
    fig = px.scatter(
        sample, x=x_col, y=y_col, color="_y",
        color_discrete_map={0: "#6B7280", 1: COLOR_PRODUCTION},
        labels={"_y": "Subscribed"},
        opacity=0.5,
    )
    fig.update_layout(template=PLOTLY_TEMPLATE, title=f"{x_col} vs {y_col}", height=400)
    return fig


def main():
    parser = argparse.ArgumentParser(description="Run EDA on Bank Marketing data")
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    print("=" * 60)
    print("  UCI Bank Marketing — Exploratory Data Analysis")
    print("=" * 60)
    run_eda(out_dir=args.output_dir.resolve(), verbose=True)
    print("=" * 60)


if __name__ == "__main__":
    main()
