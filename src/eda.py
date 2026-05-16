"""
EDA — UCI Bank Marketing Dataset
==================================
Produces figures used in the notebook and README.
Run standalone: python src/eda.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)


def load() -> pd.DataFrame:
    path = DATA_DIR / "bank-additional-full.csv"
    return pd.read_csv(path, sep=";")


def plot_eda(df: pd.DataFrame, out_dir: Path):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    # 1. Class imbalance
    counts = df["y"].value_counts()
    axes[0].bar(counts.index, counts.values, color=["#6B7280", "#2563EB"], width=0.5)
    axes[0].set(title="Target Class Imbalance", xlabel="Subscribed?", ylabel="Count")
    for bar, val in zip(axes[0].patches, counts.values):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                     f"{val:,}\n({val/len(df)*100:.1f}%)", ha="center", fontsize=10)

    # 2. Age distribution by outcome
    df.groupby("y")["age"].plot(kind="kde", ax=axes[1], legend=True)
    axes[1].set(title="Age Distribution by Outcome", xlabel="Age")
    axes[1].legend(["No", "Yes"], title="Subscribed")

    # 3. Euribor3m vs subscription rate
    df["euribor_bin"] = pd.cut(df["euribor3m"], bins=6)
    rate = df.groupby("euribor_bin")["y"].apply(lambda x: (x=="yes").mean() * 100)
    axes[2].bar(range(len(rate)), rate.values, color="#2563EB", alpha=0.8)
    axes[2].set(xticks=range(len(rate)), title="Subscription Rate by Euribor3m",
                xlabel="Euribor3m (binned)", ylabel="Subscription Rate (%)")
    axes[2].set_xticklabels([str(b) for b in rate.index], rotation=30, ha="right", fontsize=8)

    # 4. Contact month heatmap
    month_order = ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"]
    pivot = df.groupby("month")["y"].apply(lambda x: (x=="yes").mean()).reindex(month_order)
    axes[3].bar(pivot.index, pivot.values * 100, color="#DC2626", alpha=0.8)
    axes[3].set(title="Subscription Rate by Month", ylabel="Rate (%)", xlabel="Month")
    plt.setp(axes[3].xaxis.get_majorticklabels(), rotation=45)

    # 5. Duration vs outcome (with leakage warning)
    df_yes = df[df["y"] == "yes"]["duration"]
    df_no = df[df["y"] == "no"]["duration"].sample(len(df_yes)*2, random_state=42)
    axes[4].hist(df_no.clip(0, 1500), bins=40, alpha=0.6, color="#6B7280",
                 label="No (n=sample)", density=True)
    axes[4].hist(df_yes.clip(0, 1500), bins=40, alpha=0.7, color="#F59E0B",
                 label="Yes", density=True)
    axes[4].set(title="⚠️ Duration (DATA LEAKAGE — do not use in prod)",
                xlabel="Call duration (s)", ylabel="Density")
    axes[4].legend(fontsize=9)
    axes[4].set_facecolor("#FFF7ED")

    # 6. poutcome vs subscription
    pout = df.groupby("poutcome")["y"].apply(lambda x: (x=="yes").mean() * 100)
    axes[5].bar(pout.index, pout.values, color="#10B981", alpha=0.85)
    axes[5].set(title="Subscription Rate by Previous Outcome", ylabel="Rate (%)", xlabel="poutcome")

    fig.suptitle("UCI Bank Marketing — Exploratory Data Analysis", fontsize=14, y=1.01)
    plt.tight_layout()
    out = out_dir / "eda_overview.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    df = load()
    plot_eda(df, OUT_DIR)
    print("\nKey stats:")
    print(f"  Rows: {len(df):,}")
    print(f"  Positive rate: {(df['y']=='yes').mean()*100:.1f}%")
    print(f"  Duration corr with y: {df.assign(yn=df['y'].eq('yes').astype(int))['yn'].corr(df['duration']):.3f}")
