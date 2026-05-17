"""
UCI Bank Marketing — Full ML Pipeline

Run:
    python src/pipeline.py
    python src/pipeline.py --no-duration --threshold 0.4 --output-dir outputs
    python src/pipeline.py --cv
"""

import argparse
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import shap
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, precision_recall_curve

import bank_ml
from bank_ml import (
    OUT_DIR,
    VARIANTS,
    cross_validate_models,
    lift_curve_data,
    save_artifacts,
    train_all_variants,
)


def make_plots(results: dict, out_dir: Path, threshold: float):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)
    colors = {"v1_with_duration": "#DC2626", "v2_no_duration": "#2563EB"}

    ax0 = fig.add_subplot(gs[0, 0])
    for key, res in results.items():
        fpr, tpr, _ = roc_curve(res["y_test"], res["proba"])
        ax0.plot(
            fpr, tpr, color=colors.get(key, "#2563EB"),
            label=f"{VARIANTS[key]['label']} (AUC={res['auc']:.3f})", lw=2,
        )
    ax0.plot([0, 1], [0, 1], "--", color="#9CA3AF", lw=1)
    ax0.set(xlabel="FPR", ylabel="TPR", title="ROC Curves")
    ax0.legend(fontsize=8)

    ax1 = fig.add_subplot(gs[0, 1])
    for key, res in results.items():
        p, r, _ = precision_recall_curve(res["y_test"], res["proba"])
        ax1.plot(r, p, color=colors.get(key, "#2563EB"), label=VARIANTS[key]["label"], lw=2)
    ax1.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curves")
    ax1.legend(fontsize=8)

    ax2 = fig.add_subplot(gs[0, 2])
    for key, res in results.items():
        pct, lifts, _ = lift_curve_data(res["y_test"], res["proba"])
        ax2.plot(pct, lifts, color=colors.get(key, "#2563EB"), label=VARIANTS[key]["label"], lw=2)
    ax2.axhline(1.0, linestyle="--", color="#9CA3AF", lw=1, label="Random baseline")
    ax2.set(xlabel="% Customers Called", ylabel="Lift", title="Cumulative Lift Curve")
    ax2.legend(fontsize=8)

    ax3 = fig.add_subplot(gs[1, 0])
    for key, res in results.items():
        pct, _, recalls = lift_curve_data(res["y_test"], res["proba"])
        ax3.plot(
            pct, [r * 100 for r in recalls], color=colors.get(key, "#2563EB"),
            label=VARIANTS[key]["label"], lw=2,
        )
    ax3.plot([0, 100], [0, 100], "--", color="#9CA3AF", lw=1, label="Random")
    ax3.set(xlabel="% Customers Called", ylabel="% Subscribers Captured", title="Recall vs. Effort")
    ax3.legend(fontsize=8)

    prod_key = "v2_no_duration" if "v2_no_duration" in results else list(results.keys())[-1]
    res = results[prod_key]
    ax4 = fig.add_subplot(gs[1, 1])
    cm = confusion_matrix(res["y_test"], (res["proba"] >= threshold).astype(int))
    im = ax4.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax4.text(
                j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=14,
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )
    ax4.set(
        xticks=[0, 1], yticks=[0, 1], xticklabels=["No", "Yes"],
        yticklabels=["No", "Yes"], xlabel="Predicted", ylabel="Actual",
        title=f"Confusion Matrix\n(threshold={threshold})",
    )
    plt.colorbar(im, ax=ax4)

    ax5 = fig.add_subplot(gs[1, 2])
    ax5.hist(res["proba"][res["y_test"] == 0], bins=50, alpha=0.6, color="#6B7280", label="No", density=True)
    ax5.hist(res["proba"][res["y_test"] == 1], bins=50, alpha=0.7, color="#2563EB", label="Yes", density=True)
    ax5.set(xlabel="Predicted Probability", ylabel="Density", title="Score Separation")
    ax5.legend(fontsize=8)

    fig.suptitle("UCI Bank Marketing — Model Evaluation Dashboard", fontsize=14, y=1.01)
    out = out_dir / "evaluation_dashboard.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: {out}")


def make_shap_plots(clf, X_proc, feature_names: list, out_dir: Path):
    print("\n  Computing SHAP values...")
    explainer = shap.TreeExplainer(clf)
    shap_vals = explainer.shap_values(X_proc[:2000])
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    plt.sca(axes[0])
    shap.summary_plot(
        shap_vals, X_proc[:2000], feature_names=feature_names,
        show=False, plot_type="dot", max_display=15,
    )
    axes[0].set_title("SHAP Summary — Feature Impact", fontsize=12)
    plt.sca(axes[1])
    shap.summary_plot(
        shap_vals, X_proc[:2000], feature_names=feature_names,
        show=False, plot_type="bar", max_display=15,
    )
    axes[1].set_title("SHAP Bar — Global Importance", fontsize=12)
    plt.tight_layout()
    out = out_dir / "shap_analysis.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def print_cv_table(cv_df: pd.DataFrame):
    print("\n  Cross-validation comparison (mean +/- std):")
    print("  " + "-" * 58)
    for _, row in cv_df.iterrows():
        tag = " [PRODUCTION]" if row["deployable"] else " [LEAKY]"
        print(
            f"  {row['model']}{tag}\n"
            f"    AUC: {row['auc_mean']:.4f} +/- {row['auc_std']:.4f}\n"
            f"    F1:  {row['f1_mean']:.4f} +/- {row['f1_std']:.4f}"
        )


def parse_args():
    p = argparse.ArgumentParser(description="UCI Bank Marketing ML pipeline")
    p.add_argument("--no-duration", action="store_true", help="Train production model only (v2)")
    p.add_argument("--with-duration-only", action="store_true", help="Train leaky benchmark only (v1)")
    p.add_argument("--threshold", type=float, default=0.5, help="Classification threshold for F1/CM")
    p.add_argument("--output-dir", type=Path, default=OUT_DIR, help="Directory for plots and artifacts")
    p.add_argument("--cv", action="store_true", help="Run 5-fold stratified CV comparison")
    p.add_argument("--skip-report", action="store_true", help="Skip HTML report generation")
    p.add_argument("--skip-plots", action="store_true", help="Skip matplotlib figures")
    p.add_argument("--skip-eda", action="store_true", help="Skip exploratory data analysis")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    bank_ml.OUT_DIR = out_dir
    bank_ml.ARTIFACTS_PATH = out_dir / "artifacts.joblib"

    if args.no_duration and args.with_duration_only:
        raise SystemExit("Choose at most one of --no-duration and --with-duration-only")

    if args.no_duration:
        variant_keys = ["v2_no_duration"]
    elif args.with_duration_only:
        variant_keys = ["v1_with_duration"]
    else:
        variant_keys = list(VARIANTS.keys())

    print("=" * 60)
    print("  UCI Bank Marketing — ML Pipeline")
    print("=" * 60)

    if not getattr(args, "skip_eda", False):
        print("\n[1/8] Exploratory data analysis...")
        from eda import run_eda
        run_eda(out_dir=out_dir, verbose=True)
    else:
        print("\n[1/8] EDA skipped")

    print("\n[2/8] Loading, engineering, and training...")
    artifacts = train_all_variants(
        verbose=True,
        variant_keys=variant_keys,
        threshold=args.threshold,
        run_cv=False,
    )
    results = artifacts["results"]

    if args.cv and len(variant_keys) == len(VARIANTS):
        print("\n[3/8] Cross-validation (5-fold StratifiedKFold)...")
        cv_df = cross_validate_models(n_splits=5, verbose=True)
        artifacts["cv_comparison"] = cv_df
        cv_df.to_csv(out_dir / "cv_comparison.csv", index=False)
        print(f"  Saved: {out_dir / 'cv_comparison.csv'}")
        print_cv_table(cv_df)
    else:
        print("\n[3/8] Cross-validation skipped (use --cv for full comparison)")

    for variant, res in results.items():
        print(f"\n{'-' * 50}")
        print(f"  {variant}")
        print(f"  AUC-ROC : {res['auc']:.4f}")
        print(f"  F1(yes) : {res['f1']:.4f}  (threshold={args.threshold})")
        pred = (res["proba"] >= args.threshold).astype(int)
        print(classification_report(res["y_test"], pred, target_names=["no", "yes"]))
        print(
            f"  Lift @ top 20%: {res['lift_20']:.2f}x  |  "
            f"Recall @ top 20%: {res['recall_20']:.1f}%"
        )

    print("\n[4/8] Saving artifacts...")
    save_artifacts(artifacts)

    if not args.skip_plots:
        print("\n[5/8] Evaluation plots...")
        make_plots(results, out_dir, args.threshold)
        prod_key = "v2_no_duration" if "v2_no_duration" in results else list(results.keys())[-1]
        res = results[prod_key]
        X_te_proc = res["preprocessor"].transform(artifacts["X_test"])
        print("\n[6/8] SHAP plots...")
        make_shap_plots(res["clf"], X_te_proc, res["feature_names"], out_dir)
    else:
        print("\n[5-6/8] Plots skipped")

    print("\n[7/8] Lift curve CSV...")
    prod_key = "v2_no_duration" if "v2_no_duration" in results else list(results.keys())[-1]
    pct, lifts, recalls = lift_curve_data(
        results[prod_key]["y_test"], results[prod_key]["proba"],
    )
    pd.DataFrame({
        "pct_called": pct, "lift": lifts, "recall_pct": [r * 100 for r in recalls],
    }).to_csv(out_dir / "lift_curve_data.csv", index=False)

    print("\n[8/8] Model card + report...")
    from generate_model_card import write_model_card
    card_path = write_model_card(artifacts)
    print(f"  Saved: {card_path}")

    if not args.skip_report:
        from generate_report import generate_report
        report_path = generate_report(artifacts)
        print(f"  Saved: {report_path}")

    print("\n" + "=" * 60)
    print("  Done.")
    print(f"  Artifacts: {out_dir / 'artifacts.joblib'}")
    print("  Score customers: python src/score_customers.py -i data.csv -o ranked.csv")
    print("  Dashboard:       streamlit run app.py")
    print("  Report:          reports/report.html")
    print("=" * 60)


if __name__ == "__main__":
    main()
