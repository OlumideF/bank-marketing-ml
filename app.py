"""
Streamlit dashboard — UCI Bank Marketing ML
Run: streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st
from sklearn.metrics import precision_recall_curve, roc_curve

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from bank_ml import (  # noqa: E402
    ARTIFACTS_PATH,
    COLOR_LEAKY,
    COLOR_PRODUCTION,
    VARIANTS,
    confusion_at_threshold,
    load_artifacts,
    roi_estimates,
    score_customers,
)

st.set_page_config(
    page_title="Bank Marketing ML",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background-color: #0f172a; color: #e2e8f0; }
    .metric-card {
        background: #1e293b; border-radius: 10px; padding: 1rem;
        border: 1px solid #334155; text-align: center;
    }
    .metric-card h3 { margin: 0; color: #94a3b8; font-size: 0.85rem; }
    .metric-card p { margin: 0.25rem 0 0; font-size: 1.6rem; font-weight: 700; }
    .header-title { font-size: 1.8rem; font-weight: 700; color: #f8fafc; }
    .header-sub { color: #94a3b8; margin-bottom: 1.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<p class="header-title">UCI Bank Marketing — Call Center Optimizer</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="header-sub">Rank customers by subscription probability — call the top 20% first, not everyone.</p>',
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading models (run pipeline if first time)...")
def get_artifacts():
    return load_artifacts(retrain_if_missing=True)


def metric_cards(res):
    cols = st.columns(3)
    metrics = [
        ("AUC-ROC", f"{res['auc']:.3f}", COLOR_PRODUCTION),
        ("F1 (minority)", f"{res['f1']:.3f}", COLOR_PRODUCTION),
        ("Lift @ 20%", f"{res['lift_20']:.2f}x", COLOR_PRODUCTION),
    ]
    for col, (label, val, color) in zip(cols, metrics):
        col.markdown(
            f'<div class="metric-card"><h3>{label}</h3>'
            f'<p style="color:{color}">{val}</p></div>',
            unsafe_allow_html=True,
        )


artifacts = get_artifacts()
results = artifacts["results"]
stats = artifacts["dataset_stats"]
shap_info = artifacts["shap"]

with st.sidebar:
    st.header("Controls")
    model_choice = st.radio(
        "Model",
        ["v2_no_duration", "v1_with_duration"],
        format_func=lambda k: "v2 — Production (no duration)" if k == "v2_no_duration" else "v1 — With duration (leaky)",
    )
    if model_choice == "v1_with_duration":
        st.warning(
            "⚠️ This model uses call duration — a feature only known AFTER the call. "
            "It cannot be deployed. Shown here for benchmarking only."
        )

    page = st.radio(
        "Page",
        [
            "Model Performance",
            "Lift & Business Value",
            "SHAP Explorer",
            "Score New Customers",
        ],
    )

    st.divider()
    st.subheader("Dataset stats")
    st.write(f"**Total rows:** {stats['n_rows']:,}")
    st.write(f"**Positive rate:** {stats['positive_rate']*100:.1f}%")
    st.write(f"**Train size:** {stats['train_size']:,}")
    st.write(f"**Test size:** {stats['test_size']:,}")

    if not ARTIFACTS_PATH.exists():
        st.info("Run `python src/pipeline.py` to refresh artifacts.")

res_sel = results[model_choice]
color_sel = COLOR_PRODUCTION if model_choice == "v2_no_duration" else COLOR_LEAKY

# ── Page 1: Model Performance ──────────────────────────────────────────────
if page == "Model Performance":
    st.subheader("Model Performance")
    metric_cards(res_sel)

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        for key in ["v1_with_duration", "v2_no_duration"]:
            r = results[key]
            fpr, tpr, _ = roc_curve(r["y_test"], r["proba"])
            c = COLOR_LEAKY if key == "v1_with_duration" else COLOR_PRODUCTION
            fig.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines",
                name=f"{VARIANTS[key]['label']} (AUC={r['auc']:.3f})",
                line=dict(color=c, width=2),
            ))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="#6B7280")))
        fig.update_layout(template="plotly_dark", title="AUC-ROC", height=400)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = go.Figure()
        for key in ["v1_with_duration", "v2_no_duration"]:
            r = results[key]
            p, rec, _ = precision_recall_curve(r["y_test"], r["proba"])
            c = COLOR_LEAKY if key == "v1_with_duration" else COLOR_PRODUCTION
            fig.add_trace(go.Scatter(x=rec, y=p, mode="lines", name=VARIANTS[key]["label"], line=dict(color=c)))
        fig.update_layout(template="plotly_dark", title="Precision-Recall", height=400)
        st.plotly_chart(fig, use_container_width=True)

    prod = results["v2_no_duration"]
    threshold = st.slider("Classification threshold (production model)", 0.3, 0.7, 0.5, 0.05)
    cm = confusion_at_threshold(prod["y_test"], prod["proba"], threshold)

    c3, c4 = st.columns(2)
    with c3:
        fig = go.Figure(data=go.Heatmap(
            z=cm, x=["Pred No", "Pred Yes"], y=["Actual No", "Actual Yes"],
            colorscale="Blues", text=cm, texttemplate="%{text:,}",
        ))
        fig.update_layout(template="plotly_dark", title=f"Confusion Matrix (threshold={threshold})", height=380)
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        proba, y = prod["proba"], prod["y_test"]
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=proba[y == 0], name="No", opacity=0.6, marker_color="#6B7280", histnorm="probability density"))
        fig.add_trace(go.Histogram(x=proba[y == 1], name="Yes", opacity=0.7, marker_color=COLOR_PRODUCTION, histnorm="probability density"))
        fig.update_layout(template="plotly_dark", title="Score Distribution (production)", barmode="overlay", height=380)
        st.plotly_chart(fig, use_container_width=True)

# ── Page 2: Lift & Business Value ────────────────────────────────────────────
elif page == "Lift & Business Value":
    st.subheader("Lift & Business Value")
    prod = results["v2_no_duration"]
    lc = prod["lift_curve"]

    fig = go.Figure()
    for key in ["v1_with_duration", "v2_no_duration"]:
        r = results[key]
        lcc = r["lift_curve"]
        c = COLOR_LEAKY if key == "v1_with_duration" else COLOR_PRODUCTION
        fig.add_trace(go.Scatter(x=lcc["pct"], y=lcc["lift"], mode="lines", name=VARIANTS[key]["label"], line=dict(color=c)))
    fig.add_vline(x=20, line_dash="dash", line_color="#9CA3AF", annotation_text="20%")
    fig.add_hline(y=1.0, line_dash="dot", line_color="#6B7280")
    fig.update_layout(template="plotly_dark", title="Cumulative Lift Curve", xaxis_title="% Called", yaxis_title="Lift", height=420)
    st.plotly_chart(fig, use_container_width=True)

    fig2 = go.Figure()
    for key in ["v1_with_duration", "v2_no_duration"]:
        r = results[key]
        lcc = r["lift_curve"]
        c = COLOR_LEAKY if key == "v1_with_duration" else COLOR_PRODUCTION
        fig2.add_trace(go.Scatter(
            x=lcc["pct"], y=[x * 100 for x in lcc["recall"]], mode="lines",
            name=VARIANTS[key]["label"], line=dict(color=c),
        ))
    fig2.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode="lines", line=dict(dash="dash", color="#6B7280"), name="Random"))
    fig2.update_layout(template="plotly_dark", title="Recall vs Effort", xaxis_title="% Called", yaxis_title="% Subscribers Captured", height=420)
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### ROI Calculator")
    n_customers = st.slider("Customers in your list", 1000, 50000, 10000, 500)
    pct_call = st.slider("% of list you can afford to call", 5, 50, 20, 1)

    roi = roi_estimates(n_customers, pct_call, prod["y_test"], prod["proba"])
    r1, r2, r3 = st.columns(3)
    r1.metric("Calls placed", f"{roi['n_call']:,}")
    r2.metric("Est. subscribers (model ranking)", f"{roi['model_subscribers']:.0f}")
    r3.metric("Est. subscribers (random order)", f"{roi['random_subscribers']:.0f}")
    st.success(
        f"At **{pct_call}%** outreach on **{n_customers:,}** customers, model-ranked calling "
        f"captures ~**{roi['recall_pct']:.0f}%** of all subscribers vs ~**{pct_call}%** with random dialing. "
        f"Estimated **{roi['calls_saved']:,}** calls saved to reach the same subscriber count."
    )

# ── Page 3: SHAP Explorer ──────────────────────────────────────────────────────
elif page == "SHAP Explorer":
    st.subheader("SHAP Explorer (production model)")
    importance = shap_info["importance"]
    shap_vals = shap_info["values"]
    X_sample = shap_info["X_sample"]
    feat_names = shap_info["feature_names"]

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(8, 6), facecolor="#0f172a")
        ax.set_facecolor("#0f172a")
        plt.sca(ax)
        shap.summary_plot(
            shap_vals, X_sample, feature_names=feat_names,
            show=False, plot_type="dot", max_display=15,
        )
        ax.set_title("SHAP Beeswarm (Top 15)", color="#e2e8f0")
        st.pyplot(fig, clear_figure=True)

    with c2:
        top = importance.head(15).sort_values("mean_abs_shap", ascending=True)
        fig = go.Figure(go.Bar(
            x=top["mean_abs_shap"], y=top["feature"], orientation="h",
            marker_color=COLOR_PRODUCTION,
        ))
        fig.update_layout(template="plotly_dark", title="Global SHAP Importance", height=480, margin=dict(l=120))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Plain-English: Top 5 Features")
    for item in shap_info["narratives"]:
        arrow = "↑" if item["direction"] == "increases" else "↓"
        st.markdown(f"**{item['feature']}** {arrow} — {item['text']}")

# ── Page 4: Score New Customers ──────────────────────────────────────────────
else:
    st.subheader("Score New Customers")
    st.caption("Upload a CSV with the UCI schema (semicolon-separated). Target column `y` is optional.")

    uploaded = st.file_uploader("Customer CSV", type=["csv"])
    if uploaded is not None:
        try:
            raw = pd.read_csv(uploaded, sep=";")
        except Exception:
            raw = pd.read_csv(uploaded)

        prod = results["v2_no_duration"]
        ranked = score_customers(raw, prod["clf"], prod["preprocessor"])

        st.success(f"Scored **{len(ranked):,}** customers with production model (v2).")

        fig = go.Figure(go.Histogram(x=ranked["subscription_probability"], nbinsx=40, marker_color=COLOR_PRODUCTION))
        fig.update_layout(template="plotly_dark", title="Score Distribution (uploaded batch)", height=350)
        st.plotly_chart(fig, use_container_width=True)

        display_cols = [c for c in ranked.columns if c != "subscription_probability"][:6]
        display_cols.append("subscription_probability")
        st.markdown("**Top 10 highest-probability customers**")
        st.dataframe(
            ranked[display_cols].head(10).style.format({"subscription_probability": "{:.4f}"}),
            use_container_width=True,
        )

        csv_out = ranked.to_csv(index=False)
        st.download_button(
            "Download ranked customer list as CSV",
            csv_out,
            file_name="ranked_customers.csv",
            mime="text/csv",
        )
