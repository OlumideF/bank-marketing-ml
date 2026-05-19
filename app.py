"""
Streamlit dashboard — UCI Bank Marketing ML
Run: python -m streamlit run app.py
     Or double-click preview.bat (Windows)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
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
    OUT_DIR,
    VARIANTS,
    confusion_at_threshold,
    load_artifacts,
    load_data,
    roi_estimates,
    score_customers,
)
from eda import (  # noqa: E402
    CAT_FEATURES,
    NUM_FEATURES,
    plotly_class_balance,
    plotly_corr_heatmap,
    plotly_numeric_hist,
    plotly_scatter_macro,
    plotly_subscription_by,
    plotly_subscription_by_binned,
    summary_statistics,
)

PAGES = [
    "Exploratory Data Analysis",
    "Model Performance",
    "Lift & Business Value",
    "SHAP Explorer",
    "Score New Customers",
]

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


@st.cache_resource(show_spinner="Loading trained models...")
def get_artifacts():
    if not ARTIFACTS_PATH.exists():
        raise FileNotFoundError(
            f"Missing {ARTIFACTS_PATH}. Run: python src/pipeline.py"
        )
    return load_artifacts(retrain_if_missing=False)


@st.cache_data(show_spinner="Loading dataset...")
def get_raw_data():
    return load_data()


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


# ── Sidebar (page first — models load only when needed) ─────────────────────
with st.sidebar:
    st.header("Navigation")
    page = st.radio("Page", PAGES, index=0)

    st.divider()
    if page == "Exploratory Data Analysis":
        st.info("EDA uses raw data only — no model load required.")
        if st.button("Open static EDA images folder"):
            st.caption(str(OUT_DIR.resolve()))
    else:
        model_choice = st.radio(
            "Model",
            ["v2_no_duration", "v1_with_duration"],
            format_func=lambda k: (
                "v2 — Production (no duration)" if k == "v2_no_duration"
                else "v1 — With duration (leaky)"
            ),
        )
        if model_choice == "v1_with_duration":
            st.warning(
                "This model uses call duration — known only AFTER the call. "
                "Benchmark only, not deployable."
            )

    st.divider()
    st.caption("Dashboard URL: http://localhost:8501")

# ── EDA (no ML artifacts) ───────────────────────────────────────────────────
if page == "Exploratory Data Analysis":
    st.subheader("Exploratory Data Analysis")
    raw_df = get_raw_data()
    eda_summary = summary_statistics(raw_df)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Rows", f"{eda_summary['n_rows']:,}")
    m2.metric("Columns", eda_summary["n_columns"])
    m3.metric("Subscribe rate", f"{eda_summary['positive_rate_pct']}%")
    m4.metric("Class ratio", f"{eda_summary['class_ratio']}:1")
    m5.metric("Duration corr.", eda_summary["duration_corr_target"])

    st.warning(
        "Call **duration** correlates with the target (~0.4) but is only known after a call ends."
    )

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Overview", "Numeric", "Categorical", "Correlations", "Campaign", "Summary tables",
    ])

    with tab1:
        c1, c2 = st.columns([1, 2])
        with c1:
            st.plotly_chart(plotly_class_balance(raw_df), width="stretch")
        with c2:
            st.plotly_chart(plotly_subscription_by(raw_df, "month"), width="stretch")
        st.plotly_chart(plotly_subscription_by_binned(raw_df, "euribor3m", bins=10), width="stretch")
        for img_name in ["eda_overview.png", "eda_campaign.png"]:
            img_path = OUT_DIR / img_name
            if img_path.exists():
                st.image(str(img_path), caption=img_name.replace("_", " ").title())

    with tab2:
        num_col = st.selectbox("Numeric feature", NUM_FEATURES, index=0)
        st.plotly_chart(plotly_numeric_hist(raw_df, num_col), width="stretch")
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plotly_scatter_macro(raw_df, "age", "euribor3m"), width="stretch")
        with c2:
            st.plotly_chart(plotly_scatter_macro(raw_df, "campaign", "cons.conf.idx"), width="stretch")
        if (OUT_DIR / "eda_numeric.png").exists():
            st.image(str(OUT_DIR / "eda_numeric.png"))

    with tab3:
        cat_col = st.selectbox("Categorical feature", CAT_FEATURES, index=0)
        top_n = st.slider("Top N categories", 5, 25, 12)
        st.plotly_chart(plotly_subscription_by(raw_df, cat_col, top_n=top_n), width="stretch")
        if (OUT_DIR / "eda_categorical.png").exists():
            st.image(str(OUT_DIR / "eda_categorical.png"))

    with tab4:
        st.plotly_chart(plotly_corr_heatmap(raw_df), width="stretch")
        corr_df = eda_summary["corr_with_target"].reset_index()
        corr_df.columns = ["feature", "correlation"]
        st.dataframe(corr_df, width="stretch", hide_index=True)

    with tab5:
        st.plotly_chart(plotly_subscription_by(raw_df, "poutcome", top_n=None), width="stretch")
        st.plotly_chart(plotly_numeric_hist(raw_df, "campaign"), width="stretch")

    with tab6:
        st.dataframe(eda_summary["numeric_summary"], width="stretch")
        if len(eda_summary["missing"]):
            st.dataframe(
                pd.DataFrame({"count": eda_summary["missing"], "pct": eda_summary["missing_pct"]}),
                width="stretch",
            )
        else:
            st.success("No missing values.")

    st.stop()

# ── ML pages require artifacts ────────────────────────────────────────────────
if not ARTIFACTS_PATH.exists():
    st.error("Trained models not found.")
    st.code("python src/pipeline.py", language="bash")
    st.markdown(
        "Or double-click **`preview.bat`** in the project folder (trains + opens browser)."
    )
    st.stop()

try:
    artifacts = get_artifacts()
except Exception as e:
    st.error(f"Could not load models: {e}")
    st.stop()

results = artifacts["results"]
stats = artifacts["dataset_stats"]
shap_info = artifacts["shap"]

with st.sidebar:
    st.subheader("Dataset stats")
    st.write(f"**Total rows:** {stats['n_rows']:,}")
    st.write(f"**Positive rate:** {stats['positive_rate']*100:.1f}%")

res_sel = results[model_choice]

if page == "Model Performance":
    st.subheader("Model Performance")
    metric_cards(res_sel)

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        for key in results:
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
        st.plotly_chart(fig, width="stretch")

    with c2:
        fig = go.Figure()
        for key in results:
            r = results[key]
            p, rec, _ = precision_recall_curve(r["y_test"], r["proba"])
            c = COLOR_LEAKY if key == "v1_with_duration" else COLOR_PRODUCTION
            fig.add_trace(go.Scatter(x=rec, y=p, mode="lines", name=VARIANTS[key]["label"], line=dict(color=c)))
        fig.update_layout(template="plotly_dark", title="Precision-Recall", height=400)
        st.plotly_chart(fig, width="stretch")

    if "v2_no_duration" in results:
        prod = results["v2_no_duration"]
        threshold = st.slider("Threshold (production model)", 0.3, 0.7, 0.5, 0.05)
        cm = confusion_at_threshold(prod["y_test"], prod["proba"], threshold)
        c3, c4 = st.columns(2)
        with c3:
            fig = go.Figure(data=go.Heatmap(
                z=cm, x=["Pred No", "Pred Yes"], y=["Actual No", "Actual Yes"],
                colorscale="Blues", text=cm, texttemplate="%{text}",
            ))
            fig.update_layout(template="plotly_dark", title=f"Confusion Matrix ({threshold})", height=380)
            st.plotly_chart(fig, width="stretch")
        with c4:
            proba, y = prod["proba"], prod["y_test"]
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=proba[y == 0], name="No", opacity=0.6, marker_color="#6B7280", histnorm="probability density"))
            fig.add_trace(go.Histogram(x=proba[y == 1], name="Yes", opacity=0.7, marker_color=COLOR_PRODUCTION, histnorm="probability density"))
            fig.update_layout(template="plotly_dark", title="Score Distribution", barmode="overlay", height=380)
            st.plotly_chart(fig, width="stretch")

elif page == "Lift & Business Value":
    st.subheader("Lift & Business Value")
    prod = results["v2_no_duration"]

    fig = go.Figure()
    for key in results:
        lcc = results[key]["lift_curve"]
        c = COLOR_LEAKY if key == "v1_with_duration" else COLOR_PRODUCTION
        fig.add_trace(go.Scatter(x=lcc["pct"], y=lcc["lift"], mode="lines", name=VARIANTS[key]["label"], line=dict(color=c)))
    fig.add_vline(x=20, line_dash="dash", line_color="#9CA3AF")
    fig.add_hline(y=1.0, line_dash="dot", line_color="#6B7280")
    fig.update_layout(template="plotly_dark", title="Cumulative Lift", height=420)
    st.plotly_chart(fig, width="stretch")

    n_customers = st.slider("Customers in list", 1000, 50000, 10000, 500)
    pct_call = st.slider("% to call", 5, 50, 20, 1)
    roi = roi_estimates(n_customers, pct_call, prod["y_test"], prod["proba"])
    c1, c2, c3 = st.columns(3)
    c1.metric("Calls placed", f"{roi['n_call']:,}")
    c2.metric("Subscribers (model)", f"{roi['model_subscribers']:.0f}")
    c3.metric("Subscribers (random)", f"{roi['random_subscribers']:.0f}")

elif page == "SHAP Explorer":
    st.subheader("SHAP Explorer")
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(8, 6))
        shap.summary_plot(
            shap_info["values"], shap_info["X_sample"],
            feature_names=shap_info["feature_names"],
            show=False, plot_type="dot", max_display=15,
        )
        st.pyplot(fig, clear_figure=True)
    with c2:
        top = shap_info["importance"].head(15).sort_values("mean_abs_shap", ascending=True)
        fig = go.Figure(go.Bar(x=top["mean_abs_shap"], y=top["feature"], orientation="h", marker_color=COLOR_PRODUCTION))
        fig.update_layout(template="plotly_dark", title="SHAP Importance", height=480)
        st.plotly_chart(fig, width="stretch")
    for item in shap_info["narratives"]:
        arrow = "↑" if item["direction"] == "increases" else "↓"
        st.markdown(f"**{item['feature']}** {arrow} — {item['text']}")

elif page == "Score New Customers":
    st.subheader("Score New Customers")
    uploaded = st.file_uploader("Customer CSV", type=["csv"])
    if uploaded:
        try:
            raw = pd.read_csv(uploaded, sep=";")
        except Exception:
            raw = pd.read_csv(uploaded)
        prod = results["v2_no_duration"]
        ranked = score_customers(raw, prod["clf"], prod["preprocessor"])
        st.success(f"Scored {len(ranked):,} customers.")
        st.plotly_chart(
            go.Figure(go.Histogram(x=ranked["subscription_probability"], marker_color=COLOR_PRODUCTION)),
            width="stretch",
        )
        st.dataframe(ranked.head(10), width="stretch")
        st.download_button("Download ranked CSV", ranked.to_csv(index=False), "ranked_customers.csv", "text/csv")
