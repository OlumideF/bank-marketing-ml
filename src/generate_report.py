"""
Generate a self-contained HTML executive report (Plotly + inline CSS).
Run: python src/generate_report.py
"""

from __future__ import annotations

import html
import sys
from pathlib import Path

import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bank_ml import (  # noqa: E402
    COLOR_LEAKY,
    COLOR_PRODUCTION,
    REPORTS_DIR,
    VARIANTS,
    load_artifacts,
    save_artifacts,
    train_all_variants,
)

PLOTLY_TEMPLATE = "plotly_dark"


def _fig_to_div(fig) -> str:
    return fig.to_html(
        full_html=False, include_plotlyjs="cdn", config={"displayModeBar": False},
    )


def _lift_figure(results: dict) -> go.Figure:
    fig = go.Figure()
    colors = {"v1_with_duration": COLOR_LEAKY, "v2_no_duration": COLOR_PRODUCTION}
    for key, res in results.items():
        lc = res["lift_curve"]
        fig.add_trace(go.Scatter(
            x=lc["pct"], y=lc["lift"], mode="lines", name=VARIANTS[key]["label"],
            line=dict(color=colors[key], width=2),
        ))
    prod = results["v2_no_duration"]
    fig.add_vline(x=20, line_dash="dash", line_color="#9CA3AF", annotation_text="20%")
    fig.add_annotation(
        x=20, y=prod["lift_20"],
        text=f"Lift {prod['lift_20']:.2f}x",
        showarrow=True, arrowhead=2, bgcolor="#1F2937",
    )
    fig.add_hline(y=1.0, line_dash="dot", line_color="#6B7280")
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title="Cumulative Lift Curve",
        xaxis_title="% Customers Called",
        yaxis_title="Lift vs Random",
        height=420,
        legend=dict(orientation="h", y=-0.2),
    )
    return fig


def _roc_figure(results: dict) -> go.Figure:
    fig = go.Figure()
    colors = {"v1_with_duration": COLOR_LEAKY, "v2_no_duration": COLOR_PRODUCTION}
    for key, res in results.items():
        roc = res["roc"]
        fig.add_trace(go.Scatter(
            x=roc["fpr"], y=roc["tpr"], mode="lines",
            name=f"{VARIANTS[key]['label']} (AUC={res['auc']:.3f})",
            line=dict(color=colors[key], width=2),
        ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines", name="Random",
        line=dict(dash="dash", color="#6B7280"),
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title="ROC Curves",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=420,
    )
    return fig


def _score_dist_figure(results: dict) -> go.Figure:
    res = results["v2_no_duration"]
    proba, y = res["proba"], res["y_test"]
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=proba[y == 0], name="No (did not subscribe)",
        opacity=0.65, marker_color="#6B7280", histnorm="probability density",
    ))
    fig.add_trace(go.Histogram(
        x=proba[y == 1], name="Yes (subscribed)",
        opacity=0.75, marker_color=COLOR_PRODUCTION, histnorm="probability density",
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title="Score Distribution (Production Model)",
        xaxis_title="Predicted Probability",
        yaxis_title="Density",
        barmode="overlay",
        height=420,
    )
    return fig


def _shap_bar_figure(importance_df) -> go.Figure:
    top = importance_df.head(15).sort_values("mean_abs_shap", ascending=True)
    fig = go.Figure(go.Bar(
        x=top["mean_abs_shap"],
        y=top["feature"],
        orientation="h",
        marker_color=COLOR_PRODUCTION,
    ))
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title="SHAP Global Importance (Top 15)",
        xaxis_title="Mean |SHAP|",
        height=480,
        margin=dict(l=120),
    )
    return fig


def generate_report(artifacts: dict | None = None) -> Path:
    if artifacts is None:
        try:
            artifacts = load_artifacts(retrain_if_missing=False)
        except FileNotFoundError:
            artifacts = train_all_variants(verbose=True)
            save_artifacts(artifacts)

    results = artifacts["results"]
    prod = results["v2_no_duration"]
    shap_info = artifacts["shap"]
    narratives = shap_info["narratives"]

    recall_pct = prod["recall_20"]
    lift_20 = prod["lift_20"]

    narrative = (
        f"Calling the top 20% of customers ranked by this model captures "
        f"{recall_pct:.0f}% of all subscribers — a {lift_20:.1f}x improvement "
        f"over random calling. At a call center processing 10,000 contacts per "
        f"campaign, this translates to reaching the same number of subscribers "
        f"with roughly {lift_20:.0f}x fewer calls."
    )

    v1, v2 = results["v1_with_duration"], results["v2_no_duration"]
    auc_gap = v1["auc"] - v2["auc"]

    comparison_rows = [
        ("AUC-ROC", f"{v1['auc']:.4f}", f"{v2['auc']:.4f}"),
        ("F1 (minority)", f"{v1['f1']:.4f}", f"{v2['f1']:.4f}"),
        ("Lift @ 20%", f"{v1['lift_20']:.2f}x", f"{v2['lift_20']:.2f}x"),
        ("Recall @ 20%", f"{v1['recall_20']:.1f}%", f"{v2['recall_20']:.1f}%"),
    ]

    table_html = (
        '<table class="compare-table"><thead><tr>'
        "<th>Metric</th><th>v1 (with duration)</th><th>v2 (production)</th>"
        "</tr></thead><tbody>"
    )
    for metric, a, b in comparison_rows:
        table_html += f"<tr><td>{metric}</td><td>{a}</td><td>{b}</td></tr>"
    table_html += "</tbody></table>"

    leakage_section = f"""
    <p><strong>What is <code>duration</code>?</strong> Call length in seconds, recorded only after a call ends.</p>
    <p><strong>Why is it leaky?</strong> You cannot know duration before deciding to dial a customer. Using it
    inflates metrics and creates a model that is impossible to deploy for prioritization.</p>
    <p><strong>What the AUC gap reveals:</strong> v1 AUC {v1['auc']:.3f} vs v2 AUC {v2['auc']:.3f}
    (Δ = {auc_gap:.3f}). Much of v1's apparent power comes from post-call information, not pre-call signals.
    v2 still achieves strong ranking (AUC {v2['auc']:.3f}) with only information available at campaign time.</p>
    """

    css = """
    :root {
      --bg: #0f172a; --card: #1e293b; --text: #e2e8f0; --muted: #94a3b8;
      --blue: #2563EB; --red: #DC2626; --green: #16a34a;
    }
    * { box-sizing: border-box; }
    body {
      font-family: 'Segoe UI', system-ui, sans-serif;
      background: var(--bg); color: var(--text);
      margin: 0; padding: 2rem; line-height: 1.6;
    }
    h1, h2 { color: #f8fafc; }
    h2 { border-bottom: 1px solid #334155; padding-bottom: 0.5rem; margin-top: 2.5rem; }
    .subtitle { color: var(--muted); font-size: 1.05rem; }
    .stat-row { display: flex; gap: 1rem; flex-wrap: wrap; margin: 1.5rem 0; }
    .stat-card {
      flex: 1; min-width: 180px; background: var(--card);
      border-radius: 12px; padding: 1.25rem; border: 1px solid #334155;
    }
    .stat-card .value { font-size: 2rem; font-weight: 700; color: var(--blue); }
    .stat-card .label { color: var(--muted); font-size: 0.9rem; }
    .narrative { background: var(--card); padding: 1.25rem; border-radius: 12px; border-left: 4px solid var(--blue); }
    .badge { display: inline-block; padding: 0.25rem 0.6rem; border-radius: 6px; font-size: 0.75rem; font-weight: 600; }
    .badge-red { background: #7f1d1d; color: #fecaca; }
    .badge-green { background: #14532d; color: #bbf7d0; }
    .compare-table { width: 100%; border-collapse: collapse; margin-top: 1rem; }
    .compare-table th, .compare-table td {
      border: 1px solid #334155; padding: 0.75rem; text-align: center;
    }
    .compare-table th { background: #1e293b; }
    .chart-block { background: var(--card); border-radius: 12px; padding: 1rem; margin: 1rem 0; border: 1px solid #334155; }
    .shap-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1rem; }
    .shap-card {
      background: var(--card); border-radius: 12px; padding: 1rem;
      border: 1px solid #334155;
    }
    .shap-card h4 { margin: 0 0 0.5rem; color: #f8fafc; }
    .shap-card .direction { font-size: 0.85rem; color: var(--blue); }
    .shap-card p { color: var(--muted); font-size: 0.95rem; margin: 0.5rem 0 0; }
    .leakage { background: #1c1917; border: 1px solid #44403c; border-radius: 12px; padding: 1.25rem; }
    footer { margin-top: 3rem; text-align: center; color: var(--muted); font-size: 0.85rem; }
    """

    shap_cards_clean = ""
    for item in narratives:
        direction_label = (
            "↑ Increases probability" if item["direction"] == "increases"
            else "↓ Decreases probability"
        )
        shap_cards_clean += f"""
        <div class="shap-card">
          <h4>{html.escape(item['feature'])}</h4>
          <span class="direction">{direction_label}</span>
          <p>{html.escape(item['text'])}</p>
        </div>
        """

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>UCI Bank Marketing — ML Report</title>
  <style>{css}</style>
</head>
<body>
  <h1>UCI Bank Marketing — Executive Report</h1>
  <p class="subtitle">Customer ranking for call-center prioritization (production model: v2, no duration)</p>

  <h2>1. Executive Summary</h2>
  <div class="stat-row">
    <div class="stat-card"><div class="value">{prod['auc']:.3f}</div><div class="label">AUC-ROC</div></div>
    <div class="stat-card"><div class="value">{prod['f1']:.3f}</div><motion-div class="label">F1 (minority class)</motion-div></div>
    <motion-div class="stat-card"><div class="value">{lift_20:.2f}x</motion-div><div class="label">Lift @ top 20%</motion-div></motion-div>
  </motion-div>

  <p class="narrative">{html.escape(narrative)}</p>

  <h2>2. Model Comparison</h2>
  <p>
    <span class="badge badge-red">DATA LEAKAGE — not deployable</span> v1_with_duration &nbsp;
    <span class="badge badge-green">PRODUCTION SAFE</span> v2_no_duration
  </p>
  {table_html}

  <h2>3. Interactive Charts</h2>
  <div class="chart-block">{_fig_to_div(_lift_figure(results))}</div>
  <div class="chart-block">{_fig_to_div(_roc_figure(results))}</div>
  <div class="chart-block">{_fig_to_div(_score_dist_figure(results))}</div>
  <div class="chart-block">{_fig_to_div(_shap_bar_figure(shap_info['importance']))}</div>

  <h2>4. SHAP Economic Interpretation</h2>
  <div class="shap-grid">{shap_cards_clean}</div>

  <h2>5. Data Leakage Documentation</h2>
  <div class="leakage">{leakage_section}</motion-div>

  <footer>Generated by UCI Bank Marketing ML Pipeline</footer>
</body>
</html>
"""
    html_doc = html_doc.replace("motion-div", "div")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "report.html"
    out_path.write_text(html_doc, encoding="utf-8")
    return out_path


if __name__ == "__main__":
    path = generate_report()
    print(f"Report written to {path}")
