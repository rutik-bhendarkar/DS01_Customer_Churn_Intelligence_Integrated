"""Model Performance page — reports metrics from actual saved evaluation artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import REPORTS_DIR


def _load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_csv(path: Path):
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def render(customers: pd.DataFrame) -> None:
    """Render the model performance page."""
    st.markdown(
        """
        <div style="margin-bottom:1.5rem;">
          <h1 style="margin:0;">Model Performance</h1>
          <p style="color:var(--color-text-secondary);margin:0.25rem 0 0;">
            Evaluation metrics from the saved Parts 2 & 3 pipeline artifacts. Metrics appear only
            when the corresponding model has been trained and results exist on disk.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metrics_path = REPORTS_DIR / "churn_test_metrics.json"
    importance_path = REPORTS_DIR / "churn_feature_importance.csv"
    metadata_path = Path("models") / "churn_model_metadata.json"
    clv_report = REPORTS_DIR / "survival_summary.json"

    metrics = _load_json(metrics_path)
    importance = _load_csv(importance_path)
    metadata = _load_json(metadata_path)
    clv_summary = _load_json(clv_report)

    has_churn = metrics is not None

    # ---- Model information section ----
    st.markdown('<div class="section-header"><span class="section-icon">🤖</span><h2>Model information</h2></div>', unsafe_allow_html=True)
    info_col1, info_col2 = st.columns(2)
    with info_col1:
        st.markdown(
            f"""
            <div class="premium-card">
              <div style="font-size:0.8rem;text-transform:uppercase;letter-spacing:0.05em;color:var(--color-text-muted);font-weight:600;margin-bottom:0.5rem;">Churn model</div>
              <div style="color:var(--color-text-secondary);font-size:0.9rem;">
                <div><strong>Type:</strong> Logistic Regression / LightGBM pipeline</div>
                <div><strong>Task:</strong> Binary churn classification</div>
                <div><strong>Status:</strong> {'Trained artifacts found' if metadata else 'No trained model on disk'}</div>
                <div><strong>Source:</strong> Part 2 — <code>src/churn_model.py</code></div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with info_col2:
        st.markdown(
            f"""
            <div class="premium-card">
              <div style="font-size:0.8rem;text-transform:uppercase;letter-spacing:0.05em;color:var(--color-text-muted);font-weight:600;margin-bottom:0.5rem;">CLV / survival model</div>
              <div style="color:var(--color-text-secondary);font-size:0.9rem;">
                <div><strong>Method:</strong> Kaplan–Meier conditional survival</div>
                <div><strong>Horizon:</strong> 60 months, discounted</div>
                <div><strong>Status:</strong> {'Estimates found' if clv_summary else 'No CLV estimates on disk'}</div>
                <div><strong>Source:</strong> Part 3 — <code>src/clv_model.py</code></div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ---- Churn metrics ----
    if has_churn:
        st.markdown('<div class="section-header"><span class="section-icon">📈</span><h2>Churn model test metrics (held-out set)</h2></div>', unsafe_allow_html=True)
        m = metrics.get("test_metrics_at_selected_threshold", {})
        default_m = metrics.get("test_metrics_at_0_5", {})
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Precision", f"{m.get('precision', 0):.3f}")
        c2.metric("Recall", f"{m.get('recall', 0):.3f}")
        c3.metric("F1-score", f"{m.get('f1', 0):.3f}")
        c4.metric("PR-AUC", f"{m.get('pr_auc', 0):.3f}")
        c5, c6, c7, c8 = st.columns(4)
        c5.metric("ROC-AUC", f"{m.get('roc_auc', 0):.3f}")
        c6.metric("Balanced accuracy", f"{m.get('balanced_accuracy', 0):.3f}")
        c7.metric("Threshold", f"{metrics.get('selected_threshold', 0):.2f}")
        c8.metric("Default threshold", "0.50")

        # Confusion matrix
        cm = m.get("confusion_matrix")
        if cm:
            st.markdown('<div class="section-header"><span class="section-icon">🧮</span><h2>Confusion matrix</h2></div>', unsafe_allow_html=True)
            import plotly.graph_objects as go
            fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=["Pred: No", "Pred: Yes"],
                y=["Actual: No", "Actual: Yes"],
                colorscale=[[0, "#0b1021"], [1, "#3b82f6"]],
                showscale=False,
                text=cm,
                texttemplate="%{text}",
                hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
            ))
            fig.update_layout(
                height=300,
                plot_bgcolor="#11182a", paper_bgcolor="#11182a",
                font=dict(color="#f1f5f9"),
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis=dict(gridcolor="#1f2a4a"),
                yaxis=dict(gridcolor="#1f2a4a"),
            )
            st.plotly_chart(fig, width="stretch")

        st.caption(
            f"Selected model: <strong>{metrics.get('selected_model','—')}</strong> · "
            f"Validation PR-AUC used for candidate selection · "
            f"Test set not used for selection or threshold tuning.",
            unsafe_allow_html=True,
        )

        if importance is not None and not importance.empty:
            st.markdown('<div class="section-header"><span class="section-icon">🔍</span><h2>Top feature importance</h2></div>', unsafe_allow_html=True)
            top = importance.head(12).iloc[::-1]
            import plotly.graph_objects as go2
            fig = go2.Figure(go2.Bar(
                x=top["importance"], y=top["feature"], orientation="h",
                marker=dict(color="#8b5cf6"),
                hovertemplate="%{y}: %{x:.4f}<extra></extra>",
            ))
            fig.update_layout(
                height=400,
                plot_bgcolor="#11182a", paper_bgcolor="#11182a",
                font=dict(color="#f1f5f9"),
                xaxis=dict(title="Importance", gridcolor="#1f2a4a"),
                yaxis=dict(title="Feature", gridcolor="#1f2a4a", automargin=True),
                margin=dict(l=10, r=10, t=10, b=10),
            )
            st.plotly_chart(fig, width="stretch")

    else:
        st.info("No churn model evaluation metrics found on disk. Run `python -m src.train_churn_model` and `python -m src.predict_churn` to generate Part 2 artifacts, then refresh this page.")

    # ---- CLV summary ----
    st.markdown('<div class="section-header"><span class="section-icon">💎</span><h2>CLV / survival summary</h2></div>', unsafe_allow_html=True)
    if clv_summary:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Customers", f"{clv_summary.get('customers', 0):,}")
        c2.metric("Median remaining months", f"{clv_summary.get('median_expected_remaining_months', 0):.1f}")
        c3.metric("Total estimated CLV", f"₹{clv_summary.get('total_estimated_clv', 0):,.0f}")
        c4.metric("Horizon", f"{clv_summary.get('horizon_months', 60)} months")
        st.caption(clv_summary.get("limitations", ""), )
    else:
        st.info("No CLV estimates found. Run `python -m src.clv_model` to generate Part 3 artifacts.")

