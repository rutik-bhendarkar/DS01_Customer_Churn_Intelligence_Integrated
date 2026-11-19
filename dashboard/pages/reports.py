"""Reports & Downloads page."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import REPORTS_DIR


def _button(label, data, file_name, mime):
    st.download_button(label, data, file_name=file_name, mime=mime, width="stretch")


def render(customers: pd.DataFrame, scored: pd.DataFrame) -> None:
    """Render the reports & downloads page."""
    st.markdown(
        """
        <div style="margin-bottom:1.5rem;">
          <h1 style="margin:0;">Reports & Downloads</h1>
          <p style="color:var(--color-text-secondary);margin:0.25rem 0 0;">
            Export filtered results, retention recommendations and KPI summaries.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    show_cols = [c for c in ["customerID", "churn_probability", "churn_risk", "clv",
                             "value_segment", "expected_at_risk_value", "recommended_action",
                             "scenario_net_value", "tenure", "contract"] if c in scored.columns]

    st.markdown('<div class="section-header"><span class="section-icon">📥</span><h2>Customer intelligence exports</h2></div>', unsafe_allow_html=True)
    _button("Download filtered customer intelligence (CSV)", scored[show_cols].to_csv(index=False).encode("utf-8"), "ds01_customer_intelligence_results.csv", "text/csv")
    st.caption(f"{len(scored):,} rows × {len(show_cols)} columns.")

    st.divider()
    st.markdown('<div class="section-header"><span class="section-icon">🚨</span><h2>High-risk customers</h2></div>', unsafe_allow_html=True)
    high_risk = scored[scored["churn_risk"] == "High"].copy()
    if high_risk.empty:
        st.info("No high-risk customers in the current filter set.")
        _button("Download high-risk customer CSV", pd.DataFrame(columns=show_cols).to_csv(index=False).encode("utf-8"), "ds01_high_risk_customers.csv", "text/csv")
    else:
        hr_cols = [c for c in ["customerID", "churn_probability", "churn_risk", "clv", "expected_at_risk_value", "recommended_action", "contract", "tenure"] if c in high_risk.columns]
        st.caption(f"{len(high_risk):,} high-risk customers.")
        _button("Download high-risk customer CSV", high_risk[hr_cols].sort_values("churn_probability", ascending=False).to_csv(index=False).encode("utf-8"), "ds01_high_risk_customers.csv", "text/csv")

    st.divider()
    st.markdown('<div class="section-header"><span class="section-icon">💎</span><h2>CLV segmentation</h2></div>', unsafe_allow_html=True)
    seg_cols = [c for c in ["customerID", "clv", "value_segment", "churn_probability", "churn_risk"] if c in scored.columns]
    _button("Download CLV segmentation CSV", scored[seg_cols].sort_values("clv", ascending=False).to_csv(index=False).encode("utf-8"), "ds01_clv_segmentation.csv", "text/csv")
    st.caption("All customers with CLV value segment.")

    st.divider()
    st.markdown('<div class="section-header"><span class="section-icon">🛡️</span><h2>Retention recommendations</h2></div>', unsafe_allow_html=True)
    ret_cols = [c for c in ["customerID", "churn_probability", "churn_risk", "clv", "expected_at_risk_value", "scenario_retained_value", "scenario_net_value", "recommended_action"] if c in scored.columns]
    _button("Download retention recommendation CSV", scored[ret_cols].sort_values("scenario_net_value", ascending=False).to_csv(index=False).encode("utf-8"), "ds01_retention_recommendations.csv", "text/csv")
    st.caption("Retention priority ranking with scenario economics.")

    st.divider()
    st.markdown('<div class="section-header"><span class="section-icon">📊</span><h2>Business KPI summary</h2></div>', unsafe_allow_html=True)
    kpi = pd.DataFrame([{
        "total_customers": len(scored),
        "high_risk_customers": int((scored["churn_risk"] == "High").sum()),
        "total_estimated_clv": float(scored["clv"].sum()),
        "expected_at_risk_value": float(scored["expected_at_risk_value"].sum()),
        "average_churn_probability": float(scored["churn_probability"].mean()),
    }])
    _button("Download business KPI summary (CSV)", kpi.to_csv(index=False).encode("utf-8"), "ds01_business_kpi_summary.csv", "text/csv")

    st.divider()
    st.markdown('<div class="section-header"><span class="section-icon">📋</span><h2>Model evaluation report</h2></div>', unsafe_allow_html=True)
    metrics_path = REPORTS_DIR / "churn_test_metrics.json"
    importance_path = REPORTS_DIR / "churn_feature_importance.csv"
    if metrics_path.exists():
        st.caption("Saved Part 2 evaluation artifacts found.")
        c1, c2 = st.columns(2)
        with c1:
            _button("Download churn test metrics (JSON)", metrics_path.read_bytes(), "churn_test_metrics.json", "application/json")
        with c2:
            if importance_path.exists():
                _button("Download feature importance (CSV)", importance_path.read_bytes(), "churn_feature_importance.csv", "text/csv")
    else:
        st.info("No model evaluation report on disk. Train the Part 2 model to generate metrics.")
