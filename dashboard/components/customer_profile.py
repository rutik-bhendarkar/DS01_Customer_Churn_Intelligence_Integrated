"""Customer 360 profile card and risk gauge for the Customer Explorer page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.components.charts import risk_gauge, RISK_COLORS


def risk_badge_html(risk: str) -> str:
    """Return HTML for a risk badge."""
    cls = risk.lower()
    return (
        f'<span class="risk-badge {cls}">'
        f'<span class="badge-dot"></span>{risk}</span>'
    )


def render_customer_profile(selected: pd.Series) -> None:
    """Render a premium customer profile panel."""
    churn_probability = float(selected["churn_probability"])
    clv = float(selected["clv"])
    expected_at_risk = churn_probability * clv
    risk = str(selected.get("churn_risk", "Medium"))
    segment = str(selected.get("value_segment", "Standard"))
    customer_id = str(selected.get("customerID", "—"))

    # Header row: ID + risk badge + segment
    st.markdown(
        f"""
        <div class="premium-card">
          <div class="premium-card-header">
            <span style="font-size:1.5rem;">👤</span>
            <div>
              <div style="font-size:1.125rem;font-weight:700;color:var(--color-text-primary);">{customer_id}</div>
              <div style="font-size:0.8rem;color:var(--color-text-muted);">Customer profile · {segment} segment</div>
            </div>
            <div style="margin-left:auto;">{risk_badge_html(risk)}</div>
          </div>
        """,
        unsafe_allow_html=True,
    )

    # KPI row (4 metrics)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Churn probability", f"{churn_probability:.1%}")
    c2.metric("Estimated CLV", f"₹{clv:,.2f}")
    c3.metric("Expected at-risk value", f"₹{expected_at_risk:,.2f}")
    c4.metric("Risk level", risk)

    # Secondary info row
    info_rows = []
    if "tenure" in selected.index and pd.notna(selected.get("tenure")):
        info_rows.append(("Tenure", f"{float(selected['tenure']):.0f} months"))
    if "contract" in selected.index and pd.notna(selected.get("contract")):
        info_rows.append(("Contract", str(selected["contract"])))
    if "MonthlyCharges" in selected.index and pd.notna(selected.get("MonthlyCharges")):
        info_rows.append(("Monthly charges", f"₹{float(selected['MonthlyCharges']):,.2f}"))
    if "recommended_action" in selected.index:
        info_rows.append(("Recommended action", str(selected["recommended_action"])))

    st.markdown("#### Customer details")
    for i in range(0, len(info_rows), 2):
        cols = st.columns(2)
        for j, (k, v) in enumerate(info_rows[i:i+2]):
            with cols[j]:
                st.markdown(
                    f'<div style="background:var(--color-bg-card);border:1px solid #1f2a4a;border-radius:var(--radius-md);padding:0.75rem 1rem;margin-bottom:0.5rem;">'
                    f'<div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.05em;color:var(--color-text-muted);font-weight:700;">{k}</div>'
                    f'<div style="font-size:1rem;color:var(--color-text-primary);">{v}</div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )

    # Recommendation card
    st.markdown("#### Retention recommendation")
    action = str(selected.get("recommended_action", "No immediate intervention"))
    st.markdown(
        f"""
        <div class="premium-card" style="border-left:4px solid var(--accent-blue);margin-top:0.75rem;">
          <div style="font-size:0.75rem;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;color:var(--color-text-muted);margin-bottom:0.5rem;">Why this action</div>
          <div style="color:var(--color-text-secondary);font-size:0.95rem;line-height:1.5;">
            <strong>{action}</strong> — based on a churn probability of
            <strong>{churn_probability:.1%}</strong> and an estimated CLV of
            <strong>₹{clv:,.2f}</strong>. Expected at-risk value is
            <strong>₹{expected_at_risk:,.2f}</strong> (probability × CLV).
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)
