"""Retention recommendation cards and priority table for the Retention Intelligence page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.components.charts import RISK_COLORS


def risk_badge_html(risk: str) -> str:
    cls = risk.lower()
    return f'<span class="risk-badge {cls}"><span class="badge-dot"></span>{risk}</span>'


def render_retention_table(df: pd.DataFrame, offer_cost: float, save_rate: float) -> pd.DataFrame:
    """Render the retention priority table and return the ranked frame."""
    cols = ["customerID", "churn_probability", "churn_risk", "clv",
            "expected_at_risk_value", "scenario_retained_value", "scenario_net_value",
            "recommended_action"]
    present = [c for c in cols if c in df.columns]
    show = df[present].copy()
    show = show.sort_values("scenario_net_value", ascending=False).reset_index(drop=True)

    st.markdown("#### Retention priority ranking")
    st.caption("Ranked by scenario net value (retained value − offer cost). Positive values suggest a worthwhile intervention.")

    for _, row in show.iterrows():
        risk = str(row.get("churn_risk", "Medium"))
        p = float(row.get("churn_probability", 0))
        clv = float(row.get("clv", 0))
        earv = float(row.get("expected_at_risk_value", p * clv))
        retained = float(row.get("scenario_retained_value", earv * save_rate))
        net = float(row.get("scenario_net_value", retained - offer_cost))
        action = str(row.get("recommended_action", "No immediate intervention"))
        cid = str(row.get("customerID", "—"))

        net_cls = "text-success" if net >= 0 else ""
        st.markdown(
            f"""
            <div class="premium-card" style="padding:0.875rem 1.125rem;margin-bottom:0.625rem;border-left:4px solid var(--accent-blue);">
              <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:0.5rem;">
                <div style="min-width:140px;">
                  <div style="font-size:0.875rem;font-weight:700;color:var(--color-text-primary);">{cid}</div>
                  <div style="font-size:0.78rem;color:var(--color-text-muted);">{risk_badge_html(risk)}</div>
                </div>
                <div style="flex:1;display:flex;gap:1rem;flex-wrap:wrap;">
                  <div style="font-size:0.8rem;text-transform:uppercase;letter-spacing:0.04em;color:var(--color-text-muted);">Churn prob.</div>
                  <div style="font-size:0.95rem;font-weight:600;color:var(--color-text-primary);">{p:.1%}</div>
                  <div style="font-size:0.8rem;text-transform:uppercase;letter-spacing:0.04em;color:var(--color-text-muted);">Est. CLV</div>
                  <div style="font-size:0.95rem;font-weight:600;color:var(--color-text-primary);">₹{clv:,.0f}</div>
                  <div style="font-size:0.8rem;text-transform:uppercase;letter-spacing:0.04em;color:var(--color-text-muted);">At-risk value</div>
                  <div style="font-size:0.95rem;font-weight:600;color:var(--color-text-primary);">₹{earv:,.0f}</div>
                </div>
                <div style="text-align:right;">
                  <div style="font-size:0.8rem;text-transform:uppercase;letter-spacing:0.04em;color:var(--color-text-muted);">Net scenario value</div>
                  <div style="font-size:1rem;font-weight:700;color:{'#10b981' if net >= 0 else '#f59e0b'};">₹{net:,.0f}</div>
                  <div style="font-size:0.78rem;color:var(--color-text-muted);">{action}</div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    return show
