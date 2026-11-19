"""Retention Intelligence page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.components.recommendation_cards import render_retention_table
import plotly.graph_objects as go


def render(customers: pd.DataFrame) -> None:
    """Render the retention intelligence page."""
    st.markdown(
        """
        <div style="margin-bottom:1.5rem;">
          <h1 style="margin:0;">Retention Intelligence</h1>
          <p style="color:var(--color-text-secondary);margin:0.25rem 0 0;">
            Prioritize retention outreach using churn probability, CLV and scenario economics.
            All figures are scenario assumptions — not measured causal impact.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---- Scenario controls ----
    with st.expander("Scenario assumptions", expanded=True):
        s1, s2, s3 = st.columns(3)
        with s1:
            offer_cost = st.number_input("Retention offer cost per customer", min_value=0.0, value=20.0, step=5.0)
        with s2:
            save_rate = st.slider("Assumed save-rate scenario", 0.0, 1.0, 0.20, 0.05)
        with s3:
            high_risk_threshold = st.slider("High-risk threshold", 0.0, 1.0, 0.60, 0.05)

    st.markdown(
        '<div class="premium-card" style="margin-bottom:1rem;">'
        '<div style="font-size:0.78rem;text-transform:uppercase;letter-spacing:0.05em;color:var(--color-text-muted);font-weight:600;margin-bottom:0.5rem;">'
        'Scenario assumptions — not causal impact</div>'
        '<div style="color:var(--color-text-secondary);font-size:0.9rem;line-height:1.6;">'
        f'Offer cost: <strong>₹{offer_cost:,.0f}</strong> per customer · '
        f'Assumed save rate: <strong>{save_rate:.0%}</strong> · '
        f'High-risk threshold: <strong>{high_risk_threshold:.0%}</strong> churn probability. '
        'Scenario net value = (expected at-risk value × save rate) − offer cost. '
        'Positive net value suggests the intervention is worth considering under these assumptions.'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ---- Recompute scenario columns ----
    scored = customers.copy()
    scored["expected_at_risk_value"] = scored["churn_probability"] * scored["clv"]
    scored["scenario_retained_value"] = scored["expected_at_risk_value"] * save_rate
    scored["scenario_net_value"] = scored["scenario_retained_value"] - offer_cost

    def action(row):
        risk = row["churn_probability"]
        clv = row["clv"]
        if risk >= high_risk_threshold and clv >= scored["clv"].median():
            return "Personal retention outreach"
        if risk >= high_risk_threshold:
            return "Low-cost retention message"
        if risk >= high_risk_threshold / 2:
            return "Engagement check-in"
        return "No immediate intervention"

    scored["recommended_action"] = scored.apply(action, axis=1)

    # ---- KPI strip ----
    high_val_high_risk = scored[
        (scored["churn_probability"] >= high_risk_threshold) & (scored["clv"] >= scored["clv"].median())
    ]
    total_at_risk = float(scored["expected_at_risk_value"].sum())
    total_retained = float(scored["scenario_retained_value"].sum())
    total_net = float(scored["scenario_net_value"].sum())

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("High-risk & high-value", f"{len(high_val_high_risk):,}", help="Above risk threshold and above median CLV")
    k2.metric("Total at-risk value", f"₹{total_at_risk:,.0f}", help="Sum of probability × CLV")
    k3.metric("Scenario retained value", f"₹{total_retained:,.0f}", help="At-risk value × assumed save rate")
    k4.metric("Scenario net value", f"₹{total_net:,.0f}", help="Retained value − offer cost")

    # ---- Offer cost vs scenario value chart ----
    st.markdown('<div class="section-header"><span class="section-icon">📉</span><h2>Offer cost vs scenario value</h2></div>', unsafe_allow_html=True)
    sweep = pd.DataFrame({"offer_cost": [i * 5 for i in range(0, 21)]})
    sweep["net_value"] = sweep["offer_cost"].apply(
        lambda c: float((scored["expected_at_risk_value"] * save_rate - c).clip(lower=0).sum())
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sweep["offer_cost"], y=sweep["net_value"],
        mode="lines+markers",
        line=dict(color="#3b82f6", width=2),
        marker=dict(size=6),
        name="Scenario net value",
        hovertemplate="Offer cost: ₹%{x:,.0f}<br>Net value: ₹%{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[offer_cost], y=[total_net],
        mode="markers",
        marker=dict(color="#ef4444", size=12, symbol="diamond"),
        name="Current setting",
        hovertemplate="Current: ₹%{y:,.0f}<extra></extra>",
    ))
    fig.update_layout(
        height=320,
        plot_bgcolor="#11182a", paper_bgcolor="#11182a",
        font=dict(color="#f1f5f9", family="Inter, system-ui, sans-serif"),
        title=dict(text="Break-even analysis — net value as offer cost varies", x=0.0, xanchor="left", font=dict(size=14)),
        xaxis=dict(title="Offer cost per customer (₹)", gridcolor="#1f2a4a", tickformat=",.0f"),
        yaxis=dict(title="Scenario net value (₹)", gridcolor="#1f2a4a", tickformat=",.0f"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8", size=11)),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig, width="stretch")

    # ---- Priority table ----
    render_retention_table(scored, offer_cost, save_rate)

    # ---- Download ----
    show_cols = [c for c in ["customerID", "churn_probability", "churn_risk", "clv",
                             "expected_at_risk_value", "scenario_retained_value",
                             "scenario_net_value", "recommended_action"] if c in scored.columns]
    csv = scored[show_cols].sort_values("scenario_net_value", ascending=False).to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download retention recommendation CSV",
        csv,
        file_name="ds01_retention_recommendations.csv",
        mime="text/csv",
        width="stretch",
    )

