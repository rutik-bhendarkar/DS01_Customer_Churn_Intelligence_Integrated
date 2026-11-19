"""Overview / executive home page."""

from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from dashboard.components.kpi_cards import kpi_grid
from dashboard.components.charts import risk_bar_chart, probability_histogram, clv_box_by_risk, value_segment_pie


def render(customers: pd.DataFrame) -> None:
    """Render the executive overview page."""
    st.markdown(
        """
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:1.5rem;">
          <div>
            <h1 style="margin:0;">Customer Intelligence Platform</h1>
            <p style="color:var(--color-text-secondary);margin:0.25rem 0 0;">
              Understand customer behavior, predict churn and maximize customer lifetime value.
            </p>
          </div>
          <div style="text-align:right;">
            <div style="font-size:0.875rem;color:var(--color-text-muted);">Reporting period</div>
            <div style="font-size:1rem;font-weight:600;color:var(--color-text-primary);">{today}</div>
          </div>
        </div>
        """.format(today=date.today().strftime("%B %d, %Y")),
        unsafe_allow_html=True,
    )

    # ---- KPI cards ----
    total_customers = len(customers)
    high_risk = int((customers["churn_risk"] == "High").sum())
    total_clv = float(customers["clv"].sum())
    at_risk = float(customers["expected_at_risk_value"].sum())
    avg_prob = float(customers["churn_probability"].mean())

    cards = [
        ("Total Customers", f"{total_customers:,}", "Active customer records", "👥", "blue"),
        ("High-Risk Customers", f"{high_risk:,}", f"{high_risk/total_customers:.1%} of base", "⚠️", "rose"),
        ("Total Estimated CLV", f"₹{total_clv:,.0f}", "Sum of CLV estimates", "💎", "emerald"),
        ("Expected At-Risk Value", f"₹{at_risk:,.0f}", "Probability × CLV", "📉", "violet"),
        ("Average Churn Probability", f"{avg_prob:.1%}", "Mean risk across base", "📊", "cyan"),
    ]
    if "tenure" in customers.columns and customers["tenure"].notna().any():
        avg_tenure = float(customers["tenure"].mean())
        cards.append(("Average Tenure", f"{avg_tenure:.0f} months", "Mean customer tenure", "⏱️", "blue"))

    kpi_grid(cards)

    st.markdown('<div class="section-header"><span class="section-icon">📈</span><h2>Risk & value distribution</h2></div>', unsafe_allow_html=True)

    left, right = st.columns(2)
    with left:
        st.plotly_chart(risk_bar_chart(customers), width="stretch")
    with right:
        st.plotly_chart(value_segment_pie(customers), width="stretch")

    left2, right2 = st.columns(2)
    with left2:
        st.plotly_chart(probability_histogram(customers), width="stretch")
    with right2:
        st.plotly_chart(clv_box_by_risk(customers), width="stretch")
