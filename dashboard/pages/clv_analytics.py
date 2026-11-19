"""CLV Analytics page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.components.kpi_cards import kpi_grid
from dashboard.components.charts import clv_box_by_risk, value_segment_pie, probability_vs_clv_scatter


def render(customers: pd.DataFrame) -> None:
    """Render the CLV analytics page."""
    st.markdown(
        """
        <div style="margin-bottom:1.5rem;">
          <h1 style="margin:0;">CLV Analytics</h1>
          <p style="color:var(--color-text-secondary);margin:0.25rem 0 0;">
            Customer lifetime value estimates based on Kaplan–Meier survival analysis and
            discounted future revenue. These are model-based projections, not historical revenue.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    total_clv = float(customers["clv"].sum())
    avg_clv = float(customers["clv"].mean())
    median_clv = float(customers["clv"].median())
    premium_count = int((customers.get("value_segment", pd.Series(["Standard"] * len(customers))) == "Premium").sum())

    cards = [
        ("Total Estimated CLV", f"₹{total_clv:,.0f}", "Sum of all CLV estimates", "💎", "emerald"),
        ("Average CLV", f"₹{avg_clv:,.0f}", "Mean CLV per customer", "📊", "blue"),
        ("Median CLV", f"₹{median_clv:,.0f}", "Midpoint of CLV distribution", "📐", "cyan"),
        ("Premium Customers", f"{premium_count:,}", f"{premium_count/len(customers):.1%} of base", "👑", "violet"),
    ]
    kpi_grid(cards)

    st.markdown('<div class="section-header"><span class="section-icon">📊</span><h2>Value segmentation</h2></div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="premium-card" style="margin-bottom:1rem;">'
        '<div style="font-size:0.8rem;text-transform:uppercase;letter-spacing:0.05em;color:var(--color-text-muted);font-weight:600;margin-bottom:0.5rem;">Segmentation approach</div>'
        '<div style="color:var(--color-text-secondary);font-size:0.9rem;line-height:1.5;">'
        'Customers are ranked by estimated CLV and split into three equal-sized buckets using a '
        'data-driven quantile approach: <strong>Standard</strong> (lower CLV), '
        '<strong>Valuable</strong> (mid CLV), and <strong>Premium</strong> (top CLV). '
        'Thresholds adapt to the current dataset, so segment boundaries shift as the customer base changes.'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns(2)
    with left:
        st.plotly_chart(value_segment_pie(customers), width="stretch")
    with right:
        st.plotly_chart(clv_box_by_risk(customers), width="stretch")

    st.plotly_chart(probability_vs_clv_scatter(customers), width="stretch")

    # High-value customer identification
    st.markdown('<div class="section-header"><span class="section-icon">👑</span><h2>High-value customer identification</h2></div>', unsafe_allow_html=True)
    top_n = st.slider("Show top N customers by CLV", 5, 50, 10, step=5)
    top_customers = customers.sort_values("clv", ascending=False).head(top_n)
    show_cols = [c for c in ["customerID", "clv", "churn_probability", "churn_risk", "value_segment", "expected_at_risk_value", "contract"] if c in top_customers.columns]
    st.dataframe(
        top_customers[show_cols],
        width="stretch", hide_index=True,
        column_config={
            "clv": st.column_config.NumberColumn("CLV (₹)", format="₹%,.0f"),
            "churn_probability": st.column_config.NumberColumn("Churn prob.", format=":.2%"),
            "expected_at_risk_value": st.column_config.NumberColumn("At-risk (₹)", format="₹%,.0f"),
        },
    )
