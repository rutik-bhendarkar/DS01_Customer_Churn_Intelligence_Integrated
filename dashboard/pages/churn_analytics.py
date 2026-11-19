"""Churn Analytics page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.components.charts import (
    risk_bar_chart,
    probability_histogram,
    clv_box_by_risk,
    probability_vs_clv_scatter,
    RISK_COLORS,
    RISK_ORDER,
)


def render(customers: pd.DataFrame) -> None:
    """Render the churn analytics page."""
    st.markdown(
        """
        <div style="margin-bottom:1.5rem;">
          <h1 style="margin:0;">Churn Analytics</h1>
          <p style="color:var(--color-text-secondary);margin:0.25rem 0 0;">
            Understand churn risk across segments and identify customers most likely to leave.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Top row: risk distribution + probability histogram
    left, right = st.columns(2)
    with left:
        st.plotly_chart(risk_bar_chart(customers), width="stretch")
    with right:
        st.plotly_chart(probability_histogram(customers), width="stretch")

    # Second row: CLV by risk + probability vs CLV
    left2, right2 = st.columns(2)
    with left2:
        st.plotly_chart(clv_box_by_risk(customers), width="stretch")
    with right2:
        st.plotly_chart(probability_vs_clv_scatter(customers), width="stretch")

    # Churn by contract type (if available)
    if "contract" in customers.columns:
        st.markdown('<div class="section-header"><span class="section-icon">📄</span><h2>Churn by contract type</h2></div>', unsafe_allow_html=True)
        contract_stats = (
            customers.groupby("contract")
            .agg(customers=("customerID", "count"),
                 high_risk=("churn_risk", lambda s: int((s == "High").sum())),
                 avg_prob=("churn_probability", "mean"))
            .reset_index()
        )
        contract_stats["high_risk_pct"] = contract_stats["high_risk"] / contract_stats["customers"] * 100
        st.dataframe(
            contract_stats.sort_values("avg_prob", ascending=False),
            width="stretch", hide_index=True,
            column_config={
                "customers": st.column_config.NumberColumn("Customers", format="%d"),
                "high_risk": st.column_config.NumberColumn("High-risk", format="%d"),
                "high_risk_pct": st.column_config.NumberColumn("High-risk %", format="%.1f%%"),
                "avg_prob": st.column_config.NumberColumn("Avg churn prob.", format=":.2%"),
            },
        )

    # Churn by tenure (if available)
    if "tenure" in customers.columns:
        st.markdown('<div class="section-header"><span class="section-icon">⏱️</span><h2>Churn by tenure</h2></div>', unsafe_allow_html=True)
        tenure_bins = [0, 12, 24, 48, 72, float("inf")]
        tenure_labels = ["0–12 mo", "13–24 mo", "25–48 mo", "49–72 mo", "73+ mo"]
        customers_copy = customers.copy()
        customers_copy["tenure_group"] = pd.cut(customers_copy["tenure"], bins=tenure_bins, labels=tenure_labels, right=False)
        tenure_stats = (
            customers_copy.groupby("tenure_group", observed=True)
            .agg(customers=("customerID", "count"),
                 avg_prob=("churn_probability", "mean"))
            .reset_index()
        )
        st.dataframe(
            tenure_stats,
            width="stretch", hide_index=True,
            column_config={"avg_prob": st.column_config.NumberColumn("Avg churn prob.", format=":.2%")},
        )

    # High-risk customer list
    st.markdown('<div class="section-header"><span class="section-icon">🚨</span><h2>High-risk customer list</h2></div>', unsafe_allow_html=True)
    high_risk_df = customers[customers["churn_risk"] == "High"].copy()
    if high_risk_df.empty:
        st.info("No high-risk customers in the current filter set.")
    else:
        show_cols = [c for c in ["customerID", "churn_probability", "churn_risk", "clv", "expected_at_risk_value", "contract", "tenure"] if c in high_risk_df.columns]
        st.dataframe(
            high_risk_df[show_cols].sort_values("churn_probability", ascending=False),
            width="stretch", hide_index=True,
            column_config={
                "churn_probability": st.column_config.NumberColumn("Churn prob.", format=":.2%"),
                "clv": st.column_config.NumberColumn("CLV (₹)", format="₹%,.0f"),
                "expected_at_risk_value": st.column_config.NumberColumn("At-risk (₹)", format="₹%,.0f"),
            },
        )
        csv = high_risk_df[show_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download high-risk customer CSV",
            csv,
            file_name="ds01_high_risk_customers.csv",
            mime="text/csv",
            width="stretch",
        )
