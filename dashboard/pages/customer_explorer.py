"""Customer 360 Explorer — search, filter, and customer profiles."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.components.charts import risk_gauge
from dashboard.components.customer_profile import render_customer_profile


def render(customers: pd.DataFrame) -> None:
    """Render the customer explorer page."""
    st.markdown(
        """
        <div style="margin-bottom:1.5rem;">
          <h1 style="margin:0;">Customer 360 Explorer</h1>
          <p style="color:var(--color-text-secondary);margin:0.25rem 0 0;">
            Search, filter, and inspect individual customer profiles with churn risk,
            lifetime value and retention recommendations.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---- Filters ----
    with st.expander("Filters", expanded=True):
        f1, f2, f3 = st.columns(3)
        with f1:
            search_id = st.text_input("Customer ID search", placeholder="e.g. C1001")
            risk_options = ["Low", "Medium", "High"]
            selected_risks = st.multiselect("Churn risk", risk_options, default=risk_options)
        with f2:
            contract_options = sorted(customers["contract"].dropna().astype(str).unique()) if "contract" in customers.columns else []
            selected_contracts = st.multiselect("Contract type", contract_options, default=contract_options) if contract_options else None
            min_clv, max_clv = st.columns(2)
            with min_clv:
                min_clv_v = st.number_input("Min CLV", min_value=0.0, value=0.0, step=100.0)
            with max_clv:
                max_clv_v = st.number_input("Max CLV", min_value=0.0, value=float(customers["clv"].max()), step=100.0)
        with f3:
            if "tenure" in customers.columns and customers["tenure"].notna().any():
                min_t, max_t = st.columns(2)
                with min_t:
                    min_tenure = st.number_input("Min tenure", min_value=0.0, value=0.0, step=1.0)
                with max_t:
                    max_tenure = st.number_input("Max tenure", min_value=0.0, value=float(customers["tenure"].max()), step=1.0)
            else:
                min_tenure = max_tenure = None

    # ---- Apply filters ----
    filtered = customers.copy()
    if search_id:
        filtered = filtered[filtered["customerID"].astype(str).str.contains(search_id, case=False, na=False)]
    if selected_risks:
        filtered = filtered[filtered["churn_risk"].isin(selected_risks)]
    if selected_contracts and "contract" in filtered.columns:
        filtered = filtered[filtered["contract"].isin(selected_contracts)]
    filtered = filtered[(filtered["clv"] >= min_clv_v) & (filtered["clv"] <= max_clv_v)]
    if min_tenure is not None and "tenure" in filtered.columns:
        filtered = filtered[(filtered["tenure"].fillna(0) >= min_tenure) & (filtered["tenure"].fillna(0) <= max_tenure)]

    st.markdown(
        f'<div style="font-size:0.85rem;color:var(--color-text-secondary);margin-bottom:1rem;">{len(filtered):,} of {len(customers):,} customers match</div>',
        unsafe_allow_html=True,
    )

    if filtered.empty:
        st.warning("No customers match the selected filters. Adjust the filters above.")
        return

    # ---- Customer selection ----
    st.markdown('<div class="section-header"><span class="section-icon">🔍</span><h2>Customer profile</h2></div>', unsafe_allow_html=True)
    selected_id = st.selectbox(
        "Select a customer",
        filtered["customerID"].astype(str).tolist(),
        index=0,
    )
    matches = filtered[filtered["customerID"].astype(str) == str(selected_id)]
    if len(matches) != 1:
        st.error("Customer ID must match exactly one customer.")
        return
    selected = matches.iloc[0]

    # ---- Profile: risk gauge + metrics ----
    g_col, p_col = st.columns([1, 1.6])
    with g_col:
        st.plotly_chart(risk_gauge(float(selected["churn_probability"])), width="stretch")
    with p_col:
        render_customer_profile(selected)

    # ---- Similar customers table ----
    st.markdown("#### Similar customers (same risk tier)")
    same_risk = filtered[filtered["churn_risk"] == str(selected.get("churn_risk", "Medium"))]
    same_risk = same_risk[same_risk["customerID"].astype(str) != str(selected_id)].head(6)
    if not same_risk.empty:
        show_cols = [c for c in ["customerID", "churn_probability", "churn_risk", "clv", "expected_at_risk_value", "value_segment"] if c in same_risk.columns]
        st.dataframe(
            same_risk[show_cols],
            width="stretch", hide_index=True,
            column_config={
                "churn_probability": st.column_config.NumberColumn("Churn prob.", format=":.2%"),
                "clv": st.column_config.NumberColumn("CLV (₹)", format="₹%,.0f"),
                "expected_at_risk_value": st.column_config.NumberColumn("At-risk (₹)", format="₹%,.0f"),
            },
        )
    else:
        st.info("No other customers in this risk tier.")
