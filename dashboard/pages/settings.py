"""Settings & data management page."""

from __future__ import annotations

import pandas as pd
import streamlit as st



def render(customers: pd.DataFrame, source_label: str) -> None:
    """Render the settings page."""
    st.markdown(
        """
        <div style="margin-bottom:1.5rem;">
          <h1 style="margin:0;">Settings & Data Management</h1>
          <p style="color:var(--color-text-secondary);margin:0.25rem 0 0;">
            Manage data sources, validation status and dashboard preferences. No credentials
            or API keys are stored or displayed here.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---- Data source section ----
    st.markdown('<div class="section-header"><span class="section-icon">🗄️</span><h2>Data source</h2></div>', unsafe_allow_html=True)
    st.info("Use the Data Management card in the sidebar to upload or clear a customer CSV.")
    st.markdown(
        f"""
        <div class="premium-card" style="margin-top:1rem;">
          <div style="font-size:0.8rem;text-transform:uppercase;letter-spacing:0.05em;color:var(--color-text-muted);font-weight:600;margin-bottom:0.5rem;">Current source</div>
          <div style="color:var(--color-text-secondary);font-size:0.9rem;">
            <div><strong>Source:</strong> {source_label or 'Integrated outputs (Parts 1–3)'}</div>
            <div><strong>Records:</strong> {len(customers):,} customers</div>
            <div><strong>Fields:</strong> {len(customers.columns)} columns</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---- Validation status ----
    st.markdown('<div class="section-header"><span class="section-icon">✅</span><h2>Data validation status</h2></div>', unsafe_allow_html=True)
    dup_cols = customers.columns[customers.columns.duplicated()].tolist()
    dup_ids = int(customers["customerID"].duplicated().any()) if "customerID" in customers.columns else 0
    missing_required = {"customerID", "churn_probability", "clv"} - set(customers.columns)
    valid = (not dup_cols and not dup_ids and not missing_required)

    v1, v2, v3, v4 = st.columns(4)
    v1.metric("Duplicate columns", len(dup_cols), delta=None, delta_color="inverse")
    v2.metric("Duplicate customer IDs", dup_ids, delta=None, delta_color="inverse")
    v3.metric("Missing required cols", len(missing_required), delta=None, delta_color="inverse")
    v4.metric("Validation status", "Valid" if valid else "Issues", delta=None)

    if dup_cols:
        st.error(f"Duplicate column names detected: {', '.join(map(str, dup_cols))}")
    if dup_ids:
        st.error("Duplicate customer IDs detected — each customer must appear once.")
    if missing_required:
        st.error(f"Missing required columns: {', '.join(sorted(missing_required))}")
    if valid:
        st.success("Data passes all validation checks.")

    st.divider()

    # ---- Preferences ----
    st.markdown('<div class="section-header"><span class="section-icon">⚙️</span><h2>Dashboard preferences</h2></div>', unsafe_allow_html=True)
    p1, p2 = st.columns(2)
    with p1:
        currency = st.selectbox("Currency display", ["₹ (INR)", "$ (USD)", "€ (EUR)", "£ (GBP)"], index=0)
        st.session_state["currency"] = currency
    with p2:
        theme = st.selectbox("Theme", ["Dark (default)", "Light"], index=0)
        st.session_state["theme"] = theme

    st.markdown(
        '<div class="premium-card" style="margin-top:1rem;">'
        '<div style="font-size:0.78rem;text-transform:uppercase;letter-spacing:0.05em;color:var(--color-text-muted);font-weight:600;margin-bottom:0.5rem;">Risk thresholds</div>'
        '<div style="color:var(--color-text-secondary);font-size:0.9rem;line-height:1.6;">'
        'Low risk: churn probability < 30% · Medium: 30–60% · High: ≥ 60%. '
        'These thresholds are used consistently across churn badges, segmentations and retention prioritization.'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ---- Reset filters ----
    st.divider()
    st.markdown('<div class="section-header"><span class="section-icon">🔄</span><h2>Filter management</h2></div>', unsafe_allow_html=True)
    if st.button("Reset all filters", width="stretch"):
        for key in list(st.session_state.keys()):
            if key.startswith(("filter_", "dashboard_")):
                del st.session_state[key]
        st.success("Filters reset.")
