"""Premium sidebar components for the RevenuePulse AI dashboard."""

from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st


def render_brand_header() -> None:
    """Render the compact RevenuePulse AI brand header."""
    environment = "LIVE" if st.runtime.exists() else "LOCAL"
    st.markdown(
        f"""
        <section class="rp-brand" aria-label="RevenuePulse AI">
          <div class="rp-brand__mark" aria-hidden="true"><svg viewBox="0 0 32 32" fill="none"><path d="M7 22.5V9.5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2v13a2 2 0 0 1-2 2H9a2 2 0 0 1-2-2Z" stroke="currentColor" stroke-width="2"/><path d="m10.5 19 3.5-4 3.2 2.8 4.3-5" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg></div>
          <div class="rp-brand__copy"><div class="rp-brand__title">RevenuePulse AI</div><div class="rp-brand__subtitle">Customer Intelligence Platform</div></div>
          <span class="rp-environment">{environment}</span>
        </section>
        """,
        unsafe_allow_html=True,
    )


render_sidebar_brand = render_brand_header


def _source_details(source_label: str | None) -> tuple[str, str]:
    label = source_label or ""
    lower = label.lower()
    if "illustrative sample" in lower:
        return "Illustrative sample data", "Sample dataset"
    if "integrated" in lower or "parts 1" in lower:
        return "Integrated model outputs", "Parts 1–3 outputs"
    return "Uploaded customer data", Path(label).name or "Customer CSV"


def _dataset_status(customers: pd.DataFrame) -> dict[str, int | float | str]:
    """Calculate display-only health metrics from the active dataset."""
    rows, columns = customers.shape
    duplicate_columns = int(customers.columns.duplicated().sum())
    duplicate_records = int(customers["customerID"].duplicated(keep=False).sum()) if "customerID" in customers.columns else 0
    missing_values = int(customers.isna().sum().sum())
    missing_required = len({"customerID", "churn_probability", "clv"} - set(customers.columns))
    cell_count = rows * columns
    quality = 100.0 if cell_count == 0 else 100 * (1 - missing_values / cell_count)
    if duplicate_columns or duplicate_records or missing_required:
        state, label = "invalid", "Invalid dataset"
    elif missing_values:
        state, label = "warning", "Validation warning"
    else:
        state, label = "valid", "Validated dataset"
    return {"rows": rows, "columns": columns, "duplicates": duplicate_records, "missing": missing_values, "quality": quality, "state": state, "label": label}


def render_dataset_health(customers: pd.DataFrame, source_label: str | None) -> None:
    """Render data-backed dataset validation and provenance information."""
    health = _dataset_status(customers)
    source_type, source_name = _source_details(source_label)
    st.markdown(
        f"""
        <section class="rp-sidebar-card rp-health-card">
          <div class="rp-card-heading"><span>Dataset health</span><span class="rp-status rp-status--{health['state']}"><i></i>{health['label']}</span></div>
          <div class="rp-health-grid"><div><b>{health['rows']:,}</b><span>Customers</span></div><div><b>{health['columns']}</b><span>Columns</span></div><div><b>{health['duplicates']:,}</b><span>Duplicate records</span></div><div><b>{health['missing']:,}</b><span>Missing values</span></div></div>
          <div class="rp-quality"><div><span>Data quality</span><b>{health['quality']:.1f}%</b></div><div class="rp-quality__track"><i style="width:{health['quality']:.1f}%"></i></div></div>
          <div class="rp-source"><span>Current source</span><b>{escape(source_type)}</b><small>{escape(source_name)}</small></div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_data_source_status(customers: pd.DataFrame, source_label: str | None) -> None:
    """Compatibility wrapper for the previous sidebar API."""
    render_dataset_health(customers, source_label)


def render_data_management() -> tuple[object | None, bool]:
    """Render the validated CSV upload control and return upload/reset intents."""
    epoch = st.session_state.setdefault("upload_widget_epoch", 0)
    st.markdown('<section class="rp-sidebar-card rp-upload-card"><div class="rp-card-heading"><span>Data management</span><span class="rp-format">CSV</span></div><p>Upload and validate your customer dataset.</p></section>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload customer dataset", type=["csv"], key=f"customer_upload_{epoch}", label_visibility="collapsed", help="Required columns: customerID, churn_probability (0–1), and clv (non-negative).")
    if uploaded is not None:
        st.caption(f"{uploaded.name} · {uploaded.size / 1024:.1f} KB")
    reset = st.button("Clear uploaded data", key="clear_customer_upload", width="stretch")
    return uploaded, reset


def render_upload_section():
    """Backward-compatible upload accessor for legacy callers."""
    uploaded, _ = render_data_management()
    return uploaded


def render_project_profile() -> None:
    """Render the compact project identity at the bottom of sidebar content."""
    st.markdown('<section class="rp-project-profile"><span class="rp-project-profile__avatar">DS</span><span><b>DS-01</b><small>Customer Intelligence Project<br>Data Science Track</small></span></section>', unsafe_allow_html=True)


def render_scenario_controls() -> tuple[float, float]:
    """Retention scenario assumptions, unchanged from the original workflow."""
    with st.expander("Retention scenario", expanded=False):
        offer_cost = st.number_input("Offer cost per customer", min_value=0.0, value=20.0, step=5.0)
        save_rate = st.slider("Assumed save-rate", 0.0, 1.0, 0.20, 0.05)
        st.caption("Scenario assumptions, not measured causal impact.")
    return offer_cost, save_rate
