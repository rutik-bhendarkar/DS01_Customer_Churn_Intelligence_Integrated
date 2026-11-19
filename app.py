"""DS-01 Customer Intelligence Dashboard — premium SaaS shell (Part 5).

st.navigation-based multi-page app preserving Parts 1–4 model logic and the
Phase 1 data-integrity fixes (no duplicate columns, scalar customer values).
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import FEATURE_DATA_PATH, REPORTS_DIR
from src.dashboard_utils import (
    normalize_customer_data,
    apply_dashboard_filters,
    enrich_recommendations,
)

from dashboard.components.theme_css import inject_theme_css
from dashboard.theme import init_theme, toggle_theme, get_theme_colors
from dashboard.components.sidebar import (
    render_brand_header,
    render_dataset_health,
    render_data_management,
    render_project_profile,
    render_scenario_controls,
)
from dashboard.pages import (
    overview, churn_analytics, clv_analytics, customer_explorer,
    retention_intelligence, model_performance, reports, settings,
)

logger = logging.getLogger(__name__)

@st.cache_data(show_spinner="Loading customer intelligence data...")
def load_customer_data():
    """Join Part 1 features, Part 2 churn scores and Part 3 CLV estimates."""
    feature_path = FEATURE_DATA_PATH
    churn_path = REPORTS_DIR / "churn_predictions.csv"
    clv_path = REPORTS_DIR / "clv_estimates.csv"

    if feature_path.exists() and churn_path.exists() and clv_path.exists():
        features = pd.read_csv(feature_path)
        churn = pd.read_csv(churn_path)
        clv = pd.read_csv(clv_path)
        if "churn_probability" not in churn.columns:
            raise ValueError("Part 2 output must contain churn_probability. Run python -m src.predict_churn.")
        clv_col = "estimated_clv_60m" if "estimated_clv_60m" in clv.columns else "clv"
        if clv_col not in clv.columns:
            raise ValueError("Part 3 output must contain estimated_clv_60m or clv. Run python -m src.clv_model.")
        merged = features.merge(churn[["customerID", "churn_probability"]], on="customerID", how="inner")
        merged = merged.merge(clv[["customerID", clv_col]], on="customerID", how="inner")
        if clv_col != "clv":
            merged = merged.rename(columns={clv_col: "clv"})
        if merged.empty:
            raise ValueError("No matching customerID values across Parts 1, 2 and 3 outputs.")
        return merged, "Integrated outputs from Parts 1–3"

    sample_path = ROOT / "data" / "sample" / "customer_scores_sample.csv"
    if sample_path.exists():
        return pd.read_csv(sample_path), "Illustrative sample data (not model predictions)"
    return None, None



def main() -> None:
    """Run the Streamlit dashboard."""
    st.set_page_config(
        page_title="RevenuePulse AI — Customer Intelligence",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    # Initialise / restore theme (default dark, preserved on reruns)
    init_theme()
    inject_theme_css()
    # Sidebar state, restored across reruns and page navigation.
    st.session_state.setdefault("customers", None)
    st.session_state.setdefault("source_label", None)

    with st.sidebar:
        render_brand_header()
    theme_colors = get_theme_colors()
    is_dark = theme_colors["background"] == "#0b1021"
    _, theme_col = st.columns([0.88, 0.12])
    with theme_col:
        theme_label = "🌙 Dark" if is_dark else "☀️ Light"
        theme_help = "Switch to light mode" if is_dark else "Switch to dark mode"
        if st.button(
            theme_label,
            key="theme_toggle_btn",
            help=theme_help,
        ):
            toggle_theme()



    try:
        if st.session_state["customers"] is None:
            raw, source_label = load_customer_data()
            if raw is None:
                st.error("No scored customer data found. Run Parts 1–3 or upload a scored CSV.")
                st.stop()
            customers = normalize_customer_data(raw)
            st.session_state["customers"] = customers
            st.session_state["source_label"] = source_label
        else:
            customers = st.session_state["customers"]
            source_label = st.session_state["source_label"]

        logger.info("Data source: %s · %d customer records", source_label, len(customers))
        duplicate_columns = customers.columns[customers.columns.duplicated()].tolist()
        if duplicate_columns:
            logger.warning("Duplicate column names detected: %s", duplicate_columns)
            st.error("Data integrity error: duplicate column names detected.")
            st.stop()
    except Exception as exc:
        logger.exception("Failed to load customer data")
        st.error(f"Could not load customer data: {exc}")
        st.stop()

    with st.sidebar:
        render_dataset_health(customers, source_label)
        uploaded, clear_upload = render_data_management()
        if clear_upload:
            st.session_state["upload_widget_epoch"] += 1
            st.session_state["uploaded_signature"] = None
            st.session_state["customers"] = None
            st.session_state["source_label"] = None
            st.rerun()
        upload_signature = (uploaded.name, uploaded.size) if uploaded is not None else None
        if uploaded is not None and upload_signature != st.session_state.get("uploaded_signature"):
            try:
                upload_customers = normalize_customer_data(pd.read_csv(uploaded))
                st.session_state["customers"] = upload_customers
                st.session_state["source_label"] = uploaded.name
                st.session_state["uploaded_signature"] = upload_signature
                st.success(f"Uploaded {len(upload_customers):,} customer records.")
                st.rerun()
            except Exception as exc:
                st.error(f"Invalid uploaded file: {exc}")
        st.divider()
        offer_cost, save_rate = render_scenario_controls()
        st.session_state["offer_cost"] = offer_cost
        st.session_state["save_rate"] = save_rate
        with st.expander("Global filters", expanded=False):
            risk_options = ["Low", "Medium", "High"]
            selected_risks = st.multiselect("Churn risk", risk_options, default=risk_options)
            contract_options = sorted(customers["contract"].dropna().astype(str).unique()) if "contract" in customers.columns else []
            selected_contracts = st.multiselect("Contract", contract_options, default=contract_options) if contract_options else None
            min_clv = st.number_input("Minimum CLV", min_value=0.0, value=0.0, step=100.0)
            search = st.text_input("Search customer ID")
        render_project_profile()

    filtered = apply_dashboard_filters(customers, selected_risks, selected_contracts, min_clv, search)
    st.session_state["filtered"] = filtered
    scored = enrich_recommendations(filtered, offer_cost, save_rate) if not filtered.empty else filtered
    st.session_state["scored"] = scored

    # Page registry — st.navigation multi-page app
    pages = [
        st.Page(lambda: overview.render(st.session_state["customers"]), title="Overview", icon="🏠", url_path="overview", default=True),
        st.Page(lambda: churn_analytics.render(st.session_state["customers"]), title="Churn Analytics", icon="📉", url_path="churn-analytics"),
        st.Page(lambda: clv_analytics.render(st.session_state["customers"]), title="CLV Analytics", icon="💎", url_path="clv-analytics"),
        st.Page(lambda: customer_explorer.render(st.session_state["customers"]), title="Customer 360 Explorer", icon="🔍", url_path="customer-explorer"),
        st.Page(lambda: retention_intelligence.render(st.session_state["customers"]), title="Retention Intelligence", icon="🛡️", url_path="retention-intelligence"),
        st.Page(lambda: model_performance.render(st.session_state["customers"]), title="Model Performance", icon="📊", url_path="model-performance"),
        st.Page(lambda: reports.render(st.session_state["customers"], st.session_state["scored"]), title="Reports & Downloads", icon="⬇️", url_path="reports"),
        st.Page(lambda: settings.render(st.session_state["customers"], st.session_state["source_label"]), title="Settings", icon="⚙️", url_path="settings"),
    ]

    # Native Streamlit navigation remains the only navigation system; groups
    # provide the requested visual hierarchy without changing page routes.
    pg = st.navigation({
        "Workspace": pages[:5],
        "Intelligence": pages[5:7],
        "Configuration": pages[7:],
    })
    pg.run()


if __name__ == "__main__":
    main()
