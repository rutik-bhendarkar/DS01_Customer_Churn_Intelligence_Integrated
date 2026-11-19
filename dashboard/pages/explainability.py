"""Explainable Churn Intelligence page (DS-01 advanced feature).

SHAP-based explanations of the trained DS-01 churn model. Reuses the exact
trained artifact and preprocessing pipeline from src.churn_model; never trains
a model and never fabricates explanations.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src.explainability import (
    ExplainabilityError,
    data_version_key,
    explain_customer_cached,
    explain_global_model_cached,
    get_customer_explanation_summary,
    get_top_churn_drivers,
    load_explainability_model_cached,
    register_background,
)
from dashboard.components.charts import RISK_COLORS, risk_gauge
from dashboard.components.shap_charts import (
    beeswarm_chart,
    contribution_bar_chart,
    global_importance_bar,
    high_vs_low_risk_chart,
    waterfall_chart,
)


def render(customers: pd.DataFrame) -> None:
    """Render the Explainable Churn Intelligence page."""
    st.markdown(
        """
        <div style="margin-bottom:1.5rem;">
          <h1 style="margin:0;">Explainable Churn Intelligence</h1>
          <p style="color:var(--color-text-secondary);margin:0.25rem 0 0;">
            Why the model predicts an individual customer is likely to churn, using SHAP
            values computed on the trained Part 2 pipeline and its exact preprocessing.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---- Load the trained model (existing artifact only) ----
    try:
        package = load_explainability_model_cached()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.code("python -m src.train_churn_model", language="bash")
        return

    feature_columns = package["feature_columns"]
    missing = [c for c in feature_columns if c not in customers.columns]
    if missing:
        st.warning(
            "Explainability requires the full Part 1 feature schema. "
            f"Missing columns: {', '.join(missing)}. Regenerate Parts 1-3 outputs "
            "or upload a dataset with the full feature schema."
        )
        return

    data_version = data_version_key(customers)
    background = customers.dropna(subset=feature_columns).copy()
    if background.empty:
        st.error("No complete customer rows available to explain.")
        return
    register_background(data_version, background)

    # ---- Customer selector ----
    options = background["customerID"].astype(str).tolist()
    default_id = (
        str(background.loc[background["churn_probability"].idxmax(), "customerID"])
        if "churn_probability" in background.columns
        else options[0]
    )
    default_index = options.index(default_id) if default_id in options else 0
    selected_id = st.selectbox(
        "Select a customer to explain",
        options,
        index=default_index,
        help="Defaults to the highest-churn-probability customer in the current dataset.",
    )

    # ---- Individual explanation ----
    try:
        explanation = explain_customer_cached(selected_id, package["model_version"], data_version)
    except ExplainabilityError as exc:
        st.error(str(exc))
        return

    st.markdown(
        '<div class="section-header"><span class="section-icon">👤</span>'
        f"<h2>Customer {explanation['customer_id']}</h2></div>",
        unsafe_allow_html=True,
    )

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Churn probability", f"{explanation['churn_probability']:.1%}")
    k2.metric("Predicted class", "Churn" if explanation["predicted_class"] == 1 else "Stay",
              help=f"Decision threshold: {explanation['decision_threshold']:.2f}")
    k3.metric("Risk category", explanation["risk_category"],
              help="Bands: Low < 0.30, Medium 0.30-0.60, High >= 0.60")
    k4.metric("SHAP value space", explanation["value_space"].replace("_", " ").title(),
              help="Units of the SHAP contributions shown below")
    # ---- Summary ----
    st.markdown("#### Explanation summary")
    for line in get_customer_explanation_summary(explanation):
        st.markdown(line)

    # ---- Waterfall + contributors ----
    col_left, col_right = st.columns([1.15, 1])
    with col_left:
        st.plotly_chart(waterfall_chart(explanation), width="stretch")
    with col_right:
        st.plotly_chart(contribution_bar_chart(explanation), width="stretch")

    # ---- Positive / negative drivers tables ----
    st.markdown("#### Churn drivers for this customer")
    t1, t2 = st.columns(2)
    with t1:
        st.markdown("**Top factors increasing churn risk** (red)")
        if explanation["top_positive"]:
            rows = []
            for c in explanation["top_positive"][:8]:
                rows.append(
                    {
                        "Feature": c["feature"],
                        "Actual value": _fmt_value(c["actual_value"]),
                        "SHAP contribution": f"{c['contribution']:+.3f}",
                    }
                )
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        else:
            st.caption("None")
    with t2:
        st.markdown("**Top factors reducing churn risk** (green)")
        if explanation["top_negative"]:
            rows = []
            for c in explanation["top_negative"][:8]:
                rows.append(
                    {
                        "Feature": c["feature"],
                        "Actual value": _fmt_value(c["actual_value"]),
                        "SHAP contribution": f"{c['contribution']:+.3f}",
                    }
                )
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        else:
            st.caption("None")

    st.caption(
        "Actual feature values are taken from the customer's record. SHAP contributions are "
        f"in {explanation['value_space'].replace('_', ' ')} space — they explain the model output; "
        "they are not percentage-point probability changes unless the value space is probability."
    )

    # ---- Actual feature values ----
    with st.expander("Actual feature values used by the model", expanded=False):
        value_rows = [
            {"Feature": k, "Value": _fmt_value(v)} for k, v in explanation["actual_values"].items()
        ]
        st.dataframe(pd.DataFrame(value_rows), width="stretch", hide_index=True)

    with k5:
        st.plotly_chart(risk_gauge(explanation["churn_probability"]), width="stretch")

    # ---- Global model explanation ----
    st.markdown("#### Global model explanation")
    st.caption(
        "Aggregate SHAP impact across a background sample of customers. "
        "Feature influence is associational (how the model uses features), not causal evidence."
    )
    try:
        global_explanation = explain_global_model_cached(
            package["model_version"], data_version, min(200, len(background))
        )
    except ExplainabilityError as exc:
        st.error(str(exc))
        global_explanation = None

    if global_explanation:
        g1, g2 = st.columns(2)
        with g1:
            st.plotly_chart(global_importance_bar(global_explanation), width="stretch")
        with g2:
            st.plotly_chart(beeswarm_chart(global_explanation), width="stretch")
        st.plotly_chart(high_vs_low_risk_chart(global_explanation), width="stretch")

        st.markdown("##### Most influential features")
        drivers = get_top_churn_drivers(global_explanation, top_k=10)
        st.dataframe(drivers, width="stretch", hide_index=True)

    # ---- Suggested retention actions (rule-based, explanation-grounded) ----
    st.markdown("#### Suggested retention actions")
    st.info(
        "The actions below are **rule-based suggestions grounded in the actual SHAP contributors**; "
        "they are not model-generated causal recommendations."
    )
    for action in _retention_actions(explanation):
        st.markdown(f"- {action}")


def _fmt_value(value) -> str:
    import numpy as np

    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "n/a"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.2f}"
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return f"{int(value)}"
    return str(value)


def _retention_actions(explanation: dict) -> list[str]:
    """Rule-based retention suggestions grounded in this customer's SHAP contributors."""
    actions: list[str] = []
    p = explanation["churn_probability"]
    names = [c["feature"] for c in explanation["top_positive"][:4]]
    if p >= 0.60:
        actions.append("Prioritise for personal retention outreach (high churn risk).")
    elif p >= 0.30:
        actions.append("Schedule an engagement check-in (medium churn risk).")
    else:
        actions.append("No immediate intervention indicated by risk level.")
    if any("is_month_to_month" in n or n == "Contract" for n in names):
        actions.append(
            "Contract type (month-to-month) increases risk: consider offering an annual commitment with a discount."
        )
    if any(n == "MonthlyCharges" for n in names):
        actions.append(
            "Higher monthly charges increase risk: review pricing or propose a downgrade/loyalty credit."
        )
    if any(n in ("tenure", "tenure_group") for n in names):
        actions.append(
            "Short tenure is a risk factor: assign a dedicated onboarding/success touchpoint."
        )
    if any("is_electronic_check" in n for n in names):
        actions.append(
            "Electronic-check payment increases risk: incentivise autopay or bank-transfer billing."
        )
    if not any(
        key in " ".join(names)
        for key in ("Contract", "MonthlyCharges", "tenure", "is_electronic_check", "is_month_to_month")
    ):
        actions.append("Review the customer's top SHAP contributors for targeted engagement ideas.")
    actions.append("Measure outcomes after any intervention; assumed save rates must be validated empirically.")
    return actions
