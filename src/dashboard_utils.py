from __future__ import annotations

import logging

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

REQUIRED = {"customerID", "churn_probability", "clv"}

# Columns that the retention scenario adds per row. When enrich_recommendations
# concatenates its output onto the scored frame, these must not already exist in
# the frame - otherwise duplicate columns are created. They are listed explicitly
# so the source of any conflict is visible and nothing is silently dropped.
RECOMMENDATION_COLUMNS = (
    "recommended_action",
    "expected_at_risk_value",
    "scenario_retained_value",
    "scenario_net_value",
)


def validate_customer_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Validate scored customer data integrity for the dashboard.

    Raises ValueError (never silently drops data) on:
      * duplicate column names - reported by name so their source (merge /
        concat / enrichment) can be investigated and fixed,
      * missing required columns,
      * blank or duplicate customerID values,
      * non-numeric or out-of-range churn_probability (must lie in [0, 1]),
      * non-numeric or negative clv,
      * expected_at_risk_value present but not a scalar numeric column.

    If expected_at_risk_value is missing it is derived exactly once as
    churn_probability * clv.
    """
    out = df.copy()

    duplicate_columns = out.columns[out.columns.duplicated()].tolist()
    if duplicate_columns:
        raise ValueError(
            "Duplicate column names detected: "
            + ", ".join(map(str, duplicate_columns))
            + ". Investigate the source (merge/concat/enrichment) before "
            "proceeding; duplicate columns were not silently removed."
        )

    missing = REQUIRED - set(out.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    out["customerID"] = out["customerID"].astype(str).str.strip()
    if (out["customerID"] == "").any():
        raise ValueError("customerID cannot be blank")
    if out["customerID"].duplicated().any():
        duplicates = (
            out.loc[out["customerID"].duplicated(keep=False), "customerID"]
            .unique()
            .tolist()
        )
        raise ValueError(
            "customerID values must be unique; duplicates found: "
            + ", ".join(map(str, duplicates[:10]))
        )

    out["churn_probability"] = pd.to_numeric(out["churn_probability"], errors="coerce")
    out["clv"] = pd.to_numeric(out["clv"], errors="coerce")
    if out[["churn_probability", "clv"]].isna().any().any():
        raise ValueError("churn_probability and clv must be numeric and non-null")
    if not out["churn_probability"].between(0, 1).all():
        raise ValueError("churn_probability must be between 0 and 1")
    if (out["clv"] < 0).any():
        raise ValueError("clv cannot be negative")

    if "expected_at_risk_value" in out.columns:
        out["expected_at_risk_value"] = pd.to_numeric(
            out["expected_at_risk_value"], errors="coerce"
        )
        if out["expected_at_risk_value"].isna().any():
            raise ValueError("expected_at_risk_value must be numeric and non-null")
    else:
        out["expected_at_risk_value"] = out["churn_probability"] * out["clv"]

    logger.debug(
        "Validated %d customer records; columns: %s",
        len(out),
        ", ".join(map(str, out.columns)),
    )
    return out


def normalize_customer_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize scored customer data for dashboard use."""
    out = validate_customer_scores(df)

    out["churn_risk"] = pd.cut(
        out["churn_probability"],
        bins=[-0.000001, 0.30, 0.60, 1.000001],
        labels=["Low", "Medium", "High"],
        include_lowest=True,
    ).astype(str)

    if "tenure" in out:
        out["tenure"] = pd.to_numeric(out["tenure"], errors="coerce")
    if "contract" in out:
        out["contract"] = out["contract"].fillna("Unknown").astype(str)
    if "MonthlyCharges" in out:
        out["MonthlyCharges"] = pd.to_numeric(out["MonthlyCharges"], errors="coerce")

    segment_count = min(3, len(out))
    if segment_count >= 1 and len(out) >= 3:
        out["value_segment"] = pd.qcut(
            out["clv"].rank(method="first"),
            q=segment_count,
            labels=["Standard", "Valuable", "Premium"][:segment_count],
            duplicates="drop",
        ).astype(str)
    else:
        out["value_segment"] = "Standard"
    return out

def apply_dashboard_filters(df, risk_levels=None, contracts=None, min_clv=0.0, search=""):
    out = df.copy()
    if risk_levels:
        out = out[out["churn_risk"].isin(risk_levels)]
    if contracts and "contract" in out:
        out = out[out["contract"].isin(contracts)]
    out = out[out["clv"] >= float(min_clv)]
    if search:
        out = out[out["customerID"].str.contains(search, case=False, na=False)]
    return out

def customer_recommendation(row, offer_cost=20.0, assumed_save_rate=0.20):
    """Transparent scenario heuristic, not a measured causal treatment effect."""
    risk = float(row["churn_probability"])
    clv = float(row["clv"])
    expected_at_risk = risk * clv
    scenario_retained_value = expected_at_risk * float(assumed_save_rate)
    net_scenario_value = scenario_retained_value - float(offer_cost)
    if risk >= .60 and clv >= 1000:
        action = "Personal retention outreach"
    elif risk >= .60:
        action = "Low-cost retention message"
    elif risk >= .30:
        action = "Engagement check-in"
    else:
        action = "No immediate intervention"
    return pd.Series({
        "recommended_action": action,
        "expected_at_risk_value": expected_at_risk,
        "scenario_retained_value": scenario_retained_value,
        "scenario_net_value": net_scenario_value
    })

def enrich_recommendations(df, offer_cost=20.0, assumed_save_rate=0.20):
    """Append retention recommendations without creating duplicate columns.

    customer_recommendation re-derives expected_at_risk_value and adds the
    recommended_action / scenario_* columns. Any column recs will add is
    removed from `out` before concatenation: expected_at_risk_value is
    identical in both frames, and the scenario columns are authoritative in
    recs. This avoids duplicate columns and duplicate recommendation
    calculations while keeping every model output column.
    """
    out = df.copy()
    recs = out.apply(
        lambda row: customer_recommendation(row, offer_cost, assumed_save_rate),
        axis=1,
    )
    conflicts = [col for col in RECOMMENDATION_COLUMNS if col in out.columns]
    if conflicts:
        out = out.drop(columns=conflicts)
    return pd.concat([out, recs], axis=1)

def individual_customer_record(customer_id, probability, clv, tenure=None, contract=None):
    record = {"customerID": str(customer_id), "churn_probability": float(probability), "clv": float(clv)}
    if tenure is not None:
        record["tenure"] = float(tenure)
    if contract:
        record["contract"] = contract
    return normalize_customer_data(pd.DataFrame([record]))
