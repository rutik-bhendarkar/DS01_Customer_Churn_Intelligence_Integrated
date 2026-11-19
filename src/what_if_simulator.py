"""Assumption-driven what-if retention simulation for the DS-01 platform.

All calculations are transparent, vectorised Pandas/NumPy operations driven by
explicit user assumptions. These are scenario projections, not proven causal
effects. Every assumption is labelled as such in the outputs.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

DEFAULT_HIGH_RISK_THRESHOLD = 0.60
DEFAULT_SAVE_RATE = 0.20
DEFAULT_OFFER_COST = 20.0
DEFAULT_CHURN_PROB_REDUCTION = 0.10

SEGMENT_ALL_CUSTOMERS = "All customers"
SEGMENT_ALL_HIGH_RISK = "All high-risk customers"
SEGMENT_HIGH_RISK_HIGH_VALUE = "High-risk and high-value customers"
SEGMENT_PREMIUM = "Premium customers"
SEGMENT_SPECIFIC_CUSTOMERS = "Specifically selected customers"

SEGMENTS = [
    SEGMENT_ALL_HIGH_RISK,
    SEGMENT_HIGH_RISK_HIGH_VALUE,
    SEGMENT_PREMIUM,
    SEGMENT_SPECIFIC_CUSTOMERS,
    SEGMENT_ALL_CUSTOMERS,
]

SAVE_RATE_MIN = 0.0
SAVE_RATE_MAX = 1.0
CHURN_REDUCTION_MIN = 0.0
CHURN_REDUCTION_MAX = 1.0


class SimulationError(ValueError):
    """Raised when scenario assumptions are invalid or a calculation is not supported."""


def _validate_customer_frame(customers: pd.DataFrame) -> pd.DataFrame:
    required = {"customerID", "churn_probability", "clv"}
    missing = required - set(customers.columns)
    if missing:
        raise SimulationError(f"Missing required columns: {', '.join(sorted(missing))}")
    out = customers.copy()
    out["customerID"] = out["customerID"].astype(str).str.strip()
    if (out["customerID"] == "").any():
        raise SimulationError("customerID cannot be blank")
    if out["customerID"].duplicated().any():
        dupes = out.loc[out["customerID"].duplicated(keep=False), "customerID"].unique().tolist()
        raise SimulationError(
            "customerID values must be unique; duplicates found: " + ", ".join(map(str, dupes[:10]))
        )
    out["churn_probability"] = pd.to_numeric(out["churn_probability"], errors="coerce")
    out["clv"] = pd.to_numeric(out["clv"], errors="coerce")
    if out[["churn_probability", "clv"]].isna().any().any():
        raise SimulationError("churn_probability and clv must contain numeric, non-null values")
    if not out["churn_probability"].between(0, 1).all():
        raise SimulationError("churn_probability must be between 0 and 1")
    if (out["clv"] < 0).any():
        raise SimulationError("clv cannot be negative")
    return out


def _validate_save_rate(save_rate: float) -> float:
    save_rate = float(save_rate)
    if not SAVE_RATE_MIN <= save_rate <= SAVE_RATE_MAX:
        raise SimulationError(f"save_rate must be between {SAVE_RATE_MIN} and {SAVE_RATE_MAX}")
    return save_rate


def _validate_offer_cost(offer_cost: float) -> float:
    offer_cost = float(offer_cost)
    if offer_cost < 0:
        raise SimulationError("offer_cost cannot be negative")
    return offer_cost

def _segment_mask(
    customers: pd.DataFrame,
    segment: str,
    high_risk_threshold: float,
    selected_customer_ids: list[str] | None = None,
) -> pd.Series:
    """Boolean mask selecting the customers targeted by a scenario."""
    prob = customers["churn_probability"]
    clv = customers["clv"]
    if segment == SEGMENT_ALL_CUSTOMERS:
        return pd.Series(True, index=customers.index)
    if segment == SEGMENT_ALL_HIGH_RISK:
        return prob >= high_risk_threshold
    if segment == SEGMENT_HIGH_RISK_HIGH_VALUE:
        return (prob >= high_risk_threshold) & (clv >= clv.median())
    if segment == SEGMENT_PREMIUM:
        if "value_segment" in customers.columns:
            return customers["value_segment"].astype(str).str.casefold().eq("premium")
        return pd.Series(False, index=customers.index)
    if segment == SEGMENT_SPECIFIC_CUSTOMERS:
        selected = {str(cid).strip() for cid in (selected_customer_ids or []) if str(cid).strip()}
        if not selected:
            raise SimulationError("At least one customer must be selected for the 'Specifically selected customers' segment")
        return customers["customerID"].isin(selected)
    raise SimulationError(f"Unknown segment: {segment}")


def calculate_baseline_metrics(
    customers: pd.DataFrame,
    high_risk_threshold: float = DEFAULT_HIGH_RISK_THRESHOLD,
) -> dict[str, Any]:
    """Baseline business performance from actual customer records.

    No historical baseline values are assumed; everything is computed from the
    provided scored customer frame.
    """
    if customers is None or len(customers) == 0:
        return {
            "total_customers": 0,
            "high_risk_count": 0,
            "high_risk_share": 0.0,
            "baseline_churn_exposure": 0.0,
            "total_estimated_clv": 0.0,
            "average_churn_probability": 0.0,
            "average_clv": 0.0,
            "high_risk_threshold": float(high_risk_threshold),
            "current_retention_assumptions": None,
            "limitation": "Empty dataset: no baseline metrics can be calculated.",
        }
    out = _validate_customer_frame(customers)
    threshold = float(high_risk_threshold)
    if not 0 <= threshold <= 1:
        raise SimulationError("high_risk_threshold must be between 0 and 1")
    high_risk_mask = out["churn_probability"] >= threshold
    total_clv = float(out["clv"].sum())
    at_risk_value = float((out["churn_probability"] * out["clv"]).sum())
    return {
        "total_customers": int(len(out)),
        "high_risk_count": int(high_risk_mask.sum()),
        "high_risk_share": float(high_risk_mask.mean()),
        "baseline_churn_exposure": at_risk_value,
        "total_estimated_clv": total_clv,
        "average_churn_probability": float(out["churn_probability"].mean()),
        "average_clv": float(out["clv"].mean()),
        "high_risk_threshold": threshold,
        "current_retention_assumptions": None,
        "limitation": None,
    }



def _validate_churn_prob_reduction(reduction: float) -> float:
    reduction = float(reduction)
    if not CHURN_REDUCTION_MIN <= reduction <= CHURN_REDUCTION_MAX:
        raise SimulationError(
            f"churn_probability_reduction must be between {CHURN_REDUCTION_MIN} and {CHURN_REDUCTION_MAX}"
        )
    return reduction


def calculate_retention_cost(n_targeted_customers: int, offer_cost_per_customer: float) -> float:
    """Total retention campaign expenditure = targeted customers x offer cost."""
    n = int(n_targeted_customers)
    c = _validate_offer_cost(offer_cost_per_customer)
    if n < 0:
        raise SimulationError("n_targeted_customers cannot be negative")
    return float(n * c)


def simulate_retention_scenario(
    customers: pd.DataFrame,
    segment: str = SEGMENT_ALL_HIGH_RISK,
    save_rate: float = DEFAULT_SAVE_RATE,
    offer_cost_per_customer: float = DEFAULT_OFFER_COST,
    churn_probability_reduction: float = 0.0,
    high_risk_threshold: float = DEFAULT_HIGH_RISK_THRESHOLD,
    selected_customer_ids: list[str] | None = None,
    mechanism: str = "save_rate",
    simulation_horizon_months: int | None = None,
) -> dict[str, Any]:
    """Simulate one retention scenario with transparent assumptions.

    Two mechanisms (pick exactly one via ``mechanism``):

    * ``"save_rate"``: of the N targeted customers, an assumed save rate S are
      saved. Expected retained CLV = S x sum(CLV of targeted customers).
      Expected saved customers = N x S.
    * ``"probability_reduction"``: each targeted customer's churn probability is
      assumed to drop by delta (clipped at 0). Expected retained CLV =
      sum(CLV x (p_baseline - p_adjusted)); expected saved customers =
      sum(p_baseline - p_adjusted).

    Net scenario value = expected retained CLV - total retention cost.
    ROI = net value / cost (only when cost > 0).

    CLV estimates from Part 3 already represent discounted lifetime value, so
    they are never discounted again.
    """
    if mechanism not in ("save_rate", "probability_reduction"):
        raise SimulationError(f"Unknown mechanism: {mechanism}")
    out = _validate_customer_frame(customers)
    save_rate = _validate_save_rate(save_rate)
    offer_cost = _validate_offer_cost(offer_cost_per_customer)
    reduction = _validate_churn_prob_reduction(churn_probability_reduction)
    threshold = float(high_risk_threshold)
    if not 0 <= threshold <= 1:
        raise SimulationError("high_risk_threshold must be between 0 and 1")

    mask = _segment_mask(out, segment, threshold, selected_customer_ids)
    targeted = out.loc[mask]
    n_targeted = int(len(targeted))

    total_retention_cost = calculate_retention_cost(n_targeted, offer_cost)
    targeted_clv_sum = float(targeted["clv"].sum())

    if mechanism == "save_rate":
        expected_saved_customers = n_targeted * save_rate
        expected_retained_clv = save_rate * targeted_clv_sum
        assumption_note = (
            f"Assumes {save_rate:.1%} of the {n_targeted} targeted customers are saved; "
            "not a measured causal result."
        )
    else:
        adjusted_probability = (targeted["churn_probability"] - reduction).clip(lower=0.0)
        probability_reduction = targeted["churn_probability"] - adjusted_probability
        expected_saved_customers = float(probability_reduction.sum())
        expected_retained_clv = float((targeted["clv"] * probability_reduction).sum())
        assumption_note = (
            f"Assumes churn probability drops by {reduction:.1%} (floored at 0) for each targeted customer; "
            "not a measured causal result."
        )

    net_scenario_value = expected_retained_clv - total_retention_cost
    roi = (net_scenario_value / total_retention_cost) if total_retention_cost > 0 else None

    break_even_save_rate = None
    if mechanism == "save_rate" and targeted_clv_sum > 0:
        be = total_retention_cost / targeted_clv_sum
        break_even_save_rate = be if be <= 1.0 else None

    return {
        "segment": segment,
        "mechanism": mechanism,
        "n_targeted_customers": n_targeted,
        "save_rate": save_rate,
        "offer_cost_per_customer": offer_cost,
        "churn_probability_reduction": reduction,
        "high_risk_threshold": threshold,
        "total_retention_cost": float(total_retention_cost),
        "expected_saved_customers": float(expected_saved_customers),
        "expected_retained_clv": float(expected_retained_clv),
        "targeted_clv_sum": targeted_clv_sum,
        "net_scenario_value": float(net_scenario_value),
        "scenario_roi": None if roi is None else float(roi),
        "break_even_save_rate": break_even_save_rate,
        "assumption_note": assumption_note,
        "simulation_horizon_months": simulation_horizon_months,
        "horizon_note": (
            "Part 3 CLV is already a 60-month discounted lifetime estimate; no additional "
            "horizon discounting is applied."
        )
        if simulation_horizon_months is not None
        else None,
        "targeted_customer_ids": targeted["customerID"].tolist(),
    }


def compare_retention_scenarios(
    customers: pd.DataFrame,
    scenarios: list[dict[str, Any]],
    high_risk_threshold: float = DEFAULT_HIGH_RISK_THRESHOLD,
) -> pd.DataFrame:
    """Compare multiple scenarios side by side, including a baseline row."""
    rows = []
    baseline = calculate_baseline_metrics(customers, high_risk_threshold)
    rows.append(
        {
            "scenario": "Baseline (actuals)",
            "mechanism": "-",
            "n_targeted_customers": baseline["total_customers"],
            "save_rate": None,
            "offer_cost_per_customer": 0.0,
            "total_retention_cost": 0.0,
            "expected_saved_customers": None,
            "expected_retained_clv": None,
            "targeted_clv_sum": baseline["total_estimated_clv"],
            "net_scenario_value": None,
            "scenario_roi": None,
            "break_even_save_rate": None,
        }
    )
    for scenario in scenarios:
        result = simulate_retention_scenario(
            customers,
            segment=scenario.get("segment", SEGMENT_ALL_HIGH_RISK),
            save_rate=scenario.get("save_rate", DEFAULT_SAVE_RATE),
            offer_cost_per_customer=scenario.get("offer_cost_per_customer", DEFAULT_OFFER_COST),
            churn_probability_reduction=scenario.get("churn_probability_reduction", 0.0),
            high_risk_threshold=high_risk_threshold,
            selected_customer_ids=scenario.get("selected_customer_ids"),
            mechanism=scenario.get("mechanism", "save_rate"),
            simulation_horizon_months=scenario.get("simulation_horizon_months"),
        )
        rows.append(
            {
                "scenario": scenario.get("name", "Scenario"),
                "mechanism": result["mechanism"],
                "n_targeted_customers": result["n_targeted_customers"],
                "save_rate": result["save_rate"],
                "offer_cost_per_customer": result["offer_cost_per_customer"],
                "total_retention_cost": result["total_retention_cost"],
                "expected_saved_customers": result["expected_saved_customers"],
                "expected_retained_clv": result["expected_retained_clv"],
                "targeted_clv_sum": result["targeted_clv_sum"],
                "net_scenario_value": result["net_scenario_value"],
                "scenario_roi": result["scenario_roi"],
                "break_even_save_rate": result["break_even_save_rate"],
                "assumption_note": result["assumption_note"],
            }
        )
    return pd.DataFrame(rows)


def calculate_scenario_roi(expected_retained_value: float, total_retention_cost: float) -> float | None:
    """Scenario ROI = net value / cost; None when cost is zero (not meaningful)."""
    value = float(expected_retained_value)
    cost = float(total_retention_cost)
    if cost <= 0:
        return None
    return (value - cost) / cost


def generate_scenario_summary(
    comparison: pd.DataFrame,
    baseline_metrics: dict[str, Any] | None = None,
) -> str:
    """Plain-language interpretation of scenario results (assumptions labelled)."""
    if comparison is None or comparison.empty:
        return "No scenarios to interpret."
    scen = comparison[comparison["scenario"] != "Baseline (actuals)"]
    if scen.empty:
        return "No retention scenarios were simulated."
    lines: list[str] = []
    best = scen.loc[scen["net_scenario_value"].idxmax()]
    lines.append(
        f"Under the stated assumptions, '{best['scenario']}' has the highest modelled net value "
        f"(₹{best['net_scenario_value']:,.0f} vs ₹{scen['net_scenario_value'].min():,.0f} lowest)."
    )
    highest_budget = scen.loc[scen["total_retention_cost"].idxmax()]
    lines.append(
        f"'{highest_budget['scenario']}' requires the highest budget: "
        f"₹{highest_budget['total_retention_cost']:,.0f} "
        f"({int(highest_budget['n_targeted_customers'])} customers x ₹{highest_budget['offer_cost_per_customer']:,.0f})."
    )
    if best["scenario_roi"] is not None:
        lines.append(
            f"'{best['scenario']}' modelled ROI: {best['scenario_roi']:.1%} "
            "(modelled net value / total retention cost)."
        )
    save_rates = scen["save_rate"].dropna().unique()
    if len(save_rates) > 1:
        lines.append(
            "Results are sensitive to the assumed save rate: lowering the assumed save rate reduces "
            "expected retained value linearly while retention cost stays fixed."
        )
    for _, row in scen.iterrows():
        be = row["break_even_save_rate"]
        if be is not None:
            lines.append(
                f"'{row['scenario']}' breaks even at an assumed save rate of {be:.1%} "
                f"(₹{row['total_retention_cost']:,.0f} cost / ₹{row['targeted_clv_sum']:,.0f} targeted CLV). "
                "Below that rate the scenario has a negative modelled net value."
            )
    lines.append(
        "All figures are assumption-driven projections, not measured causal impacts. "
        "Assumed save rates are not experimentally measured results."
    )
    return "\n".join(lines)


def customer_level_simulation(
    customers: pd.DataFrame,
    segment: str,
    save_rate: float,
    offer_cost_per_customer: float,
    high_risk_threshold: float = DEFAULT_HIGH_RISK_THRESHOLD,
    selected_customer_ids: list[str] | None = None,
) -> pd.DataFrame:
    """Per-customer simulation results for the targeted segment (save_rate mechanism)."""
    out = _validate_customer_frame(customers)
    save_rate = _validate_save_rate(save_rate)
    offer_cost = _validate_offer_cost(offer_cost_per_customer)
    mask = _segment_mask(out, segment, float(high_risk_threshold), selected_customer_ids)
    targeted = out.loc[mask].copy()
    if targeted.empty:
        return targeted
    targeted["expected_at_risk_value"] = targeted["churn_probability"] * targeted["clv"]
    targeted["expected_retained_value"] = targeted["expected_at_risk_value"] * save_rate
    targeted["retention_cost"] = offer_cost
    targeted["net_value"] = targeted["expected_retained_value"] - targeted["retention_cost"]
    targeted = targeted.sort_values(["net_value", "churn_probability"], ascending=False).reset_index(drop=True)
    targeted.insert(0, "priority_rank", range(1, len(targeted) + 1))
    return targeted
