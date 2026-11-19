from __future__ import annotations
import numpy as np
import pandas as pd

REQUIRED = {"customerID", "churn_probability", "clv"}

def validate_customers(df: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    out = df.copy()
    out["churn_probability"] = pd.to_numeric(out["churn_probability"], errors="coerce")
    out["clv"] = pd.to_numeric(out["clv"], errors="coerce")
    if out[["churn_probability", "clv"]].isna().any().any():
        raise ValueError("churn_probability and clv must contain numeric, non-null values")
    if not out["churn_probability"].between(0, 1).all():
        raise ValueError("churn_probability must be between 0 and 1")
    if (out["clv"] < 0).any():
        raise ValueError("clv cannot be negative")
    if out["customerID"].astype(str).duplicated().any():
        raise ValueError("customerID must be unique")
    return out

def recommend_retention(df: pd.DataFrame, offer_cost: float = 20.0,
                        save_rate_assumption: float = 0.20) -> pd.DataFrame:
    """Transparent prioritization heuristic; not a causal treatment-effect model."""
    if offer_cost < 0:
        raise ValueError("offer_cost cannot be negative")
    if not 0 <= save_rate_assumption <= 1:
        raise ValueError("save_rate_assumption must be between 0 and 1")
    out = validate_customers(df)
    out["risk_band"] = pd.cut(out.churn_probability,
        bins=[-0.001, 0.30, 0.60, 1.0], labels=["Low", "Medium", "High"])
    out["expected_at_risk_value"] = out.churn_probability * out.clv
    out["assumed_retained_value"] = out.expected_at_risk_value * save_rate_assumption
    out["estimated_net_value"] = out.assumed_retained_value - offer_cost
    def action(r):
        if r.churn_probability >= 0.60 and r.clv >= out.clv.median():
            return "Personal retention outreach"
        if r.churn_probability >= 0.60:
            return "Low-cost retention message"
        if r.churn_probability >= 0.30:
            return "Monitor and send engagement offer"
        return "No immediate intervention"
    out["recommended_action"] = out.apply(action, axis=1)
    out["priority"] = np.select(
        [out.estimated_net_value > 0, out.churn_probability >= 0.60],
        ["Prioritize - positive assumed value", "Review - high churn risk"],
        default="Monitor")
    return out.sort_values(["estimated_net_value", "churn_probability"], ascending=False).reset_index(drop=True)

def estimate_uplift_t_learner(df: pd.DataFrame, treatment_col: str,
                              outcome_col: str, feature_cols: list[str],
                              random_state: int = 42) -> pd.DataFrame:
    """Exploratory T-learner. Requires treatment and outcome for every row.

    Estimates model-based conditional differences; causal interpretation requires
    randomized assignment or defensible no-unmeasured-confounding assumptions.
    """
    from sklearn.base import clone
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.model_selection import train_test_split
    if treatment_col not in df or outcome_col not in df:
        raise ValueError("Treatment and outcome columns are required")
    if not feature_cols or any(c not in df for c in feature_cols):
        raise ValueError("Provide valid feature_cols")
    t = pd.to_numeric(df[treatment_col], errors="coerce")
    y = pd.to_numeric(df[outcome_col], errors="coerce")
    if t.isna().any() or y.isna().any() or set(t.unique()) != {0, 1}:
        raise ValueError("Treatment must contain both 0 and 1; outcome must be numeric and complete")
    if (t.value_counts() < 10).any():
        raise ValueError("At least 10 observations per treatment group are required")
    X = df[feature_cols].copy()
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in feature_cols if c not in numeric]
    transformers = []
    if numeric:
        transformers.append(("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric))
    if categorical:
        transformers.append(("cat", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore")), categorical))
    pre = ColumnTransformer(transformers)
    model0 = make_pipeline(clone(pre), RandomForestRegressor(n_estimators=150, min_samples_leaf=5, random_state=random_state))
    model1 = make_pipeline(clone(pre), RandomForestRegressor(n_estimators=150, min_samples_leaf=5, random_state=random_state))
    model0.fit(X.loc[t == 0], y.loc[t == 0])
    model1.fit(X.loc[t == 1], y.loc[t == 1])
    result = df.copy()
    result["predicted_outcome_control"] = model0.predict(X)
    result["predicted_outcome_treatment"] = model1.predict(X)
    result["estimated_uplift"] = result.predicted_outcome_treatment - result.predicted_outcome_control
    result.attrs["warning"] = "Exploratory estimate only; not causal proof without valid identification assumptions."
    return result
