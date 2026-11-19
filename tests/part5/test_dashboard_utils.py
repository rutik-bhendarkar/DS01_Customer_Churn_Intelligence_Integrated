import pandas as pd
import pytest
from src.dashboard_utils import (
    normalize_customer_data, apply_dashboard_filters, enrich_recommendations,
    individual_customer_record
)

def sample():
    return pd.DataFrame({
        "customerID": ["A", "B", "C"],
        "churn_probability": [0.1, 0.5, 0.9],
        "clv": [100, 1000, 2500],
        "contract": ["Two year", "One year", "Month-to-month"]
    })

def test_normalize_and_risk_bands():
    out = normalize_customer_data(sample())
    assert out["churn_risk"].tolist() == ["Low", "Medium", "High"]
    assert out["expected_at_risk_value"].tolist() == pytest.approx([10, 500, 2250])

def test_missing_required_column_fails():
    with pytest.raises(ValueError, match="Missing required"):
        normalize_customer_data(pd.DataFrame({"customerID": ["A"]}))

def test_probability_range_fails():
    df = sample()
    df.loc[0, "churn_probability"] = 1.2
    with pytest.raises(ValueError, match="between 0 and 1"):
        normalize_customer_data(df)

def test_duplicate_customer_fails():
    df = sample()
    df.loc[1, "customerID"] = "A"
    with pytest.raises(ValueError, match="unique"):
        normalize_customer_data(df)

def test_filter_and_recommendation():
    df = normalize_customer_data(sample())
    filtered = apply_dashboard_filters(df, ["High"], ["Month-to-month"], 0, "")
    assert filtered["customerID"].tolist() == ["C"]
    recs = enrich_recommendations(filtered, offer_cost=10, assumed_save_rate=.2)
    assert recs.iloc[0]["recommended_action"] == "Personal retention outreach"

def test_manual_customer_scenario():
    one = individual_customer_record("X", .8, 1500)
    assert one.iloc[0]["churn_risk"] == "High"
