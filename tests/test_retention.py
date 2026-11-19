import pandas as pd
import pytest
from src.retention import recommend_retention, validate_customers

@pytest.fixture
def data():
    return pd.DataFrame({"customerID": ["A", "B", "C"], "churn_probability": [0.9, 0.4, 0.1], "clv": [1000, 500, 2000]})

def test_recommendations_have_expected_columns(data):
    result = recommend_retention(data, offer_cost=10, save_rate_assumption=0.2)
    assert {"risk_band", "recommended_action", "estimated_net_value"}.issubset(result.columns)
    assert result.iloc[0].customerID == "A"

def test_probability_out_of_range_rejected(data):
    data.loc[0, "churn_probability"] = 1.5
    with pytest.raises(ValueError):
        validate_customers(data)

def test_duplicate_customer_rejected(data):
    data.loc[2, "customerID"] = "A"
    with pytest.raises(ValueError):
        validate_customers(data)

def test_negative_offer_cost_rejected(data):
    with pytest.raises(ValueError):
        recommend_retention(data, offer_cost=-1)
