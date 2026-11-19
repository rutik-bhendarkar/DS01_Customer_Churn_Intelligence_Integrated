"""Part 5 dashboard integrity tests.

Covers the data-validation contract and the customer-level prediction display:
duplicate columns, duplicate customer IDs, expected-at-risk calculation,
scalar selected-customer values, numeric formatting, filters, retention
recommendations, and CSV export columns.
"""
from io import StringIO

import pandas as pd
import pytest

from src.dashboard_utils import (
    apply_dashboard_filters,
    enrich_recommendations,
    individual_customer_record,
    normalize_customer_data,
    validate_customer_scores,
)


def base_scores():
    return pd.DataFrame(
        {
            "customerID": ["C001", "C002", "C003", "C004"],
            "churn_probability": [0.10, 0.50, 0.90, 0.30],
            "clv": [100.0, 1000.0, 2500.0, 400.0],
            "contract": ["Two year", "One year", "Month-to-month", "Two year"],
        }
    )


def test_validate_detects_duplicate_columns():
    # Duplicate columns must be reported by name, never silently dropped.
    df = base_scores()
    df = pd.concat([df, df[["clv"]]], axis=1)  # creates a duplicate 'clv' column
    with pytest.raises(ValueError, match="Duplicate column names"):
        validate_customer_scores(df)


def test_validate_rejects_duplicate_customer_ids():
    df = base_scores()
    df.loc[1, "customerID"] = "C001"
    with pytest.raises(ValueError, match="customerID values must be unique"):
        validate_customer_scores(df)


def test_validate_rejects_invalid_numeric_ranges():
    df = base_scores()
    df.loc[0, "churn_probability"] = 1.2
    with pytest.raises(ValueError, match="between 0 and 1"):
        validate_customer_scores(df)

    df = base_scores()
    df.loc[0, "clv"] = -50.0
    with pytest.raises(ValueError, match="clv cannot be negative"):
        validate_customer_scores(df)


def test_expected_at_risk_value_is_calculated_once_and_correct():
    out = normalize_customer_data(base_scores())
    expected = (out["churn_probability"] * out["clv"]).rename("expected_at_risk_value")
    pd.testing.assert_series_equal(out["expected_at_risk_value"], expected)
    assert out["expected_at_risk_value"].tolist() == pytest.approx([10.0, 500.0, 2250.0, 120.0])

    # A pre-existing expected_at_risk_value column is preserved and validated,
    # not recalculated or dropped.
    df = base_scores()
    df["expected_at_risk_value"] = df["churn_probability"] * df["clv"]
    out2 = normalize_customer_data(df)
    assert out2["expected_at_risk_value"].notna().all()
    pd.testing.assert_series_equal(out2["expected_at_risk_value"], expected)


def test_selected_customer_row_returns_scalars():
    # Mirrors the customer-level lookup in app.py.
    out = normalize_customer_data(base_scores())
    scored = enrich_recommendations(out, 20.0, 0.20)
    assert not scored.columns.duplicated().any()

    selected_id = "C003"
    matches = scored[scored["customerID"].astype(str) == str(selected_id)]
    assert len(matches) == 1, "customer lookup must match exactly one customer"
    selected = matches.iloc[0]

    assert not isinstance(selected["churn_probability"], pd.Series)
    assert not isinstance(selected["clv"], pd.Series)
    assert not isinstance(selected["expected_at_risk_value"], pd.Series)
    assert not isinstance(selected["scenario_net_value"], pd.Series)

    churn_probability = float(selected["churn_probability"])
    clv = float(selected["clv"])
    expected_at_risk_value = float(selected["expected_at_risk_value"])
    assert churn_probability == pytest.approx(0.90)
    assert clv == pytest.approx(2500.0)
    assert expected_at_risk_value == pytest.approx(churn_probability * clv)
    assert expected_at_risk_value == pytest.approx(2250.0)


def test_numeric_formatting_works():
    churn_probability, clv, expected_at_risk_value = 0.91, 1200.0, 1092.0
    assert f"{churn_probability:.1%}" == "91.0%"
    assert f"₹{clv:,.2f}" == "₹1,200.00"
    assert f"₹{expected_at_risk_value:,.2f}" == "₹1,092.00"


def test_filters_return_expected_records():
    out = normalize_customer_data(base_scores())

    filtered = apply_dashboard_filters(out, ["High"], None, 0.0, "")
    assert filtered["customerID"].tolist() == ["C003"]

    filtered = apply_dashboard_filters(out, None, ["Two year"], 0.0, "")
    assert set(filtered["customerID"]) == {"C001", "C004"}

    filtered = apply_dashboard_filters(out, None, None, 500.0, "")
    assert set(filtered["customerID"]) == {"C002", "C003"}

    filtered = apply_dashboard_filters(out, None, None, 0.0, "C001")
    assert filtered["customerID"].tolist() == ["C001"]

    # Combining filters still returns a valid, non-duplicated frame.
    assert not filtered.columns.duplicated().any()


def test_retention_recommendations_no_duplicate_columns():
    out = normalize_customer_data(base_scores())
    recs = enrich_recommendations(out, offer_cost=10.0, assumed_save_rate=0.2)
    assert not recs.columns.duplicated().any()
    assert {
        "recommended_action",
        "expected_at_risk_value",
        "scenario_retained_value",
        "scenario_net_value",
    }.issubset(recs.columns)

    high_value_high_risk = recs[recs["customerID"] == "C003"].iloc[0]
    assert high_value_high_risk["recommended_action"] == "Personal retention outreach"
    assert high_value_high_risk["expected_at_risk_value"] == pytest.approx(0.90 * 2500.0)
    assert high_value_high_risk["scenario_net_value"] == pytest.approx(
        0.90 * 2500.0 * 0.2 - 10.0
    )

    low_risk = recs[recs["customerID"] == "C001"].iloc[0]
    assert low_risk["recommended_action"] == "No immediate intervention"


def test_csv_exports_contain_valid_columns():
    out = normalize_customer_data(base_scores())
    scored = enrich_recommendations(out, 20.0, 0.20)

    customer_csv = scored.to_csv(index=False).encode("utf-8")
    reloaded = pd.read_csv(StringIO(customer_csv.decode("utf-8")))
    assert not reloaded.columns.duplicated().any()
    for col in [
        "customerID",
        "churn_probability",
        "clv",
        "expected_at_risk_value",
        "recommended_action",
        "scenario_net_value",
    ]:
        assert col in reloaded.columns, f"export missing column: {col}"

    action_summary = (
        scored.groupby("recommended_action", as_index=False)
        .agg(
            customers=("customerID", "count"),
            total_at_risk_value=("expected_at_risk_value", "sum"),
            scenario_net_value=("scenario_net_value", "sum"),
        )
        .sort_values("total_at_risk_value", ascending=False)
    )
    summary_csv = action_summary.to_csv(index=False).encode("utf-8")
    summary_reloaded = pd.read_csv(StringIO(summary_csv.decode("utf-8")))
    assert set(summary_reloaded.columns) == {
        "recommended_action",
        "customers",
        "total_at_risk_value",
        "scenario_net_value",
    }


def test_manual_customer_record_returns_scalars():
    one = individual_customer_record("MANUAL-001", 0.80, 1500.0, tenure=12.0)
    row = one.iloc[0]
    assert float(row["churn_probability"]) == pytest.approx(0.80)
    assert float(row["clv"]) == pytest.approx(1500.0)
    assert float(row["expected_at_risk_value"]) == pytest.approx(0.80 * 1500.0)
    assert row["churn_risk"] == "High"