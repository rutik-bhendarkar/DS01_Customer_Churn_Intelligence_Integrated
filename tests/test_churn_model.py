from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.churn_model import (
    EXCLUDED_FEATURE_COLUMNS,
    build_models,
    choose_threshold,
    predict_dataframe,
    prepare_features,
    score_probabilities,
    train_and_evaluate,
)


def make_customers(n: int = 160) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    tenure = rng.integers(0, 72, size=n)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], size=n, p=[0.55, 0.25, 0.20])
    payment = rng.choice(["Electronic check", "Mailed check", "Bank transfer"], size=n)
    monthly = rng.normal(75, 20, size=n).clip(20, 120)
    logit = -1.0 + (contract == "Month-to-month") * 1.1 - (tenure / 72) * 1.5 + (payment == "Electronic check") * 0.25
    probability = 1 / (1 + np.exp(-logit))
    churn = rng.binomial(1, probability, size=n)
    return pd.DataFrame({
        "customerID": [f"C{i:04d}" for i in range(n)],
        "tenure": tenure,
        "MonthlyCharges": monthly,
        "Contract": contract,
        "PaymentMethod": payment,
        "TotalCharges": tenure * monthly,
        "avg_historical_monthly_charge": monthly,
        "service_count": rng.integers(0, 8, size=n),
        "Churn": churn,
    })


def test_prepare_features_excludes_identifier_target_and_cumulative_charge():
    df = make_customers()
    X, y, columns = prepare_features(df)
    assert "customerID" not in columns
    assert "Churn" not in columns
    assert "TotalCharges" not in columns
    assert "avg_historical_monthly_charge" not in columns
    assert len(X) == len(y)
    assert set(y.unique()) == {0, 1}
    assert EXCLUDED_FEATURE_COLUMNS.issubset({"customerID", "Churn", "TotalCharges", "avg_historical_monthly_charge"})


def test_threshold_selection_returns_valid_threshold_and_metrics():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.05, 0.25, 0.45, 0.40, 0.65, 0.90])
    threshold, table = choose_threshold(y, p)
    assert 0.05 <= threshold <= 0.95
    assert not table.empty
    metrics = score_probabilities(y, p, threshold)
    assert 0 <= metrics["precision"] <= 1
    assert 0 <= metrics["recall"] <= 1
    assert 0 <= metrics["pr_auc"] <= 1


def test_models_have_preprocessing_pipeline():
    df = make_customers()
    X, _, _ = prepare_features(df)
    models = build_models(X)
    assert set(models) == {"logistic_regression", "lightgbm"}
    assert all(isinstance(model, Pipeline) for model in models.values())
    assert all("preprocessor" in model.named_steps for model in models.values())


def test_training_produces_model_and_metrics(tmp_path, monkeypatch):
    # Isolate artifacts from the project folder while exercising the real training flow.
    import src.churn_model as churn_module
    monkeypatch.setattr(churn_module, "ROOT_DIR", tmp_path)
    monkeypatch.setattr(churn_module, "MODEL_DIR", tmp_path / "models")
    monkeypatch.setattr(churn_module, "MODEL_PATH", tmp_path / "models" / "churn_model.joblib")
    monkeypatch.setattr(churn_module, "METADATA_PATH", tmp_path / "models" / "churn_model_metadata.json")
    monkeypatch.setattr(churn_module, "REPORTS_DIR", tmp_path / "reports")
    monkeypatch.setattr(churn_module, "FIGURE_DIR", tmp_path / "reports" / "figures")
    monkeypatch.setattr(churn_module, "COMPARISON_PATH", tmp_path / "reports" / "churn_model_comparison.csv")
    monkeypatch.setattr(churn_module, "TEST_METRICS_PATH", tmp_path / "reports" / "churn_test_metrics.json")
    monkeypatch.setattr(churn_module, "THRESHOLD_PATH", tmp_path / "reports" / "churn_threshold_analysis.csv")
    monkeypatch.setattr(churn_module, "IMPORTANCE_PATH", tmp_path / "reports" / "churn_feature_importance.csv")
    result = train_and_evaluate(make_customers())
    assert result["selected_model"] in {"logistic_regression", "lightgbm"}
    assert (tmp_path / "models" / "churn_model.joblib").exists()
    assert (tmp_path / "reports" / "churn_test_metrics.json").exists()
    assert (tmp_path / "reports" / "figures" / "part2_roc_curve.png").exists()
    loaded = predict_dataframe(make_customers(12), model_path=tmp_path / "models" / "churn_model.joblib",
                               metadata_path=tmp_path / "models" / "churn_model_metadata.json")
    assert len(loaded) == 12
    assert loaded["churn_probability"].between(0, 1).all()
    assert set(loaded["predicted_churn"].unique()).issubset({0, 1})
