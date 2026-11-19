import pandas as pd
from src.feature_engineering import add_features
from src.quality import clean_dataset


def sample_data():
    return pd.DataFrame([
        {"customerID":"A1", "gender":"Female", "SeniorCitizen":0, "Partner":"Yes", "Dependents":"No",
         "tenure":0, "PhoneService":"No", "MultipleLines":"No phone service", "InternetService":"DSL",
         "OnlineSecurity":"No", "OnlineBackup":"No", "DeviceProtection":"No", "TechSupport":"No",
         "StreamingTV":"No", "StreamingMovies":"No", "Contract":"Month-to-month", "PaperlessBilling":"Yes",
         "PaymentMethod":"Electronic check", "MonthlyCharges":29.85, "TotalCharges":" ", "Churn":"Yes"},
        {"customerID":"A2", "gender":"Male", "SeniorCitizen":0, "Partner":"No", "Dependents":"No",
         "tenure":12, "PhoneService":"Yes", "MultipleLines":"No", "InternetService":"No",
         "OnlineSecurity":"No internet service", "OnlineBackup":"No internet service", "DeviceProtection":"No internet service",
         "TechSupport":"No internet service", "StreamingTV":"No internet service", "StreamingMovies":"No internet service",
         "Contract":"One year", "PaperlessBilling":"No", "PaymentMethod":"Mailed check", "MonthlyCharges":20.35,
         "TotalCharges":"250.00", "Churn":"No"},
    ])


def test_clean_and_engineer_features():
    cleaned = clean_dataset(sample_data())
    assert cleaned["TotalCharges"].iloc[0] == 0.0
    assert cleaned["Churn"].tolist() == [1, 0]
    features = add_features(cleaned)
    assert features["is_month_to_month"].tolist() == [1, 0]
    assert features["is_electronic_check"].tolist() == [1, 0]
    assert features["tenure_group"].iloc[0] == "0-12 months"
    assert pd.isna(features["avg_historical_monthly_charge"].iloc[0])


def test_no_mutation_to_input():
    raw = sample_data()
    original_columns = list(raw.columns)
    _ = clean_dataset(raw)
    assert list(raw.columns) == original_columns
    assert raw["Churn"].iloc[0] == "Yes"
