from __future__ import annotations
import pandas as pd
from src.config import CLEAN_DATA_PATH, FEATURE_DATA_PATH, REPORTS_DIR
from src.data_access import load_raw_dataset
from src.quality import clean_dataset, inspect_quality, write_quality_report
from src.feature_engineering import add_features


def write_data_dictionary(df: pd.DataFrame) -> None:
    descriptions = {
        "customerID": "Unique customer identifier; exclude from model features.",
        "tenure": "Months the customer has stayed with the company.",
        "MonthlyCharges": "Current monthly amount charged to the customer.",
        "TotalCharges": "Cumulative charges to date; blank values for tenure=0 are set to 0.",
        "Churn": "Target label: 1 means churned; 0 means retained.",
        "service_count": "Count of affirmative service flags present in this snapshot.",
        "tenure_group": "Tenure bucket derived from tenure months.",
        "avg_historical_monthly_charge": "TotalCharges / tenure; undefined when tenure=0.",
        "is_month_to_month": "1 when the contract is month-to-month.",
        "is_electronic_check": "1 when payment method is electronic check.",
        "has_internet_service": "1 when InternetService is not No.",
        "has_phone_service": "1 when PhoneService is Yes.",
    }
    rows = []
    for col in df.columns:
        rows.append({"column": col, "dtype": str(df[col].dtype), "non_null": int(df[col].notna().sum()),
                     "unique_values": int(df[col].nunique(dropna=True)),
                     "description": descriptions.get(col, "Original source field; see IBM Telco dataset documentation.")})
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(REPORTS_DIR / "data_dictionary.csv", index=False)


def main() -> None:
    raw = load_raw_dataset()
    quality = inspect_quality(raw)
    clean = clean_dataset(raw)
    features = add_features(clean)
    CLEAN_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    FEATURE_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(CLEAN_DATA_PATH, index=False)
    features.to_csv(FEATURE_DATA_PATH, index=False)
    write_quality_report(quality)
    write_data_dictionary(features)
    feature_summary = features.describe(include="all").transpose()
    feature_summary.to_csv(REPORTS_DIR / "feature_summary.csv")
    print(f"Raw rows: {len(raw):,}")
    print(f"Clean rows: {len(clean):,}; columns: {clean.shape[1]}")
    print(f"Feature rows: {len(features):,}; columns: {features.shape[1]}")
    print(f"Clean data saved: {CLEAN_DATA_PATH.resolve()}")
    print(f"Feature data saved: {FEATURE_DATA_PATH.resolve()}")
    print(f"Reports saved under: {REPORTS_DIR.resolve()}")
    print("Note: No customer rows were dropped except exact duplicate rows, and duplicate IDs trigger an error.")

if __name__ == "__main__":
    main()
