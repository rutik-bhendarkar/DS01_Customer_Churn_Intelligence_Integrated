"""Data quality checks and transparent cleaning for the IBM Telco snapshot."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from src.config import REQUIRED_COLUMNS, REPORTS_DIR


def inspect_quality(df: pd.DataFrame) -> dict:
    total_charges_raw = df["TotalCharges"].astype("string").str.strip()
    total_charges_blank = int(total_charges_raw.isna().sum() + total_charges_raw.eq("").sum())
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "missing_cells_by_column": {str(k): int(v) for k, v in df.isna().sum().items()},
        "exact_duplicate_rows": int(df.duplicated().sum()),
        "duplicate_customer_ids": int(df["customerID"].duplicated().sum()),
        "blank_total_charges_before_conversion": total_charges_blank,
        "blank_total_charges_for_zero_tenure": int(((total_charges_raw.isna() | total_charges_raw.eq("")) & (pd.to_numeric(df["tenure"], errors="coerce") == 0)).sum()),
        "churn_raw_counts": {str(k): int(v) for k, v in df["Churn"].value_counts(dropna=False).items()},
        "contract_counts": {str(k): int(v) for k, v in df["Contract"].value_counts(dropna=False).items()},
        "numeric_ranges": {
            "tenure_min": float(pd.to_numeric(df["tenure"], errors="coerce").min()),
            "tenure_max": float(pd.to_numeric(df["tenure"], errors="coerce").max()),
            "monthly_charges_min": float(pd.to_numeric(df["MonthlyCharges"], errors="coerce").min()),
            "monthly_charges_max": float(pd.to_numeric(df["MonthlyCharges"], errors="coerce").max()),
        },
    }


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Return cleaned data without mutating the raw dataframe or dropping rows silently."""
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Required columns missing: {sorted(missing)}")
    cleaned = df.copy()
    cleaned.columns = [str(c).strip() for c in cleaned.columns]
    cleaned["customerID"] = cleaned["customerID"].astype("string").str.strip()

    # Remove only exact duplicate rows; duplicate customer IDs are an error because
    # each record is expected to represent one customer in this dataset.
    duplicate_rows = int(cleaned.duplicated().sum())
    if duplicate_rows:
        cleaned = cleaned.drop_duplicates().copy()
    duplicate_ids = cleaned["customerID"].duplicated(keep=False)
    if duplicate_ids.any():
        examples = cleaned.loc[duplicate_ids, "customerID"].head(10).tolist()
        raise ValueError(f"Duplicate customer IDs detected after exact-row cleanup: {examples}")

    cleaned["tenure"] = pd.to_numeric(cleaned["tenure"], errors="coerce")
    cleaned["SeniorCitizen"] = pd.to_numeric(cleaned["SeniorCitizen"], errors="coerce")
    cleaned["MonthlyCharges"] = pd.to_numeric(cleaned["MonthlyCharges"], errors="coerce")
    total_text = cleaned["TotalCharges"].astype("string").str.strip()
    cleaned["TotalCharges"] = pd.to_numeric(total_text.replace("", pd.NA), errors="coerce")
    # The public sample has blank TotalCharges for brand-new customers whose tenure is 0.
    valid_zero_tenure = cleaned["TotalCharges"].isna() & cleaned["tenure"].eq(0)
    cleaned.loc[valid_zero_tenure, "TotalCharges"] = 0.0

    # Guard against unexplained missing numeric values rather than concealing them.
    bad_total = cleaned["TotalCharges"].isna()
    if bad_total.any():
        ids = cleaned.loc[bad_total, "customerID"].head(10).tolist()
        raise ValueError(f"TotalCharges is missing for non-zero/unknown tenure records. Review IDs: {ids}")
    for col in ["tenure", "MonthlyCharges", "SeniorCitizen"]:
        if cleaned[col].isna().any():
            raise ValueError(f"Unexpected missing values in required numeric field: {col}")
    if not set(cleaned["Churn"].dropna().unique()).issubset({"Yes", "No"}):
        raise ValueError("Unexpected values in Churn; expected 'Yes' and 'No'.")
    if cleaned["Churn"].isna().any():
        raise ValueError("Churn target has missing values.")
    if not cleaned["tenure"].between(0, 1000).all():
        raise ValueError("Tenure has values outside the expected non-negative range; investigate data.")
    if (cleaned["MonthlyCharges"] < 0).any() or (cleaned["TotalCharges"] < 0).any():
        raise ValueError("Negative charge values found; investigate data.")

    cleaned["Churn"] = cleaned["Churn"].map({"Yes": 1, "No": 0}).astype("int8")
    cleaned["SeniorCitizen"] = cleaned["SeniorCitizen"].astype("int8")
    cleaned["tenure"] = cleaned["tenure"].astype("int16")
    for col in cleaned.select_dtypes(include=["object", "string"]).columns:
        if col != "customerID":
            cleaned[col] = cleaned[col].astype("string").str.strip()
    return cleaned


def write_quality_report(report: dict, path: Path | None = None) -> Path:
    path = path or (REPORTS_DIR / "data_quality_report.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
