from __future__ import annotations
import json
from src.data_access import load_raw_dataset
from src.quality import inspect_quality, write_quality_report
from src.config import REPORTS_DIR

if __name__ == "__main__":
    df = load_raw_dataset()
    report = inspect_quality(df)
    path = write_quality_report(report)
    print(f"Rows: {len(df):,} | Columns: {df.shape[1]}")
    print(f"Exact duplicate rows: {report['exact_duplicate_rows']}")
    print(f"Duplicate customer IDs: {report['duplicate_customer_ids']}")
    print(f"Blank TotalCharges before conversion: {report['blank_total_charges_before_conversion']}")
    print(f"Churn counts: {report['churn_raw_counts']}")
    print(f"Quality report: {path.resolve()}")
