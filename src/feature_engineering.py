"""Create interpretable Part 1 features from fields present in the IBM Telco snapshot."""
from __future__ import annotations
import pandas as pd

SERVICE_YES_COLUMNS = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Count affirmative service flags. Values such as 'No internet service' count as 0.
    for col in SERVICE_YES_COLUMNS:
        if col not in out.columns:
            continue
        out[f"{col}_flag"] = out[col].astype("string").str.casefold().eq("yes").astype("int8")
    flag_columns = [f"{col}_flag" for col in SERVICE_YES_COLUMNS if f"{col}_flag" in out.columns]
    out["service_count"] = out[flag_columns].sum(axis=1).astype("int8") if flag_columns else 0
    out["has_internet_service"] = out["InternetService"].astype("string").str.casefold().ne("no").astype("int8")
    out["has_phone_service"] = out["PhoneService"].astype("string").str.casefold().eq("yes").astype("int8")
    out["is_month_to_month"] = out["Contract"].astype("string").str.casefold().eq("month-to-month").astype("int8")
    out["is_electronic_check"] = out["PaymentMethod"].astype("string").str.casefold().eq("electronic check").astype("int8")
    out["tenure_group"] = pd.cut(
        out["tenure"], bins=[-1, 12, 24, 48, 72, float("inf")],
        labels=["0-12 months", "13-24 months", "25-48 months", "49-72 months", "73+ months"],
        ordered=True,
    ).astype("string")
    # A descriptive average charge measure. For zero-tenure customers, leave as missing
    # rather than dividing by zero or fabricating an average.
    out["avg_historical_monthly_charge"] = out["TotalCharges"].div(out["tenure"].where(out["tenure"].gt(0)))
    # Snapshot supports cumulative totals and current monthly charges, not monthly usage trends.
    return out
