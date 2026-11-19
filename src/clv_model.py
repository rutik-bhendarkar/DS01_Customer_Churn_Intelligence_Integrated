"""Part 3: Kaplan-Meier survival analysis and transparent CLV estimates."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from src.config import FEATURE_DATA_PATH, REPORTS_DIR, ROOT_DIR

MODEL_DIR = ROOT_DIR / "models"
CLV_OUTPUT = REPORTS_DIR / "clv_estimates.csv"
SURVIVAL_REPORT = REPORTS_DIR / "survival_summary.json"
HORIZON_MONTHS = 60

def estimate_clv(df: pd.DataFrame, horizon_months: int = HORIZON_MONTHS, annual_discount_rate: float = 0.10) -> tuple[pd.DataFrame, dict]:
    required = {"customerID", "tenure", "Churn", "MonthlyCharges"}
    missing = required - set(df.columns)
    if missing: raise ValueError(f"Missing required columns: {sorted(missing)}")
    if horizon_months < 1: raise ValueError("horizon_months must be positive")
    if annual_discount_rate < 0: raise ValueError("annual_discount_rate cannot be negative")
    data = df.copy()
    data["tenure"] = pd.to_numeric(data["tenure"], errors="coerce")
    data["MonthlyCharges"] = pd.to_numeric(data["MonthlyCharges"], errors="coerce")
    churn = data["Churn"]
    if churn.dtype == object:
        event = churn.astype(str).str.strip().str.lower().map({"yes":1,"no":0,"1":1,"0":0})
    else: event = pd.to_numeric(churn, errors="coerce")
    data["event"] = event
    if data[["tenure","MonthlyCharges","event"]].isna().any().any(): raise ValueError("tenure, MonthlyCharges and Churn must be valid")
    if not data["event"].isin([0,1]).all(): raise ValueError("Churn must map to 0/1")
    if (data["tenure"] < 0).any() or (data["MonthlyCharges"] < 0).any(): raise ValueError("Negative tenure/charges are invalid")
    km = KaplanMeierFitter().fit(data["tenure"], event_observed=data["event"], label="customer_survival")
    timeline = np.arange(0, horizon_months + 1, dtype=float)
    surv = km.predict(timeline).to_numpy(dtype=float)
    # Conditional survival: P(survive another u months | already reached current tenure).
    rows=[]
    discount_monthly=(1+annual_discount_rate)**(1/12)-1
    for r in data.itertuples(index=False):
        t=float(r.tenure); s_t=float(km.predict(t))
        future=np.arange(1,horizon_months+1,dtype=float)
        future_surv=np.array([float(km.predict(t+u))/s_t if s_t>0 else 0.0 for u in future])
        future_surv=np.clip(future_surv,0,1)
        disc=np.power(1+discount_monthly, future)
        expected_months=float(future_surv.sum())
        expected_revenue=float(np.sum(future_surv*float(r.MonthlyCharges)/disc))
        rows.append({"customerID":r.customerID,"tenure_months":t,"monthly_charges":float(r.MonthlyCharges),"observed_churn":int(r.event),"survival_probability_at_snapshot":s_t,"expected_remaining_months_60m":expected_months,"estimated_clv_60m":expected_revenue})
    result=pd.DataFrame(rows)
    summary={"method":"Kaplan-Meier conditional survival with discounted monthly charges","horizon_months":horizon_months,"annual_discount_rate":annual_discount_rate,"customers":int(len(result)),"median_expected_remaining_months":float(result.expected_remaining_months_60m.median()),"total_estimated_clv":float(result.estimated_clv_60m.sum()),"limitations":"Cross-sectional tenure/churn snapshot; estimates are descriptive model-based projections, not individual causal or contract-level forecasts."}
    return result,summary

def main():
    if not FEATURE_DATA_PATH.exists():
        from src.run_part1 import main as run_part1
        run_part1()
    df=pd.read_csv(FEATURE_DATA_PATH)
    estimates,summary=estimate_clv(df)
    REPORTS_DIR.mkdir(parents=True,exist_ok=True)
    estimates.to_csv(CLV_OUTPUT,index=False)
    SURVIVAL_REPORT.write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(f"Saved CLV estimates: {CLV_OUTPUT.resolve()}")
    print(json.dumps(summary,indent=2))
if __name__=="__main__": main()
