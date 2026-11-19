from pathlib import Path
import sys
import pandas as pd
import streamlit as st
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from src.retention import recommend_retention
from src.config import FEATURE_DATA_PATH, REPORTS_DIR
from src.churn_model import predict_dataframe
from src.clv_model import estimate_clv

st.set_page_config(page_title="DS-01 Customer Intelligence",page_icon="📊",layout="wide")
st.title("Customer Lifetime Value & Churn Intelligence")
st.caption("Integrated DS-01 dashboard | Parts 1–4")
st.info("Run Parts 1–3 first, or upload a scored customer CSV containing customerID, churn_probability and clv/estimated_clv_60m.")
upload=st.file_uploader("Upload scored customers CSV (optional)",type=["csv"])
if upload:
    customers=pd.read_csv(upload)
else:
    feature_path=FEATURE_DATA_PATH
    churn_path=REPORTS_DIR/"churn_predictions.csv"
    clv_path=REPORTS_DIR/"clv_estimates.csv"
    if feature_path.exists() and churn_path.exists() and clv_path.exists():
        base=pd.read_csv(feature_path)
        pred=pd.read_csv(churn_path)
        clv=pd.read_csv(clv_path)
        customers=base.merge(pred,on="customerID",how="inner").merge(clv[["customerID","estimated_clv_60m"]],on="customerID",how="inner")
    else:
        sample=ROOT/"data"/"sample"/"customer_scores_sample.csv"
        customers=pd.read_csv(sample)
        st.warning("Showing illustrative sample data. Train Part 2 and run Part 3 to load actual project outputs.")
if "clv" not in customers.columns and "estimated_clv_60m" in customers.columns: customers=customers.rename(columns={"estimated_clv_60m":"clv"})
if "clv" not in customers.columns and "MonthlyCharges" in customers.columns:
    st.warning("CLV estimates not found; run `python -m src.clv_model` or upload a scored file with CLV.")
    st.stop()
if "churn_probability" not in customers.columns:
    st.warning("Churn probabilities not found; run `python -m src.predict_churn` or upload a scored file.")
    st.stop()
a,b=st.columns(2)
with a: offer_cost=st.number_input("Offer cost per customer",min_value=0.0,value=20.0,step=5.0)
with b: save_rate=st.slider("Assumed save-rate scenario (not learned from data)",0.0,1.0,0.20,0.05)
try: result=recommend_retention(customers[["customerID","churn_probability","clv"]],offer_cost,save_rate)
except Exception as e: st.error(f"Could not score customers: {e}"); st.stop()
cols=st.columns(4)
cols[0].metric("Customers",f"{len(result):,}")
cols[1].metric("High churn risk",f"{int((result.churn_probability>=.60).sum()):,}")
cols[2].metric("Total estimated CLV",f"{result.clv.sum():,.2f}")
cols[3].metric("At-risk value",f"{result.expected_at_risk_value.sum():,.2f}")
st.subheader("Retention recommendations")
st.dataframe(result,width="stretch",hide_index=True)
st.download_button("Download recommendations CSV",result.to_csv(index=False).encode(),"retention_recommendations.csv","text/csv")
x,y=st.columns(2)
with x: st.subheader("Customers by risk band"); st.bar_chart(result.risk_band.value_counts().reindex(["Low","Medium","High"]).fillna(0))
with y: st.subheader("At-risk value by recommended action"); st.bar_chart(result.groupby("recommended_action",observed=False).expected_at_risk_value.sum())
st.subheader("Offer economics — scenario, not measured impact")
st.metric("Estimated net value under assumptions",f"{result.estimated_net_value.sum():,.2f}")
st.caption("This prioritization heuristic is not proof that an offer causes retention. Causal/uplift modeling requires valid historical treatment assignment and outcomes.")
