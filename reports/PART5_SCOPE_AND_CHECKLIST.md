# Part 5 — Interactive Dashboard

## Goal
Turn outputs from Parts 1–3 into a usable Streamlit customer intelligence application. Part 5 is an additional deliverable and preserves Parts 1–4.

## Features
- Business KPIs: customer count, high-risk count/share, total estimated CLV, expected at-risk value and average churn probability.
- Customer-level probability, risk band, CLV and retention suggestion.
- Value segmentation into Standard, Valuable and Premium groups.
- Filters for churn risk, contract, minimum CLV and customer ID.
- Plotly charts: churn-risk distribution, CLV by risk, churn probability versus CLV, and value segment mix.
- Manual scenario form using an existing Part 2 probability and Part 3 CLV estimate.
- Download filtered customer data and retention action summary as CSV.

## Data contract
Required: `customerID`, `churn_probability` (0–1), `clv` (non-negative). Optional: `tenure`, `contract`, `MonthlyCharges`.

The main app automatically joins `data/processed/telco_customers_features.csv`, `reports/churn_predictions.csv` and `reports/clv_estimates.csv` by `customerID` when all exist. Otherwise it uses `data/sample/customer_scores_sample.csv` and labels it illustrative. A scored CSV can also be uploaded.

## Run
```powershell
pip install -r requirements-dashboard.txt
streamlit run app.py
```

## Test
```powershell
python -m pytest tests/part5 -q
```

## Acceptance checklist
- [ ] Run Parts 1–3 to generate actual scores and CLV estimates.
- [ ] Launch the main dashboard with `streamlit run app.py`.
- [ ] Confirm dashboard data source indicates integrated outputs, not sample data.
- [ ] Test all filters and individual customer selection.
- [ ] Download both CSV exports.
- [ ] Capture screenshots for the internship report.

## Limitations
Sample data is illustrative. CLV is a survival-based estimate from a cross-sectional dataset. Offer/save-rate values are scenario assumptions and not measured causal effects. Currency is displayed in ₹ by default; confirm business currency before external use.
