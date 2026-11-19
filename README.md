# DS-01 — Customer Lifetime Value and Churn Intelligence System

Unified VS Code project containing five distinct deliverables. Parts 1–4 are retained; Part 5 adds a customer intelligence dashboard and becomes the main application entry point. The previous Part 4 dashboard remains available as `app_part4.py`.

## Setup (Windows PowerShell)
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-modeling.txt
pip install -r requirements-dashboard.txt
```

## Run the integrated pipeline
```powershell
python run_all.py
```

Then launch the Part 5 dashboard:
```powershell
streamlit run app.py
```

The dashboard automatically joins Part 1 features, Part 2 churn probabilities and Part 3 CLV estimates by `customerID` when those outputs exist. Otherwise it displays clearly labelled illustrative sample data. You can upload a scored CSV with `customerID`, `churn_probability`, and `clv`.

## Run individual parts
```powershell
python -m src.run_part1
python -m src.train_churn_model
python -m src.predict_churn
python -m src.clv_model
streamlit run app_part4.py  # legacy Part 4 UI
streamlit run app.py        # Part 5 dashboard
```

## Tests
```powershell
python -m pytest -q
```

## Part map
1. Data collection, cleaning, EDA and feature engineering.
2. Customer churn prediction and evaluation.
3. CLV estimation and survival analysis.
4. Retention intelligence and scenario prioritization.
5. Interactive dashboard, KPIs, customer-level views, filters, charts and exports.

## Important limitations
The IBM Telco dataset is a cross-sectional snapshot. CLV is a survival-based estimate using current monthly charges, not transaction-history CLV. Retention save-rate and offer economics are scenario assumptions. Causal/uplift effects cannot be established without suitable treatment assignment and outcome data. Never report sample-data dashboard values as real model performance.
