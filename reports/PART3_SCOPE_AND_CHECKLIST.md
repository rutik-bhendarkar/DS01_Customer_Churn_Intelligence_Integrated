# Part 3 — CLV & Survival Analysis

Implemented with Lifelines Kaplan-Meier. The model estimates conditional remaining survival over a configurable 60-month horizon and discounts expected monthly charges to form a transparent CLV proxy.

Run: `python -m src.clv_model`
Outputs: `reports/clv_estimates.csv`, `reports/survival_summary.json`.

Limitations: the source is a single cross-sectional snapshot, not longitudinal transaction history. Estimates are projections and should not be presented as observed lifetime revenue.
