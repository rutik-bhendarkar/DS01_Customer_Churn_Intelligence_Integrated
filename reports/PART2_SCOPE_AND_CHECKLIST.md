# Part 2 Scope and Acceptance Checklist — Customer Churn Prediction

## Objective
Create a reproducible churn classification workflow using the Part 1 dataset, compare a transparent baseline to LightGBM, evaluate on held-out data, and save a reusable inference pipeline.

## In scope
- [x] Define `Churn=1` as the positive class.
- [x] Exclude identifier and target from predictors.
- [x] Exclude cumulative `TotalCharges` and its derived average-charge ratio in the primary baseline due to timestamp ambiguity.
- [x] Split stratified data into train/validation/test partitions.
- [x] Keep imputation, encoding and scaling inside sklearn pipelines to prevent preprocessing leakage.
- [x] Compare Logistic Regression and LightGBM using validation PR-AUC.
- [x] Tune a decision threshold on validation data, with F1 as the objective and recall as tie-breaker.
- [x] Report default and tuned-threshold test metrics.
- [x] Save model pipeline, metadata, comparison report, metrics, threshold scan, model card and plots.
- [x] Provide a CLI prediction script for new rows matching the feature schema.
- [x] Add unit/integration tests.

## To complete locally
- [ ] Run Part 1 to download the dataset and create `data/processed/telco_customers_features.csv`.
- [ ] Run `python -m src.train_churn_model` on the actual sample.
- [ ] Review validation comparison and held-out test metrics; do not claim quality based on code/test fixtures.
- [ ] Review confusion matrix for false negatives and false positives.
- [ ] Run prediction command and verify output rows.
- [ ] Run `python -m pytest -q` in the local environment.
- [ ] Record actual run environment and observed metrics in the internship submission.

## Exit criteria
Part 2 is complete once the model can be trained from the Part 1 output, artifacts are produced, a CSV can be scored with the saved pipeline, all tests pass, and the model card is updated/reviewed using the actual run outputs.
