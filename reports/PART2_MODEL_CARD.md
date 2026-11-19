# Part 2 — Churn Model Card

## Purpose
Classify customers with observed churn labels as a churn-risk screening aid. This is not a guarantee that a particular customer will leave.

## Dataset and split
- Rows: 7,043
- Observed churn rate: 26.54%
- Split: stratified 60% train / 20% validation / 20% held-out test, random state 42.
- Candidate selection: validation PR-AUC (average precision). The test set was not used for candidate selection or threshold tuning.

## Validation model comparison

| model               |   training_seconds |   validation_threshold |   validation_accuracy |   validation_balanced_accuracy |   validation_precision |   validation_recall |   validation_specificity |   validation_f1 |   validation_roc_auc |   validation_pr_auc |   validation_true_negative |   validation_false_positive |   validation_false_negative |   validation_true_positive |
|:--------------------|-------------------:|-----------------------:|----------------------:|-------------------------------:|-----------------------:|--------------------:|-------------------------:|----------------:|---------------------:|--------------------:|---------------------------:|----------------------------:|----------------------------:|---------------------------:|
| logistic_regression |             0.0590 |                 0.5000 |                0.7473 |                         0.7700 |                 0.5152 |              0.8182 |                   0.7217 |          0.6322 |               0.8584 |              0.6712 |                        747 |                         288 |                          68 |                        306 |
| lightgbm            |             0.2230 |                 0.5000 |                0.8070 |                         0.7175 |                 0.6747 |              0.5267 |                   0.9082 |          0.5916 |               0.8512 |              0.6693 |                        940 |                          95 |                         177 |                        197 |

## Final held-out test metrics
- Selected model: `logistic_regression`
- Threshold selected using validation F1: 0.60
- Precision at selected threshold: 0.540
- Recall at selected threshold: 0.709
- F1 at selected threshold: 0.613
- PR-AUC: 0.625
- ROC-AUC: 0.833
- Balanced accuracy: 0.745
- At default threshold 0.50, precision=0.506, recall=0.797, F1=0.619.

## Top model features

- `numeric__tenure` — 0.6256
- `categorical__Contract_Two year` — 0.562
- `categorical__InternetService_DSL` — 0.5368
- `categorical__InternetService_Fiber optic` — 0.5308
- `numeric__is_month_to_month` — 0.4405
- `categorical__tenure_group_13-24 months` — 0.2863
- `categorical__tenure_group_0-12 months` — 0.2846
- `categorical__Contract_One year` — 0.2819
- `numeric__MonthlyCharges` — 0.2676
- `categorical__PaperlessBilling_No` — 0.2487

Importance reflects how the fitted model uses features; it is not proof that changing a feature will change churn.

## Files generated
- `models/churn_model.joblib`: trained preprocessing + classifier pipeline.
- `models/churn_model_metadata.json`: input schema, threshold and model metadata.
- `reports/churn_model_comparison.csv`: candidate validation comparison.
- `reports/churn_test_metrics.json`: split details and final test metrics.
- `reports/churn_threshold_analysis.csv`: validation-only threshold scan.
- `reports/churn_feature_importance.csv`: encoded feature importance.
- `reports/figures/part2_*.png`: plots.

## Intended use and limitations
- Use as a decision-support demonstration; validate on current, temporally separated company data before operational use.
- This dataset is a single snapshot. Random holdout metrics do not measure true future-period generalization.
- The data does not establish causal impact from retention offers.
- `customerID`, `Churn`, `TotalCharges`, and `avg_historical_monthly_charge` are excluded from predictors. Cumulative billing may reflect tenure and a decision-time ambiguity; exclusion is a deliberate conservative choice for this baseline.
- The classification threshold is tuned on validation data for F1, then locked before test scoring. Business stakeholders should select a threshold based on outreach costs and capacity in a real deployment.
- Probabilities are not guaranteed calibrated; calibrate on representative validation data before interpreting them as true event probabilities.
