"""Leakage-aware churn-model training, evaluation, artifact persistence, and inference."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import FEATURE_DATA_PATH, REPORTS_DIR, ROOT_DIR

RANDOM_STATE = 42
TARGET_COLUMN = "Churn"
ID_COLUMNS = ["customerID"]
# TotalCharges is cumulative billing and highly correlated with tenure in this one-time
# snapshot. Exclude it and its derived ratio in the primary model to reduce timestamp
# ambiguity and avoid letting cumulative billing dominate the churn signal.
EXCLUDED_FEATURE_COLUMNS = {TARGET_COLUMN, *ID_COLUMNS, "TotalCharges", "avg_historical_monthly_charge"}
MODEL_DIR = ROOT_DIR / "models"
FIGURE_DIR = REPORTS_DIR / "figures"
COMPARISON_PATH = REPORTS_DIR / "churn_model_comparison.csv"
TEST_METRICS_PATH = REPORTS_DIR / "churn_test_metrics.json"
THRESHOLD_PATH = REPORTS_DIR / "churn_threshold_analysis.csv"
IMPORTANCE_PATH = REPORTS_DIR / "churn_feature_importance.csv"
MODEL_PATH = MODEL_DIR / "churn_model.joblib"
METADATA_PATH = MODEL_DIR / "churn_model_metadata.json"


def read_feature_data(path: Path = FEATURE_DATA_PATH) -> pd.DataFrame:
    """Read the Part 1 engineered CSV, running Part 1 when its output is absent."""
    if not path.exists():
        if path != FEATURE_DATA_PATH:
            raise FileNotFoundError(f"Feature dataset not found: {path}")
        from src.run_part1 import main as run_part1
        print("Part 1 feature data not found; running the Part 1 pipeline first...")
        run_part1()
    df = pd.read_csv(path)
    required = {TARGET_COLUMN, "customerID"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Feature dataset is missing required columns: {sorted(missing)}")
    if df[TARGET_COLUMN].isna().any():
        raise ValueError("Churn target contains missing values.")
    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="raise").astype(int)
    unique_labels = set(df[TARGET_COLUMN].unique())
    if unique_labels != {0, 1}:
        raise ValueError(f"Expected binary Churn labels 0/1; found {sorted(unique_labels)}")
    if df["customerID"].duplicated().any():
        raise ValueError("Duplicate customerID values found; resolve duplicate records before modeling.")
    return df


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Separate target and approved predictors; never allow ID/target as model inputs."""
    feature_columns = [col for col in df.columns if col not in EXCLUDED_FEATURE_COLUMNS]
    if not feature_columns:
        raise ValueError("No usable feature columns remain after exclusions.")
    X = df[feature_columns].copy()
    y = df[TARGET_COLUMN].astype(int).copy()
    if X.shape[1] == 0:
        raise ValueError("Predictor matrix is empty.")
    return X, y, feature_columns


def make_preprocessor(X: pd.DataFrame, scale_numeric: bool) -> ColumnTransformer:
    numeric_columns = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_columns = [col for col in X.columns if col not in numeric_columns]
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric_columns:
        numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median", add_indicator=True))]
        if scale_numeric:
            numeric_steps.append(("scaler", StandardScaler()))
        transformers.append(("numeric", Pipeline(numeric_steps), numeric_columns))
    if categorical_columns:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append(("categorical", categorical_pipeline, categorical_columns))
    if not transformers:
        raise ValueError("No numeric or categorical predictors available.")
    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=True)


def build_models(X: pd.DataFrame) -> dict[str, Pipeline]:
    """Return candidate pipelines with preprocessing fitted only on the training fold."""
    logistic = Pipeline([
        ("preprocessor", make_preprocessor(X, scale_numeric=True)),
        ("classifier", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="liblinear",
            random_state=RANDOM_STATE,
        )),
    ])
    lightgbm = Pipeline([
        ("preprocessor", make_preprocessor(X, scale_numeric=False)),
        ("classifier", LGBMClassifier(
            objective="binary",
            n_estimators=400,
            learning_rate=0.03,
            num_leaves=15,
            min_child_samples=25,
            reg_lambda=1.0,
            colsample_bytree=0.9,
            subsample=0.9,
            subsample_freq=1,
            importance_type="gain",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=-1,
        )),
    ])
    return {"logistic_regression": logistic, "lightgbm": lightgbm}


def score_probabilities(y_true: pd.Series | np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float | int]:
    y_pred = (np.asarray(probabilities) >= threshold).astype(int)
    y_true_array = np.asarray(y_true).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true_array, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true_array, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_array, y_pred)),
        "precision": float(precision_score(y_true_array, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true_array, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "f1": float(f1_score(y_true_array, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true_array, probabilities)),
        "pr_auc": float(average_precision_score(y_true_array, probabilities)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def choose_threshold(y_true: pd.Series | np.ndarray, probabilities: np.ndarray) -> tuple[float, pd.DataFrame]:
    """Choose a threshold by best validation F1 (not by evaluating the test set)."""
    rows: list[dict[str, float]] = []
    for threshold in np.round(np.arange(0.05, 0.951, 0.01), 2):
        metric = score_probabilities(y_true, probabilities, float(threshold))
        rows.append({"threshold": float(threshold), "precision": float(metric["precision"]),
                     "recall": float(metric["recall"]), "f1": float(metric["f1"]),
                     "balanced_accuracy": float(metric["balanced_accuracy"])})
    table = pd.DataFrame(rows)
    best_rows = table[table["f1"] == table["f1"].max()]
    # Resolve F1 ties in favor of higher recall for a churn-risk screening use case.
    best = best_rows.sort_values(["recall", "threshold"], ascending=[False, True]).iloc[0]
    return float(best["threshold"]), table


def _save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def _plot_results(y_test: pd.Series, probabilities: np.ndarray, threshold: float, model_name: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    y_pred = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    image = ax.imshow(matrix, interpolation="nearest")
    ax.figure.colorbar(image, ax=ax)
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Stayed", "Churned"],
           yticklabels=["Stayed", "Churned"], ylabel="Actual", xlabel="Predicted",
           title=f"Held-out test confusion matrix\n{model_name} | threshold={threshold:.2f}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{matrix[i, j]:,}", ha="center", va="center",
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "part2_confusion_matrix.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    PrecisionRecallDisplay.from_predictions(y_test, probabilities, ax=ax, name=model_name)
    ax.set_title("Held-out test precision-recall curve")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "part2_precision_recall_curve.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    RocCurveDisplay.from_predictions(y_test, probabilities, ax=ax, name=model_name)
    ax.set_title("Held-out test ROC curve")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "part2_roc_curve.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _write_feature_importance(pipeline: Pipeline) -> pd.DataFrame:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    IMPORTANCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    preprocessor = pipeline.named_steps["preprocessor"]
    estimator = pipeline.named_steps["classifier"]
    names = preprocessor.get_feature_names_out()
    if hasattr(estimator, "feature_importances_"):
        importance = np.asarray(estimator.feature_importances_, dtype=float)
        table = pd.DataFrame({"feature": names, "importance": importance, "signed_effect": np.nan,
                              "importance_method": "LightGBM gain"})
    else:
        coefficients = np.asarray(estimator.coef_).reshape(-1)
        table = pd.DataFrame({"feature": names, "importance": np.abs(coefficients),
                              "signed_effect": coefficients, "importance_method": "absolute logistic coefficient"})
    table = table.sort_values("importance", ascending=False).reset_index(drop=True)
    table.to_csv(IMPORTANCE_PATH, index=False)

    top = table.head(20).sort_values("importance", ascending=True)
    fig, ax = plt.subplots(figsize=(9, max(5, len(top) * 0.32)))
    ax.barh(top["feature"].astype(str), top["importance"].astype(float))
    ax.set(title="Top model feature importances (not causal effects)", xlabel="Importance", ylabel="Model feature")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "part2_feature_importance.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return table


def train_and_evaluate(df: pd.DataFrame) -> dict[str, Any]:
    """Fit candidates on train, select on validation PR-AUC, test once, and persist outputs."""
    X, y, feature_columns = prepare_features(df)
    X_train, X_remaining, y_train, y_remaining = train_test_split(
        X, y, test_size=0.40, random_state=RANDOM_STATE, stratify=y
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_remaining, y_remaining, test_size=0.50, random_state=RANDOM_STATE, stratify=y_remaining
    )
    if min(y_train.value_counts().min(), y_valid.value_counts().min(), y_test.value_counts().min()) < 2:
        raise ValueError("Not enough examples in each class for stratified train/validation/test split.")

    comparison_rows: list[dict[str, Any]] = []
    fitted: dict[str, Pipeline] = {}
    validation_probabilities: dict[str, np.ndarray] = {}
    for name, pipeline in build_models(X_train).items():
        print(f"Training {name}...")
        start = time.perf_counter()
        pipeline.fit(X_train, y_train)
        elapsed = time.perf_counter() - start
        fitted[name] = pipeline
        probabilities = pipeline.predict_proba(X_valid)[:, 1]
        validation_probabilities[name] = probabilities
        metrics = score_probabilities(y_valid, probabilities, 0.5)
        comparison_rows.append({"model": name, "training_seconds": round(elapsed, 3),
                                **{f"validation_{key}": value for key, value in metrics.items()}})

    comparison = pd.DataFrame(comparison_rows).sort_values("validation_pr_auc", ascending=False).reset_index(drop=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(COMPARISON_PATH, index=False)
    selected_name = str(comparison.iloc[0]["model"])
    selected_pipeline = fitted[selected_name]
    selected_valid_probabilities = validation_probabilities[selected_name]
    selected_threshold, threshold_table = choose_threshold(y_valid, selected_valid_probabilities)
    threshold_table.to_csv(THRESHOLD_PATH, index=False)

    # The test set is used only for final evaluation after candidate selection and threshold selection.
    test_probabilities = selected_pipeline.predict_proba(X_test)[:, 1]
    test_at_default = score_probabilities(y_test, test_probabilities, 0.5)
    test_at_selected = score_probabilities(y_test, test_probabilities, selected_threshold)
    test_results = {
        "selected_model": selected_name,
        "selection_metric": "validation PR-AUC (average precision)",
        "threshold_selection": "maximum validation F1; recall breaks ties; test data not used to select threshold",
        "selected_threshold": selected_threshold,
        "data_split": {"train_rows": int(len(X_train)), "validation_rows": int(len(X_valid)), "test_rows": int(len(X_test)),
                       "train_fraction": round(len(X_train) / len(X), 4),
                       "validation_fraction": round(len(X_valid) / len(X), 4),
                       "test_fraction": round(len(X_test) / len(X), 4),
                       "random_state": RANDOM_STATE, "stratified": True},
        "features_used": feature_columns,
        "excluded_from_features": sorted(EXCLUDED_FEATURE_COLUMNS),
        "validation_metrics_at_0_5": score_probabilities(y_valid, selected_valid_probabilities, 0.5),
        "test_metrics_at_0_5": test_at_default,
        "test_metrics_at_selected_threshold": test_at_selected,
        "dataset_rows": int(len(df)),
        "churn_rate": float(y.mean()),
        "caveats": [
            "This public dataset is a single snapshot, not a prospective time-series validation cohort.",
            "Test metrics estimate performance on a random stratified holdout from this dataset, not future production performance.",
            "Feature importance is associative/model-based and not causal evidence.",
            "TotalCharges and avg_historical_monthly_charge are excluded in the primary model to reduce cumulative-billing timestamp ambiguity.",
        ],
    }
    _save_json(test_results, TEST_METRICS_PATH)

    metadata = {
        "model_name": selected_name,
        "target_column": TARGET_COLUMN,
        "positive_label": 1,
        "positive_label_meaning": "Churn = Yes",
        "feature_columns": feature_columns,
        "excluded_feature_columns": sorted(EXCLUDED_FEATURE_COLUMNS),
        "decision_threshold": selected_threshold,
        "selection_metric": "validation PR-AUC / average precision",
        "validation_threshold_objective": "maximum F1; recall breaks ties",
        "random_state": RANDOM_STATE,
        "dataset_source": "IBM Telco Customer Churn public sample",
        "test_metrics_at_selected_threshold": test_at_selected,
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(selected_pipeline, MODEL_PATH)
    _save_json(metadata, METADATA_PATH)
    importance = _write_feature_importance(selected_pipeline)
    _plot_results(y_test, test_probabilities, selected_threshold, selected_name)
    _write_model_card(test_results, comparison, importance)

    print("\nPART 2 COMPLETE")
    print(f"Selected model: {selected_name}")
    print(f"Selected validation threshold: {selected_threshold:.2f}")
    print(f"Held-out test PR-AUC: {test_at_selected['pr_auc']:.4f}")
    print(f"Held-out test recall: {test_at_selected['recall']:.4f}")
    print(f"Held-out test precision: {test_at_selected['precision']:.4f}")
    print(f"Model saved: {MODEL_PATH.resolve()}")
    print(f"Metrics saved: {TEST_METRICS_PATH.resolve()}")
    return {"selected_model": selected_name, "threshold": selected_threshold,
            "test_metrics": test_results, "comparison": comparison}


def _write_model_card(results: dict[str, Any], comparison: pd.DataFrame, importance: pd.DataFrame) -> None:
    test = results["test_metrics_at_selected_threshold"]
    default = results["test_metrics_at_0_5"]
    top_features = importance.head(10)[["feature", "importance"]].to_dict("records")
    feature_lines = "\n".join(f"- `{item['feature']}` — {item['importance']:.4g}" for item in top_features)
    comparison_table = comparison.to_markdown(index=False, floatfmt=".4f")
    content = f"""# Part 2 — Churn Model Card\n\n## Purpose\nClassify customers with observed churn labels as a churn-risk screening aid. This is not a guarantee that a particular customer will leave.\n\n## Dataset and split\n- Rows: {results['dataset_rows']:,}\n- Observed churn rate: {results['churn_rate']:.2%}\n- Split: stratified 60% train / 20% validation / 20% held-out test, random state {RANDOM_STATE}.\n- Candidate selection: validation PR-AUC (average precision). The test set was not used for candidate selection or threshold tuning.\n\n## Validation model comparison\n\n{comparison_table}\n\n## Final held-out test metrics\n- Selected model: `{results['selected_model']}`\n- Threshold selected using validation F1: {results['selected_threshold']:.2f}\n- Precision at selected threshold: {test['precision']:.3f}\n- Recall at selected threshold: {test['recall']:.3f}\n- F1 at selected threshold: {test['f1']:.3f}\n- PR-AUC: {test['pr_auc']:.3f}\n- ROC-AUC: {test['roc_auc']:.3f}\n- Balanced accuracy: {test['balanced_accuracy']:.3f}\n- At default threshold 0.50, precision={default['precision']:.3f}, recall={default['recall']:.3f}, F1={default['f1']:.3f}.\n\n## Top model features\n\n{feature_lines}\n\nImportance reflects how the fitted model uses features; it is not proof that changing a feature will change churn.\n\n## Files generated\n- `models/churn_model.joblib`: trained preprocessing + classifier pipeline.\n- `models/churn_model_metadata.json`: input schema, threshold and model metadata.\n- `reports/churn_model_comparison.csv`: candidate validation comparison.\n- `reports/churn_test_metrics.json`: split details and final test metrics.\n- `reports/churn_threshold_analysis.csv`: validation-only threshold scan.\n- `reports/churn_feature_importance.csv`: encoded feature importance.\n- `reports/figures/part2_*.png`: plots.\n\n## Intended use and limitations\n- Use as a decision-support demonstration; validate on current, temporally separated company data before operational use.\n- This dataset is a single snapshot. Random holdout metrics do not measure true future-period generalization.\n- The data does not establish causal impact from retention offers.\n- `customerID`, `Churn`, `TotalCharges`, and `avg_historical_monthly_charge` are excluded from predictors. Cumulative billing may reflect tenure and a decision-time ambiguity; exclusion is a deliberate conservative choice for this baseline.\n- The classification threshold is tuned on validation data for F1, then locked before test scoring. Business stakeholders should select a threshold based on outreach costs and capacity in a real deployment.\n- Probabilities are not guaranteed calibrated; calibrate on representative validation data before interpreting them as true event probabilities.\n"""
    (REPORTS_DIR / "PART2_MODEL_CARD.md").write_text(content, encoding="utf-8")


def predict_dataframe(df: pd.DataFrame, model_path: Path = MODEL_PATH, metadata_path: Path = METADATA_PATH) -> pd.DataFrame:
    """Predict churn risk for customer rows from the Part 1 feature schema."""
    if not model_path.exists() or not metadata_path.exists():
        raise FileNotFoundError("Model artifacts are missing. Run `python -m src.train_churn_model` first.")
    model = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    features = metadata["feature_columns"]
    missing = [col for col in features if col not in df.columns]
    if missing:
        raise ValueError(f"Input data is missing model features: {missing}")
    X = df[features].copy()
    probabilities = model.predict_proba(X)[:, 1]
    threshold = float(metadata["decision_threshold"])
    output = df[[col for col in ID_COLUMNS if col in df.columns]].copy()
    output["churn_probability"] = probabilities
    output["predicted_churn"] = (probabilities >= threshold).astype(int)
    output["risk_band"] = pd.cut(
        output["churn_probability"], bins=[-np.inf, 0.25, 0.50, np.inf],
        labels=["Low", "Medium", "High"], right=False,
    ).astype(str)
    output["decision_threshold"] = threshold
    return output
