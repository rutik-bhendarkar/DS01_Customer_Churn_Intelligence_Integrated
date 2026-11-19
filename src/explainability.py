"""SHAP-based explainability for the DS-01 churn prediction pipeline.

Reuses the *exact* trained artifact produced by ``src.churn_model``
(``models/churn_model.joblib`` = preprocessing pipeline + classifier) and its
metadata. SHAP values explain the churn probability of the positive class
(Churn = Yes, ``positive_label == 1``) and never fabricate contributions.

SHAP value space:
  * Tree estimators (LightGBM): ``model_output="probability"`` -> values are in
    probability space and sum to ``predicted_probability - base_probability``.
    Contributions can be described as percentage-point effects on churn risk.
  * Linear estimators (LogisticRegression): values are in log-odds space. They
    indicate direction and relative magnitude only; they are *not* percentage
    point changes and are labelled as such.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st

from src.churn_model import (
    EXCLUDED_FEATURE_COLUMNS,
    ID_COLUMNS,
    METADATA_PATH,
    MODEL_PATH,
    TARGET_COLUMN,
)
from src.config import REPORTS_DIR

# RevenuePulse AI palette: red increases churn risk, green reduces it,
# blue is baseline / neutral.
RED = "#ef4444"
GREEN = "#10b981"
BLUE = "#3b82f6"
NEUTRAL = "#94a3b8"

MAX_GLOBAL_SAMPLES = 200
MAX_WATERFALL_FEATURES = 10
RISK_BINS = [-0.000001, 0.30, 0.60, 1.000001]
RISK_LABELS = ["Low", "Medium", "High"]

# In-memory, version-keyed caches (Streamlit reruns keep module state alive
# within a session). Keys are model/data version strings so cached results are
# invalidated whenever the artifact or dataset changes.
_model_package_cache: dict[str, dict[str, Any]] = {}
_explainer_cache: dict[str, Any] = {}
_background_cache: dict[str, pd.DataFrame] = {}
_explanation_cache: dict[str, dict[str, Any]] = {}
_global_explanation_cache: dict[str, dict[str, Any]] = {}


class ExplainabilityError(RuntimeError):
    """Raised when an explanation cannot be produced from available artifacts."""


def _file_hash(path: Path | None) -> str:
    if path is None or not path.exists():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def register_background(data_version: str, customers_df: pd.DataFrame) -> None:
    """Register the customer frame used as SHAP background for a data version."""
    _background_cache[data_version] = customers_df.copy()


def model_version_key(model_path: Path = MODEL_PATH, metadata_path: Path = METADATA_PATH) -> str:
    """Version key that changes whenever the model artifact or metadata changes."""
    return f"{_file_hash(model_path)}.{_file_hash(metadata_path)}"


def data_version_key(customers_df: pd.DataFrame) -> str:
    """Version key for the customer dataset used as explanation background."""
    h = hashlib.sha256()
    h.update(str(len(customers_df)).encode("utf-8"))
    if "customerID" in customers_df.columns:
        for cid in customers_df["customerID"].astype(str).head(50).tolist():
            h.update(f"|{cid}".encode("utf-8"))
    for col in ("churn_probability", "clv"):
        if col in customers_df.columns:
            total = float(pd.to_numeric(customers_df[col], errors="coerce").fillna(0.0).sum())
            h.update(f"|{col}:{total:.6f}".encode("utf-8"))
    return h.hexdigest()[:16]


def _detect_estimator_kind(classifier: Any) -> str:
    name = type(classifier).__name__.lower()
    tree_hints = ("lgbm", "xgb", "randomforest", "decisiontree", "extratrees", "histgradient")
    linear_hints = ("logisticregression", "sgdclassifier", "linearsvc", "ridgeclassifier", "passiveaggressive")
    if any(hint in name for hint in tree_hints):
        return "tree"
    if any(hint in name for hint in linear_hints):
        return "linear"
    return "other"


def _output_widths_per_column(transformer: Any, columns: list[str]) -> list[int]:
    """Number of transformed output columns produced per input column."""
    onehot = None
    if hasattr(transformer, "named_steps"):
        for step in transformer.named_steps.values():
            if type(step).__name__ == "OneHotEncoder":
                onehot = step
                break
    if onehot is not None and hasattr(onehot, "categories_"):
        return [len(cats) for cats in onehot.categories_]
    return [1] * len(columns)


def _build_original_feature_map(preprocessor: Any, transformed_names: list[str]) -> dict[str, str]:
    """Map each transformed feature name back to its original input column.

    Position-based: ColumnTransformer emits transformed columns block by block
    (one block per transformer), so widths from the fitted encoders let us
    assign every output column to an original input column without parsing
    generated names. Missing-indicator columns left over after the per-column
    widths are exhausted keep their transformed name as the label.
    """
    original_of: dict[str, str] = {}
    offset = 0
    for _tname, transformer, columns in getattr(preprocessor, "transformers_", []):
        if _tname == "remainder":
            continue
        cols = list(columns)
        try:
            out_names = [str(n) for n in transformer.get_feature_names_out(cols)]
        except Exception:
            out_names = [str(c) for c in cols]
        block = transformed_names[offset: offset + len(out_names)]
        offset += len(out_names)
        if len(block) != len(out_names):
            continue
        widths = _output_widths_per_column(transformer, cols)
        pos = 0
        for col, width in zip(cols, widths):
            for _j in range(width):
                if pos < len(block):
                    original_of[block[pos]] = str(col)
                pos += 1
        # Leftovers (e.g. imputation missingness indicators) keep own name.
        while pos < len(block):
            original_of[block[pos]] = block[pos]
            pos += 1
    return original_of


@st.cache_resource(show_spinner=False)

def load_explainability_model(
    model_path: Path = MODEL_PATH,
    metadata_path: Path = METADATA_PATH,
) -> dict[str, Any]:
    """Load the trained DS-01 churn pipeline and metadata. Never retrains."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Churn model artifact not found: {model_path}. "
            "Run `python -m src.train_churn_model` first."
        )
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Model metadata not found: {metadata_path}. "
            "Run `python -m src.train_churn_model` first."
        )
    model = joblib.load(model_path)
    if not hasattr(model, "named_steps") or "preprocessor" not in model.named_steps or "classifier" not in model.named_steps:
        raise ValueError("Saved artifact is not the expected preprocessing + classifier pipeline.")
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    feature_columns = [str(c) for c in metadata["feature_columns"]]
    positive_label = int(metadata.get("positive_label", 1))
    if positive_label != 1:
        raise ValueError(f"Explainability requires positive_label == 1; found {positive_label}.")
    estimator_kind = _detect_estimator_kind(classifier)
    if estimator_kind == "other":
        raise ValueError(
            f"Unsupported estimator type for SHAP explanations: {type(classifier).__name__}"
        )
    transformed_names = [str(n) for n in preprocessor.get_feature_names_out()]
    original_map = _build_original_feature_map(preprocessor, transformed_names)
    return {
        "model": model,
        "metadata": metadata,
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "feature_columns": feature_columns,
        "preprocessor": preprocessor,
        "classifier": classifier,
        "estimator_kind": estimator_kind,
        "transformed_feature_names": transformed_names,
        "original_feature_of": original_map,
        "value_space": "probability" if estimator_kind == "tree" else "log_odds",
        "model_version": model_version_key(model_path, metadata_path),
        "positive_label": positive_label,
        "decision_threshold": float(metadata.get("decision_threshold", 0.5)),
    }

def load_explainability_model_cached(
    model_path_str: str = str(MODEL_PATH),
    metadata_path_str: str = str(METADATA_PATH),
) -> dict[str, Any]:
    """Cached loader for the explainability model (resource cached per session)."""
    package = load_explainability_model(Path(model_path_str), Path(metadata_path_str))
    _model_package_cache[package["model_version"]] = package
    return package


def _transform_frame(preprocessor: Any, X: pd.DataFrame) -> np.ndarray:
    """Transform raw features through the *training* preprocessor and densify."""
    Xt = preprocessor.transform(X)
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()
    return np.asarray(Xt, dtype=float)


def create_shap_explainer(
    package: dict[str, Any],
    background_X: pd.DataFrame,
) -> tuple[Any, np.ndarray]:
    """Create a SHAP explainer matched to the trained estimator."""
    estimator_kind = package["estimator_kind"]
    if estimator_kind not in ("tree", "linear"):
        raise ExplainabilityError(
            f"No SHAP explainer supported for estimator kind '{estimator_kind}'."
        )
    preprocessor = package["preprocessor"]
    classifier = package["classifier"]
    X_trans = _transform_frame(preprocessor, background_X)
    if estimator_kind == "tree":
        explainer = shap.TreeExplainer(
            classifier,
            X_trans,
            feature_perturbation="tree_path_dependent",
            model_output="probability",
        )
    else:
        explainer = shap.LinearExplainer(
            classifier,
            X_trans,
            feature_perturbation="interventional",
        )
    return explainer, X_trans


def _get_cached_explainer(package, data_version, background_X):
    key = f"{package['model_version']}.{data_version}"
    if key not in _explainer_cache:
        explainer, _ = create_shap_explainer(package, background_X)
        _explainer_cache[key] = explainer
    return _explainer_cache[key]


def _raw_shap_values(explainer, X_trans_row, n_features):
    sv = explainer.shap_values(X_trans_row)
    sv = np.asarray(sv, dtype=float)
    if sv.ndim == 1:
        sv = sv.reshape(1, -1)
    if sv.ndim == 3:
        classes = getattr(explainer, "classes_", None)
        if classes is not None and 1 in classes:
            pos_idx = int(list(classes).index(1))
        else:
            pos_idx = sv.shape[1] - 1
        sv = sv[:, pos_idx, :]
    if sv.shape != (1, n_features):
        raise ExplainabilityError(
            f"Unexpected SHAP output shape {sv.shape}; expected (1, {n_features})."
        )
    return sv


def _aggregate_contributions(shap_values_row, transformed_names, original_map, actual_row):
    aggregated = {}
    for tname, contribution in zip(transformed_names, shap_values_row):
        original = original_map.get(tname, tname)
        aggregated[original] = aggregated.get(original, 0.0) + float(contribution)
    contributions = []
    for feature, contribution in aggregated.items():
        actual_value = actual_row.get(feature, np.nan) if feature in actual_row.index else np.nan
        contributions.append(
            {
                "feature": str(feature),
                "actual_value": actual_value,
                "contribution": float(contribution),
                "abs_contribution": abs(float(contribution)),
            }
        )
    contributions.sort(key=lambda item: item["abs_contribution"], reverse=True)
    return contributions


def explain_customer(
    customer_id: str,
    customers_df: pd.DataFrame,
    package: dict[str, Any],
    data_version: str,
) -> dict[str, Any]:
    """Explain the churn prediction for a single customer."""
    feature_columns = package["feature_columns"]
    missing = [c for c in feature_columns if c not in customers_df.columns]
    if missing:
        raise ExplainabilityError(
            f"Customer data is missing model features: {missing}. "
            "Regenerate Parts 1-3 outputs or upload a dataset with the full feature schema."
        )
    matches = customers_df[customers_df["customerID"].astype(str) == str(customer_id)]
    if matches.empty:
        raise ExplainabilityError(f"Customer '{customer_id}' not found in the dataset.")
    if len(matches) > 1:
        raise ExplainabilityError(f"Customer '{customer_id}' is not unique in the dataset.")
    row = matches.iloc[0]

    X = row[feature_columns].to_frame().T
    X_trans = _transform_frame(package["preprocessor"], X)
    n_features = len(package["transformed_feature_names"])

    probability = float(package["model"].predict_proba(X)[0, 1])
    threshold = float(package["decision_threshold"])
    predicted_class = int(probability >= threshold)
    risk_category = str(
        pd.cut(
            pd.Series([probability]),
            bins=RISK_BINS,
            labels=RISK_LABELS,
            include_lowest=True,
        ).iloc[0]
    )

    background = customers_df[feature_columns]
    explainer = _get_cached_explainer(package, data_version, background)
    base_value = float(getattr(explainer, "expected_value", 0.0))
    if hasattr(base_value, "__len__") and not isinstance(base_value, (str, bytes)):
        base_value = float(np.asarray(base_value).reshape(-1)[0])

    shap_values_row = _raw_shap_values(explainer, X_trans, n_features)[0]
    contributions = _aggregate_contributions(
        shap_values_row,
        package["transformed_feature_names"],
        package["original_feature_of"],
        row,
    )

    raw = {
        name: float(value)
        for name, value in zip(package["transformed_feature_names"], shap_values_row)
    }

    return {
        "customer_id": str(customer_id),
        "churn_probability": probability,
        "model_output": probability,
        "predicted_class": predicted_class,
        "decision_threshold": threshold,
        "risk_category": risk_category,
        "base_value": base_value,
        "value_space": package["value_space"],
        "total_contribution": float(shap_values_row.sum()),
        "contributions": contributions,
        "top_positive": [c for c in contributions if c["contribution"] > 0],
        "top_negative": [c for c in contributions if c["contribution"] < 0],
        "raw_shap_values": raw,
        "transformed_feature_names": package["transformed_feature_names"],
        "actual_values": {str(k): (None if pd.isna(v) else v) for k, v in row[feature_columns].items()},
    }


@st.cache_data(show_spinner=False)
def explain_customer_cached(
    customer_id: str,
    model_version: str,
    data_version: str,
) -> dict[str, Any]:
    """Cached per-customer explanation (invalidated when model or data changes)."""
    package = _model_package_cache[model_version]
    background = _background_cache[data_version]
    return explain_customer(customer_id, background, package, data_version)


def get_customer_explanation_summary(explanation: dict[str, Any]) -> list[str]:
    """Plain-language, explanation-grounded summary bullets."""
    lines: list[str] = []
    space = explanation["value_space"]
    unit = "percentage points" if space == "probability" else "log-odds"
    p = explanation["churn_probability"]
    lines.append(
        f"Customer **{explanation['customer_id']}** has a predicted churn probability of "
        f"**{p:.1%}** (risk category: **{explanation['risk_category']}**)."
    )
    if explanation["top_positive"]:
        items = []
        for c in explanation["top_positive"][:5]:
            val = c["actual_value"]
            val_text = f"{val:.2f}" if isinstance(val, (int, float, np.floating, np.integer)) else str(val)
            items.append(f"**{c['feature']}** (value: {val_text}) adds {c['contribution']:+.3f} {unit}")
        lines.append("Factors *increasing* churn risk for this customer:")
        lines.extend(f"- {item}" for item in items)
    else:
        lines.append("No material factors increase this customer's churn risk according to the explanation.")
    if explanation["top_negative"]:
        items = []
        for c in explanation["top_negative"][:5]:
            val = c["actual_value"]
            val_text = f"{val:.2f}" if isinstance(val, (int, float, np.floating, np.integer)) else str(val)
            items.append(f"**{c['feature']}** (value: {val_text}) reduces risk by {abs(c['contribution']):.3f} {unit}")
        lines.append("Factors *reducing* churn risk for this customer:")
        lines.extend(f"- {item}" for item in items)
    else:
        lines.append("No material factors reduce this customer's churn risk according to the explanation.")
    strongest = max(explanation["contributions"], key=lambda c: c["abs_contribution"]) if explanation["contributions"] else None
    if strongest:
        direction = "increases" if strongest["contribution"] > 0 else "reduces"
        lines.append(
            f"Strongest contributor: **{strongest['feature']}**, which {direction} churn risk "
            f"({strongest['contribution']:+.3f} {unit})."
        )
    lines.append(
        "These contributions are derived from the fitted model's SHAP values: they are "
        "associational explanations of a prediction, not causal evidence that changing "
        "a feature will change churn."
    )
    return lines


def explain_global_model(
    package: dict[str, Any],
    customers_df: pd.DataFrame,
    data_version: str,
    n_samples: int = MAX_GLOBAL_SAMPLES,
) -> dict[str, Any]:
    """Global explanation: aggregate SHAP impact across a background sample."""
    feature_columns = package["feature_columns"]
    missing = [c for c in feature_columns if c not in customers_df.columns]
    if missing:
        raise ExplainabilityError(f"Customer data is missing model features: {missing}.")
    background = customers_df.dropna(subset=feature_columns)
    if background.empty:
        raise ExplainabilityError("No complete customer rows available for a global explanation.")
    sample = background.sample(n=min(n_samples, len(background)), random_state=42)
    X_trans = _transform_frame(package["preprocessor"], sample[feature_columns])
    n_features = len(package["transformed_feature_names"])

    explainer = _get_cached_explainer(package, data_version, sample[feature_columns])
    base_value = float(getattr(explainer, "expected_value", 0.0))
    if hasattr(base_value, "__len__") and not isinstance(base_value, (str, bytes)):
        base_value = float(np.asarray(base_value).reshape(-1)[0])

    shap_values = explainer.shap_values(X_trans)
    shap_values = np.asarray(shap_values, dtype=float)
    if shap_values.ndim == 3:
        classes = getattr(explainer, "classes_", None)
        pos_idx = int(list(classes).index(1)) if classes is not None and 1 in classes else shap_values.shape[1] - 1
        shap_values = shap_values[:, pos_idx, :]
    if shap_values.ndim == 1:
        shap_values = shap_values.reshape(1, -1)
    if shap_values.shape[1] != n_features:
        raise ExplainabilityError(
            f"Global SHAP output has {shap_values.shape[1]} features; expected {n_features}."
        )

    mean_abs = np.abs(shap_values).mean(axis=0)
    aggregated: dict[str, float] = {}
    for tname, value in zip(package["transformed_feature_names"], mean_abs):
        original = package["original_feature_of"].get(tname, tname)
        aggregated[original] = aggregated.get(original, 0.0) + float(value)

    feature_importance = pd.DataFrame(
        [
            {"feature": feature, "mean_abs_shap": value}
            for feature, value in aggregated.items()
        ]
    ).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    feature_details = []
    for feature in feature_importance["feature"].head(15):
        tcols = [t for t, o in package["original_feature_of"].items() if o == feature]
        if not tcols:
            continue
        tcol = tcols[0]
        tcol_idx = package["transformed_feature_names"].index(tcol)
        feature_details.append(
            {
                "feature": str(feature),
                "shap_values": shap_values[:, tcol_idx].tolist(),
                "feature_values": sample[feature].tolist() if feature in sample.columns else [np.nan] * len(sample),
            }
        )

    probabilities = package["model"].predict_proba(sample[feature_columns])[:, 1]
    high_mask = probabilities >= float(np.median(probabilities))
    high_rows = sample.loc[high_mask]
    low_rows = sample.loc[~high_mask]
    contrast_rows = []
    for feature in feature_importance["feature"].head(10):
        if feature not in sample.columns:
            continue
        high_mean = pd.to_numeric(high_rows[feature], errors="coerce").mean()
        low_mean = pd.to_numeric(low_rows[feature], errors="coerce").mean()
        contrast_rows.append(
            {
                "feature": str(feature),
                "high_risk_mean": None if pd.isna(high_mean) else float(high_mean),
                "low_risk_mean": None if pd.isna(low_mean) else float(low_mean),
            }
        )

    return {
        "base_value": base_value,
        "value_space": package["value_space"],
        "n_customers": int(len(sample)),
        "feature_importance": feature_importance,
        "feature_details": feature_details,
        "median_probability": float(np.mean(probabilities)),
        "high_vs_low_risk": contrast_rows,
        "shap_values": shap_values.tolist(),
        "transformed_feature_names": package["transformed_feature_names"],
    }


@st.cache_data(show_spinner=False)
def explain_global_model_cached(
    model_version: str,
    data_version: str,
    n_samples: int,
) -> dict[str, Any]:
    """Cached global explanation (invalidated when model or data changes)."""
    package = _model_package_cache[model_version]
    background = _background_cache[data_version]
    return explain_global_model(package, background, data_version, n_samples)


def get_top_churn_drivers(global_explanation: dict[str, Any], top_k: int = 10) -> pd.DataFrame:
    """Return the most influential features from a global explanation."""
    importance = global_explanation["feature_importance"].copy()
    importance["rank"] = range(1, len(importance) + 1)
    return importance.head(top_k)[["rank", "feature", "mean_abs_shap"]]
