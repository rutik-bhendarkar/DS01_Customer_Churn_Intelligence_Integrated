"""SHAP visualisations for the Explainable AI page.

All figures use the RevenuePulse AI dark theme. Colours:
  * red (#ef4444)   -> contribution increases churn risk
  * green (#10b981) -> contribution reduces churn risk
  * blue (#3b82f6)  -> baseline / neutral
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import plotly.graph_objects as go

from dashboard.components.charts import DARK_LAYOUT

RED = "#ef4444"
GREEN = "#10b981"
BLUE = "#3b82f6"
NEUTRAL = "#94a3b8"

MAX_WATERFALL_FEATURES = 10


def _unit_label(value_space: str) -> str:
    return "percentage points" if value_space == "probability" else "log-odds"


def _fmt_value(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.2f}"
    if isinstance(value, (int, np.integer)):
        return f"{int(value)}"
    return str(value)


def waterfall_chart(explanation: dict, max_features: int = MAX_WATERFALL_FEATURES, height: int = 520) -> go.Figure:
    """SHAP waterfall: base value + contributions -> model output.

    Contributions and the model output are in the same value space as the
    explainer (probability for tree models, log-odds for linear models).
    """
    value_space = explanation.get("value_space", "log_odds")
    unit = _unit_label(value_space)
    base = float(explanation["base_value"])
    contributions = sorted(
        explanation["contributions"], key=lambda c: abs(c["contribution"]), reverse=True
    )
    top = contributions[:max_features]

    measure = ["total"] + ["relative"] * len(top) + ["total"]
    x = ["Average churn risk (base value)"] + [c["feature"] for c in top] + ["Model output"]
    y = [base] + [c["contribution"] for c in top] + [base + float(explanation["total_contribution"])]
    text = [
        f"base: {base:+.3f} {unit}"
    ] + [
        f"{c['contribution']:+.3f} {unit} (value: {_fmt_value(c['actual_value'])})"
        for c in top
    ] + [
        f"output: {base + float(explanation['total_contribution']):+.3f} {unit}"
    ]

    fig = go.Figure(
        go.Waterfall(
            name="SHAP contribution",
            orientation="v",
            measure=measure,
            x=x,
            y=y,
            text=text,
            textposition="outside",
            textfont=dict(color="#f1f5f9", size=10),
            increasing=dict(marker=dict(color=RED)),
            decreasing=dict(marker=dict(color=GREEN)),
            totals=dict(marker=dict(color=BLUE)),
            connector=dict(line=dict(color=NEUTRAL)),
            hovertemplate="%{x}<br>%{text}<extra></extra>",
        )
    )
    fig.update_layout(
        height=height,
        **DARK_LAYOUT,
        title=dict(
            text=f"SHAP waterfall — contributions in {unit} ({value_space} space)",
            x=0.0,
            xanchor="left",
        ),
        yaxis=dict(title=f"Contribution ({unit})", gridcolor="#1f2a4a", tickformat="+.2f"),
        xaxis=dict(title="Feature", gridcolor="#1f2a4a"),
        showlegend=False,
    )
    return fig


def contribution_bar_chart(explanation: dict, height: int = 400) -> go.Figure:
    """Horizontal bar chart of the strongest SHAP contributors."""
    contributions = sorted(
        explanation["contributions"], key=lambda c: abs(c["contribution"]), reverse=True
    )[:12]
    if not contributions:
        fig = go.Figure()
        fig.update_layout(height=height, **DARK_LAYOUT, title=dict(text="No SHAP contributions available", x=0.0))
        return fig
    features = [c["feature"] for c in contributions][::-1]
    values = [c["contribution"] for c in contributions][::-1]
    colors = [RED if v > 0 else GREEN for v in values]
    fig = go.Figure(
        go.Bar(
            x=values,
            y=features,
            orientation="h",
            marker_color=colors,
            hovertemplate="%{y}: %{x:+.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=height,
        **DARK_LAYOUT,
        title=dict(
            text=f"Top contributors — {explanation.get('value_space', 'log_odds')} space",
            x=0.0,
            xanchor="left",
        ),
        xaxis=dict(title="SHAP contribution", gridcolor="#1f2a4a", tickformat="+.2f"),
        yaxis=dict(title="Feature", gridcolor="#1f2a4a", automargin=True),
        showlegend=False,
    )
    return fig


def beeswarm_chart(global_explanation: dict, height: int = 520) -> go.Figure:
    """Beeswarm-style plot: SHAP values per background customer vs feature value."""
    details = global_explanation.get("feature_details", [])
    if not details:
        fig = go.Figure()
        fig.update_layout(height=height, **DARK_LAYOUT, title=dict(text="No global SHAP detail available", x=0.0))
        return fig
    value_space = global_explanation.get("value_space", "log_odds")
    unit = _unit_label(value_space)

    fig = go.Figure()
    y_step = 1.0
    for i, detail in enumerate(details):
        shap_values = np.asarray(detail["shap_values"], dtype=float)
        feature_values = np.asarray(detail["feature_values"], dtype=float)
        mask = ~np.isnan(shap_values)
        shap_values = shap_values[mask]
        feature_values = feature_values[mask]
        if shap_values.size == 0:
            continue
        rng = np.random.default_rng(7)
        jitter = rng.normal(0, 0.16, size=shap_values.size)
        fmin, fmax = np.nanmin(feature_values), np.nanmax(feature_values)
        if fmax > fmin:
            colour = (feature_values - fmin) / (fmax - fmin)
        else:
            colour = np.full(shap_values.size, 0.5)
        fig.add_trace(
            go.Scatter(
                x=shap_values,
                y=[i * y_step] * shap_values.size + jitter,
                mode="markers",
                name=detail["feature"],
                marker=dict(
                    size=6,
                    color=colour,
                    colorscale=[[0, BLUE], [0.5, NEUTRAL], [1, RED]],
                    showscale=False,
                    line=dict(width=0),
                    opacity=0.75,
                ),
                hovertemplate=(
                    f"Feature: {detail['feature']}<br>"
                    f"Feature value: %{{customdata:.2f}}<br>"
                    f"SHAP value: %{{x:+.3f}} {unit}<extra></extra>"
                ),
                customdata=feature_values,
            )
        )

    fig.update_layout(
        height=height,
        **DARK_LAYOUT,
        title=dict(
            text=f"SHAP beeswarm — global feature contributions ({unit})",
            x=0.0,
            xanchor="left",
        ),
        yaxis=dict(
            title="Feature",
            tickmode="array",
            tickvals=[i * y_step for i in range(len(details))],
            ticktext=[d["feature"] for d in details],
            gridcolor="#1f2a4a",
        ),
        xaxis=dict(title=f"SHAP value ({unit})", gridcolor="#1f2a4a", tickformat="+.2f"),
        showlegend=False,
    )
    return fig


def global_importance_bar(global_explanation: dict, height: int = 450, top_k: int = 12) -> go.Figure:
    """Global mean |SHAP| feature importance ranking (associational, not causal)."""
    importance = global_explanation["feature_importance"].head(top_k).iloc[::-1]
    fig = go.Figure(
        go.Bar(
            x=importance["mean_abs_shap"],
            y=importance["feature"],
            orientation="h",
            marker_color=BLUE,
            hovertemplate="%{y}: %{x:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=height,
        **DARK_LAYOUT,
        title=dict(
            text="Global mean |SHAP| — feature influence ranking (model-based, not causal)",
            x=0.0,
            xanchor="left",
        ),
        xaxis=dict(title="Mean |SHAP value|", gridcolor="#1f2a4a"),
        yaxis=dict(title="Feature", gridcolor="#1f2a4a", automargin=True),
        showlegend=False,
    )
    return fig


def high_vs_low_risk_chart(global_explanation: dict, height: int = 360) -> go.Figure:
    """Mean feature values for high-risk vs low-risk background customers."""
    contrast = global_explanation.get("high_vs_low_risk", [])
    if not contrast:
        fig = go.Figure()
        fig.update_layout(height=height, **DARK_LAYOUT, title=dict(text="No risk contrast available", x=0.0))
        return fig
    features = [c["feature"] for c in contrast][::-1]
    high = [c["high_risk_mean"] for c in contrast][::-1]
    low = [c["low_risk_mean"] for c in contrast][::-1]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="High-risk customers",
            x=high,
            y=features,
            orientation="h",
            marker_color=RED,
            opacity=0.85,
            hovertemplate="%{y}: %{x:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Low-risk customers",
            x=low,
            y=features,
            orientation="h",
            marker_color=GREEN,
            opacity=0.85,
            hovertemplate="%{y}: %{x:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        height=height,
        barmode="group",
        **DARK_LAYOUT,
        title=dict(
            text="High-risk vs low-risk customers — mean feature values (associational)",
            x=0.0,
            xanchor="left",
        ),
        xaxis=dict(title="Mean feature value", gridcolor="#1f2a4a"),
        yaxis=dict(title="Feature", gridcolor="#1f2a4a", automargin=True),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8", size=11)),
    )
    return fig
