"""What-if simulation visualisations for the What-if Simulator page.

All figures use the RevenuePulse AI dark theme.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from dashboard.components.charts import DARK_LAYOUT, dark_figure

RED = "#ef4444"
GREEN = "#10b981"
BLUE = "#3b82f6"
AMBER = "#f59e0b"
NEUTRAL = "#94a3b8"


def baseline_vs_scenario_chart(comparison: pd.DataFrame, height: int = 360) -> go.Figure:
    """Net scenario value by strategy, baseline alongside scenarios."""
    plot = comparison.copy()
    plot["net_value_filled"] = plot["net_scenario_value"].fillna(0.0)
    is_baseline = plot["scenario"] == "Baseline (actuals)"
    colors = [AMBER if b else BLUE for b in is_baseline]
    fig = go.Figure(
        go.Bar(
            x=plot["scenario"],
            y=plot["net_value_filled"],
            marker_color=colors,
            hovertemplate="%{x}<br>Net value: ₹%{y:,.0f}<extra></extra>",
            text=[f"₹{v:,.0f}" if pd.notna(v) else "—" for v in plot["net_scenario_value"]],
            textposition="outside",
            textfont=dict(color="#f1f5f9", size=10),
        )
    )
    fig.update_layout(
        height=height,
        **DARK_LAYOUT,
        title=dict(text="Net scenario value — baseline vs strategies", x=0.0, xanchor="left"),
        yaxis=dict(title="Modelled net value (₹)", gridcolor="#1f2a4a", tickformat=",.0f"),
        xaxis=dict(title="Scenario", gridcolor="#1f2a4a", tickangle=-15),
        showlegend=False,
    )
    return fig


def cost_vs_retained_value_chart(comparison: pd.DataFrame, height: int = 360) -> go.Figure:
    """Retention cost vs expected retained value (bubble size = targeted customers)."""
    plot = comparison[comparison["scenario"] != "Baseline (actuals)"].copy()
    if plot.empty:
        return dark_figure(go.Figure(), height=height, title="No scenarios to plot")
    sizes = (plot["n_targeted_customers"] / plot["n_targeted_customers"].max() * 30 + 10).clip(lower=10)
    fig = go.Figure(
        go.Scatter(
            x=plot["total_retention_cost"],
            y=plot["expected_retained_clv"],
            mode="markers+text",
            text=plot["scenario"],
            textposition="top center",
            textfont=dict(color="#94a3b8", size=10),
            marker=dict(
                size=sizes,
                color=plot["net_scenario_value"],
                colorscale=[[0, RED], [0.5, AMBER], [1, GREEN]],
                colorbar=dict(title="Net value (₹)", tickformat=",.0f"),
                line=dict(width=1, color="#11182a"),
            ),
            hovertemplate=(
                "%{text}<br>Cost: ₹%{x:,.0f}<br>Retained value: ₹%{y:,.0f}<br>"
                "Targeted: %{customdata}<extra></extra>"
            ),
            customdata=plot["n_targeted_customers"],
        )
    )
    fig.update_layout(
        height=height,
        **DARK_LAYOUT,
        title=dict(text="Retention cost vs expected retained value", x=0.0, xanchor="left"),
        xaxis=dict(title="Total retention cost (₹)", gridcolor="#1f2a4a", tickformat=",.0f"),
        yaxis=dict(title="Expected retained value (₹)", gridcolor="#1f2a4a", tickformat=",.0f"),
    )
    return fig


def save_rate_sensitivity_chart(
    comparison: pd.DataFrame,
    customers: pd.DataFrame,
    segment: str,
    offer_cost: float,
    height: int = 360,
) -> go.Figure:
    """Modelled net value as the assumed save rate varies."""
    from src.what_if_simulator import simulate_retention_scenario, SEGMENT_ALL_HIGH_RISK

    segment = segment or SEGMENT_ALL_HIGH_RISK
    save_rates = np.round(np.arange(0.0, 1.01, 0.05), 2)
    net_values = []
    for rate in save_rates:
        result = simulate_retention_scenario(
            customers,
            segment=segment,
            save_rate=float(rate),
            offer_cost_per_customer=float(offer_cost),
        )
        net_values.append(result["net_scenario_value"])
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=save_rates,
            y=net_values,
            mode="lines+markers",
            line=dict(color=BLUE, width=2),
            marker=dict(size=6),
            name="Modelled net value",
            hovertemplate="Save rate: %{x:.0%}<br>Net value: ₹%{y:,.0f}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line=dict(color=NEUTRAL, width=1, dash="dot"))
    fig.update_layout(
        height=height,
        **DARK_LAYOUT,
        title=dict(text="Save-rate sensitivity — modelled net value", x=0.0, xanchor="left"),
        xaxis=dict(title="Assumed save rate", gridcolor="#1f2a4a", tickformat=".0%"),
        yaxis=dict(title="Modelled net value (₹)", gridcolor="#1f2a4a", tickformat=",.0f"),
    )
    return fig


def offer_cost_sensitivity_chart(
    comparison: pd.DataFrame,
    customers: pd.DataFrame,
    segment: str,
    save_rate: float,
    height: int = 360,
) -> go.Figure:
    """Modelled net value as the offer cost per customer varies."""
    from src.what_if_simulator import simulate_retention_scenario, SEGMENT_ALL_HIGH_RISK

    segment = segment or SEGMENT_ALL_HIGH_RISK
    max_cost = max(200.0, float(save_rate) * 1000.0)
    costs = np.arange(0.0, max_cost + 1.0, max(1.0, max_cost / 40.0))
    net_values = []
    for cost in costs:
        result = simulate_retention_scenario(
            customers,
            segment=segment,
            save_rate=float(save_rate),
            offer_cost_per_customer=float(cost),
        )
        net_values.append(result["net_scenario_value"])
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=costs,
            y=net_values,
            mode="lines+markers",
            line=dict(color=BLUE, width=2),
            marker=dict(size=4),
            name="Modelled net value",
            hovertemplate="Offer cost: ₹%{x:,.0f}<br>Net value: ₹%{y:,.0f}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line=dict(color=NEUTRAL, width=1, dash="dot"))
    fig.update_layout(
        height=height,
        **DARK_LAYOUT,
        title=dict(text="Offer-cost sensitivity — modelled net value", x=0.0, xanchor="left"),
        xaxis=dict(title="Offer cost per customer (₹)", gridcolor="#1f2a4a", tickformat=",.0f"),
        yaxis=dict(title="Modelled net value (₹)", gridcolor="#1f2a4a", tickformat=",.0f"),
    )
    return fig


def customer_priority_chart(customer_results: pd.DataFrame, height: int = 400) -> go.Figure:
    """Customer-level retention priority (net value ranking)."""
    if customer_results is None or customer_results.empty:
        return dark_figure(go.Figure(), height=height, title="No customers in the selected segment")
    plot = customer_results.head(25).iloc[::-1]
    colors = [GREEN if v >= 0 else RED for v in plot["net_value"]]
    fig = go.Figure(
        go.Bar(
            x=plot["net_value"],
            y=[f"{r['priority_rank']}. {r['customerID']}" for _, r in plot.iterrows()],
            orientation="h",
            marker_color=colors,
            hovertemplate="%{y}<br>Net value: ₹%{x:,.0f}<br>Churn prob: %{customdata:.1%}<extra></extra>",
            customdata=plot["churn_probability"],
        )
    )
    fig.update_layout(
        height=height,
        **DARK_LAYOUT,
        title=dict(text="Customer-level retention priority (top 25)", x=0.0, xanchor="left"),
        xaxis=dict(title="Modelled net value per customer (₹)", gridcolor="#1f2a4a", tickformat=",.0f"),
        yaxis=dict(title="Customer (priority rank)", gridcolor="#1f2a4a", automargin=True),
        showlegend=False,
    )
    return fig
