"""Plotly chart helpers with a consistent premium dark-theme palette."""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# Risk color convention used across all pages
RISK_COLORS = {"Low": "#10b981", "Medium": "#f59e0b", "High": "#ef4444"}
RISK_ORDER = ["Low", "Medium", "High"]

# Discrete palette for categorical charts
CATEGORY_PALETTE = ["#3b82f6", "#06b6d4", "#8b5cf6", "#10b981", "#f59e0b", "#ef4444"]

DARK_LAYOUT = dict(
    plot_bgcolor="#11182a",
    paper_bgcolor="#11182a",
    font=dict(color="#f1f5f9", family="Inter, system-ui, sans-serif"),
    title_font=dict(color="#f1f5f9", size=14),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        bordercolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8", size=11),
    ),
    margin=dict(l=10, r=10, t=40, b=10),
    xaxis=dict(
        gridcolor="#1f2a4a", zeroline=False,
        title_font=dict(color="#94a3b8", size=11),
        tickfont=dict(color="#94a3b8", size=10),
    ),
    yaxis=dict(
        gridcolor="#1f2a4a", zeroline=False,
        title_font=dict(color="#94a3b8", size=11),
        tickfont=dict(color="#94a3b8", size=10),
    ),
)


def dark_figure(fig: go.Figure, height: int = 360, title: str | None = None) -> go.Figure:
    """Apply the dark theme layout to a figure."""
    fig.update_layout(height=height, **DARK_LAYOUT)
    if title:
        fig.update_layout(title=dict(text=title, x=0.0, xanchor="left"))
    fig.update_traces(
        hovertemplate="%{y}<extra></extra>",
        marker=dict(line=dict(width=0)),
    )
    return fig


def risk_bar_chart(df: pd.DataFrame, height: int = 340) -> go.Figure:
    """Churn risk distribution bar chart."""
    counts = (
        df["churn_risk"].value_counts()
        .reindex(RISK_ORDER, fill_value=0)
        .rename_axis("risk")
        .reset_index(name="customers")
    )
    fig = go.Figure(
        data=go.Bar(
            x=counts["risk"],
            y=counts["customers"],
            marker_color=[RISK_COLORS[r] for r in counts["risk"]],
            text=counts["customers"],
            textposition="outside",
            textfont=dict(color="#f1f5f9", size=11),
            hovertemplate="Risk: %{x}<br>Customers: %{y}<extra></extra>",
        )
    )
    fig.update_yaxes(title="Customers", gridcolor="#1f2a4a")
    fig.update_xaxes(title="Churn risk", gridcolor="#1f2a4a")
    return dark_figure(fig, height=height, title="Customers by churn risk")


def probability_histogram(df: pd.DataFrame, height: int = 340) -> go.Figure:
    """Churn probability distribution histogram."""
    fig = go.Figure(
        data=go.Histogram(
            x=df["churn_probability"],
            nbinsx=20,
            marker_color="#3b82f6",
            opacity=0.85,
            hovertemplate="Probability: %{x:.0%}<br>Count: %{y}<extra></extra>",
        )
    )
    fig.update_yaxes(title="Customers", gridcolor="#1f2a4a")
    fig.update_xaxes(title="Churn probability", tickformat=".0%", gridcolor="#1f2a4a")
    return dark_figure(fig, height=height, title="Churn probability distribution")


def clv_box_by_risk(df: pd.DataFrame, height: int = 360) -> go.Figure:
    """Compare numeric CLV distributions across churn-risk categories."""
    required = {"churn_risk", "clv"}
    missing = required - set(df.columns)
    if missing and not df.empty:
        raise ValueError(f"CLV-by-risk chart requires columns: {', '.join(sorted(missing))}")

    plot_data = df.reindex(columns=["churn_risk", "clv"]).copy()
    plot_data["clv"] = pd.to_numeric(plot_data["clv"], errors="coerce")
    plot_data = plot_data.dropna(subset=["churn_risk", "clv"])

    fig = go.Figure()
    for risk in RISK_ORDER:
        values = plot_data.loc[plot_data["churn_risk"] == risk, "clv"]
        if values.empty:
            continue
        fig.add_trace(
            go.Box(
                y=values,
                name=risk,
                marker_color=RISK_COLORS[risk],
                boxmean=True,
                hovertemplate=f"{risk}<br>CLV: ₹%{{y:,.2f}}<extra></extra>",
            )
        )

    fig.update_xaxes(title="Churn risk")
    fig.update_yaxes(title="Customer lifetime value (₹)", tickformat=",.0f")
    return dark_figure(fig, height=height, title="Customer lifetime value by churn risk")


def value_segment_pie(df: pd.DataFrame, height: int = 360) -> go.Figure:
    """Show customer counts by estimated-CLV value segment."""
    if "value_segment" not in df.columns and not df.empty:
        raise ValueError("Value-segment chart requires the value_segment column")

    segments = df.get("value_segment", pd.Series(dtype="object")).fillna("Unknown")
    counts = segments.value_counts(sort=False)
    palette = {
        "Standard": "#3b82f6",
        "Valuable": "#8b5cf6",
        "Premium": "#f59e0b",
        "Unknown": "#64748b",
    }
    colors = [palette.get(str(segment), CATEGORY_PALETTE[i % len(CATEGORY_PALETTE)])
              for i, segment in enumerate(counts.index)]
    fig = go.Figure(
        data=go.Pie(
            labels=counts.index.astype(str),
            values=counts.values,
            hole=0.55,
            marker_colors=colors,
            textinfo="label+percent",
            hovertemplate="%{label}<br>Customers: %{value}<br>Share: %{percent}<extra></extra>",
        )
    )
    dark_figure(fig, height=height, title="Customers by value segment")
    fig.update_traces(
        hovertemplate="%{label}<br>Customers: %{value}<br>Share: %{percent}<extra></extra>"
    )
    return fig


def probability_vs_clv_scatter(df: pd.DataFrame, height: int = 400) -> go.Figure:
    """Plot churn probability against CLV, colored by churn risk when available."""
    required = {"churn_probability", "clv"}
    missing = required - set(df.columns)
    if missing and not df.empty:
        raise ValueError(f"Probability-versus-CLV chart requires columns: {', '.join(sorted(missing))}")

    columns = ["churn_probability", "clv"]
    if "churn_risk" in df.columns:
        columns.append("churn_risk")
    plot_data = df.reindex(columns=columns).copy()
    plot_data["churn_probability"] = pd.to_numeric(
        plot_data["churn_probability"], errors="coerce"
    )
    plot_data["clv"] = pd.to_numeric(plot_data["clv"], errors="coerce")
    plot_data = plot_data.dropna(subset=["churn_probability", "clv"])

    fig = go.Figure()
    if "churn_risk" in plot_data.columns:
        for risk in RISK_ORDER:
            subset = plot_data.loc[plot_data["churn_risk"] == risk]
            if subset.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=subset["churn_probability"],
                    y=subset["clv"],
                    mode="markers",
                    name=risk,
                    marker=dict(color=RISK_COLORS[risk], size=8, opacity=0.8),
                    hovertemplate=(
                        f"Risk: {risk}<br>Churn probability: %{{x:.1%}}"
                        "<br>CLV: ₹%{y:,.2f}<extra></extra>"
                    ),
                )
            )
    elif not plot_data.empty:
        fig.add_trace(
            go.Scatter(
                x=plot_data["churn_probability"],
                y=plot_data["clv"],
                mode="markers",
                name="Customers",
                marker=dict(color=CATEGORY_PALETTE[0], size=8, opacity=0.8),
                hovertemplate="Churn probability: %{x:.1%}<br>CLV: ₹%{y:,.2f}<extra></extra>",
            )
        )

    fig.update_xaxes(title="Churn probability", tickformat=".0%", range=[0, 1])
    fig.update_yaxes(title="Customer lifetime value (₹)", tickformat=",.0f")
    dark_figure(fig, height=height, title="Churn probability vs. customer lifetime value")
    for trace in fig.data:
        if trace.name in RISK_COLORS:
            trace.hovertemplate = (
                f"Risk: {trace.name}<br>Churn probability: %{{x:.1%}}"
                "<br>CLV: ₹%{y:,.2f}<extra></extra>"
            )
        else:
            trace.hovertemplate = "Churn probability: %{x:.1%}<br>CLV: ₹%{y:,.2f}<extra></extra>"
    return fig


def risk_gauge(probability: float, height: int = 260) -> go.Figure:
    """Display an individual customer's churn probability as a risk gauge."""
    value = max(0.0, min(1.0, float(probability)))
    if value < 0.30:
        bar_color = RISK_COLORS["Low"]
    elif value < 0.60:
        bar_color = RISK_COLORS["Medium"]
    else:
        bar_color = RISK_COLORS["High"]

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number=dict(valueformat=".0%"),
            title=dict(text="Churn probability"),
            gauge=dict(
                axis=dict(range=[0, 1], tickformat=".0%"),
                bar=dict(color=bar_color),
                bgcolor="#11182a",
                bordercolor="#1f2a4a",
                steps=[
                    dict(range=[0, 0.30], color=RISK_COLORS["Low"]),
                    dict(range=[0.30, 0.60], color=RISK_COLORS["Medium"]),
                    dict(range=[0.60, 1], color=RISK_COLORS["High"]),
                ],
            ),
        )
    )
    fig.update_layout(height=height, **DARK_LAYOUT)
    fig.update_layout(margin=dict(l=20, r=20, t=55, b=10))
    return fig
