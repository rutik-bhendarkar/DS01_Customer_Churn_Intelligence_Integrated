"""Import and chart smoke tests for the Streamlit dashboard."""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from streamlit.testing.v1 import AppTest

import app
import dashboard
import dashboard.components.charts
import dashboard.pages.overview
import dashboard.pages.churn_analytics
import dashboard.pages.clv_analytics
import dashboard.pages.customer_explorer
import dashboard.pages.retention_intelligence
import dashboard.pages.model_performance
import dashboard.pages.reports
import dashboard.pages.settings
from dashboard.components.charts import (
    clv_box_by_risk,
    probability_histogram,
    probability_vs_clv_scatter,
    risk_bar_chart,
    risk_gauge,
    value_segment_pie,
)


def test_required_overview_chart_imports():
    from dashboard.components.charts import (
        risk_bar_chart,
        probability_histogram,
        clv_box_by_risk,
        value_segment_pie,
    )

    assert all(callable(chart) for chart in (
        risk_bar_chart,
        probability_histogram,
        clv_box_by_risk,
        value_segment_pie,
    ))


def test_dashboard_chart_helpers_return_plotly_figures():
    customers = pd.DataFrame(
        {
            "churn_risk": ["Low", "Medium", "High"],
            "churn_probability": [0.1, 0.5, 0.9],
            "clv": [100.0, 1000.0, 2500.0],
            "value_segment": ["Standard", "Valuable", "Premium"],
        }
    )
    figures = [
        risk_bar_chart(customers),
        probability_histogram(customers),
        clv_box_by_risk(customers),
        value_segment_pie(customers),
        probability_vs_clv_scatter(customers),
        risk_gauge(0.9),
    ]

    assert all(isinstance(figure, go.Figure) for figure in figures)


def test_clv_box_by_risk_coerces_values_and_preserves_input():
    customers = pd.DataFrame(
        {
            "churn_risk": ["Low", "Medium", "High", "High", None],
            "clv": ["100", "not-a-value", "2500", None, "700"],
        }
    )
    original = customers.copy(deep=True)

    figure = clv_box_by_risk(customers)

    pd.testing.assert_frame_equal(customers, original)
    assert [trace.name for trace in figure.data] == ["Low", "High"]
    assert figure.data[0].marker.color == "#10b981"
    assert figure.data[1].marker.color == "#ef4444"
    assert figure.layout.title.text == "Customer lifetime value by churn risk"


def test_clv_box_by_risk_handles_empty_data():
    figure = clv_box_by_risk(pd.DataFrame(columns=["churn_risk", "clv"]))

    assert isinstance(figure, go.Figure)
    assert not figure.data
    assert figure.layout.title.text == "Customer lifetime value by churn risk"


def test_streamlit_app_renders_overview_charts():
    app_test = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run()

    assert not app_test.exception
    assert len(app_test.get("plotly_chart")) == 4
