"""Animated KPI cards for the executive overview."""

from __future__ import annotations

from typing import Iterable

import streamlit as st

# (label, value, subtitle, icon_emoji, accent_css_class)
KPI_SPEC = tuple[str, str, str, str, str]

_ACCENT_MAP = {
    "blue": "accent-blue",
    "cyan": "accent-cyan",
    "violet": "accent-violet",
    "emerald": "accent-emerald",
    "amber": "accent-amber",
    "rose": "accent-rose",
}


def kpi_card(label: str, value: str, subtitle: str = "", icon: str = "📊", accent: str = "blue") -> None:
    """Render a single premium KPI card."""
    accent_cls = _ACCENT_MAP.get(accent, "accent-blue")
    st.markdown(
        f"""
        <div class="kpi-card {accent_cls}">
          <div class="kpi-label">
            <span>{icon}</span> {label}
          </div>
          <div class="kpi-value">{value}</div>
          <div class="kpi-subtitle">{subtitle}</div>
          <div class="kpi-icon">{icon}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_grid(cards: Iterable[KPI_SPEC]) -> None:
    """Render a responsive grid of KPI cards.

    Each card spec: (label, value, subtitle, icon, accent)
    """
    cards = list(cards)
    n = len(cards)
    cols_per_row = min(3, n) if n else 1
    # Build rows of up to 3 cards
    for row_start in range(0, n, cols_per_row):
        row = cards[row_start:row_start + cols_per_row]
        cols = st.columns(len(row))
        for col, (label, value, subtitle, icon, accent) in zip(cols, row):
            with col:
                kpi_card(label, value, subtitle, icon, accent)
