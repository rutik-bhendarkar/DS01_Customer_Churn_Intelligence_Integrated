"""Inject the RevenuePulse AI themed CSS into the Streamlit app.

The CSS file (``dashboard/styles/theme.css``) is a *base* stylesheet that
references CSS custom properties via e.g. ``var(--color-bg)``.  This module
reads the active theme colours from Streamlit session state and re-injects
the same file, overriding the custom properties with the user's current
theme so both dark and light modes work from one CSS file.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from dashboard.theme import DARK_THEME

_CSS_PATH = Path(__file__).resolve().parents[1] / "styles" / "theme.css"


def _build_theme_vars(colors: dict[str, str]) -> str:
    """Return stylesheet variable overrides for the active theme palette."""
    css_colors = {
        "--color-bg": colors["background"],
        "--color-bg-panel": colors["secondary_background"],
        "--color-bg-card": colors["card_background"],
        "--color-bg-hover": colors["border"],
        "--color-text-primary": colors["text"],
        "--color-text-secondary": colors["secondary_text"],
        "--color-text-muted": colors["secondary_text"],
        "--accent-blue": colors["primary"],
        "--accent-cyan": colors["accent"],
        "--accent-emerald": colors["success"],
        "--accent-amber": colors["warning"],
        "--accent-rose": colors["danger"],
        "--border": colors["border"],
    }
    declarations = "; ".join(
        f"{name}: {value};" for name, value in css_colors.items()
    )
    return f":root {{{declarations}}}"


def _active_theme_vars() -> str:
    """Build CSS aliases from the active session theme."""
    colors = st.session_state.get("theme_colors")
    if not isinstance(colors, dict) or not colors:
        colors = DARK_THEME
    return _build_theme_vars(colors)


def inject_theme_css() -> None:
    """Load the base ``theme.css`` and inject theme colour variables.

    The base file is a single stylesheet that works for both dark and light
    themes; the colour variables are applied as inline ``:root`` overrides.
    """
    if not _CSS_PATH.exists():
        # Last-resort fallback: supply a minimal dark theme inline.
        st.markdown(
            "<style>"
            ":root { --color-bg: #0b1021; --color-bg-panel: #11182a; "
            "--color-bg-card: #16213e; --color-text-primary: #f1f5f9; "
            "--color-text-secondary: #94a3b8; }"
            "body { background: #0b1021; color: #f1f5f9; }"
            "</style>",
            unsafe_allow_html=True,
        )
        return

    css = _CSS_PATH.read_text(encoding="utf-8")
    theme_vars = _active_theme_vars()
    st.markdown(
        f"<style>\n{css}\n{theme_vars}\n</style>",
        unsafe_allow_html=True,
    )
