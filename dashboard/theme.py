"""Theme management module for the RevenuePulse AI dashboard.

Provides a unified theming system supporting both dark and light modes.
All theme colors are stored in Streamlit session state and injected
into the page via CSS custom properties.

Workflow:
    1. init_theme()      — Initialize / restore theme in session state
    2. toggle_theme()    — Switch between dark and light modes
    3. get_theme_colors()— Return the active color dictionary
"""

from __future__ import annotations

import streamlit as st

# ---------------------------------------------------------------------------
# Theme palettes
# ---------------------------------------------------------------------------

DARK_THEME: dict[str, str] = {
    "background": "#0b1021",
    "secondary_background": "#11182a",
    "sidebar_background": "#11182a",
    "card_background": "#16213e",
    "text": "#f1f5f9",
    "secondary_text": "#94a3b8",
    "border": "#1e2a4a",
    "primary": "#3b82f6",
    "accent": "#06b6d4",
    "success": "#10b981",
    "warning": "#f59e0b",
    "danger": "#ef4444",
    "chart_background": "#11182a",
    "grid_color": "#1e2a4a",
}

LIGHT_THEME: dict[str, str] = {
    "background": "#f8fafc",
    "secondary_background": "#f1f5f9",
    "sidebar_background": "#f1f5f9",
    "card_background": "#ffffff",
    "text": "#0f172a",
    "secondary_text": "#475569",
    "border": "#e2e8f0",
    "primary": "#3b82f6",
    "accent": "#06b6d4",
    "success": "#10b981",
    "warning": "#f59e0b",
    "danger": "#ef4444",
    "chart_background": "#f8fafc",
    "grid_color": "#cbd5e1",
}

_THEME_MAP: dict[str, dict[str, str]] = {
    "dark": DARK_THEME,
    "light": LIGHT_THEME,
}

_LIGHT_MODE_DEFAULT = "light"


def init_theme() -> None:
    """Initialise (or restore) the theme in Streamlit session state.

    The default theme is *dark*, but if the user previously selected a
    different mode it is restored across reruns.  The active colour
    dictionary is written to ``st.session_state["theme_colors"]`` and
    CSS custom properties are injected so the theme stylesheet can use
    them throughout the app.
    """
    # Respect a previously selected mode
    if "theme_mode" not in st.session_state:
        st.session_state["theme_mode"] = "dark"

    # Apply & cache the palette
    _apply_theme(st.session_state["theme_mode"])


def _apply_theme(mode: str) -> None:
    """Apply a theme mode and expose CSS variables on ``:root``.

    Parameters
    ----------
    mode:
        Either ``"dark"`` or ``"light"``.

    Side Effects
    ------------
    * Writes ``st.session_state["theme_colors"]``.
    * Injects a ``<style>`` block that sets CSS custom properties.
    """
    colors = _THEME_MAP.get(mode)
    if colors is None:
        st.warning(f"Unknown theme mode {mode!r}, falling back to 'light'.")
        colors = LIGHT_THEME
        mode = "light"

    st.session_state["theme_colors"] = colors

    # Map snake_case keys to kebab-case CSS variables
    declarations = "; ".join(
        f"--color-{k.replace('_', '-')}: {v};"
        for k, v in colors.items()
    )
    st.markdown(
        f"<style>\n:root {{{declarations}}}\n</style>",
        unsafe_allow_html=True,
    )


def toggle_theme() -> str:
    """Toggle between dark and light mode.

    Returns
    -------
    str
        The *new* mode after the toggle.
    """
    current = st.session_state.get("theme_mode", "dark")
    new_mode = "light" if current == "dark" else "dark"
    st.session_state["theme_mode"] = new_mode
    _apply_theme(new_mode)
    return new_mode


def get_theme_colors() -> dict[str, str]:
    """Return the active colour dictionary.

    Returns
    -------
    dict[str, str]
        A dictionary containing all theme keys
        (background, secondary_background, …).
    """
    colors = st.session_state.get("theme_colors")
    if not isinstance(colors, dict) or not colors:
        # Safe-fix: fall back to the default dark palette (or light, if
        # the user previously set a "light" session state without the
        # explicit colour dictionary).
        fallback = (
            LIGHT_THEME if st.session_state.get("theme_mode") == "light"
            else DARK_THEME
        )
        st.session_state["theme_colors"] = fallback
        colors = fallback
    return colors
