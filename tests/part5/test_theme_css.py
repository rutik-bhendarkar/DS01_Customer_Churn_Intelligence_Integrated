"""Regression tests for dashboard stylesheet injection."""

from dashboard.components import theme_css
from dashboard.theme import LIGHT_THEME


def test_theme_stylesheet_is_injected_inside_style_tag(tmp_path, monkeypatch):
    css_path = tmp_path / "theme.css"
    css_content = ":root { --color-bg: #0b1021; }\n.kpi-card { color: white; }"
    css_path.write_text(css_content, encoding="utf-8")
    rendered = []

    monkeypatch.setattr(theme_css, "_CSS_PATH", css_path)
    monkeypatch.setattr(theme_css, "_active_theme_vars", lambda: ":root { --color-bg: #123456; }")
    monkeypatch.setattr(theme_css.st, "markdown", lambda body, **kwargs: rendered.append((body, kwargs)))

    theme_css.inject_theme_css()

    assert len(rendered) == 1
    html, options = rendered[0]
    assert html.startswith("<style>\n")
    assert html.endswith("\n</style>")
    assert css_content in html
    assert html.index(css_content) < html.rindex("</style>")
    assert options == {"unsafe_allow_html": True}


def test_theme_variables_override_stylesheet_colors():
    theme_vars = theme_css._build_theme_vars(LIGHT_THEME)

    assert "--color-bg: #f8fafc;" in theme_vars
    assert "--color-text-primary: #0f172a;" in theme_vars
    assert "--color-bg-card: #ffffff;" in theme_vars
