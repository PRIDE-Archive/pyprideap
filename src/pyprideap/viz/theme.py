"""Consistent color palettes and plot styling for pyprideap visualizations.

All palette functions return lists of hex color strings. The ``set_plot_theme``
helper applies a uniform look-and-feel to any Plotly figure so that reports and
standalone plots share the same visual identity.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from plotly.graph_objects import Figure

# ---------------------------------------------------------------------------
# Named colour constants
# ---------------------------------------------------------------------------

PRIDE_COLORS: dict[str, str] = {
    "primary": "#2c3e50",
    "accent": "#3498db",
    "bg": "#f5f7fa",
    "card": "#ffffff",
    "border": "#e1e8ed",
    "text": "#2c3e50",
    "text_muted": "#7f8c8d",
    "success": "#2ecc71",
    "warning": "#f39c12",
    "error": "#e74c3c",
    "info": "#3498db",
    # Semantic aliases used in QC/LOD plots
    "above_lod": "#2ecc71",
    "below_lod": "#f39c12",
    "pass": "#2ecc71",
    "warn": "#f39c12",
    "fail": "#e74c3c",
    "na": "#95a5a6",
}

# ---------------------------------------------------------------------------
# Discrete palettes
# ---------------------------------------------------------------------------

_DISCRETE_PALETTE: list[str] = [
    "#3498db",  # blue
    "#2ecc71",  # green
    "#e67e22",  # orange
    "#9b59b6",  # purple
    "#1abc9c",  # teal
    "#e74c3c",  # red
    "#f1c40f",  # yellow
    "#34495e",  # dark slate
    "#16a085",  # dark teal
    "#d35400",  # burnt orange
    "#8e44ad",  # dark purple
    "#2980b9",  # dark blue
]

_FILL_PALETTE: list[str] = [
    "#5dade2",  # muted blue
    "#58d68d",  # muted green
    "#f0b27a",  # muted orange
    "#bb8fce",  # muted purple
    "#48c9b0",  # muted teal
    "#ec7063",  # muted red
    "#f7dc6f",  # muted yellow
    "#5d6d7e",  # muted slate
    "#45b39d",  # muted dark teal
    "#e59866",  # muted burnt orange
    "#a569bd",  # muted dark purple
    "#5499c7",  # muted dark blue
]


def _cycle(palette: list[str], n: int | None) -> list[str]:
    """Return *n* colours from *palette*, cycling if necessary."""
    if n is None:
        return list(palette)
    if n <= 0:
        return []
    full, remainder = divmod(n, len(palette))
    return palette * full + palette[:remainder]


def pride_color_discrete(n: int | None = None) -> list[str]:
    """Return a list of visually distinct hex colours.

    Parameters
    ----------
    n : int or None
        Number of colours to return.  If *None* the full base palette
        (12 colours) is returned.  When *n* exceeds the palette length the
        colours are cycled.
    """
    return _cycle(_DISCRETE_PALETTE, n)


def pride_fill_discrete(n: int | None = None) -> list[str]:
    """Return a list of slightly muted hex colours suitable for fills.

    Follows the same interface as :func:`pride_color_discrete`.
    """
    return _cycle(_FILL_PALETTE, n)


# ---------------------------------------------------------------------------
# Continuous / gradient palette
# ---------------------------------------------------------------------------

def pride_color_gradient(n: int = 10) -> list[str]:
    """Return *n* hex colours interpolated along a blue-to-red gradient.

    The gradient passes through the intermediate stops:
    blue -> teal -> green -> yellow -> orange -> red.

    Parameters
    ----------
    n : int
        Number of colours to produce (minimum 1).
    """
    if n <= 0:
        return []
    if n == 1:
        return ["#3498db"]

    # Anchor RGB stops (6 stops for a rich blue-to-red ramp)
    _stops: list[tuple[int, int, int]] = [
        (52, 152, 219),   # #3498db  blue
        (26, 188, 156),   # #1abc9c  teal
        (46, 204, 113),   # #2ecc71  green
        (241, 196, 15),   # #f1c40f  yellow
        (230, 126, 34),   # #e67e22  orange
        (231, 76, 60),    # #e74c3c  red
    ]

    colors: list[str] = []
    for i in range(n):
        t = i / (n - 1)  # 0..1
        # Map t onto the stop segments
        seg = t * (len(_stops) - 1)
        idx = int(seg)
        if idx >= len(_stops) - 1:
            idx = len(_stops) - 2
        frac = seg - idx
        r0, g0, b0 = _stops[idx]
        r1, g1, b1 = _stops[idx + 1]
        r = int(r0 + (r1 - r0) * frac)
        g = int(g0 + (g1 - g0) * frac)
        b = int(b0 + (b1 - b0) * frac)
        colors.append(f"#{r:02x}{g:02x}{b:02x}")
    return colors


# ---------------------------------------------------------------------------
# Plot theme
# ---------------------------------------------------------------------------

def set_plot_theme(fig: Figure) -> Figure:
    """Apply the pyprideap house style to a Plotly figure *in place*.

    Adjusts fonts, background colour, grid style, and margins to match
    the CSS variables used in the HTML report (``--primary``, ``--accent``,
    ``--bg``).

    Returns the figure so calls can be chained:
    ``fig = set_plot_theme(go.Figure(...))``
    """
    fig.update_layout(
        font=dict(
            family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
            size=13,
            color=PRIDE_COLORS["text"],
        ),
        title_font=dict(
            size=16,
            color=PRIDE_COLORS["primary"],
        ),
        paper_bgcolor=PRIDE_COLORS["bg"],
        plot_bgcolor=PRIDE_COLORS["card"],
        legend=dict(
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor=PRIDE_COLORS["border"],
            borderwidth=1,
            font=dict(size=11),
        ),
        margin=dict(l=60, r=30, t=60, b=50),
    )

    # Axis styling — light grid, subtle tick marks
    _axis_style = dict(
        gridcolor="#e1e8ed",
        gridwidth=1,
        linecolor=PRIDE_COLORS["border"],
        linewidth=1,
        showgrid=True,
        zeroline=False,
        title_font=dict(size=13, color=PRIDE_COLORS["primary"]),
        tickfont=dict(size=11, color=PRIDE_COLORS["text_muted"]),
    )

    fig.update_xaxes(**_axis_style)
    fig.update_yaxes(**_axis_style)

    return fig
