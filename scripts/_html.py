"""Shared HTML output helpers for the resource-level plot scripts."""

from __future__ import annotations

import logging
from pathlib import Path

import plotly.graph_objects as go

logger = logging.getLogger(__name__)

_PAGE_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{title}</title>
    <style>
        :root {{
            --bg: #f5f7fa; --card: #ffffff; --border: #e1e8ed;
        }}
        body {{
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            margin: 0; padding: 20px; background: var(--bg);
        }}
        .plot-container {{
            background: var(--card); border: 1px solid var(--border);
            border-radius: 8px; padding: 16px; margin-bottom: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        }}
    </style>
</head>
<body>
    <div class="plot-container">
        {plot}
    </div>
</body>
</html>
"""

_PLOT_CONFIG = {
    "responsive": True,
    "displayModeBar": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}


def save_plot(fig: go.Figure, path: Path, title: str) -> Path:
    """Write *fig* as a standalone HTML page.

    plotly.js is loaded from the CDN at the version matching the installed
    plotly package, so the serialised figure and the renderer never drift apart.
    The figure is written as-is: apply ``set_plot_theme`` *before* any
    per-plot layout tweaks so those tweaks (e.g. wide left margins) survive.
    """
    plot = fig.to_html(full_html=False, include_plotlyjs="cdn", config=_PLOT_CONFIG)
    path.write_text(_PAGE_TEMPLATE.format(title=title, plot=plot), encoding="utf-8")
    logger.info("Saved %s", path)
    return path
