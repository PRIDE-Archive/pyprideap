#!/usr/bin/env python3
"""Generate global resource-level plots for the PRIDE Affinity Proteomics page.

Fetches metadata for all PAD (affinity) projects from the PRIDE Archive API and
produces interactive Plotly HTML charts summarising the entire resource.

The fetched metadata is cached as ``pad_projects.json`` in the output directory;
pass ``--from-cache`` to re-render the plots offline from that snapshot.

Usage:
    python scripts/generate_global_plots.py [--output-dir reports/global] [--from-cache]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from collections.abc import Iterable
from datetime import date
from pathlib import Path

import plotly.graph_objects as go
import requests
from _html import save_plot
from plotly.subplots import make_subplots

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from pyprideap.viz.theme import PRIDE_COLORS, pride_color_discrete, set_plot_theme

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PRIDE API
# ---------------------------------------------------------------------------

_BASE = "https://www.ebi.ac.uk/pride/ws/archive/v3"
_TIMEOUT = 60
_PAGE_SIZE = 100
_CACHE_NAME = "pad_projects.json"

_OLINK_COLOR = PRIDE_COLORS["accent"]
_SOMASCAN_COLOR = PRIDE_COLORS["success"]
_OTHER_COLOR = PRIDE_COLORS["na"]

# Disease annotations that describe controls rather than a studied condition
_NON_DISEASES = {"disease free", "normal", "healthy", "not applicable", "not available"}


def fetch_pad_projects() -> list[dict]:
    """Fetch metadata for every affinity (PAD) project in PRIDE.

    ``submissionType==AFFINITY`` selects exactly the PAD accessions; a keyword
    search for "affinity proteomics" would also match MS projects such as AP-MS.
    Search results omit some fields (e.g. ``countries``), so each project's full
    record is then fetched from ``/projects/{accession}``.
    """
    projects: list[dict] = []
    with requests.Session() as session:
        page = 0
        while True:
            resp = session.get(
                f"{_BASE}/search/projects",
                params={
                    "keyword": "",
                    "filter": "submissionType==AFFINITY",
                    "pageSize": _PAGE_SIZE,
                    "page": page,
                    "sortDirection": "ASC",
                    "sortFields": "submissionDate",
                },
                timeout=_TIMEOUT,
            )
            resp.raise_for_status()
            batch = resp.json()
            projects.extend(batch)
            if len(batch) < _PAGE_SIZE:
                break
            page += 1

        non_pad = [p["accession"] for p in projects if not p["accession"].startswith("PAD")]
        if non_pad:
            logger.warning("Dropping %d non-PAD projects returned by the API: %s", len(non_pad), non_pad)

        detailed = []
        for p in projects:
            if not p["accession"].startswith("PAD"):
                continue
            resp = session.get(f"{_BASE}/projects/{p['accession']}", timeout=_TIMEOUT)
            resp.raise_for_status()
            detailed.append(resp.json())
    return detailed


def load_projects(output_dir: Path, from_cache: bool) -> list[dict]:
    cache = output_dir / _CACHE_NAME
    if from_cache:
        logger.info("Loading project metadata from %s", cache)
        return json.loads(cache.read_text(encoding="utf-8"))  # type: ignore[no-any-return]
    projects = fetch_pad_projects()
    cache.write_text(json.dumps(projects, indent=1), encoding="utf-8")
    logger.info("Fetched %d PAD projects (cached in %s)", len(projects), cache)
    return projects


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------


def _names(project: dict, field: str) -> list[str]:
    """Return the distinct names of a CvParam-list (or string-list) field."""
    values = project.get(field) or []
    names = (v.get("name", "") if isinstance(v, dict) else str(v) for v in values)
    return sorted({n.strip() for n in names if n and n.strip()})


def _count(projects: Iterable[dict], field: str) -> Counter[str]:
    """Number of projects annotated with each value of *field*."""
    return Counter(name for p in projects for name in _names(p, field))


def _family(label: str) -> str:
    low = label.lower()
    if "olink" in low:
        return "Olink"
    if "somascan" in low:
        return "SomaScan"
    return "Other"


def platform_family(project: dict) -> str:
    """Classify a project as Olink, SomaScan or Other from its experiment types / instruments."""
    families = {_family(n) for n in _names(project, "experimentTypes") + _names(project, "instruments")}
    families.discard("Other")
    if len(families) == 1:
        return families.pop()
    return "Other" if not families else "Mixed"


def _diseases(project: dict) -> list[str]:
    return [d for d in _names(project, "diseases") if d.lower() not in _NON_DISEASES]


def _submission_months(projects: list[dict]) -> Counter[str]:
    return Counter(p["submissionDate"][:7] for p in projects if p.get("submissionDate"))


def _month_range(first: str, last: str) -> list[str]:
    """All YYYY-MM months from *first* to *last* inclusive."""
    y, m = int(first[:4]), int(first[5:7])
    months = []
    while f"{y:04d}-{m:02d}" <= last:
        months.append(f"{y:04d}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return months


def _new_figure(*args: object, **kwargs: object) -> go.Figure:
    """Create a themed figure; per-plot layout set afterwards overrides the theme."""
    return set_plot_theme(go.Figure(*args, **kwargs))


# ---------------------------------------------------------------------------
# Plot generators
# ---------------------------------------------------------------------------


def generate_kpi_cards(projects: list[dict], output_dir: Path) -> Path:
    """Summary KPI cards with key resource metrics."""
    families = Counter(platform_family(p) for p in projects)
    diseases = {d for p in projects for d in _diseases(p)}
    countries = {c for p in projects for c in _names(p, "countries")}
    with_pubs = sum(1 for p in projects if p.get("references"))
    dates = sorted(p["submissionDate"] for p in projects if p.get("submissionDate"))
    since = f"Since {dates[0][:4]}" if dates else ""

    cards_data = [
        (len(projects), "Total Projects", f"PAD datasets · {since}"),
        (families["Olink"], "Olink Projects", "Explore, Explore HT, Target, Reveal"),
        (families["SomaScan"], "SomaScan Projects", "SomaScan assay"),
        (len(diseases), "Diseases Studied", "Distinct annotated conditions"),
        (len(countries), "Countries", "Of submitting labs"),
        (with_pubs, "With Publications", "Projects linked to a paper"),
    ]
    accent_colors = ["#3498db", "#3498db", "#2ecc71", "#e67e22", "#9b59b6", "#1abc9c"]

    cards_html = "".join(
        f"""
        <div class="kpi-card">
            <div class="kpi-value" style="color: {color}">{value}</div>
            <div class="kpi-label">{label}</div>
            <div class="kpi-sub">{sub}</div>
        </div>"""
        for (value, label, sub), color in zip(cards_data, accent_colors)
    )

    path = output_dir / "summary_kpi.html"
    path.write_text(_KPI_HTML_TEMPLATE.format(cards=cards_html), encoding="utf-8")
    logger.info("Saved %s", path)
    return path


def plot_resource_growth(projects: list[dict], output_dir: Path) -> Path:
    """Monthly new submissions (bars) and cumulative total (line, secondary axis)."""
    monthly = _submission_months(projects)
    months = _month_range(min(monthly), max(monthly))
    counts = [monthly.get(m, 0) for m in months]
    cumulative = []
    total = 0
    for c in counts:
        total += c
        cumulative.append(total)

    colors = pride_color_discrete(2)
    fig = set_plot_theme(make_subplots(specs=[[{"secondary_y": True}]]))
    fig.add_trace(
        go.Bar(
            x=months,
            y=counts,
            marker_color=colors[1],
            name="New Submissions",
            opacity=0.8,
            hovertemplate="<b>%{x}</b><br>New: %{y}<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=months,
            y=cumulative,
            mode="lines+markers",
            line=dict(color=colors[0], width=2.5),
            name="Cumulative Projects",
            hovertemplate="<b>%{x}</b><br>Total: %{y} projects<extra></extra>",
        ),
        secondary_y=True,
    )
    fig.update_layout(
        title="Resource Growth Over Time",
        xaxis_title="Submission Month",
        height=450,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_yaxes(title_text="New Projects", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative Projects", secondary_y=True, showgrid=False, rangemode="tozero")

    return save_plot(fig, output_dir / "resource_growth.html", "Resource Growth Over Time")


def plot_submission_timeline_heatmap(projects: list[dict], output_dir: Path) -> Path:
    """Calendar heatmap of submissions by month/year."""
    monthly = _submission_months(projects)
    first_year, last_year = int(min(monthly)[:4]), int(max(monthly)[:4])
    years = [str(y) for y in range(first_year, last_year + 1)]
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    z = [[monthly.get(f"{yr}-{m:02d}", 0) for m in range(1, 13)] for yr in years]

    fig = _new_figure(
        go.Heatmap(
            z=z,
            x=month_names,
            y=years,
            colorscale=[
                [0, "#f5f7fa"],
                [0.01, "#d4e6f1"],
                [0.25, "#85c1e9"],
                [0.5, "#3498db"],
                [0.75, "#2471a3"],
                [1, "#1b4f72"],
            ],
            text=[[v if v > 0 else "" for v in row] for row in z],
            texttemplate="%{text}",
            hovertemplate="<b>%{y} %{x}</b><br>Submissions: %{z}<extra></extra>",
            colorbar=dict(title="Submissions", thickness=15),
            xgap=2,
            ygap=2,
        )
    )
    fig.update_layout(
        title="Submission Activity by Month",
        height=max(260, len(years) * 45 + 140),
        yaxis=dict(autorange="reversed", type="category"),
        margin=dict(l=60, r=80, t=60, b=50),
    )

    return save_plot(fig, output_dir / "submission_heatmap.html", "Submission Activity Heatmap")


def plot_disease_landscape(projects: list[dict], output_dir: Path, top_n: int = 30) -> Path:
    """Horizontal bar chart of the most frequently studied diseases."""
    counts = Counter(d for p in projects for d in _diseases(p))
    top = sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:top_n]
    top.reverse()  # largest at the top of a horizontal bar chart
    labels = [d for d, _ in top]
    values = [c for _, c in top]
    title = "Disease Landscape" if len(counts) <= top_n else f"Disease Landscape (top {top_n} of {len(counts)})"

    fig = _new_figure(
        go.Bar(
            y=labels,
            x=values,
            orientation="h",
            marker=dict(color=values, colorscale=[[0, "#aed6f1"], [1, "#2980b9"]], line=dict(color="white", width=1)),
            hovertemplate="<b>%{y}</b><br>Projects: %{x}<extra></extra>",
            text=values,
            textposition="outside",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Number of Projects",
        xaxis=dict(dtick=1),
        height=max(400, len(labels) * 26 + 120),
        margin=dict(l=320, r=40, t=60, b=50),
    )

    return save_plot(fig, output_dir / "disease_landscape.html", "Disease Landscape")


def plot_technology_platforms(projects: list[dict], output_dir: Path) -> Path:
    """Bar chart of instrument/assay versions, coloured by platform family."""
    counts = _count(projects, "instruments")
    fig = _new_figure()
    for family, color in (("Olink", _OLINK_COLOR), ("SomaScan", _SOMASCAN_COLOR), ("Other", _OTHER_COLOR)):
        items = sorted(((k, v) for k, v in counts.items() if _family(k) == family), key=lambda x: (-x[1], x[0]))
        if not items:
            continue
        fig.add_trace(
            go.Bar(
                x=[k for k, _ in items],
                y=[v for _, v in items],
                name=family,
                marker_color=color,
                text=[v for _, v in items],
                textposition="outside",
                hovertemplate="<b>%{x}</b><br>Projects: %{y}<extra></extra>",
            )
        )
    fig.update_layout(
        title="Affinity Platform Distribution",
        yaxis_title="Number of Projects",
        height=450,
        barmode="group",
        bargap=0.25,
        xaxis_tickangle=-30,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return save_plot(fig, output_dir / "platform_distribution.html", "Affinity Platform Distribution")


def plot_quantification_methods(projects: list[dict], output_dir: Path) -> Path:
    """Projects per quantification method (a project may report several)."""
    items = sorted(_count(projects, "quantificationMethods").items(), key=lambda x: (x[1], x[0]))
    labels = [k for k, _ in items]
    values = [v for _, v in items]

    fig = _new_figure(
        go.Bar(
            y=labels,
            x=values,
            orientation="h",
            marker_color=pride_color_discrete(1)[0],
            text=values,
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Projects: %{x}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Quantification Methods",
        xaxis_title="Number of Projects",
        height=max(320, len(labels) * 32 + 120),
        margin=dict(l=220, r=40, t=60, b=50),
    )

    return save_plot(fig, output_dir / "quantification_methods.html", "Quantification Methods")


def plot_countries(projects: list[dict], output_dir: Path) -> Path:
    """Projects per submitting country."""
    items = sorted(_count(projects, "countries").items(), key=lambda x: (x[1], x[0]))
    labels = [k for k, _ in items]
    values = [v for _, v in items]

    fig = _new_figure(
        go.Bar(
            y=labels,
            x=values,
            orientation="h",
            marker_color=pride_color_discrete(4)[3],
            text=values,
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Projects: %{x}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Projects by Country",
        xaxis_title="Number of Projects",
        height=max(320, len(labels) * 28 + 120),
        margin=dict(l=180, r=40, t=60, b=50),
    )

    return save_plot(fig, output_dir / "country_distribution.html", "Projects by Country")


# ---------------------------------------------------------------------------
# KPI page
# ---------------------------------------------------------------------------

_KPI_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>PRIDE Affinity Proteomics — Resource Summary</title>
    <style>
        :root {{
            --bg: #f5f7fa;
            --card: #ffffff;
            --border: #e1e8ed;
            --text: #2c3e50;
            --text-muted: #7f8c8d;
        }}
        body {{
            font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
            margin: 0; padding: 20px;
            background: var(--bg); color: var(--text);
        }}
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            max-width: 1100px;
            margin: 0 auto;
        }}
        .kpi-card {{
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 24px 20px;
            text-align: center;
            box-shadow: 0 2px 6px rgba(0,0,0,0.04);
        }}
        .kpi-value {{
            font-size: 2.4em;
            font-weight: 700;
            line-height: 1.1;
        }}
        .kpi-label {{
            font-size: 0.85em;
            color: var(--text-muted);
            margin-top: 6px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        .kpi-sub {{
            font-size: 0.75em;
            color: var(--text-muted);
            margin-top: 4px;
        }}
    </style>
</head>
<body>
    <div class="kpi-grid">
        {cards}
    </div>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--output-dir",
        "-o",
        default="reports/global",
        help="Directory for output HTML files (default: reports/global)",
    )
    parser.add_argument(
        "--from-cache",
        action="store_true",
        help=f"Render from <output-dir>/{_CACHE_NAME} instead of querying the PRIDE API",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    projects = load_projects(output_dir, args.from_cache)
    if not projects:
        sys.exit("No PAD projects found; nothing to plot.")

    logger.info("Generating global resource plots for %d projects (as of %s)...", len(projects), date.today())
    generated = [
        generate_kpi_cards(projects, output_dir),
        plot_resource_growth(projects, output_dir),
        plot_submission_timeline_heatmap(projects, output_dir),
        plot_technology_platforms(projects, output_dir),
        plot_disease_landscape(projects, output_dir),
        plot_quantification_methods(projects, output_dir),
        plot_countries(projects, output_dir),
    ]

    print(f"\nGenerated {len(generated)} plots in {output_dir}/:")
    for p in generated:
        print(f"  - {p.name}")


if __name__ == "__main__":
    main()
