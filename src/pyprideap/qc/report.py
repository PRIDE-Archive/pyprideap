from __future__ import annotations

from pathlib import Path

from pyprideap.core import AffinityDataset
from pyprideap.qc.compute import (
    CorrelationData,
    CvDistributionData,
    DetectionRateData,
    DistributionData,
    LodAnalysisData,
    MissingFrequencyData,
    MissingValuesData,
    PcaData,
    QcLodSummaryData,
    compute_all,
)

_HELP_TEXT: dict[str, str] = {
    "distribution": (
        "Shows the intensity distribution of expression values for each sample as overlaid histograms. "
        "All samples should have similar shapes and ranges. A sample with a shifted or bimodal distribution "
        "may indicate a technical issue (e.g. failed plate, low protein yield). Compare the spread and center "
        "of each sample to identify outliers."
    ),
    "missing_frequency": (
        "Displays how many assays fall into each missing-frequency bin. The x-axis is the fraction of samples "
        "where a given assay has no measured value; the y-axis (log scale) is the count of assays. The red "
        "dashed line marks the 30%% threshold — assays above this line are missing in more than 30%% of "
        "samples and may need to be excluded from downstream analysis."
    ),
    "qc_summary": (
        "Stacked bar showing the percentage of sample–assay measurements in each QC category. Categories "
        "combine the Olink QC flag (PASS / WARN / FAIL) with whether the value is above or below the Limit "
        "of Detection (LOD). A high proportion of PASS & NPX > LOD (green) indicates good data quality. "
        "Large WARN or FAIL fractions suggest systematic issues."
    ),
    "lod_analysis": (
        "Bar chart of the percentage of samples above the Limit of Detection for each assay, coloured by "
        "panel. Assays with a low percentage above LOD have weak signal and may be unreliable. This helps "
        "identify low-abundance proteins that are difficult to quantify on the platform."
    ),
    "pca": (
        "Principal Component Analysis projects the high-dimensional expression data onto two axes that "
        "capture the most variance. Each point is a sample. Samples that cluster together are similar; "
        "outliers far from the main cluster may have quality issues. The percentage on each axis shows "
        "how much of the total variance that component explains."
    ),
    "correlation": (
        "Heatmap of pairwise Pearson correlations between samples. Values range from −1 (inverse) to +1 "
        "(perfect correlation). In a well-behaved experiment most sample pairs should show high positive "
        "correlation (warm colours). A sample with consistently low correlation against all others is a "
        "potential outlier."
    ),
    "missing_values": (
        "Two side-by-side bar charts. Left: the fraction of features missing per sample — a sample with "
        "unusually high missingness may have failed. Right: the fraction of samples missing per feature "
        "(top 50 shown) — features missing across many samples may be below detection limits."
    ),
    "cv_distribution": (
        "Histogram of the coefficient of variation (CV = std / mean) across all features. CV measures "
        "relative variability: values below 0.2 indicate tight reproducibility, while a long right tail "
        "suggests some features have high technical or biological variability."
    ),
    "detection_rate": (
        "Bar chart showing the fraction of features successfully detected (non-missing) per sample. "
        "The red dashed line marks the median detection rate. Samples well below the median may have "
        "lower input amounts or technical problems and should be inspected."
    ),
}

_SECTION_ORDER = [
    ("Quality Overview", ["qc_summary", "detection_rate"]),
    ("Signal & Distribution", ["distribution", "lod_analysis"]),
    ("Missing Data", ["missing_values", "missing_frequency"]),
    ("Sample Relationships", ["pca", "correlation"]),
    ("Variability", ["cv_distribution"]),
]

_CSS = """\
:root {
    --primary: #2c3e50;
    --accent: #3498db;
    --bg: #f5f7fa;
    --card: #ffffff;
    --border: #e1e8ed;
    --text: #2c3e50;
    --text-muted: #7f8c8d;
}
* { box-sizing: border-box; }
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    max-width: 1280px; margin: 0 auto; padding: 24px 32px;
    background: var(--bg); color: var(--text); line-height: 1.5;
}
header {
    background: linear-gradient(135deg, var(--primary), var(--accent));
    color: white; padding: 28px 32px; border-radius: 12px;
    margin-bottom: 24px;
}
header h1 { margin: 0 0 12px 0; font-size: 1.6em; font-weight: 600; }
.stats { display: flex; gap: 24px; flex-wrap: wrap; }
.stat-item {
    background: rgba(255,255,255,0.15); padding: 8px 16px;
    border-radius: 8px; font-size: 0.95em;
}
.stat-item strong { font-weight: 600; }
nav.toc {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 10px; padding: 16px 24px; margin-bottom: 28px;
}
nav.toc h2 {
    font-size: 0.85em; text-transform: uppercase; letter-spacing: 0.08em;
    color: var(--text-muted); margin: 0 0 10px 0;
}
nav.toc ul { list-style: none; padding: 0; margin: 0; display: flex; flex-wrap: wrap; gap: 6px 16px; }
nav.toc a {
    color: var(--accent); text-decoration: none; font-size: 0.9em;
    padding: 2px 0; border-bottom: 1px solid transparent;
    transition: border-color 0.2s;
}
nav.toc a:hover { border-bottom-color: var(--accent); }
.section-group { margin-bottom: 36px; }
.section-group > h2 {
    font-size: 1.15em; color: var(--primary); margin: 0 0 16px 0;
    padding-bottom: 8px; border-bottom: 2px solid var(--accent);
}
.plot-card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 10px; padding: 20px 24px; margin-bottom: 20px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}
.plot-header { display: flex; align-items: center; gap: 10px; margin-bottom: 4px; }
.plot-header h3 { margin: 0; font-size: 1.05em; color: var(--primary); }
.help-toggle {
    display: inline-flex; align-items: center; justify-content: center;
    width: 22px; height: 22px; border-radius: 50%;
    background: var(--accent); color: white; font-size: 13px;
    font-weight: 700; cursor: pointer; border: none;
    flex-shrink: 0; line-height: 1; user-select: none;
    transition: background 0.2s;
}
.help-toggle:hover { background: #2980b9; }
.help-text {
    display: none; background: #eef6fd; border-left: 3px solid var(--accent);
    padding: 10px 14px; margin: 8px 0 12px 0; border-radius: 0 6px 6px 0;
    font-size: 0.88em; color: #34495e; line-height: 1.55;
}
.help-text.open { display: block; }
footer {
    text-align: center; padding: 20px 0 8px 0;
    color: var(--text-muted); font-size: 0.82em;
}
"""

_JS = """\
document.addEventListener('DOMContentLoaded', function() {
    document.querySelectorAll('.help-toggle').forEach(function(btn) {
        btn.addEventListener('click', function() {
            var helpEl = this.closest('.plot-card').querySelector('.help-text');
            if (helpEl) {
                helpEl.classList.toggle('open');
                this.textContent = helpEl.classList.contains('open') ? '\\u00d7' : '?';
            }
        });
    });
});
"""


def _lod_source_info(dataset: AffinityDataset) -> dict[str, object]:
    """Detect which LOD sources are available and which one is active."""
    from pyprideap.lod import (
        _MIN_CONTROLS_FOR_LOD,
        _find_negative_controls,
        get_bundled_fixed_lod_path,
        get_reported_lod,
    )

    info: dict[str, object] = {"active": None, "sources": []}
    sources: list[dict[str, str]] = []

    # 1. Reported LOD
    reported = get_reported_lod(dataset)
    if reported is not None:
        n_assays = int(reported.notna().any(axis=0).sum()) if hasattr(reported, "shape") and reported.ndim == 2 else int(reported.notna().sum())
        sources.append({
            "name": "Reported LOD",
            "status": "available",
            "detail": f"LOD column in NPX file ({n_assays} assays)",
        })
        if info["active"] is None:
            info["active"] = "Reported LOD"
    else:
        sources.append({"name": "Reported LOD", "status": "unavailable", "detail": "No LOD column in data file"})

    # 2. NCLOD from negative controls
    try:
        nc_mask = _find_negative_controls(dataset)
        n_controls = int(nc_mask.sum())
        if n_controls >= _MIN_CONTROLS_FOR_LOD:
            sources.append({
                "name": "NCLOD",
                "status": "available",
                "detail": f"Computed from {n_controls} negative control samples",
            })
            if info["active"] is None:
                info["active"] = "NCLOD"
        else:
            sources.append({
                "name": "NCLOD",
                "status": "insufficient",
                "detail": f"Only {n_controls} negative controls (need \u2265{_MIN_CONTROLS_FOR_LOD})",
            })
    except (ValueError, KeyError):
        has_st = "SampleType" in dataset.samples.columns
        sources.append({
            "name": "NCLOD",
            "status": "unavailable",
            "detail": "No SampleType column" if not has_st else "No negative control samples found",
        })

    # 3. FixedLOD from bundled configs
    fixed_path = get_bundled_fixed_lod_path(dataset.platform)
    if fixed_path is not None:
        sources.append({
            "name": "FixedLOD",
            "status": "available",
            "detail": f"Bundled reference file ({fixed_path.name})",
        })
    else:
        sources.append({
            "name": "FixedLOD",
            "status": "unavailable",
            "detail": f"No bundled file for {dataset.platform.value}",
        })

    info["sources"] = sources
    return info


def _render_lod_card(lod_info: dict[str, object]) -> str:
    """Render the LOD source summary as an HTML card."""
    status_icons = {"available": "\u2705", "unavailable": "\u274c", "insufficient": "\u26a0\ufe0f"}
    rows = []
    for src in lod_info["sources"]:
        icon = status_icons.get(src["status"], "")
        active = " (active)" if src["name"] == lod_info["active"] else ""
        name_style = "font-weight:600;" if active else ""
        rows.append(
            f"<tr>"
            f'<td style="padding:4px 10px;">{icon}</td>'
            f'<td style="padding:4px 10px;{name_style}">{src["name"]}{active}</td>'
            f'<td style="padding:4px 10px;color:#555;">{src["detail"]}</td>'
            f"</tr>"
        )

    return (
        '<div class="plot-card" id="lod-sources" style="margin-bottom:24px;">'
        '<div class="plot-header">'
        "<h3>LOD Sources</h3>"
        '<button class="help-toggle" title="About LOD sources" aria-label="Help">?</button>'
        "</div>"
        '<div class="help-text">'
        "The Limit of Detection (LOD) determines whether a measured protein signal is above background noise. "
        "Three LOD sources are supported: "
        "<strong>Reported LOD</strong> comes from the LOD column in the original NPX file (per sample and assay). "
        "<strong>NCLOD</strong> is computed from negative control samples in the dataset using the formula "
        "LOD = median(NC) + max(0.2, 3&times;SD(NC)), following the OlinkAnalyze R package. "
        "Requires &ge;10 negative controls. "
        "<strong>FixedLOD</strong> is a pre-computed reference from Olink, specific to the reagent lot and "
        "Data Analysis Reference ID. "
        "The report uses the first available source (Reported &gt; NCLOD &gt; FixedLOD)."
        "</div>"
        '<table style="width:100%;border-collapse:collapse;margin-top:8px;">'
        f'{"".join(rows)}'
        "</table>"
        "</div>"
    )


def _count_proteins_above_lod(dataset: AffinityDataset) -> int | None:
    """Count unique proteins where >50% of samples are above LOD."""
    import pandas as pd

    try:
        from pyprideap.lod import _above_lod_matrix, get_lod_values
    except ImportError:
        return None

    lod = get_lod_values(dataset)
    if lod is None:
        return None

    numeric = dataset.expression.apply(pd.to_numeric, errors="coerce")
    above_lod, has_lod = _above_lod_matrix(numeric, lod)

    if "UniProt" not in dataset.features.columns:
        return None

    oid_col = "OlinkID" if "OlinkID" in dataset.features.columns else dataset.features.columns[0]
    oid_to_uniprot = dict(zip(dataset.features[oid_col], dataset.features["UniProt"]))

    proteins_above: set[str] = set()
    for col in numeric.columns:
        valid = numeric[col].notna() & has_lod[col]
        n = int(valid.sum())
        if n == 0:
            continue
        pct = float(above_lod.loc[valid, col].sum() / n * 100)
        if pct > 50:
            up = oid_to_uniprot.get(col)
            if up and pd.notna(up):
                proteins_above.add(up)

    return len(proteins_above)


def qc_report(dataset: AffinityDataset, output: str | Path) -> Path:
    """Generate a complete QC HTML report for the dataset."""
    try:
        import plotly  # noqa: F401
    except ImportError:
        raise ImportError("Plotly is required for HTML reports. Install with: pip install pyprideap[plots]") from None

    from pyprideap.qc import render as R

    output = Path(output)
    plot_data = compute_all(dataset)

    _RENDERERS = {
        "distribution": (DistributionData, R.render_distribution),
        "missing_frequency": (MissingFrequencyData, R.render_missing_frequency),
        "qc_summary": (QcLodSummaryData, R.render_qc_summary),
        "lod_analysis": (LodAnalysisData, R.render_lod_analysis),
        "pca": (PcaData, R.render_pca),
        "correlation": (CorrelationData, R.render_correlation),
        "missing_values": (MissingValuesData, R.render_missing_values),
        "cv_distribution": (CvDistributionData, R.render_cv_distribution),
        "detection_rate": (DetectionRateData, R.render_detection_rate),
    }

    # Determine display order from _SECTION_ORDER so first-displayed plot gets plotly.js
    display_order: list[str] = []
    for _, keys in _SECTION_ORDER:
        for key in keys:
            display_order.append(key)

    # Build rendered sections keyed by plot id
    rendered: dict[str, tuple[str, str]] = {}  # key -> (title, html)
    first_key = None
    for key in display_order:
        if key in _RENDERERS:
            data = plot_data.get(key)
            if data is not None and first_key is None:
                first_key = key
                break

    for key, (_dtype, renderer) in _RENDERERS.items():
        data = plot_data.get(key)
        if data is None:
            continue
        fig = renderer(data)
        fig.update_layout(height=500)
        js = "cdn" if key == first_key else False
        plot_html = fig.to_html(full_html=False, include_plotlyjs=js, default_height="500px")
        rendered[key] = (data.title, plot_html)

    # Build grouped sections and TOC
    toc_items: list[str] = []
    group_sections: list[str] = []

    for group_title, keys in _SECTION_ORDER:
        cards: list[str] = []
        for key in keys:
            if key not in rendered:
                continue
            title, plot_html = rendered[key]
            help_html = _HELP_TEXT.get(key, "")
            toc_items.append(f'<li><a href="#{key}">{title}</a></li>')
            help_block = (
                f'<div class="help-text">{help_html}</div>' if help_html else ""
            )
            cards.append(
                f'<div class="plot-card" id="{key}">'
                f'<div class="plot-header">'
                f"<h3>{title}</h3>"
                f'<button class="help-toggle" title="How to read this plot" aria-label="Help">?</button>'
                f"</div>"
                f"{help_block}"
                f"{plot_html}"
                f"</div>"
            )
        if cards:
            group_sections.append(
                f'<div class="section-group">'
                f"<h2>{group_title}</h2>"
                f"{''.join(cards)}"
                f"</div>"
            )

    platform_label = dataset.platform.value.replace("_", " ").title()
    n_samples = len(dataset.samples)
    n_features = len(dataset.features)

    # Unique protein accessions
    n_proteins = 0
    if "UniProt" in dataset.features.columns:
        n_proteins = int(dataset.features["UniProt"].dropna().nunique())

    # Proteins above LOD (>50% of samples)
    n_proteins_above_lod = _count_proteins_above_lod(dataset)

    # LOD source summary card
    lod_info = _lod_source_info(dataset)
    lod_card_html = _render_lod_card(lod_info)

    stat_items = [
        f'<span class="stat-item"><strong>Platform</strong> {platform_label}</span>',
        f'<span class="stat-item"><strong>Samples</strong> {n_samples}</span>',
        f'<span class="stat-item"><strong>Assays</strong> {n_features}</span>',
    ]
    if n_proteins > 0:
        stat_items.append(f'<span class="stat-item"><strong>Proteins</strong> {n_proteins}</span>')
    if n_proteins_above_lod is not None:
        stat_items.append(
            f'<span class="stat-item"><strong>Proteins &gt; LOD</strong> {n_proteins_above_lod}</span>'
        )

    html = (
        "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n"
        '    <meta charset="utf-8">\n'
        '    <meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"    <title>QC Report \u2014 {platform_label}</title>\n"
        f"    <style>\n{_CSS}    </style>\n"
        "</head>\n<body>\n"
        "    <header>\n"
        f"        <h1>QC Report &mdash; {platform_label}</h1>\n"
        f'        <div class="stats">\n'
        f'            {"".join(stat_items)}\n'
        f"        </div>\n"
        "    </header>\n"
        f'    <nav class="toc"><h2>Contents</h2><ul>{"".join(toc_items)}</ul></nav>\n'
        f"    {lod_card_html}\n"
        f"    {''.join(group_sections)}\n"
        f'    <footer>Generated by <strong>pyprideap</strong></footer>\n'
        f"    <script>\n{_JS}    </script>\n"
        "</body>\n</html>"
    )

    output.write_text(html)
    return output
