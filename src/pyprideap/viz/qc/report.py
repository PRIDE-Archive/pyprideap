from __future__ import annotations

from pathlib import Path
from typing import Any

from pyprideap.core import AffinityDataset
from pyprideap.viz.qc.compute import (
    CorrelationData,
    CvDistributionData,
    DataCompletenessData,
    DistributionData,
    HeatmapData,
    LodAnalysisData,
    LodComparisonData,
    NormScaleData,
    PcaData,
    PlateCvData,
    QcLodSummaryData,
    UmapData,
    compute_all,
)

_HELP_TEXT: dict[str, str] = {
    "distribution": (
        "Shows the intensity distribution of expression values for each sample as overlaid histograms. "
        "Each protein (assay) produces one NPX value per sample, so this plot shows how all protein "
        "measurements are distributed within each sample. All samples should have similar shapes and "
        "ranges. A sample with a shifted or bimodal distribution may indicate a technical issue "
        "(e.g. failed plate, low protein yield)."
    ),
    "qc_summary": (
        "Stacked bar showing the percentage of sample–protein measurements in each QC category. "
        "Categories combine the Olink QC flag (PASS / WARN / FAIL) with whether the value is above "
        "or below the Limit of Detection (LOD). A high proportion of PASS & NPX > LOD (green) "
        "indicates good data quality. Large WARN or FAIL fractions suggest systematic issues."
    ),
    "lod_analysis": (
        "Two-panel view. Left: proteins ranked by %% of samples above the Limit of Detection, "
        "coloured by panel — a steep drop-off reveals how many proteins have weak signal. Right: "
        "histogram of the same percentages showing the overall distribution. Proteins with low "
        "%% above LOD are unreliable and may need filtering."
    ),
    "pca": (
        "Principal Component Analysis projects the high-dimensional protein expression data onto "
        "two axes that capture the most variance. Each point is a sample. Samples that cluster "
        "together have similar protein profiles; outliers far from the main cluster may have "
        "quality issues. The percentage on each axis shows how much of the total variance that "
        "component explains."
    ),  # kept for standalone use; report uses 'dimreduction' instead
    "correlation": (
        "Heatmap of pairwise Pearson correlations between samples based on their protein expression "
        "profiles. Values range from −1 (inverse) to +1 (perfect correlation). In a well-behaved "
        "experiment most sample pairs should show high positive correlation (warm colours). A sample "
        "with consistently low correlation against all others is a potential outlier."
    ),
    "data_completeness": (
        "Two side-by-side panels showing data completeness based on the Limit of Detection. "
        "Left: per-sample stacked bar showing Above LOD (green, reliable signal) vs Below LOD "
        "(orange, measured but below detection limit). A sample with a large orange fraction may "
        "indicate low protein input or technical issues. Right: Missing Frequency distribution — "
        "a histogram of per-protein missing rate (%% of samples where NPX is below LOD). "
        "Proteins clustered near 0%% are reliably detected; those near 100%% may need filtering."
    ),
    "dimreduction": (
        "Dimensionality reduction projects the high-dimensional protein expression data onto two axes. "
        "Use the toggle switch to switch between PCA and t-SNE. "
        "<strong>PCA</strong> (Principal Component Analysis) captures the directions of maximum variance — "
        "the percentage on each axis shows how much total variance that component explains. "
        "<strong>t-SNE</strong> (t-distributed Stochastic Neighbor Embedding) is a non-linear technique "
        "that preserves local neighbourhood structure, useful for identifying clusters. "
        "In both views, each point is a sample; samples that cluster together have similar protein "
        "profiles. Outliers far from the main cluster may have quality issues. "
        "Note: t-SNE is stochastic and distances between distant clusters should not be over-interpreted."
    ),
    "heatmap": (
        "Clustered expression heatmap showing Z-scored protein values across samples. "
        "Rows (samples) and columns (proteins) are reordered by hierarchical clustering "
        "so that similar profiles appear adjacent. Blue indicates below-average expression, "
        "red indicates above-average. Only the most variable proteins are shown (up to 200). "
        "Blocks of correlated colour reveal co-regulated protein groups or sample batches."
    ),
    "cv_distribution": (
        "Histogram of the coefficient of variation (CV = std / mean) across all proteins. CV measures "
        "relative variability: values below 0.2 indicate tight reproducibility, while a long right tail "
        "suggests some proteins have high technical or biological variability."
    ),
    "norm_scale": (
        "Shows the hybridization control normalization scale factor (HybControlNormScale) per sample, "
        "sorted by value and coloured by plate. This SomaScan QC metric reflects how much each sample's "
        "hybridization signal deviates from the reference. A value of 1.0 (green line) is ideal. "
        "Values outside 0.8–1.2 (orange lines) indicate moderate deviation, and values outside "
        "0.4–2.5 (red lines) indicate poor hybridization that may compromise data quality. "
        "Samples failing this check should be flagged for potential exclusion."
    ),
    "plate_cv": (
        "Two-panel view of plate-level variability. "
        "<strong>Top — Intra-plate CV:</strong> for each plate, the CV (SD / mean) is computed "
        "per analyte across samples within that plate. Each violin shows the distribution of "
        "these CVs. A plate with a notably higher distribution suggests worse reproducibility. "
        "<strong>Bottom — Inter-plate CV:</strong> for each analyte, the CV of plate medians "
        "across all plates. This measures how consistently an analyte is measured between plates. "
        "Lower CV indicates better reproducibility in both panels."
    ),
    "lod_comparison": (
        "Scatter plot comparing LOD values from different sources for each protein. "
        "Up to three LOD sources are available: Reported LOD (from the NPX data file), "
        "NCLOD (computed from negative controls), and FixedLOD (from Olink reference files). "
        "Each point represents one protein — points on the diagonal indicate agreement between "
        "the two sources. Use the dropdown to switch between available LOD source pairs. "
        "The Pearson correlation (r) and number of proteins (n) are shown. "
        "Large deviations from the diagonal may indicate batch effects or sample quality issues."
    ),
}

_SECTION_ORDER = [
    ("Quality Overview", ["lod_comparison", "qc_summary"]),
    ("Signal & Distribution", ["distribution", "lod_analysis"]),
    ("Data Completeness", ["data_completeness"]),
    ("Sample Relationships", ["dimreduction", "correlation", "heatmap"]),
    ("Normalization QC", ["norm_scale"]),
    ("Variability", ["cv_distribution", "plate_cv"]),
]

_CSS = """\
:root {
    --primary: #5bc0be;
    --primary-dark: #4a9e9c;
    --accent: #5bc0be;
    --link: #3b94d9;
    --bg: #f8f8f9;
    --card: #ffffff;
    --border: #e8eaec;
    --text: #495060;
    --text-muted: #888;
    --heading: #454548;
    --sidebar-width: 230px;
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    margin: 0; padding: 0;
    background: var(--bg); color: var(--text); line-height: 1.5;
}
/* --- Layout --- */
.layout-wrapper {
    display: flex; min-height: 100vh;
}
/* --- Sidebar TOC --- */
nav.toc {
    position: sticky; top: 0; align-self: flex-start;
    width: var(--sidebar-width); min-width: var(--sidebar-width);
    height: 100vh; overflow-y: auto;
    background: var(--heading); color: white;
    padding: 20px 0; flex-shrink: 0;
    scrollbar-width: thin; scrollbar-color: rgba(255,255,255,0.2) transparent;
}
nav.toc .toc-header {
    padding: 0 18px 14px 18px; margin-bottom: 6px;
    border-bottom: 1px solid rgba(255,255,255,0.1);
    font-size: 0.78em; text-transform: uppercase; letter-spacing: 0.1em;
    color: rgba(255,255,255,0.5); font-weight: 600;
}
nav.toc .toc-group-label {
    padding: 12px 18px 4px 18px; margin: 0;
    font-size: 0.7em; text-transform: uppercase; letter-spacing: 0.08em;
    color: rgba(255,255,255,0.4); font-weight: 600;
}
nav.toc ul { list-style: none; padding: 0; margin: 0; }
nav.toc li { margin: 0; }
nav.toc a {
    color: rgba(255,255,255,0.8); text-decoration: none; font-size: 0.82em;
    padding: 5px 18px 5px 22px; display: block;
    border-left: 3px solid transparent;
    transition: all 0.15s ease;
}
nav.toc a:hover {
    background: rgba(91,192,190,0.15); color: white;
    border-left-color: var(--primary);
}
nav.toc a.active {
    background: rgba(91,192,190,0.2); color: white;
    border-left-color: var(--primary); font-weight: 500;
}
/* --- Main content --- */
.main-content {
    flex: 1; max-width: 1100px; padding: 28px 36px; min-width: 0;
}
/* --- Header --- */
header {
    background: linear-gradient(135deg, #5bc0be 0%, #4a9e9c 100%);
    color: white; padding: 28px 32px; border-radius: 8px;
    margin-bottom: 28px;
    box-shadow: 0 2px 12px rgba(91,192,190,0.2);
}
header h1 { margin: 0 0 14px 0; font-size: 1.5em; font-weight: 600; }
.stats { display: flex; gap: 12px; flex-wrap: wrap; }
.stat-item {
    background: rgba(255,255,255,0.15); padding: 7px 14px;
    border-radius: 6px; font-size: 0.88em; backdrop-filter: blur(4px);
}
.stat-item strong { font-weight: 600; margin-right: 4px; }
/* --- Section groups --- */
.section-group { margin-bottom: 32px; }
.section-group > h2 {
    font-size: 1.05em; color: var(--heading); margin: 0 0 14px 0;
    padding-bottom: 6px; border-bottom: 2px solid var(--primary);
    letter-spacing: 0.02em;
}
/* --- Plot cards --- */
.plot-card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 8px; padding: 18px 22px; margin-bottom: 18px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    transition: box-shadow 0.2s;
}
.plot-card:hover { box-shadow: 0 2px 8px rgba(0,0,0,0.07); }
.plot-header { display: flex; align-items: center; gap: 10px; margin-bottom: 2px; }
.plot-header h3 { margin: 0; font-size: 0.98em; color: var(--heading); font-weight: 600; }
.help-toggle {
    display: inline-flex; align-items: center; justify-content: center;
    width: 20px; height: 20px; border-radius: 50%;
    background: var(--primary); color: white; font-size: 12px;
    font-weight: 700; cursor: pointer; border: none;
    flex-shrink: 0; line-height: 1; user-select: none;
    transition: background 0.2s;
}
.help-toggle:hover { background: var(--primary-dark); }
.help-text {
    display: none; background: #f5fbfb; border-left: 3px solid var(--primary);
    padding: 10px 14px; margin: 8px 0 10px 0; border-radius: 0 6px 6px 0;
    font-size: 0.84em; color: var(--text); line-height: 1.6;
}
.help-text.open { display: block; }
/* --- Footer --- */
footer {
    text-align: center; padding: 24px 0 12px 0;
    color: var(--text-muted); font-size: 0.8em;
    border-top: 1px solid var(--border); margin-top: 16px;
}
/* --- Responsive --- */
@media (max-width: 900px) {
    .layout-wrapper { flex-direction: column; }
    nav.toc {
        position: static; width: 100%; height: auto;
        padding: 10px 0;
    }
    nav.toc .toc-group-label { display: none; }
    nav.toc ul { display: flex; flex-wrap: wrap; gap: 2px 4px; padding: 0 12px; }
    nav.toc a { padding: 4px 10px; border-left: none; border-radius: 4px; font-size: 0.8em; }
    nav.toc a:hover, nav.toc a.active { border-left: none; }
    .main-content { padding: 16px; }
}
/* --- Dim-reduction toggle switch --- */
.dimred-toggle {
    display: inline-flex; align-items: center; gap: 8px;
    margin-left: auto; margin-right: 8px;
}
.dimred-label {
    font-size: 0.82em; font-weight: 500; color: var(--text-muted);
    transition: color 0.2s;
}
.dimred-label-active { color: var(--heading); font-weight: 600; }
.toggle-switch {
    position: relative; display: inline-block;
    width: 40px; height: 22px; cursor: pointer;
}
.toggle-switch input { opacity: 0; width: 0; height: 0; }
.toggle-slider {
    position: absolute; inset: 0;
    background: var(--primary); border-radius: 22px;
    transition: background 0.3s;
}
.toggle-slider::before {
    content: ""; position: absolute;
    width: 16px; height: 16px; left: 3px; bottom: 3px;
    background: white; border-radius: 50%;
    transition: transform 0.3s;
}
.toggle-switch input:checked + .toggle-slider::before {
    transform: translateX(18px);
}
/* --- PRIDE embedded mode --- */
body.pride-embedded {
    --bg: transparent;
    --border: #e8eaec;
    background: transparent;
}
body.pride-embedded .layout-wrapper { display: block; }
body.pride-embedded nav.toc,
body.pride-embedded header,
body.pride-embedded footer { display: none; }
body.pride-embedded .main-content { max-width: none; padding: 8px; }
body.pride-embedded .plot-card { border-color: #e8eaec; box-shadow: none; }
body.pride-embedded .pride-embedded-empty { display: block; }
"""

_JS = """\
document.addEventListener('DOMContentLoaded', function() {
    // Help toggle
    document.querySelectorAll('.help-toggle').forEach(function(btn) {
        btn.addEventListener('click', function() {
            var helpEl = this.closest('.plot-card').querySelector('.help-text');
            if (helpEl) {
                helpEl.classList.toggle('open');
                this.textContent = helpEl.classList.contains('open') ? '\\u00d7' : '?';
            }
        });
    });
    // Scroll-spy: highlight active TOC link
    var tocLinks = document.querySelectorAll('nav.toc a[href^=\"#\"]');
    if (tocLinks.length > 0) {
        var sections = [];
        tocLinks.forEach(function(a) {
            var id = a.getAttribute('href').slice(1);
            var el = document.getElementById(id);
            if (el) sections.push({el: el, link: a});
        });
        function updateActive() {
            var scrollY = window.scrollY + 120;
            var active = null;
            for (var i = sections.length - 1; i >= 0; i--) {
                if (sections[i].el.offsetTop <= scrollY) { active = sections[i]; break; }
            }
            tocLinks.forEach(function(a) { a.classList.remove('active'); });
            if (active) active.link.classList.add('active');
        }
        window.addEventListener('scroll', updateActive, {passive: true});
        updateActive();
    }
    // Dimensionality reduction toggle (PCA ↔ t-SNE)
    window.toggleDimRed = function(checkbox) {
        var pcaPanel = document.getElementById('dimred-pca');
        var tsnePanel = document.getElementById('dimred-tsne');
        var lblPca = document.getElementById('dimred-lbl-pca');
        var lblTsne = document.getElementById('dimred-lbl-tsne');
        if (!pcaPanel || !tsnePanel) return;
        if (checkbox.checked) {
            pcaPanel.style.display = 'none';
            tsnePanel.style.display = 'block';
            lblPca.classList.remove('dimred-label-active');
            lblTsne.classList.add('dimred-label-active');
            // Trigger Plotly resize so the chart fills its container
            var tsPlot = tsnePanel.querySelector('.plotly-graph-div');
            if (tsPlot && window.Plotly) Plotly.Plots.resize(tsPlot);
        } else {
            tsnePanel.style.display = 'none';
            pcaPanel.style.display = 'block';
            lblTsne.classList.remove('dimred-label-active');
            lblPca.classList.add('dimred-label-active');
            var pcaPlot = pcaPanel.querySelector('.plotly-graph-div');
            if (pcaPlot && window.Plotly) Plotly.Plots.resize(pcaPlot);
        }
    };
    // PRIDE iframe embedding
    if (window.self !== window.top) {
        document.body.classList.add('pride-embedded');
        var emptyEl = document.querySelector('.pride-embedded-empty');
        if (emptyEl && document.querySelectorAll('.plot-card').length > 0) {
            emptyEl.style.display = 'none';
        }
        function notifyHeight() {
            window.parent.postMessage(
                {type: 'pride-qc-resize', height: document.body.scrollHeight}, '*'
            );
        }
        if (typeof ResizeObserver !== 'undefined') {
            new ResizeObserver(notifyHeight).observe(document.body);
        }
        window.addEventListener('load', notifyHeight);
        window.addEventListener('resize', notifyHeight);
    }
});
"""


def _lod_source_info(dataset: AffinityDataset) -> dict[str, Any]:
    """Detect which LOD sources are available and which one is active."""
    from pyprideap.processing.lod import (
        _MIN_CONTROLS_FOR_LOD,
        _find_negative_controls,
        get_bundled_fixed_lod_path,
        get_reported_lod,
    )

    info: dict[str, Any] = {"active": None, "sources": []}
    sources: list[dict[str, str]] = []

    # 1. Reported LOD
    reported = get_reported_lod(dataset)
    if reported is not None:
        if hasattr(reported, "shape") and reported.ndim == 2:
            n_assays = int(reported.notna().any(axis=0).sum())
        else:
            n_assays = int(reported.notna().sum())
        sources.append(
            {
                "name": "Reported LOD",
                "status": "available",
                "detail": f"LOD column in NPX file ({n_assays} assays)",
            }
        )
        if info["active"] is None:
            info["active"] = "Reported LOD"
    else:
        sources.append({"name": "Reported LOD", "status": "unavailable", "detail": "No LOD column in data file"})

    # 2. NCLOD from negative controls
    try:
        nc_mask = _find_negative_controls(dataset)
        n_controls = int(nc_mask.sum())
        if n_controls >= _MIN_CONTROLS_FOR_LOD:
            sources.append(
                {
                    "name": "NCLOD",
                    "status": "available",
                    "detail": f"Computed from {n_controls} negative control samples",
                }
            )
            if info["active"] is None:
                info["active"] = "NCLOD"
        else:
            sources.append(
                {
                    "name": "NCLOD",
                    "status": "insufficient",
                    "detail": f"Only {n_controls} negative controls (need \u2265{_MIN_CONTROLS_FOR_LOD})",
                }
            )
    except (ValueError, KeyError):
        has_st = "SampleType" in dataset.samples.columns
        sources.append(
            {
                "name": "NCLOD",
                "status": "unavailable",
                "detail": "No SampleType column" if not has_st else "No negative control samples found",
            }
        )

    # 3. Platform-specific LOD sources
    from pyprideap.core import Platform

    if dataset.platform == Platform.SOMASCAN:
        # SomaScan: eLOD from buffer samples (not FixedLOD)
        try:
            from pyprideap.processing.lod import compute_soma_elod

            compute_soma_elod(dataset)
            sources.append(
                {
                    "name": "eLOD",
                    "status": "available",
                    "detail": "Computed from buffer samples (MAD formula)",
                }
            )
            if info["active"] is None:
                info["active"] = "eLOD"
        except (ValueError, KeyError, ImportError):
            sources.append(
                {
                    "name": "eLOD",
                    "status": "unavailable",
                    "detail": "No buffer samples found for eLOD computation",
                }
            )
    else:
        # Olink: FixedLOD from bundled configs
        fixed_path = get_bundled_fixed_lod_path(dataset.platform)
        if fixed_path is not None:
            sources.append(
                {
                    "name": "FixedLOD",
                    "status": "available",
                    "detail": f"Bundled reference file ({fixed_path.name})",
                }
            )
        else:
            sources.append(
                {
                    "name": "FixedLOD",
                    "status": "unavailable",
                    "detail": f"No bundled file for {dataset.platform.value}",
                }
            )

    info["sources"] = sources
    info["platform"] = dataset.platform.value
    return info


def _render_lod_card(lod_info: dict[str, Any]) -> str:
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

    # Warning banner when no LOD source is available
    no_lod = lod_info["active"] is None
    warning_html = ""
    if no_lod:
        has_fallback = any(
            s["name"] in ("FixedLOD", "eLOD") and s["status"] == "available" for s in lod_info["sources"]
        )
        if has_fallback:
            fallback_name = next(
                s["name"]
                for s in lod_info["sources"]
                if s["name"] in ("FixedLOD", "eLOD") and s["status"] == "available"
            )
            warning_html = (
                '<div style="background:#fff3cd;border:1px solid #ffc107;border-radius:6px;'
                'padding:10px 14px;margin-top:10px;color:#856404;font-size:0.92em;">'
                "\u26a0\ufe0f <strong>Warning:</strong> No Reported LOD or NCLOD available. "
                "LOD-dependent plots (QC Summary, LOD Analysis) will not be shown. "
                f"Consider using <strong>{fallback_name}</strong> for this platform."
                "</div>"
            )
        else:
            warning_html = (
                '<div style="background:#f8d7da;border:1px solid #f5c6cb;border-radius:6px;'
                'padding:10px 14px;margin-top:10px;color:#721c24;font-size:0.92em;">'
                "\u26a0\ufe0f <strong>Warning:</strong> No LOD source is available for this dataset. "
                "No Reported LOD in the data file and insufficient negative controls for NCLOD. "
                "LOD-dependent plots (QC Summary, LOD Analysis) will not be shown."
                "</div>"
            )

    return (
        '<div class="plot-card" id="lod-sources" style="margin-bottom:24px;">'
        '<div class="plot-header">'
        "<h3>LOD Sources</h3>"
        '<button class="help-toggle" title="About LOD sources" aria-label="Help">?</button>'
        "</div>"
        '<div class="help-text">'
        "The Limit of Detection (LOD) determines whether a measured protein signal is above background noise. "
        + (
            "Three LOD sources are supported: "
            "<strong>Reported LOD</strong> comes from the LOD column in the original data file (per sample and assay). "
            "<strong>NCLOD</strong> is computed from negative control samples using the formula "
            "LOD = median(NC) + max(0.2, 3&times;SD(NC)). Requires &ge;10 negative controls. "
            "<strong>eLOD</strong> is computed from buffer samples using a MAD-based formula. "
            "The report uses the first available source (Reported &gt; NCLOD &gt; eLOD)."
            if lod_info.get("platform") == "somascan"
            else "Three LOD sources are supported: "
            "<strong>Reported LOD</strong> comes from the LOD column in the original NPX file (per sample and assay). "
            "<strong>NCLOD</strong> is computed from negative control samples using the formula "
            "LOD = median(NC) + max(0.2, 3&times;SD(NC)), following the OlinkAnalyze R package. "
            "Requires &ge;10 negative controls. "
            "<strong>FixedLOD</strong> is a pre-computed reference from Olink, specific to the reagent lot and "
            "Data Analysis Reference ID. "
            "The report uses the first available source (Reported &gt; NCLOD &gt; FixedLOD)."
        )
        + "</div>"
        '<table style="width:100%;border-collapse:collapse;margin-top:8px;">'
        f"{''.join(rows)}"
        "</table>"
        f"{warning_html}"
        "</div>"
    )


def _count_proteins_above_lod(dataset: AffinityDataset) -> int | None:
    """Count unique proteins where >50% of samples are above LOD."""
    import pandas as pd

    try:
        from pyprideap.processing.lod import _above_lod_matrix, get_lod_values
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

    from pyprideap.viz.qc import render as R

    output = Path(output)
    plot_data = compute_all(dataset)

    _RENDERERS = {
        "lod_comparison": (LodComparisonData, R.render_lod_comparison),
        "distribution": (DistributionData, R.render_distribution),
        "qc_summary": (QcLodSummaryData, R.render_qc_summary),
        "lod_analysis": (LodAnalysisData, R.render_lod_analysis),
        "heatmap": (HeatmapData, R.render_heatmap),
        "correlation": (CorrelationData, R.render_correlation),
        "data_completeness": (DataCompletenessData, R.render_data_completeness),
        "cv_distribution": (CvDistributionData, R.render_cv_distribution),
        "plate_cv": (PlateCvData, R.render_plate_cv),
        "norm_scale": (NormScaleData, R.render_norm_scale),
    }

    # Determine display order from _SECTION_ORDER so first-displayed plot gets plotly.js
    display_order: list[str] = []
    for _, keys in _SECTION_ORDER:
        for key in keys:
            display_order.append(key)

    # Build rendered sections keyed by plot id
    rendered: dict[str, tuple[str, ...]] = {}  # key -> (title, html[, extra_header])
    first_key = None

    # Handle combined dimensionality reduction (PCA + t-SNE in one panel with toggle)
    _pca_raw = plot_data.get("pca")
    _umap_raw = plot_data.get("umap")  # actually t-SNE data
    pca_data: PcaData | None = _pca_raw if isinstance(_pca_raw, PcaData) else None
    umap_data: UmapData | None = _umap_raw if isinstance(_umap_raw, UmapData) else None
    has_dimred = pca_data is not None or umap_data is not None
    if has_dimred:
        # Check if dimreduction is the first plot in display order
        for key in display_order:
            if key == "dimreduction":
                first_key = "dimreduction"
                break
            if key in _RENDERERS and plot_data.get(key) is not None:
                first_key = key
                break

        dimred_parts: list[str] = []
        need_plotly_cdn = first_key == "dimreduction"

        if pca_data is not None:
            pca_fig = R.render_pca(pca_data)
            pca_fig.update_layout(height=500)
            js = "cdn" if need_plotly_cdn else False
            need_plotly_cdn = False  # only include CDN once
            pca_html = pca_fig.to_html(full_html=False, include_plotlyjs=js, default_height="500px")
            dimred_parts.append(f'<div class="dimred-panel" id="dimred-pca">{pca_html}</div>')

        if umap_data is not None:
            tsne_fig = R.render_tsne(umap_data)
            tsne_fig.update_layout(height=500)
            js = "cdn" if need_plotly_cdn else False
            tsne_html = tsne_fig.to_html(full_html=False, include_plotlyjs=js, default_height="500px")
            hidden = ' style="display:none"' if pca_data is not None else ""
            dimred_parts.append(f'<div class="dimred-panel" id="dimred-tsne"{hidden}>{tsne_html}</div>')

        nl_method = umap_data.title if umap_data is not None else "t-SNE"
        combined_html = "".join(dimred_parts)

        # Build toggle switch only if both methods available
        if pca_data is not None and umap_data is not None:
            toggle_html = (
                '<div class="dimred-toggle">'
                '<span class="dimred-label dimred-label-active" id="dimred-lbl-pca">PCA</span>'
                '<label class="toggle-switch">'
                '<input type="checkbox" id="dimred-switch" onchange="toggleDimRed(this)">'
                '<span class="toggle-slider"></span>'
                "</label>"
                f'<span class="dimred-label" id="dimred-lbl-tsne">{nl_method}</span>'
                "</div>"
            )
        else:
            toggle_html = ""

        rendered["dimreduction"] = ("Dimensionality Reduction", combined_html, toggle_html)

    # Find first key if not already set
    if first_key is None:
        for key in display_order:
            if key in _RENDERERS and plot_data.get(key) is not None:
                first_key = key
                break

    for key, (dtype, renderer) in _RENDERERS.items():
        data = plot_data.get(key)
        if data is None or not isinstance(data, dtype):
            continue
        fig = renderer(data)  # type: ignore[operator]
        # Multi-panel renderers set their own height (e.g. 800px); only
        # apply the default for plots that haven't specified one.
        current_height = fig.layout.height
        if current_height is None:
            fig.update_layout(height=500)
        plot_height = f"{fig.layout.height}px"
        js = "cdn" if key == first_key else False
        plot_html = fig.to_html(full_html=False, include_plotlyjs=js, default_height=plot_height)
        rendered[key] = (data.title, plot_html)  # type: ignore[attr-defined]

    # Build grouped sections and TOC with section labels
    toc_html_parts: list[str] = []
    group_sections: list[str] = []

    for group_title, keys in _SECTION_ORDER:
        cards: list[str] = []
        group_toc_items: list[str] = []
        for key in keys:
            if key not in rendered:
                continue
            entry = rendered[key]
            title = entry[0]
            plot_html = entry[1]
            extra_header = entry[2] if len(entry) > 2 else ""
            help_html = _HELP_TEXT.get(key, "")
            group_toc_items.append(f'<li><a href="#{key}">{title}</a></li>')
            help_block = f'<div class="help-text">{help_html}</div>' if help_html else ""
            cards.append(
                f'<div class="plot-card" id="{key}">'
                f'<div class="plot-header">'
                f"<h3>{title}</h3>"
                f"{extra_header}"
                f'<button class="help-toggle" title="How to read this plot" aria-label="Help">?</button>'
                f"</div>"
                f"{help_block}"
                f"{plot_html}"
                f"</div>"
            )
        if cards:
            toc_html_parts.append(f'<div class="toc-group-label">{group_title}</div>')
            toc_html_parts.append(f"<ul>{''.join(group_toc_items)}</ul>")
            group_sections.append(f'<div class="section-group"><h2>{group_title}</h2>{"".join(cards)}</div>')
    toc_inner = "".join(toc_html_parts)

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
        f'<span class="stat-item"><strong>Proteins (Assays)</strong> {n_features}</span>',
    ]
    if n_proteins > 0:
        stat_items.append(f'<span class="stat-item"><strong>Proteins</strong> {n_proteins}</span>')
    if n_proteins_above_lod is not None:
        stat_items.append(f'<span class="stat-item"><strong>Proteins &gt; LOD</strong> {n_proteins_above_lod}</span>')

    html = (
        '<!DOCTYPE html>\n<html lang="en">\n<head>\n'
        '    <meta charset="utf-8">\n'
        '    <meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"    <title>QC Report \u2014 {platform_label}</title>\n"
        f"    <style>\n{_CSS}    </style>\n"
        "</head>\n<body>\n"
        '<div class="layout-wrapper">\n'
        f'    <nav class="toc"><div class="toc-header">Contents</div>{toc_inner}</nav>\n'
        '    <div class="main-content">\n'
        "        <header>\n"
        f"            <h1>QC Report &mdash; {platform_label}</h1>\n"
        f'            <div class="stats">\n'
        f"                {''.join(stat_items)}\n"
        f"            </div>\n"
        "        </header>\n"
        f"        {lod_card_html}\n"
        '        <div class="pride-embedded-empty" style="display:none;text-align:center;'
        'padding:40px;color:#777;">No QC plots available for this dataset.</div>\n'
        f"        {''.join(group_sections)}\n"
        f"        <footer>Generated by <strong>pyprideap</strong></footer>\n"
        "    </div>\n"
        "</div>\n"
        f"    <script>\n{_JS}    </script>\n"
        "</body>\n</html>"
    )

    output.write_text(html)
    return output
