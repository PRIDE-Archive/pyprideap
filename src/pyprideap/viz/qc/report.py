from __future__ import annotations

import html as html_mod
import logging
from pathlib import Path
from typing import Any

import numpy as np

from pyprideap.core import AffinityDataset
from pyprideap.viz.qc.compute import (
    ColCheckData,
    CorrelationData,
    CvDistributionData,
    DataCompletenessData,
    DistributionData,
    HeatmapData,
    IqrMedianQcData,
    LodAnalysisData,
    LodComparisonData,
    NormScaleData,
    PcaData,
    PlateCvData,
    QcLodSummaryData,
    UmapData,
    UniProtDuplicateData,
    VolcanoData,
    compute_all,
    compute_volcano,
)

logger = logging.getLogger(__name__)

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
        "Proteins ranked by the percentage of samples with signal above the Limit of Detection, "
        "coloured by panel. A steep drop-off reveals how many proteins have weak signal. "
        "Proteins with low %% above LOD are unreliable and may need filtering."
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
        "profiles. Samples are grouped by sample type so that controls, biological samples, etc. cluster "
        "together. Values range from \u22121 (inverse) to +1 (perfect correlation). In a well-behaved "
        "experiment most biological sample pairs should show high positive correlation (warm colours). "
        "Control samples (e.g. plate controls, negative controls) are expected to have low correlation "
        "with biological samples. A biological sample with consistently low correlation against all others "
        "is a potential outlier."
    ),
    "data_completeness": (
        "Two stacked panels showing data completeness based on the Limit of Detection. "
        "Note: control samples (negative controls, plate controls, etc.) are excluded — only biological "
        "samples are shown. "
        "<strong>Top:</strong> per-sample stacked bar showing Above LOD (green, reliable signal) vs Below LOD "
        "(orange, measured but below detection limit). A sample with a large orange fraction may "
        "indicate low protein input or technical issues. "
        "<strong>Bottom:</strong> Missing Frequency distribution — a histogram of per-protein missing rate "
        "(%% of samples where signal is below LOD). Olink recommends a missing frequency threshold "
        "of 30%%: proteins above this threshold may be unreliable and should be considered for filtering. "
        "Proteins clustered near 0%% are reliably detected; those near 100%% may need filtering."
    ),
    "sample_completeness": (
        "Per-sample stacked bar showing Above LOD (green, reliable signal) vs Below LOD "
        "(orange, measured but below detection limit). Control samples are excluded. "
        "A sample with a large orange fraction may indicate low protein input or technical issues."
    ),
    "missing_frequency_distribution": (
        "Histogram of per-protein missing rate (%% of samples where signal is below LOD). "
        "Olink recommends a missing frequency threshold of 30%%: proteins above this threshold "
        "may be unreliable and should be considered for filtering. "
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
        "Histogram of the coefficient of variation (CV = standard deviation / mean) across all proteins. "
        "CV measures relative variability: values below 0.2 (dashed line) indicate tight reproducibility, "
        "while a long right tail suggests some proteins have high technical or biological variability."
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
        "Available sources: <strong>Reported LOD</strong> (from the data file), "
        "<strong>NCLOD</strong> (computed from negative controls), and "
        "<strong>FixedLOD</strong> (from Olink reference files) or <strong>eLOD</strong> (SomaScan buffer-based). "
        "Each point represents one protein — points on the diagonal indicate agreement between "
        "the two sources. Use the dropdown to switch between source pairs. "
        "The Pearson correlation (r) and number of assays (n) are shown; n may be lower than total assays "
        "because only assays with valid LOD from both sources are included. "
        "When sources disagree significantly, NCLOD (experiment-specific) is generally preferred for "
        "filtering, as it reflects the actual noise level in your experiment. "
        "The LOD source used for all other analyses in this report is shown in the LOD Sources card above."
    ),
    "outlier_map": (
        "Heatmap showing MAD-based statistical outliers across all samples and analytes. "
        "Red cells indicate that a sample's RFU value for that analyte exceeded both "
        "|x &minus; median(x)| &gt; 6 &times; MAD(x) <strong>and</strong> a fold-change &gt; 5&times; "
        "from the analyte median (matching SomaDataIO's <code>calcOutlierMap()</code>). "
        "Samples with a large number of red cells across many analytes may be technical outliers. "
        "The default flagging threshold is 5%% of analytes — samples exceeding this are candidates for removal."
    ),
    "row_check": (
        "Summary of sample-level normalization QC (RowCheck). Each sample's normalization scale factors "
        "(e.g. HybControlNormScale) are checked against the acceptance range [0.4, 2.5]. "
        "Samples where <strong>all</strong> normalization scales fall within this range receive PASS; "
        "others are flagged as FLAG. Flagged samples may have experienced hybridization issues or "
        "unusually high/low overall signal and should be considered for exclusion."
    ),
    "col_check": (
        "Calibrator QC ratio for each analyte. The ratio measures how well each analyte's calibration "
        "matches the reference (ideal = 1.0). Analytes within the acceptance range [0.8, 1.2] (orange "
        "dashed lines) pass QC; those outside are flagged (red points). Flagged analytes may have poor "
        "calibration reproducibility and should be interpreted with caution. A high number of flagged "
        "analytes may indicate plate-level calibration issues."
    ),
    "control_analytes": (
        "Breakdown of control analyte types detected in the SomaScan data. "
        "<strong>HybControlElution</strong>: hybridization control probes used to monitor assay performance. "
        "<strong>Spuriomer</strong>: aptamers with known non-specific binding. "
        "<strong>NonBiotin</strong>: non-biotinylated controls. "
        "<strong>NonHuman</strong>: non-human protein targets (e.g. bacterial, viral). "
        "<strong>NonCleavable</strong>: non-cleavable linker controls. "
        "These are typically removed before downstream analysis (equivalent to "
        "<code>getAnalytes(rm.controls=TRUE)</code> in SomaDataIO)."
    ),
    "norm_scale_boxplot": (
        "Boxplots of normalization scale factors (e.g. HybControlNormScale, Med.Scale.*) grouped by "
        "a categorical variable (e.g. PlateId). Red dashed lines mark the acceptance thresholds at "
        "0.4 and 2.5 — samples outside these bounds fail RowCheck. Equivalent to the <code>data.qc</code> "
        "plots in SomaDataIO's <code>preProcessAdat()</code>. "
        "Look for systematic differences between groups that might indicate batch effects."
    ),
    "iqr_median_qc": (
        "Per-panel scatter plot of IQR (Interquartile Range) vs Median NPX per sample, mirroring "
        "OlinkAnalyze's <code>olink_qc_plot()</code>. The IQR (Q3 &minus; Q1) measures the spread of "
        "the middle 50%% of a sample's protein measurements — a large IQR indicates high variability. "
        "For each panel, IQR and median are computed per sample. "
        "Outlier thresholds (dashed lines) are set at mean &plusmn; 3 &times; SD on both axes. "
        "Samples outside these bounds on either axis are flagged as outliers (red points). "
        "Points are coloured by QC status (green = Pass, orange = Warning). Outlier samples may "
        "indicate technical issues such as failed wells, low protein input, or plate effects."
    ),
    "uniprot_duplicates": (
        "Bar chart showing proteins targeted by multiple assays (e.g. multiple OlinkIDs or aptamers "
        "mapping to the same UniProt accession). In some platforms this is <strong>by design</strong> — "
        "e.g. SomaScan includes replicate aptamers for redundancy, and Olink panels may overlap. "
        "Multiple assays per protein can be used to assess measurement concordance: "
        "high agreement between replicate assays confirms reliability, while poor concordance "
        "may indicate assay-specific issues. "
        "For downstream pathway or gene-set enrichment analyses, be aware that duplicated proteins "
        "may inflate counts unless handled (e.g. by averaging or selecting the best-performing assay)."
    ),
    "differential_expression": (
        "Volcano plots showing differentially expressed proteins between sample groups defined "
        "in the SDRF metadata file. Each dot is a protein; the x-axis shows fold change "
        "(log2 scale for Olink, log2-RFU for SomaScan) and the y-axis shows statistical "
        "significance (-log10 adjusted p-value). Coloured dots are significant (|FC| &ge; 1 "
        "and adj. p &lt; 0.05). Use the <strong>dropdown</strong> to switch between different "
        "characteristics or factor values. For variables with exactly 2 groups a Welch t-test "
        "is used; for variables with 3-10 groups every pairwise comparison is shown."
    ),
}

_SECTION_ORDER = [
    ("Quality Overview", ["lod_comparison", "qc_summary"]),
    ("Signal & Distribution", ["distribution", "lod_analysis"]),
    ("Sample Completeness", ["sample_completeness"]),
    ("Missing Frequency Distribution", ["missing_frequency_distribution"]),
    ("Sample Relationships", ["dimreduction", "correlation", "heatmap"]),
    ("Normalization QC", ["norm_scale"]),
    ("Variability", ["cv_distribution", "plate_cv"]),
    ("Assay QC", ["iqr_median_qc", "uniprot_duplicates"]),
    ("SomaScan QC", ["col_check"]),
    ("Differential Expression", ["differential_expression"]),
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
/* --- Label toggle button --- */
.label-toggle-btn {
    display: inline-flex; align-items: center; gap: 5px;
    margin-left: 8px; padding: 3px 10px;
    font-size: 0.78em; font-weight: 500;
    background: #f0f0f0; border: 1px solid var(--border);
    border-radius: 4px; cursor: pointer; color: var(--text);
    transition: all 0.2s;
}
.label-toggle-btn:hover { background: var(--primary); color: white; border-color: var(--primary); }
.label-toggle-btn.active { background: var(--primary); color: white; border-color: var(--primary); }
/* --- Summary table --- */
.summary-table {
    width: 100%; border-collapse: collapse; font-size: 0.9em;
}
.summary-table td { padding: 6px 12px; border-bottom: 1px solid var(--border); }
.summary-table .summary-group td {
    background: #e8f8f8; font-weight: 600; color: var(--heading);
    font-size: 0.88em; text-transform: uppercase; letter-spacing: 0.04em;
    padding: 10px 12px; border-bottom: 2px solid var(--primary);
}
.summary-table .metric-name { color: var(--text); }
.summary-table .metric-value { font-weight: 500; text-align: right; }
.summary-table .metric-sub {
    font-size: 0.85em; color: var(--text-muted); padding-left: 8px;
}
.status-dot {
    display: inline-block; width: 12px; height: 12px; border-radius: 50%;
    vertical-align: middle;
}
.dot-green { background: #2ecc71; }
.dot-amber { background: #f39c12; }
.dot-red { background: #e74c3c; }
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
.volcano-controls { margin: 12px 0; display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.volcano-controls select {
    padding: 6px 10px; border: 1px solid var(--border); border-radius: 6px;
    font-size: 0.92em; background: var(--card);
}
.volcano-comp-select { margin: 4px 0 8px 0; }
.volcano-comp-select select {
    padding: 5px 8px; border: 1px solid var(--border); border-radius: 6px;
    font-size: 0.88em; background: var(--card);
}
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
    // Label toggle for dimensionality reduction plots
    window.toggleDimRedLabels = function(btn) {
        var card = btn.closest('.plot-card');
        if (!card) return;
        var plots = card.querySelectorAll('.plotly-graph-div');
        var showLabels = !btn.classList.contains('active');
        btn.classList.toggle('active');
        btn.textContent = showLabels ? 'Hide Labels' : 'Show Labels';
        plots.forEach(function(plot) {
            if (!plot.data) return;
            var update = {mode: showLabels ? 'markers+text' : 'markers'};
            var indices = [];
            for (var i = 0; i < plot.data.length; i++) {
                if (plot.data[i].type === 'scatter' || plot.data[i].type === 'scattergl') {
                    indices.push(i);
                }
            }
            if (indices.length > 0) {
                Plotly.restyle(plot, update, indices);
            }
        });
    };
    // PRIDE iframe embedding
    if (window.self !== window.top) {
        document.body.classList.add('pride-embedded');
        var emptyEl = document.querySelector('.pride-embedded-empty');
        if (emptyEl && document.querySelectorAll('.plot-card').length > 0) {
            emptyEl.style.display = 'none';
        }
        function notifyHeight() {
            // Wildcard origin is intentional: the report can be embedded on
            // any host and the payload contains only a non-sensitive height value.
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

// Volcano plot variable / comparison switching
function switchVolcanoVar(varName) {
    // Hide all volcano plots and comparison selectors
    document.querySelectorAll('.volcano-plot').forEach(function(el) {
        el.style.display = 'none';
    });
    document.querySelectorAll('.volcano-comp-select').forEach(function(el) {
        el.style.display = 'none';
    });
    // Show first comparison for selected variable
    var plots = document.querySelectorAll('.volcano-plot[data-var="' + varName + '"]');
    if (plots.length > 0) {
        plots[0].style.display = '';
        // Trigger Plotly resize for the newly visible plot
        var plotDiv = plots[0].querySelector('.plotly-graph-div');
        if (plotDiv && window.Plotly) { window.Plotly.Plots.resize(plotDiv); }
    }
    // Show comparison selector if multiple comparisons
    var compSel = document.querySelector('.volcano-comp-select[data-var="' + varName + '"]');
    if (compSel) {
        compSel.style.display = '';
        // Reset to first option
        var sel = compSel.querySelector('select');
        if (sel) sel.value = '0';
    }
}

function switchVolcanoComp(varName, compIdx) {
    document.querySelectorAll('.volcano-plot[data-var="' + varName + '"]').forEach(function(el) {
        el.style.display = (el.getAttribute('data-comp') === compIdx) ? '' : 'none';
        if (el.getAttribute('data-comp') === compIdx) {
            var plotDiv = el.querySelector('.plotly-graph-div');
            if (plotDiv && window.Plotly) { window.Plotly.Plots.resize(plotDiv); }
        }
    });
}
"""


def _compact_fig(fig: Any, precision: int = 4) -> Any:
    """Round float arrays in a Plotly figure to reduce HTML size."""
    for trace in fig.data:
        for attr in ("x", "y", "z"):
            arr = getattr(trace, attr, None)
            if arr is None:
                continue
            try:
                rounded = np.around(np.asarray(arr, dtype=float), decimals=precision)
                setattr(trace, attr, rounded.tolist())
            except (TypeError, ValueError):
                pass  # non-numeric data (e.g. string labels)
    return fig


def _lod_source_info(dataset: AffinityDataset) -> dict[str, Any]:
    """Detect which LOD sources are available and which one is active.

    For Olink datasets we consider three possible sources (Reported LOD,
    NCLOD, FixedLOD) and pick the first available in that priority order.

    For SomaScan datasets, the primary and typically only source is the
    estimated LOD from buffer samples (eLOD). SomaScan ADAT files do not
    usually contain per-sample reported LOD values and there is no
    FixedLOD reference CSV equivalent; instead, LOD is derived from the
    distribution of buffer RFU values.
    """
    from pyprideap.core import Platform
    from pyprideap.processing.lod import (
        _MIN_CONTROLS_FOR_LOD,
        _find_negative_controls,
        get_bundled_fixed_lod_path,
        get_reported_lod,
    )

    info: dict[str, Any] = {"active": None, "sources": []}
    sources: list[dict[str, str]] = []

    if dataset.platform == Platform.SOMASCAN:
        # SomaScan: only consider eLOD from buffer samples.
        try:
            from pyprideap.processing.lod import compute_soma_elod

            compute_soma_elod(dataset)
            sources.append(
                {
                    "name": "eLOD",
                    "status": "available",
                    "detail": "Estimated from buffer RFU using a robust MAD-based formula",
                }
            )
            info["active"] = "eLOD"
        except (ValueError, KeyError, ImportError):
            sources.append(
                {
                    "name": "eLOD",
                    "status": "unavailable",
                    "detail": "No buffer samples (SampleType = 'Buffer') found for eLOD computation",
                }
            )
    else:
        # Olink: Reported LOD, NCLOD, and FixedLOD
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

        # 3. FixedLOD from bundled configs
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
    """Render the LOD source summary as an HTML card.
    Use HTML entities for status icons so the output is ASCII-safe for writing with any encoding.
    """
    status_icons = {"available": "&#9989;", "unavailable": "&#10060;", "insufficient": "&#9888;"}
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
    platform = lod_info.get("platform", "")
    if no_lod:
        if platform == "somascan":
            warning_html = (
                '<div style="background:#f8d7da;border:1px solid #f5c6cb;border-radius:6px;'
                'padding:10px 14px;margin-top:10px;color:#721c24;font-size:0.92em;">'
                "&#9888; <strong>Warning:</strong> No buffer samples (SampleType = 'Buffer') found; "
                "eLOD could not be computed. LOD-dependent plots (QC Summary, LOD Analysis) will not be shown."
                "</div>"
            )
        else:
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
                    "&#9888; <strong>Warning:</strong> No Reported LOD or NCLOD available. "
                    "LOD-dependent plots (QC Summary, LOD Analysis) will not be shown. "
                    f"Consider using <strong>{fallback_name}</strong> for this platform."
                    "</div>"
                )
            else:
                warning_html = (
                    '<div style="background:#f8d7da;border:1px solid #f5c6cb;border-radius:6px;'
                    'padding:10px 14px;margin-top:10px;color:#721c24;font-size:0.92em;">'
                    "&#9888; <strong>Warning:</strong> No LOD source is available for this dataset. "
                    "No Reported LOD in the data file and insufficient negative controls for NCLOD. "
                    "LOD-dependent plots (QC Summary, LOD Analysis) will not be shown."
                    "</div>"
                )

    if platform == "somascan":
        help_text = (
            "The Limit of Detection (LOD) determines whether a measured RFU signal is distinguishable from background "
            "buffer noise. For SomaScan, this report uses the <strong>estimated LOD (eLOD)</strong> computed from "
            "buffer samples only. For each analyte, eLOD is defined as "
            "<code>median(buffer RFU) + 3.3 &times; 1.4826 &times; MAD(buffer RFU)</code>, following the "
            "SomaDataIO <code>calc_eLOD()</code> implementation. The 1.4826 factor converts MAD to a "
            "standard-deviation equivalent; the 3.3 multiplier targets approximately 95% detection probability. "
            "At least one group of buffer samples (SampleType = 'Buffer') is required for eLOD calculation; "
            "Calibrator and QC samples are <em>not</em> used as background."
        )
    else:
        help_text = (
            "The Limit of Detection (LOD) determines whether a measured protein signal is above background noise. "
            "Three LOD sources are supported for Olink datasets: "
            "<strong>Reported LOD</strong> comes from the LOD column in the original NPX file (per sample and assay). "
            "<strong>NCLOD</strong> is computed from negative control samples using the formula "
            "LOD = median(NC) + max(0.2, 3&times;SD(NC)), following the OlinkAnalyze R package and requiring "
            "&ge;10 negative controls. "
            "<strong>FixedLOD</strong> is a pre-computed reference from Olink, specific to the reagent lot and "
            "Data Analysis Reference ID. The report uses the first available source in the order "
            "Reported &gt; NCLOD &gt; FixedLOD."
        )

    return (
        '<div class="plot-card" id="lod-sources" style="margin-bottom:24px;">'
        '<div class="plot-header">'
        "<h3>LOD Sources</h3>"
        '<button class="help-toggle" title="About LOD sources" aria-label="Help">?</button>'
        "</div>"
        f'<div class="help-text">{help_text}</div>'
        '<table style="width:100%;border-collapse:collapse;margin-top:8px;">'
        f"{''.join(rows)}"
        "</table>"
        f"{warning_html}"
        "</div>"
    )


def _count_proteins_above_lod(dataset: AffinityDataset) -> int | None:
    """Count unique proteins where >50% of samples are above LOD."""
    try:
        from pyprideap.processing.lod import get_proteins_above_lod
    except ImportError:
        return None

    proteins = get_proteins_above_lod(dataset)
    return len(proteins) if proteins is not None else None


def _status_dot(level: str) -> str:
    """Return an HTML span for a traffic-light status dot."""
    if level == "green":
        return '<span class="status-dot dot-green"></span>'
    if level == "amber":
        return '<span class="status-dot dot-amber"></span>'
    if level == "red":
        return '<span class="status-dot dot-red"></span>'
    return ""


def _summary_row(dot: str, name: str, value: str) -> str:
    return (
        f"<tr>"
        f'<td style="width:28px;text-align:center;">{dot}</td>'
        f'<td class="metric-name">{name}</td>'
        f'<td class="metric-value">{value}</td>'
        f"</tr>"
    )


def _summary_group(title: str) -> str:
    return f'<tr class="summary-group"><td colspan="3">{title}</td></tr>'


def _render_summary_table(
    dataset: AffinityDataset,
    plot_data: dict[str, object],
    lod_info: dict[str, Any],
) -> str:
    """Build the Dataset Summary HTML table with traffic-light indicators."""
    import pandas as pd

    from pyprideap.viz.qc.compute import CvDistributionData, DataCompletenessData, LodAnalysisData, PlateCvData

    rows: list[str] = []
    samples = dataset.samples
    features = dataset.features
    expr = dataset.expression
    n_samples = len(samples)
    n_features = len(features)

    # --- General ---
    rows.append(_summary_group("General"))
    platform_label = dataset.platform.value.replace("_", " ").title()
    rows.append(_summary_row("", "Platform", platform_label))

    # Plates — check both column name variants
    plate_col = None
    for col in ("PlateID", "PlateId"):
        if col in samples.columns:
            plate_col = col
            break
    if plate_col is not None:
        n_plates = int(samples[plate_col].nunique())
        rows.append(_summary_row("", "Plates", str(n_plates)))

    rows.append(_summary_row("", "Samples", str(n_samples)))

    # Sample types breakdown
    if "SampleType" in samples.columns:
        type_counts = samples["SampleType"].value_counts()
        parts = [f"{html_mod.escape(str(v))}: {c}" for v, c in type_counts.items()]
        rows.append(_summary_row("", "Sample types", ", ".join(parts)))

    rows.append(_summary_row("", "Features (assays)", str(n_features)))

    # --- LOD (right after General) ---
    lod_active = lod_info.get("active")
    lod_analysis = plot_data.get("lod_analysis")
    data_completeness = plot_data.get("data_completeness")
    has_any_lod = lod_active is not None or lod_analysis is not None or data_completeness is not None
    if has_any_lod or lod_info.get("sources"):
        rows.append(_summary_group("Limit of Detection"))

        # Inline LOD sources overview
        status_icons = {"available": "&#9989;", "unavailable": "&#10060;", "insufficient": "&#9888;"}
        for src in lod_info.get("sources", []):
            icon = status_icons.get(src["status"], "")
            active_tag = " <strong>(active)</strong>" if src["name"] == lod_info.get("active") else ""
            rows.append(_summary_row(icon, src["name"] + str(active_tag), src["detail"]))

        # Match the "Missing Frequency distribution" metric (Olink-recommended 30% missing threshold):
        # Missing frequency = fraction of samples below LOD, so <30% missing ⇔ >70% above LOD.
        n_pass_mf = None
        n_total_mf = None
        if isinstance(data_completeness, DataCompletenessData) and len(data_completeness.missing_freq) > 0:
            mf = [float(x) for x in data_completeness.missing_freq if x is not None]
            n_total_mf = len(mf)
            n_pass_mf = sum(1 for x in mf if x < 0.30)
        elif isinstance(lod_analysis, LodAnalysisData) and len(lod_analysis.above_lod_pct) > 0:
            n_total_mf = len(lod_analysis.above_lod_pct)
            n_pass_mf = sum(1 for p in lod_analysis.above_lod_pct if float(p) > 70.0)

        if n_pass_mf is not None and n_total_mf is not None and n_total_mf > 0:
            frac_pass = n_pass_mf / n_total_mf
            if frac_pass > 0.80:
                lod_dot = _status_dot("green")
            elif frac_pass >= 0.50:
                lod_dot = _status_dot("amber")
            else:
                lod_dot = _status_dot("red")
            rows.append(
                _summary_row(
                    lod_dot,
                    "Assays with &lt;30% missing (MissingFreq)",
                    f"{n_pass_mf} / {n_total_mf} ({frac_pass:.1%})",
                )
            )

    # --- Proteins ---
    rows.append(_summary_group("Proteins"))
    if "UniProt" in features.columns:
        n_proteins = int(features["UniProt"].dropna().nunique())
        rows.append(_summary_row("", "Unique proteins", str(n_proteins)))

    if "Panel" in features.columns:
        panel_counts = features["Panel"].value_counts()
        parts = [f"{html_mod.escape(str(p))}: {c}" for p, c in panel_counts.items()]
        rows.append(_summary_row("", "Panels", ", ".join(parts)))

    numeric = expr.apply(pd.to_numeric, errors="coerce")
    proteins_per_sample = numeric.notna().sum(axis=1)
    median_per_sample = float(proteins_per_sample.median())
    rows.append(_summary_row("", "Proteins per sample (median)", f"{median_per_sample:.0f}"))

    # --- Missing Data (only when no LOD info, since LOD-based completeness is more informative) ---
    if not has_any_lod:
        rows.append(_summary_group("Missing Data"))
        total_cells = numeric.size
        if total_cells > 0:
            detection_rate = float(numeric.notna().sum().sum() / total_cells)
        else:
            detection_rate = 0.0
        if detection_rate > 0.90:
            dr_dot = _status_dot("green")
        elif detection_rate >= 0.70:
            dr_dot = _status_dot("amber")
        else:
            dr_dot = _status_dot("red")
        rows.append(_summary_row(dr_dot, "Detection rate", f"{detection_rate:.1%}"))

        # Samples with >20% missing
        row_miss = numeric.isna().mean(axis=1)
        n_high_miss_samples = int((row_miss > 0.2).sum())
        if n_high_miss_samples == 0:
            hms_dot = _status_dot("green")
        elif n_samples > 0 and n_high_miss_samples / n_samples <= 0.10:
            hms_dot = _status_dot("amber")
        else:
            hms_dot = _status_dot("red")
        rows.append(_summary_row(hms_dot, "Samples with &gt;20% missing", str(n_high_miss_samples)))

        # Proteins with >50% missing
        col_miss = numeric.isna().mean(axis=0)
        n_high_miss_prot = int((col_miss > 0.5).sum())
        if n_high_miss_prot == 0:
            hmp_dot = _status_dot("green")
        elif n_features > 0 and n_high_miss_prot / n_features <= 0.05:
            hmp_dot = _status_dot("amber")
        else:
            hmp_dot = _status_dot("red")
        rows.append(_summary_row(hmp_dot, "Proteins with &gt;50% missing", str(n_high_miss_prot)))

    # --- Variability ---
    cv_data = plot_data.get("cv_distribution")
    plate_cv_data = plot_data.get("plate_cv")
    has_cv = (isinstance(cv_data, CvDistributionData) and len(cv_data.cv_values) > 0) or (
        isinstance(plate_cv_data, PlateCvData) and len(plate_cv_data.inter_cv) > 0
    )
    if has_cv:
        rows.append(_summary_group("Variability"))
        if isinstance(cv_data, CvDistributionData) and len(cv_data.cv_values) > 0:
            if n_samples < 2:
                rows.append(_summary_row("", "Median CV", "N/A (single sample)"))
                rows.append(_summary_row("", "CV range (5th\u201395th pctl)", "N/A"))
            else:
                med_cv = float(np.median(cv_data.cv_values))
                if med_cv < 0.15:
                    cv_dot = _status_dot("green")
                elif med_cv <= 0.25:
                    cv_dot = _status_dot("amber")
                else:
                    cv_dot = _status_dot("red")
                rows.append(_summary_row(cv_dot, "Median CV", f"{med_cv:.1%}"))
                p5, p95 = np.percentile(cv_data.cv_values, [5, 95])
                rows.append(_summary_row("", "CV range (5th\u201395th pctl)", f"{p5:.1%} \u2013 {p95:.1%}"))

        if isinstance(plate_cv_data, PlateCvData) and len(plate_cv_data.inter_cv) > 0:
            med_inter = float(np.median(plate_cv_data.inter_cv))
            if med_inter < 0.20:
                pi_dot = _status_dot("green")
            elif med_inter <= 0.30:
                pi_dot = _status_dot("amber")
            else:
                pi_dot = _status_dot("red")
            rows.append(_summary_row(pi_dot, "Median inter-plate CV", f"{med_inter:.1%}"))

    # --- QC Status (Olink only) ---
    if "SampleQC" in samples.columns:
        rows.append(_summary_group("QC Status"))
        qc_counts = samples["SampleQC"].value_counts()
        n_pass = int(qc_counts.get("PASS", 0))
        n_warn = int(qc_counts.get("WARN", 0))
        n_fail = int(qc_counts.get("FAIL", 0))
        fail_rate = n_fail / n_samples if n_samples > 0 else 0.0
        if fail_rate == 0:
            qc_dot = _status_dot("green")
        elif fail_rate < 0.10:
            qc_dot = _status_dot("amber")
        else:
            qc_dot = _status_dot("red")
        rows.append(_summary_row(qc_dot, "PASS / WARN / FAIL", f"{n_pass} / {n_warn} / {n_fail}"))

    # --- Normalization (SomaScan only) ---
    if "HybControlNormScale" in samples.columns:
        rows.append(_summary_group("Normalization"))
        hyb = pd.to_numeric(samples["HybControlNormScale"], errors="coerce")
        med_hyb = float(hyb.median())
        if 0.8 <= med_hyb <= 1.2:
            hyb_dot = _status_dot("green")
        elif 0.4 <= med_hyb <= 2.5:
            hyb_dot = _status_dot("amber")
        else:
            hyb_dot = _status_dot("red")
        rows.append(_summary_row(hyb_dot, "Median HybControlNormScale", f"{med_hyb:.3f}"))

        n_outside = int(((hyb < 0.4) | (hyb > 2.5)).sum())
        outside_rate = n_outside / n_samples if n_samples > 0 else 0.0
        if n_outside == 0:
            out_dot = _status_dot("green")
        elif outside_rate < 0.05:
            out_dot = _status_dot("amber")
        else:
            out_dot = _status_dot("red")
        rows.append(_summary_row(out_dot, "Samples outside 0.4\u20132.5", str(n_outside)))

    # --- Olink QC ---
    from pyprideap.core import Platform as _Platform

    iqr_median_data = plot_data.get("iqr_median_qc")
    if isinstance(iqr_median_data, IqrMedianQcData):
        rows.append(_summary_group("Assay QC"))
        n_outlier = iqr_median_data.n_outlier_samples
        n_total = iqr_median_data.n_total_samples
        outlier_rate = n_outlier / n_total if n_total > 0 else 0.0
        if n_outlier == 0:
            iqr_dot = _status_dot("green")
        elif outlier_rate < 0.10:
            iqr_dot = _status_dot("amber")
        else:
            iqr_dot = _status_dot("red")
        rows.append(
            _summary_row(
                iqr_dot,
                "IQR/Median outlier samples",
                f"{n_outlier} / {n_total}",
            )
        )

    # --- SomaScan QC Flags ---
    if dataset.platform == _Platform.SOMASCAN:
        col_check_data = plot_data.get("col_check")

        if isinstance(col_check_data, ColCheckData):
            rows.append(_summary_group("SomaScan QC"))
            total = col_check_data.n_pass + col_check_data.n_flag
            flag_rate = col_check_data.n_flag / total if total > 0 else 0.0
            if col_check_data.n_flag == 0:
                cc_dot = _status_dot("green")
            elif flag_rate < 0.05:
                cc_dot = _status_dot("amber")
            else:
                cc_dot = _status_dot("red")
            rows.append(
                _summary_row(
                    cc_dot,
                    "ColCheck (PASS / FLAG)",
                    f"{col_check_data.n_pass} / {col_check_data.n_flag}",
                )
            )

    table_html = f'<table class="summary-table">{"".join(rows)}</table>'

    return (
        '<div class="plot-card" id="dataset-summary">'
        '<div class="plot-header">'
        "<h3>Dataset Summary</h3>"
        '<button class="help-toggle" title="About dataset summary" aria-label="Help">?</button>'
        "</div>"
        '<div class="help-text">'
        "Overview of key dataset quality metrics. "
        "Green/amber/red indicators show whether metrics fall within "
        "recommended ranges for the platform."
        "</div>"
        f"{table_html}"
        "</div>"
    )


def _compute_sdrf_volcanoes(
    dataset: AffinityDataset,
    sdrf_path: str | Path,
) -> dict[str, list[tuple[str, VolcanoData]]]:
    """Compute volcano plot data for each suitable SDRF grouping variable.

    Returns a dict mapping variable name to a list of
    ``(comparison_label, VolcanoData)`` tuples.
    """
    from dataclasses import replace as ds_replace

    from pyprideap.io.readers.sdrf import get_grouping_columns, merge_sdrf, read_sdrf
    from pyprideap.stats.differential import linear_model, ttest

    sdrf = read_sdrf(sdrf_path)
    ds = merge_sdrf(dataset, sdrf)
    grouping_cols = get_grouping_columns(sdrf)

    results: dict[str, list[tuple[str, VolcanoData]]] = {}

    for col in grouping_cols:
        if col not in ds.samples.columns:
            continue

        # Clean out "not available" / "not applicable" values
        valid_mask = ~ds.samples[col].astype(str).str.strip().str.lower().isin(
            {"not available", "not applicable", "nan", "", "na"}
        )
        ds_clean = ds_replace(
            ds,
            samples=ds.samples[valid_mask].reset_index(drop=True),
            expression=ds.expression[valid_mask].reset_index(drop=True),
        )

        groups = ds_clean.samples[col].dropna().unique()
        n_groups = len(groups)

        # Detect potential numeric covariates from SDRF columns
        covariates = _detect_covariates(ds_clean, exclude={col})

        comparisons: list[tuple[str, VolcanoData]] = []

        if n_groups == 2:
            # Direct two-group comparison
            try:
                test_df = ttest(ds_clean, group_var=col)
                label = f"{groups[0]} vs {groups[1]}"
                n_sig = int((test_df["significant"] == True).sum()) if "significant" in test_df.columns else 0  # noqa: E712
                method = "Welch t-test"

                # Fallback: if t-test found 0 significant proteins and
                # covariates are available, re-run with linear model
                if n_sig == 0 and covariates:
                    cov_names = list(covariates.keys())
                    logger.debug(
                        "t-test found 0 significant proteins for '%s'; retrying with linear model adjusting for %s",
                        col,
                        cov_names,
                    )
                    try:
                        test_df = linear_model(ds_clean, group_var=col, covariates=cov_names)
                        method = f"Linear model (adjusted for {', '.join(cov_names)})"
                    except Exception:
                        pass  # keep the t-test result

                vdata = compute_volcano(test_df)
                if vdata is not None:
                    vdata.title = f"{col}: {label}"
                    vdata.method = method
                    comparisons.append((label, vdata))
            except (ValueError, Exception):
                pass
        elif 2 < n_groups <= 10:
            # Pairwise comparisons
            sorted_groups = sorted(groups)
            for i, g1 in enumerate(sorted_groups):
                for g2 in sorted_groups[i + 1 :]:
                    pair_mask = ds_clean.samples[col].isin({g1, g2})
                    ds_pair = ds_replace(
                        ds_clean,
                        samples=ds_clean.samples[pair_mask].reset_index(drop=True),
                        expression=ds_clean.expression[pair_mask].reset_index(drop=True),
                    )
                    try:
                        test_df = ttest(ds_pair, group_var=col)
                        label = f"{g1} vs {g2}"
                        n_sig = int((test_df["significant"] == True).sum()) if "significant" in test_df.columns else 0  # noqa: E712
                        method = "Welch t-test"

                        if n_sig == 0 and covariates:
                            cov_names = list(covariates.keys())
                            logger.debug(
                                "t-test found 0 significant proteins for '%s: %s'; "
                                "retrying with linear model adjusting for %s",
                                col,
                                label,
                                cov_names,
                            )
                            try:
                                test_df = linear_model(ds_pair, group_var=col, covariates=cov_names)
                                method = f"Linear model (adjusted for {', '.join(cov_names)})"
                            except Exception:
                                pass

                        vdata = compute_volcano(test_df)
                        if vdata is not None:
                            vdata.title = f"{col}: {label}"
                            vdata.method = method
                            comparisons.append((label, vdata))
                    except (ValueError, Exception):
                        pass

        if comparisons:
            results[col] = comparisons

    return results


def _detect_covariates(
    dataset: AffinityDataset,
    exclude: set[str],
) -> dict[str, str]:
    """Detect numeric covariates from SDRF-merged sample metadata.

    Looks for columns like ``age`` (parses '55Y' -> 55) and ``sex``
    (encodes as 0/1).  Returns a dict of {column_name: type} for
    columns that were successfully coerced to numeric.
    """
    import pandas as pd

    covariates: dict[str, str] = {}
    samples = dataset.samples

    for col in samples.columns:
        col_lower = col.lower().strip()
        if col in exclude or col_lower in exclude:
            continue

        if col_lower == "age":
            # Parse age strings like "55Y", "62Y" -> numeric
            try:
                parsed = samples[col].astype(str).str.extract(r"(\d+)", expand=False)
                numeric = pd.to_numeric(parsed, errors="coerce")
                if numeric.notna().sum() >= len(samples) * 0.5:
                    dataset.samples[col] = numeric
                    covariates[col] = "numeric"
            except Exception:
                pass
        elif col_lower == "sex":
            # Encode sex as numeric (female=0, male=1)
            try:
                mapping = samples[col].astype(str).str.strip().str.lower()
                unique_vals = set(mapping.dropna().unique()) - {"nan", "not available", "not applicable"}
                if unique_vals <= {"female", "male"} and len(unique_vals) == 2:
                    dataset.samples[col] = mapping.map({"female": 0, "male": 1})
                    covariates[col] = "binary"
            except Exception:
                pass

    if covariates:
        logger.debug("Detected covariates for adjustment: %s", covariates)

    return covariates


def qc_report(
    dataset: AffinityDataset,
    output: str | Path,
    sdrf_path: str | Path | None = None,
    strip_plot_title: bool = True,
) -> Path:
    """Generate a complete QC HTML report for the dataset.

    Parameters
    ----------
    dataset : AffinityDataset
        The dataset to report on.
    output : str | Path
        Output HTML file path.
    sdrf_path : str | Path | None
        Optional path to an SDRF TSV file.  When provided, differential
        expression volcano plots are added to the report with an
        interactive dropdown to select the grouping variable.
    strip_plot_title : bool
        Remove the Plotly figure title from each plot (default ``True``).
        The card header already displays the title, so the in-plot title
        is redundant.  Set to ``False`` to keep the Plotly title.
    """
    try:
        import plotly  # noqa: F401
    except ImportError:
        raise ImportError("Plotly is required for HTML reports. Install with: pip install pyprideap[plots]") from None

    from pyprideap.viz.qc import render as R

    output = Path(output)
    logger.debug("qc_report: starting report generation -> %s", output)
    plot_data = compute_all(dataset)

    _RENDERERS = {
        "lod_comparison": (LodComparisonData, R.render_lod_comparison),
        "distribution": (DistributionData, R.render_distribution),
        "qc_summary": (QcLodSummaryData, R.render_qc_summary),
        "lod_analysis": (LodAnalysisData, R.render_lod_analysis),
        "heatmap": (HeatmapData, R.render_heatmap),
        "correlation": (CorrelationData, R.render_correlation),
        "sample_completeness": (DataCompletenessData, R.render_sample_completeness),
        "missing_frequency_distribution": (DataCompletenessData, R.render_missing_frequency),
        "cv_distribution": (CvDistributionData, R.render_cv_distribution),
        "plate_cv": (PlateCvData, R.render_plate_cv),
        "norm_scale": (NormScaleData, R.render_norm_scale),
        # Olink-specific renderers
        "iqr_median_qc": (IqrMedianQcData, R.render_iqr_median_qc),
        "uniprot_duplicates": (UniProtDuplicateData, R.render_uniprot_duplicates),
        # SomaScan-specific renderers
        "col_check": (ColCheckData, R.render_col_check),
    }

    # Determine display order from _SECTION_ORDER so first-displayed plot gets plotly.js
    display_order: list[str] = []
    for _, keys in _SECTION_ORDER:
        for key in keys:
            display_order.append(key)

    # Build rendered sections keyed by plot id
    rendered: dict[str, tuple[str, ...]] = {}  # key -> (title, html[, extra_header])
    first_key = None

    # Map split renderer keys to their compute_all data keys
    _DATA_KEY_MAP = {
        "sample_completeness": "data_completeness",
        "missing_frequency_distribution": "data_completeness",
    }

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
            dk = _DATA_KEY_MAP.get(key, key)
            if key in _RENDERERS and plot_data.get(dk) is not None:
                first_key = key
                break

        dimred_parts: list[str] = []
        need_plotly_cdn = first_key == "dimreduction"

        if pca_data is not None:
            pca_fig = R.render_pca(pca_data)
            pca_fig.update_layout(title="" if strip_plot_title else pca_data.title, height=500)
            _compact_fig(pca_fig)
            js = "cdn" if need_plotly_cdn else False
            need_plotly_cdn = False  # only include CDN once
            pca_html = pca_fig.to_html(full_html=False, include_plotlyjs=js, default_height="500px")
            dimred_parts.append(f'<div class="dimred-panel" id="dimred-pca">{pca_html}</div>')

        if umap_data is not None:
            tsne_fig = R.render_tsne(umap_data)
            tsne_fig.update_layout(title="" if strip_plot_title else umap_data.title, height=500)
            _compact_fig(tsne_fig)
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

        # Add label toggle button (labels are hidden by default for all datasets)
        toggle_html += '<button class="label-toggle-btn" onclick="toggleDimRedLabels(this)">Show Labels</button>'

        rendered["dimreduction"] = ("Dimensionality Reduction", combined_html, toggle_html)
        logger.debug("qc_report: rendered dimreduction plot")

    # Find first key if not already set
    if first_key is None:
        for key in display_order:
            data_key = _DATA_KEY_MAP.get(key, key)
            if key in _RENDERERS and plot_data.get(data_key) is not None:
                first_key = key
                break

    for key, (dtype, renderer) in _RENDERERS.items():
        data_key = _DATA_KEY_MAP.get(key, key)
        data = plot_data.get(data_key)
        if data is None or not isinstance(data, dtype):
            continue
        fig = renderer(data)  # type: ignore[operator]
        if strip_plot_title:
            fig.update_layout(title="")
        # Multi-panel renderers set their own height (e.g. 800px); only
        # apply the default for plots that haven't specified one.
        current_height = fig.layout.height
        if current_height is None:
            fig.update_layout(height=500)
        _compact_fig(fig)
        plot_height = f"{fig.layout.height}px"
        js = "cdn" if key == first_key else False
        plot_html = fig.to_html(full_html=False, include_plotlyjs=js, default_height=plot_height)
        if key in _DATA_KEY_MAP:
            title = key.replace("_", " ").title()
        else:
            title = getattr(data, "title", key.replace("_", " ").title())
        rendered[key] = (title, plot_html)  # type: ignore[attr-defined]
        logger.debug("qc_report: rendered %s plot", key)

    # SDRF-driven differential expression volcano plots
    if sdrf_path is not None:
        try:
            volcano_results = _compute_sdrf_volcanoes(dataset, sdrf_path)
        except Exception:
            volcano_results = {}

        if volcano_results:
            volcano_html_parts: list[str] = []
            # Build dropdown options
            var_names = list(volcano_results.keys())
            options_html = "".join(
                f'<option value="{html_mod.escape(v)}">{html_mod.escape(v)}</option>' for v in var_names
            )
            dropdown_html = (
                '<div class="volcano-controls">'
                '<label for="volcano-var-select"><strong>Grouping variable:</strong></label> '
                '<select id="volcano-var-select" onchange="switchVolcanoVar(this.value)">'
                f"{options_html}"
                "</select>"
                ' <span id="volcano-comparison-label" style="margin-left:12px;color:#666;"></span>'
                "</div>"
            )
            volcano_html_parts.append(dropdown_html)

            # Render all volcano plots (hidden by default, JS toggles visibility)
            need_plotly = first_key is None  # only include CDN if not already included
            for var_idx, (var_name, comparisons) in enumerate(volcano_results.items()):
                # Comparison sub-dropdown for multi-comparison variables
                if len(comparisons) > 1:
                    comp_options = "".join(
                        f'<option value="{i}">{html_mod.escape(label)}</option>'
                        for i, (label, _) in enumerate(comparisons)
                    )
                    comp_select = (
                        f'<div class="volcano-comp-select" data-var="{html_mod.escape(var_name)}"'
                        f' style="display:none;margin:8px 0;">'
                        f"<label><strong>Comparison:</strong></label> "
                        f"<select onchange=\"switchVolcanoComp('{html_mod.escape(var_name)}', this.value)\">"
                        f"{comp_options}"
                        f"</select></div>"
                    )
                    volcano_html_parts.append(comp_select)

                for comp_idx, (label, vdata) in enumerate(comparisons):
                    fig = R.render_volcano(vdata)
                    if strip_plot_title:
                        fig.update_layout(title="")
                    fig.update_layout(height=500)
                    _compact_fig(fig)
                    js_include: Any = False
                    if need_plotly:
                        js_include = "cdn"
                        need_plotly = False
                    fig_html = fig.to_html(full_html=False, include_plotlyjs=js_include, default_height="500px")
                    visible = var_idx == 0 and comp_idx == 0
                    display = "" if visible else ' style="display:none"'
                    volcano_html_parts.append(
                        f'<div class="volcano-plot" '
                        f'data-var="{html_mod.escape(var_name)}" '
                        f'data-comp="{comp_idx}"'
                        f"{display}>{fig_html}</div>"
                    )

            rendered["differential_expression"] = (
                "Differential Expression",
                "".join(volcano_html_parts),
            )

    # LOD source info (used in summary table)
    lod_info = _lod_source_info(dataset)

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
                f'<button class="help-toggle" title="How to read this plot" aria-label="Help">?</button>'
                f"{extra_header}"
                f"</div>"
                f"{help_block}"
                f"{plot_html}"
                f"</div>"
            )
        if cards:
            toc_html_parts.append(f'<div class="toc-group-label">{group_title}</div>')
            toc_html_parts.append(f"<ul>{''.join(group_toc_items)}</ul>")
            group_sections.append(f'<div class="section-group"><h2>{group_title}</h2>{"".join(cards)}</div>')
    # Dataset Summary section (always first)
    summary_html = _render_summary_table(dataset, plot_data, lod_info)
    summary_section = f'<div class="section-group"><h2>Dataset Summary</h2>{summary_html}</div>'
    group_sections.insert(0, summary_section)
    toc_html_parts.insert(0, '<ul><li><a href="#dataset-summary">Dataset Summary</a></li></ul>')
    toc_html_parts.insert(0, '<div class="toc-group-label">Summary</div>')

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

    stat_items = [
        f'<span class="stat-item"><strong>Platform</strong> {platform_label}</span>',
        f'<span class="stat-item"><strong>Samples</strong> {n_samples}</span>',
        f'<span class="stat-item"><strong>Assays</strong> {n_features}</span>',
    ]
    if n_proteins > 0 and n_proteins != n_features:
        stat_items.append(f'<span class="stat-item"><strong>Unique Proteins</strong> {n_proteins}</span>')
    if n_proteins_above_lod is not None and n_proteins_above_lod > 0:
        stat_items.append(
            f'<span class="stat-item" title="Unique proteins where &gt;50%% of samples have signal above LOD">'
            f"<strong>Proteins &gt; LOD</strong> {n_proteins_above_lod}</span>"
        )

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
        '        <div class="pride-embedded-empty" style="display:none;text-align:center;'
        'padding:40px;color:#777;">No QC plots available for this dataset.</div>\n'
        f"        {''.join(group_sections)}\n"
        f"        <footer>Generated by <strong>pyprideap</strong></footer>\n"
        "    </div>\n"
        "</div>\n"
        f"    <script>\n{_JS}    </script>\n"
        "</body>\n</html>"
    )

    output.write_text(html, encoding="utf-8")
    logger.debug("qc_report: written %s (%d sections)", output, len(rendered))
    return output


def _wrap_standalone_html(
    title: str,
    body: str,
    include_plotlyjs: bool = True,
    no_border: bool = False,
) -> str:
    """Wrap plot HTML in a standalone page with PRIDE styling."""
    plotly_cdn = '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>\n' if include_plotlyjs else ""
    border_override = (
        "    .plot-card { border: none; box-shadow: none; }\n    .plot-card:hover { box-shadow: none; }\n"
        if no_border
        else ""
    )
    return (
        f'<!DOCTYPE html>\n<html lang="en">\n<head>\n'
        f'    <meta charset="utf-8">\n'
        f'    <meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"    <title>{title}</title>\n"
        f"    {plotly_cdn}"
        f"    <style>\n{_CSS}{border_override}    </style>\n"
        f"</head>\n<body>\n"
        f'<div style="max-width:1100px;margin:0 auto;padding:28px 36px;">\n'
        f"{body}\n"
        f"</div>\n"
        f"    <script>\n{_JS}    </script>\n"
        f"</body>\n</html>"
    )


def qc_report_split(
    dataset: AffinityDataset,
    output_dir: str | Path,
    no_border: bool = False,
    strip_plot_title: bool = True,
) -> Path:
    """Generate individual QC plot HTML files in a directory.

    Each plot is saved as a standalone HTML file named by its plot type
    (e.g., ``distribution.html``, ``correlation.html``). A ``summary.html``
    file with the dataset summary table is always generated.

    Parameters
    ----------
    dataset : AffinityDataset
        The dataset to generate plots for.
    output_dir : str | Path
        Directory to write individual HTML files into. Created if it doesn't exist.
    no_border : bool
        If True, remove card borders and shadows from standalone plot files.
    strip_plot_title : bool
        Remove the Plotly figure title from each plot (default ``True``).

    Returns
    -------
    Path
        The output directory path.
    """
    try:
        import plotly  # noqa: F401
    except ImportError:
        raise ImportError("Plotly is required for HTML reports. Install with: pip install pyprideap[plots]") from None

    from pyprideap.viz.qc import render as R

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.debug("qc_report_split: starting split report generation -> %s", output_dir)

    plot_data = compute_all(dataset)
    lod_info = _lod_source_info(dataset)
    platform_label = dataset.platform.value.replace("_", " ").title()

    _RENDERERS: dict[str, tuple[type, Any]] = {
        "lod_comparison": (LodComparisonData, R.render_lod_comparison),
        "distribution": (DistributionData, R.render_distribution),
        "qc_summary": (QcLodSummaryData, R.render_qc_summary),
        "lod_analysis": (LodAnalysisData, R.render_lod_analysis),
        "heatmap": (HeatmapData, R.render_heatmap),
        "correlation": (CorrelationData, R.render_correlation),
        "sample_completeness": (DataCompletenessData, R.render_sample_completeness),
        "missing_frequency_distribution": (DataCompletenessData, R.render_missing_frequency),
        "cv_distribution": (CvDistributionData, R.render_cv_distribution),
        "plate_cv": (PlateCvData, R.render_plate_cv),
        "norm_scale": (NormScaleData, R.render_norm_scale),
        # Olink-specific renderers
        "iqr_median_qc": (IqrMedianQcData, R.render_iqr_median_qc),
        "uniprot_duplicates": (UniProtDuplicateData, R.render_uniprot_duplicates),
        # SomaScan-specific renderers
        "col_check": (ColCheckData, R.render_col_check),
    }

    written: list[str] = []

    # Map split renderer keys to their compute_all data keys
    _DATA_KEY_MAP = {
        "sample_completeness": "data_completeness",
        "missing_frequency_distribution": "data_completeness",
    }

    # Render each plot as a standalone HTML file
    for key, (dtype, renderer) in _RENDERERS.items():
        data_key = _DATA_KEY_MAP.get(key, key)
        data = plot_data.get(data_key)
        if data is None or not isinstance(data, dtype):
            continue
        fig = renderer(data)  # type: ignore[operator]
        if strip_plot_title:
            fig.update_layout(title="")
        current_height = fig.layout.height
        if current_height is None:
            fig.update_layout(height=500)
        plot_height = f"{fig.layout.height}px"
        plot_html = fig.to_html(full_html=False, include_plotlyjs=False, default_height=plot_height)
        if key in _DATA_KEY_MAP:
            title = key.replace("_", " ").title()
        else:
            title = getattr(data, "title", key.replace("_", " ").title())
        help_html = _HELP_TEXT.get(key, "")
        help_toggle = '<button class="help-toggle" title="How to read this plot" aria-label="Help">?</button>'
        help_block = f'<div class="help-text">{help_html}</div>' if help_html else ""
        body = (
            f'<div class="plot-card"><div class="plot-header"><h3>{title}</h3>'
            f"{help_toggle}</div>{help_block}{plot_html}</div>"
        )
        page = _wrap_standalone_html(f"{title} — {platform_label}", body, no_border=no_border)
        (output_dir / f"{key}.html").write_text(page, encoding="utf-8")
        written.append(key)
        logger.debug("qc_report_split: rendered %s plot", key)

    # Combined PCA + t-SNE with toggle switch (matches full report)
    _pca_raw = plot_data.get("pca")
    _umap_raw = plot_data.get("umap")
    pca_data: PcaData | None = _pca_raw if isinstance(_pca_raw, PcaData) else None
    umap_data: UmapData | None = _umap_raw if isinstance(_umap_raw, UmapData) else None
    has_dimred = pca_data is not None or umap_data is not None

    if has_dimred:
        dimred_parts: list[str] = []

        if pca_data is not None:
            pca_fig = R.render_pca(pca_data)
            pca_fig.update_layout(title="" if strip_plot_title else pca_data.title, height=500)
            pca_html = pca_fig.to_html(full_html=False, include_plotlyjs=False, default_height="500px")
            dimred_parts.append(f'<div class="dimred-panel" id="dimred-pca">{pca_html}</div>')

        if umap_data is not None:
            tsne_fig = R.render_tsne(umap_data)
            tsne_fig.update_layout(title="" if strip_plot_title else umap_data.title, height=500)
            tsne_html = tsne_fig.to_html(full_html=False, include_plotlyjs=False, default_height="500px")
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

        toggle_html += '<button class="label-toggle-btn" onclick="toggleDimRedLabels(this)">Show Labels</button>'

        help_text = _HELP_TEXT.get("dimreduction", "")
        help_toggle = '<button class="help-toggle" title="How to read this plot" aria-label="Help">?</button>'
        body = (
            f'<div class="plot-card"><div class="plot-header">'
            f"<h3>Dimensionality Reduction</h3>{help_toggle}{toggle_html}</div>"
            f'<div class="help-text">{help_text}</div>'
            f"{combined_html}</div>"
        )
        page = _wrap_standalone_html(
            f"Dimensionality Reduction — {platform_label}",
            body,
            no_border=no_border,
        )
        (output_dir / "dimreduction.html").write_text(page, encoding="utf-8")
        written.append("dimreduction")

    # Summary table (includes LOD sources inline)
    summary_html = _render_summary_table(dataset, plot_data, lod_info)
    page = _wrap_standalone_html(
        f"Dataset Summary — {platform_label}", summary_html, include_plotlyjs=False, no_border=no_border
    )
    (output_dir / "summary.html").write_text(page, encoding="utf-8")
    written.append("summary")

    logger.debug("qc_report_split: written %d files to %s", len(written), output_dir)
    return output_dir
