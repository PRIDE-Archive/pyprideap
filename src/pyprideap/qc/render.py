from __future__ import annotations

from typing import TYPE_CHECKING

from pyprideap.qc.compute import (
    CorrelationData,
    CvDistributionData,
    DetectionRateData,
    DistributionData,
    LodAnalysisData,
    MissingValuesData,
    PcaData,
    QcSummaryData,
)

if TYPE_CHECKING:
    from plotly.graph_objects import Figure


def _import_plotly():
    try:
        import plotly.express as px
        import plotly.graph_objects as go

        return go, px
    except ImportError:
        raise ImportError("Plotly is required for rendering. Install with: pip install pyprideap[plots]") from None


_QC_COLORS = {"PASS": "#2ecc71", "WARN": "#f39c12", "FAIL": "#e74c3c", "NA": "#95a5a6"}


def render_distribution(data: DistributionData) -> Figure:
    go, _ = _import_plotly()
    fig = go.Figure(data=[go.Histogram(x=data.values, nbinsx=50)])
    fig.update_layout(title=data.title, xaxis_title=data.xlabel, yaxis_title=data.ylabel)
    return fig


def render_qc_summary(data: QcSummaryData) -> Figure:
    go, _ = _import_plotly()
    colors = [_QC_COLORS.get(c, "#3498db") for c in data.categories]
    fig = go.Figure(data=[go.Bar(x=data.categories, y=data.counts, marker_color=colors)])
    fig.update_layout(title=data.title, xaxis_title="QC Status", yaxis_title="Count")
    return fig


def render_lod_analysis(data: LodAnalysisData) -> Figure:
    go, px = _import_plotly()
    import pandas as pd

    df = pd.DataFrame({"Assay": data.assay_ids, "% Above LOD": data.above_lod_pct, "Panel": data.panel})
    fig = px.bar(df, x="Assay", y="% Above LOD", color="Panel", title=data.title)
    fig.update_layout(xaxis_tickangle=-45)
    return fig


def render_pca(data: PcaData) -> Figure:
    _, px = _import_plotly()
    import pandas as pd

    df = pd.DataFrame({"PC1": data.pc1, "PC2": data.pc2, "Label": data.labels, "Group": data.groups})
    ve = data.variance_explained
    fig = px.scatter(
        df,
        x="PC1",
        y="PC2",
        color="Group",
        hover_data=["Label"],
        title=data.title,
        labels={
            "PC1": f"PC1 ({ve[0] * 100:.1f}%)" if len(ve) > 0 else "PC1",
            "PC2": f"PC2 ({ve[1] * 100:.1f}%)" if len(ve) > 1 else "PC2",
        },
    )
    return fig


def render_correlation(data: CorrelationData) -> Figure:
    go, _ = _import_plotly()
    fig = go.Figure(
        data=go.Heatmap(
            z=data.matrix,
            x=data.labels,
            y=data.labels,
            colorscale="RdBu_r",
            zmin=-1,
            zmax=1,
        )
    )
    fig.update_layout(title=data.title)
    return fig


def render_missing_values(data: MissingValuesData) -> Figure:
    go, _ = _import_plotly()
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=1, cols=2, subplot_titles=["Per Sample", "Per Feature"])
    fig.add_trace(go.Bar(x=data.sample_ids, y=data.missing_rate_per_sample, name="Samples"), row=1, col=1)
    fig.add_trace(
        go.Bar(x=data.feature_ids[:50], y=data.missing_rate_per_feature[:50], name="Features (top 50)"), row=1, col=2
    )
    fig.update_layout(title=data.title, showlegend=False)
    fig.update_yaxes(title_text="Missing Rate", row=1, col=1)
    return fig


def render_cv_distribution(data: CvDistributionData) -> Figure:
    go, _ = _import_plotly()
    fig = go.Figure(data=[go.Histogram(x=data.cv_values, nbinsx=50)])
    fig.update_layout(title=data.title, xaxis_title="CV", yaxis_title="Count")
    return fig


def render_detection_rate(data: DetectionRateData) -> Figure:
    go, _ = _import_plotly()
    import numpy as np

    median_rate = float(np.median(data.rates))
    fig = go.Figure(data=[go.Bar(x=data.sample_ids, y=data.rates)])
    fig.add_hline(y=median_rate, line_dash="dash", line_color="red", annotation_text=f"Median: {median_rate:.2f}")
    fig.update_layout(title=data.title, xaxis_title="Sample", yaxis_title="Detection Rate")
    return fig
