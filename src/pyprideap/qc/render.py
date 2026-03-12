from __future__ import annotations

from typing import TYPE_CHECKING

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


_QC_LOD_COLORS = {
    "PASS & NPX > LOD": "#2ecc71",
    "PASS & NPX ≤ LOD": "#3498db",
    "WARN & NPX > LOD": "#f39c12",
    "WARN & NPX ≤ LOD": "#e74c3c",
    "FAIL & NPX > LOD": "#95a5a6",
    "FAIL & NPX ≤ LOD": "#7f8c8d",
    "PASS": "#2ecc71",
    "WARN": "#f39c12",
    "FAIL": "#e74c3c",
    "NA": "#95a5a6",
}


def render_distribution(data: DistributionData) -> Figure:
    """Per-sample overlaid density curves (KDE-like via histograms with histnorm)."""
    go, _ = _import_plotly()
    fig = go.Figure()

    for sid, vals in zip(data.sample_ids, data.sample_values):
        if len(vals) == 0:
            continue
        fig.add_trace(
            go.Histogram(
                x=vals,
                name=sid,
                opacity=0.6,
                nbinsx=80,
                histnorm="",
            )
        )

    fig.update_layout(
        title=data.title,
        xaxis_title=data.xlabel,
        yaxis_title=data.ylabel,
        barmode="overlay",
        legend_title="Sample",
    )
    return fig


def render_missing_frequency(data: MissingFrequencyData) -> Figure:
    """Histogram of per-assay missing frequency with 30% threshold line."""
    go, _ = _import_plotly()
    fig = go.Figure(
        data=[
            go.Histogram(
                x=data.missing_freq,
                nbinsx=40,
                marker_color="#17a2b8",
                name="Count of Assays",
            )
        ]
    )
    fig.add_vline(
        x=0.3,
        line_dash="dash",
        line_color="red",
        annotation_text="30% Threshold",
        annotation_position="top right",
    )
    fig.update_layout(
        title=data.title,
        xaxis_title="Missing Frequency",
        yaxis_title="Count of Assays",
        yaxis_type="log",
    )
    return fig


def render_qc_summary(data: QcLodSummaryData) -> Figure:
    """QC × LOD stacked bar or simple QC bar chart."""
    go, _ = _import_plotly()

    total = sum(data.counts)
    colors = [_QC_LOD_COLORS.get(c, "#3498db") for c in data.categories]

    fig = go.Figure()
    cumulative = 0.0
    for cat, cnt, color in zip(data.categories, data.counts, colors):
        pct = cnt / total * 100 if total > 0 else 0
        fig.add_trace(
            go.Bar(
                x=["Samples"],
                y=[pct],
                name=f"{cat} {cnt} ({pct:.1f}%)",
                marker_color=color,
                text=f"{pct:.1f}%",
                textposition="inside",
            )
        )
        cumulative += pct

    fig.update_layout(
        title=data.title,
        barmode="stack",
        yaxis_title="Percentage of Samples",
        yaxis=dict(range=[0, 100], ticksuffix="%"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3),
    )
    return fig


def render_lod_analysis(data: LodAnalysisData) -> Figure:
    go, px = _import_plotly()
    import pandas as pd
    from plotly.subplots import make_subplots

    df = pd.DataFrame({"Assay": data.assay_ids, "% Above LOD": data.above_lod_pct, "Panel": data.panel})
    df = df.sort_values("% Above LOD", ascending=False).reset_index(drop=True)
    df["Rank"] = range(1, len(df) + 1)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Assays Ranked by % Above LOD", "Distribution of % Above LOD"],
        column_widths=[0.6, 0.4],
    )

    panels = sorted(df["Panel"].unique())
    colors = px.colors.qualitative.Set2
    panel_colors = {p: colors[i % len(colors)] for i, p in enumerate(panels)}

    for panel in panels:
        sub = df[df["Panel"] == panel]
        fig.add_trace(
            go.Scatter(
                x=sub["Rank"],
                y=sub["% Above LOD"],
                mode="markers",
                marker=dict(size=4, color=panel_colors[panel]),
                name=panel,
                text=sub["Assay"],
                hovertemplate="%{text}<br>%{y:.1f}% above LOD<extra></extra>",
            ),
            row=1, col=1,
        )

    fig.add_trace(
        go.Histogram(
            x=df["% Above LOD"],
            nbinsx=20,
            marker_color="#3498db",
            showlegend=False,
        ),
        row=1, col=2,
    )

    fig.update_xaxes(title_text="Assay Rank", row=1, col=1)
    fig.update_yaxes(title_text="% Samples Above LOD", row=1, col=1)
    fig.update_xaxes(title_text="% Above LOD", row=1, col=2)
    fig.update_yaxes(title_text="Count of Assays", row=1, col=2)
    fig.update_layout(title=data.title, legend=dict(orientation="h", yanchor="bottom", y=-0.25))
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
        text="Label",
        hover_data=["Label"],
        title=data.title,
        labels={
            "PC1": f"PC1 ({ve[0] * 100:.1f}%)" if len(ve) > 0 else "PC1",
            "PC2": f"PC2 ({ve[1] * 100:.1f}%)" if len(ve) > 1 else "PC2",
        },
    )
    fig.update_traces(textposition="top center", marker=dict(size=10))
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

    all_zero = all(r == 0.0 for r in data.missing_rate_per_sample) and all(
        r == 0.0 for r in data.missing_rate_per_feature
    )

    fig = make_subplots(rows=1, cols=2, subplot_titles=["Per Sample", "Per Feature"])
    fig.add_trace(go.Bar(x=data.sample_ids, y=data.missing_rate_per_sample, name="Samples"), row=1, col=1)
    fig.add_trace(
        go.Bar(x=data.feature_ids[:50], y=data.missing_rate_per_feature[:50], name="Features (top 50)"), row=1, col=2
    )
    fig.update_layout(title=data.title, showlegend=False)
    fig.update_yaxes(title_text="Missing Rate", row=1, col=1)

    if all_zero:
        fig.add_annotation(
            text="No missing values detected",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=18, color="#999"),
        )

    return fig


def render_cv_distribution(data: CvDistributionData) -> Figure:
    go, _ = _import_plotly()
    fig = go.Figure(data=[go.Histogram(x=data.cv_values, nbinsx=50)])
    fig.update_layout(title=data.title, xaxis_title="CV", yaxis_title="Count")
    return fig


def render_detection_rate(data: DetectionRateData) -> Figure:
    go, _ = _import_plotly()
    import numpy as np

    fig = go.Figure(data=[go.Bar(x=data.sample_ids, y=data.rates)])
    if len(data.rates) > 0:
        median_rate = float(np.median(data.rates))
        fig.add_hline(y=median_rate, line_dash="dash", line_color="red", annotation_text=f"Median: {median_rate:.2f}")
    fig.update_layout(title=data.title, xaxis_title="Sample", yaxis_title="Detection Rate")
    return fig
