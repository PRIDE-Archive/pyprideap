from __future__ import annotations

from typing import TYPE_CHECKING

from pyprideap.viz.qc.compute import (
    BridgeabilityData,
    ColCheckData,
    ControlAnalyteData,
    CorrelationData,
    CvDistributionData,
    DataCompletenessData,
    DistributionData,
    HeatmapData,
    IqrMedianQcData,
    LodAnalysisData,
    LodComparisonData,
    NormScaleBoxplotData,
    NormScaleData,
    OutlierMapData,
    PcaData,
    PlateCvData,
    QcLodSummaryData,
    RowCheckData,
    UmapData,
    UniProtDuplicateData,
    VolcanoData,
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
    "PASS & RFU > LOD": "#2ecc71",
    "PASS & RFU ≤ LOD": "#3498db",
    "WARN & RFU > LOD": "#f39c12",
    "WARN & RFU ≤ LOD": "#e74c3c",
    "FAIL & RFU > LOD": "#95a5a6",
    "FAIL & RFU ≤ LOD": "#7f8c8d",
    # LOD-only summary when SampleQC is absent
    "NPX > LOD": "#2ecc71",
    "NPX ≤ LOD": "#f39c12",
    "RFU > LOD": "#2ecc71",
    "RFU ≤ LOD": "#f39c12",
    "PASS": "#2ecc71",
    "WARN": "#f39c12",
    "FAIL": "#e74c3c",
    "NA": "#95a5a6",
}


_DISTRIBUTION_SUMMARY_THRESHOLD = 10  # samples above this use percentile band summary


def render_distribution(data: DistributionData) -> Figure:
    """Per-sample overlaid density curves (KDE-like via histograms with histnorm).

    For datasets with >10 samples, renders percentile band summary instead
    of individual traces for clearer visualization.
    """
    go, _ = _import_plotly()

    n_samples = len(data.sample_ids)

    if n_samples > _DISTRIBUTION_SUMMARY_THRESHOLD:
        return _render_distribution_summary(data)

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


def _render_distribution_summary(data: DistributionData) -> Figure:
    """Percentile band summary for large datasets.

    Shows median, IQR (25th-75th), and 5th-95th percentile bands computed
    across all samples at shared bin edges.
    """
    go, _ = _import_plotly()
    import numpy as np

    # Collect all values to determine shared bin edges
    all_vals = []
    for vals in data.sample_values:
        if vals:
            all_vals.extend(vals)
    if not all_vals:
        fig = go.Figure()
        fig.update_layout(title=data.title)
        return fig

    all_vals_arr = np.array(all_vals)
    bin_edges = np.linspace(
        np.nanpercentile(all_vals_arr, 0.5),
        np.nanpercentile(all_vals_arr, 99.5),
        81,
    )
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Compute histogram for each sample
    histograms = []
    for vals in data.sample_values:
        if not vals:
            continue
        counts, _ = np.histogram(vals, bins=bin_edges)
        histograms.append(counts)

    if not histograms:
        fig = go.Figure()
        fig.update_layout(title=data.title)
        return fig

    hist_matrix = np.array(histograms)

    # Compute percentiles across samples
    p5 = np.percentile(hist_matrix, 5, axis=0)
    p25 = np.percentile(hist_matrix, 25, axis=0)
    p50 = np.percentile(hist_matrix, 50, axis=0)
    p75 = np.percentile(hist_matrix, 75, axis=0)
    p95 = np.percentile(hist_matrix, 95, axis=0)

    fig = go.Figure()

    # 5th-95th band (light fill)
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([bin_centers, bin_centers[::-1]]).tolist(),
            y=np.concatenate([p95, p5[::-1]]).tolist(),
            fill="toself",
            fillcolor="rgba(91,192,190,0.15)",
            line=dict(color="rgba(0,0,0,0)"),
            name="5th–95th percentile",
            hoverinfo="skip",
        )
    )

    # 25th-75th band (IQR)
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([bin_centers, bin_centers[::-1]]).tolist(),
            y=np.concatenate([p75, p25[::-1]]).tolist(),
            fill="toself",
            fillcolor="rgba(91,192,190,0.3)",
            line=dict(color="rgba(0,0,0,0)"),
            name="IQR (25th–75th)",
            hoverinfo="skip",
        )
    )

    # Median line
    fig.add_trace(
        go.Scatter(
            x=bin_centers.tolist(),
            y=p50.tolist(),
            mode="lines",
            line=dict(color="#4a9e9c", width=2.5),
            name="Median",
        )
    )

    # Add a small random subsample of individual traces for context
    rng = np.random.default_rng(42)
    n_subsample = min(10, len(data.sample_ids))
    indices = rng.choice(len(data.sample_ids), size=n_subsample, replace=False)
    for idx in sorted(indices):
        vals = data.sample_values[idx]
        if not vals:
            continue
        counts, _ = np.histogram(vals, bins=bin_edges)
        fig.add_trace(
            go.Scatter(
                x=bin_centers.tolist(),
                y=counts.tolist(),
                mode="lines",
                line=dict(width=0.8),
                opacity=0.4,
                name=data.sample_ids[idx],
                hovertemplate=f"{data.sample_ids[idx]}<br>{data.xlabel}: %{{x:.2f}}<br>Count: %{{y}}<extra></extra>",
            )
        )

    n_samples = len(data.sample_ids)
    fig.update_layout(
        title=f"{data.title} (summary of {n_samples} samples)",
        xaxis_title=data.xlabel,
        yaxis_title=data.ylabel,
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
        margin=dict(b=120),
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
        yaxis_title="% of Measurements",
        yaxis=dict(range=[0, 100], ticksuffix="%"),
        showlegend=False,
    )
    return fig


def render_lod_analysis(data: LodAnalysisData) -> Figure:
    go, px = _import_plotly()
    import pandas as pd

    df = pd.DataFrame({"Assay": data.assay_ids, "% Above LOD": data.above_lod_pct, "Panel": data.panel})
    df = df.sort_values("% Above LOD", ascending=False).reset_index(drop=True)
    df["Rank"] = range(1, len(df) + 1)

    panels = sorted(df["Panel"].unique())
    colors = px.colors.qualitative.Set2
    panel_colors = {p: colors[i % len(colors)] for i, p in enumerate(panels)}

    fig = go.Figure()
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
        )

    unit = getattr(data, "unit", "NPX")
    fig.update_xaxes(title_text="Protein Rank (by detectability)")
    fig.update_yaxes(title_text=f"% Samples with {unit} > LOD")
    fig.update_layout(title=data.title, height=500, legend=dict(orientation="h", yanchor="bottom", y=-0.15))
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
    fig.update_traces(mode="markers", textposition="top center", marker=dict(size=10))
    return fig


def render_tsne(data: UmapData) -> Figure:
    """Standalone t-SNE scatter plot."""
    _, px = _import_plotly()
    import pandas as pd

    method = data.title  # "t-SNE" or legacy "UMAP"
    x_label = f"{method} 1"
    y_label = f"{method} 2"
    df = pd.DataFrame({x_label: data.x, y_label: data.y, "Label": data.labels, "Group": data.groups})
    fig = px.scatter(
        df,
        x=x_label,
        y=y_label,
        color="Group",
        text="Label",
        hover_data=["Label"],
        title=f"{method} Projection",
    )
    fig.update_traces(mode="markers", textposition="top center", marker=dict(size=10))
    return fig


# Keep old name for backwards compatibility
render_umap = render_tsne


def render_dimreduction(pca_data: PcaData | None, umap_data: UmapData | None) -> Figure | None:
    """Combined PCA + non-linear projection with dropdown toggle.

    The non-linear method is UMAP when ``umap-learn`` is installed, otherwise
    t-SNE (from scikit-learn).  The actual method name is read from
    ``umap_data.title`` (``"UMAP"`` or ``"t-SNE"``).

    If only one method produced data, renders that alone without a dropdown.
    Returns None if neither is available.

    Text labels are hidden by default for all datasets. Labels can be
    toggled via the "Show Labels" button in the report UI.
    """
    if pca_data is None and umap_data is None:
        return None

    go, _ = _import_plotly()

    # Determine the non-linear method label
    nl_method = umap_data.title if umap_data is not None else "UMAP"
    nl_x_label = f"{nl_method}1" if nl_method == "UMAP" else f"{nl_method} 1"
    nl_y_label = f"{nl_method}2" if nl_method == "UMAP" else f"{nl_method} 2"

    # Always hide text labels by default; users can toggle via report UI
    scatter_mode = "markers"

    fig = go.Figure()
    pca_trace_range: tuple[int, int] = (0, 0)
    umap_trace_range: tuple[int, int] = (0, 0)

    # --- PCA traces ---
    if pca_data is not None:
        pca_start = len(fig.data)
        groups = sorted(set(pca_data.groups))
        from pyprideap.viz.theme import pride_color_discrete

        colors = pride_color_discrete(len(groups))
        color_map = {g: colors[i] for i, g in enumerate(groups)}

        for group in groups:
            mask = [i for i, g in enumerate(pca_data.groups) if g == group]
            fig.add_trace(
                go.Scatter(
                    x=[pca_data.pc1[i] for i in mask],
                    y=[pca_data.pc2[i] for i in mask],
                    mode=scatter_mode,
                    marker=dict(size=10, color=color_map[group]),
                    text=[pca_data.labels[i] for i in mask],
                    textposition="top center",
                    name=group,
                    hovertemplate="%{text}<extra></extra>",
                    visible=True,
                )
            )
        pca_trace_range = (pca_start, len(fig.data))

    # --- Non-linear (UMAP / t-SNE) traces ---
    if umap_data is not None:
        umap_start = len(fig.data)
        groups = sorted(set(umap_data.groups))
        from pyprideap.viz.theme import pride_color_discrete

        colors = pride_color_discrete(len(groups))
        color_map = {g: colors[i] for i, g in enumerate(groups)}
        show_nl = pca_data is None  # visible by default only if no PCA

        for group in groups:
            mask = [i for i, g in enumerate(umap_data.groups) if g == group]
            fig.add_trace(
                go.Scatter(
                    x=[umap_data.x[i] for i in mask],
                    y=[umap_data.y[i] for i in mask],
                    mode=scatter_mode,
                    marker=dict(size=10, color=color_map[group]),
                    text=[umap_data.labels[i] for i in mask],
                    textposition="top center",
                    name=group,
                    hovertemplate="%{text}<extra></extra>",
                    visible=show_nl,
                    showlegend=show_nl,
                )
            )
        umap_trace_range = (umap_start, len(fig.data))

    # --- Dropdown toggle (only if both available) ---
    if pca_data is not None and umap_data is not None:
        ve = pca_data.variance_explained
        n_traces = len(fig.data)

        pca_vis = [False] * n_traces
        for t in range(pca_trace_range[0], pca_trace_range[1]):
            pca_vis[t] = True

        umap_vis = [False] * n_traces
        for t in range(umap_trace_range[0], umap_trace_range[1]):
            umap_vis[t] = True

        fig.update_layout(
            updatemenus=[
                dict(
                    type="dropdown",
                    direction="down",
                    x=0.0,
                    xanchor="left",
                    y=1.15,
                    yanchor="top",
                    active=0,
                    buttons=[
                        dict(
                            label="PCA",
                            method="update",
                            args=[
                                {"visible": pca_vis},
                                {
                                    "xaxis.title.text": f"PC1 ({ve[0] * 100:.1f}%)" if len(ve) > 0 else "PC1",
                                    "yaxis.title.text": f"PC2 ({ve[1] * 100:.1f}%)" if len(ve) > 1 else "PC2",
                                    "title.text": "Dimensionality Reduction — PCA",
                                },
                            ],
                        ),
                        dict(
                            label=nl_method,
                            method="update",
                            args=[
                                {"visible": umap_vis},
                                {
                                    "xaxis.title.text": nl_x_label,
                                    "yaxis.title.text": nl_y_label,
                                    "title.text": f"Dimensionality Reduction — {nl_method}",
                                },
                            ],
                        ),
                    ],
                )
            ],
        )

    # Set initial axis labels and title
    if pca_data is not None:
        ve = pca_data.variance_explained
        fig.update_layout(
            title="Dimensionality Reduction — PCA",
            xaxis_title=f"PC1 ({ve[0] * 100:.1f}%)" if len(ve) > 0 else "PC1",
            yaxis_title=f"PC2 ({ve[1] * 100:.1f}%)" if len(ve) > 1 else "PC2",
        )
    else:
        fig.update_layout(
            title=f"Dimensionality Reduction — {nl_method}",
            xaxis_title=nl_x_label,
            yaxis_title=nl_y_label,
        )

    return fig


def render_heatmap(data: HeatmapData) -> Figure:
    """Clustered expression heatmap (z-scored) with reordered rows/columns.

    For large datasets, reduces precision of z-values to 2 decimal places
    to keep file size manageable.
    """
    go, _ = _import_plotly()
    import numpy as np

    z = np.array(data.values)
    z_ordered = z[data.sample_order][:, data.protein_order]
    sample_labels = [data.sample_labels[i] for i in data.sample_order]
    protein_labels = [data.protein_labels[i] for i in data.protein_order]

    # Reduce precision for large matrices
    if z_ordered.size > 50000:
        z_ordered = np.around(z_ordered, decimals=2)

    fig = go.Figure(
        data=go.Heatmap(
            z=z_ordered,
            x=protein_labels,
            y=sample_labels,
            colorscale="RdBu_r",
            zmid=0,
            colorbar=dict(title="Z-score"),
        )
    )
    fig.update_layout(
        title=data.title,
        xaxis_title="Proteins",
        yaxis_title="Samples",
        xaxis=dict(showticklabels=len(protein_labels) <= 50),
        yaxis=dict(autorange="reversed"),
    )
    return fig


def render_correlation(data: CorrelationData) -> Figure:
    go, _ = _import_plotly()

    # Truncate long labels for display; full ID on hover
    _MAX_LABEL = 20
    short_labels = [s if len(s) <= _MAX_LABEL else s[:_MAX_LABEL] + "\u2026" for s in data.labels]

    # Build customdata so hover shows full labels for both axes
    full_labels = data.labels
    n = len(full_labels)
    customdata = [[full_labels[j] for j in range(n)] for _ in range(n)]

    fig = go.Figure(
        data=go.Heatmap(
            z=data.matrix,
            x=short_labels,
            y=short_labels,
            customdata=customdata,
            colorscale="RdBu_r",
            zmin=-1,
            zmax=1,
            hovertemplate="%{customdata} vs %{y}<br>r = %{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=data.title,
        xaxis_title="Sample",
        yaxis_title="Sample",
        xaxis=dict(tickangle=-45),
    )
    return fig


def render_data_completeness(data: DataCompletenessData) -> Figure:
    """Per-sample stacked bar (above LOD / below LOD) + per-protein missing frequency histogram."""
    go, _ = _import_plotly()
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=["Per-Sample Data Completeness", "Missing Frequency Distribution"],
        vertical_spacing=0.35,
    )

    # Truncate long sample IDs for display; full ID shown on hover
    _MAX_LABEL = 20
    short_ids = [s if len(s) <= _MAX_LABEL else s[:_MAX_LABEL] + "\u2026" for s in data.sample_ids]

    # Top panel: stacked bar per sample
    fig.add_trace(
        go.Bar(
            x=short_ids,
            y=[r * 100 for r in data.above_lod_rate],
            name="Above LOD",
            marker_color="#2ecc71",
            customdata=data.sample_ids,
            hovertemplate="%{customdata}<br>Above LOD: %{y:.1f}%<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=short_ids,
            y=[r * 100 for r in data.below_lod_rate],
            name="Below LOD",
            marker_color="#f39c12",
            customdata=data.sample_ids,
            hovertemplate="%{customdata}<br>Below LOD: %{y:.1f}%<extra></extra>",
        ),
        row=1,
        col=1,
    )

    fig.update_xaxes(title_text="", tickangle=-45, row=1, col=1)
    fig.update_yaxes(title_text="% of Proteins", range=[0, 100], ticksuffix="%", row=1, col=1)
    fig.update_layout(barmode="stack")

    # Bottom panel: histogram of per-protein missing frequency
    if data.missing_freq:
        missing_pct = [f * 100 for f in data.missing_freq]
        fig.add_trace(
            go.Histogram(
                x=missing_pct,
                nbinsx=25,
                marker_color="#e74c3c",
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        # Percentage of proteins below 30% missing frequency (Olink-recommended threshold)
        n_below_30 = sum(1 for f in data.missing_freq if f < 0.30)
        pct_below_30 = n_below_30 / len(data.missing_freq) * 100 if data.missing_freq else 0
        fig.add_annotation(
            text=f"{pct_below_30:.1f}% of proteins below 30% missing",
            xref="x2 domain",
            yref="y2 domain",
            x=0.02,
            y=0.98,
            xanchor="left",
            yanchor="top",
            showarrow=False,
            font=dict(size=12),
            bgcolor="rgba(255,255,255,0.8)",
        )
        # Olink recommended 30% missing frequency threshold
        fig.add_vline(
            x=30,
            line_dash="dash",
            line_color="#e67e22",
            line_width=2,
            row=2,
            col=1,
            annotation_text="30% threshold",
            annotation_position="top right",
            annotation_font_color="#e67e22",
        )
    fig.update_xaxes(title_text="Missing Frequency (% Samples Below LOD)", range=[0, 100], row=2, col=1)
    fig.update_yaxes(title_text="Number of Proteins", row=2, col=1)

    fig.update_layout(
        title=data.title,
        height=800,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15),
    )
    return fig


def render_cv_distribution(data: CvDistributionData) -> Figure:
    go, _ = _import_plotly()
    fig = go.Figure(data=[go.Histogram(x=data.cv_values, nbinsx=50)])
    fig.add_vline(
        x=0.2,
        line_dash="dash",
        line_color="#27ae60",
        line_width=2,
        annotation_text="CV = 0.2",
        annotation_position="top right",
        annotation_font_color="#27ae60",
    )
    fig.update_layout(title=data.title, xaxis_title="Coefficient of Variation", yaxis_title="Number of Proteins")
    return fig


def render_plate_cv(data: PlateCvData) -> Figure:
    """Two-panel plot: intra-plate CV per plate (top) and inter-plate CV (bottom)."""
    go, _ = _import_plotly()
    import numpy as np
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=["Intra-plate CV (per plate)", "Inter-plate CV (across plates)"],
        vertical_spacing=0.22,
    )

    plates = data.plate_ids if data.plate_ids else sorted(set(data.intra_plate_label))

    from pyprideap.viz.theme import pride_color_discrete

    colors = pride_color_discrete(len(plates))

    # Top: intra-plate CV — one violin per plate
    for i, plate in enumerate(plates):
        vals = [v for v, p in zip(data.intra_cv, data.intra_plate_label) if p == plate]
        if not vals:
            continue
        fig.add_trace(
            go.Violin(
                x=[plate] * len(vals),
                y=vals,
                name=plate,
                marker_color=colors[i],
                box_visible=True,
                meanline_visible=True,
                scalemode="width",
            ),
            row=1,
            col=1,
        )

    # Bottom: inter-plate CV — single violin
    if data.inter_cv:
        fig.add_trace(
            go.Violin(
                x=["All analytes"] * len(data.inter_cv),
                y=data.inter_cv,
                name="Inter-plate",
                marker_color="#e74c3c",
                box_visible=True,
                meanline_visible=True,
                scalemode="width",
                showlegend=True,
            ),
            row=2,
            col=1,
        )

        median_val = float(np.median(data.inter_cv))
        fig.add_annotation(
            x="All analytes",
            y=median_val,
            text=f"median: {median_val:.3f}",
            showarrow=True,
            arrowhead=2,
            font=dict(size=11, color="#333"),
            bgcolor="rgba(255,255,255,0.8)",
            row=2,
            col=1,
        )

    fig.update_yaxes(title_text="CV (SD / Mean)", row=1, col=1)
    fig.update_yaxes(title_text="CV (SD / Mean)", row=2, col=1)
    fig.update_layout(
        title=data.title,
        height=800,
        legend=dict(orientation="h", yanchor="bottom", y=-0.1),
    )
    return fig


def render_norm_scale(data: NormScaleData) -> Figure:
    """HybControlNormScale per sample, sorted, colored by plate, with threshold lines."""
    go, _ = _import_plotly()
    import pandas as pd

    df = (
        pd.DataFrame(
            {
                "Sample": data.sample_ids,
                "NormScale": data.values,
                "Plate": data.plate_ids if data.plate_ids else [""] * len(data.sample_ids),
            }
        )
        .dropna(subset=["NormScale"])
        .sort_values("NormScale")
        .reset_index(drop=True)
    )
    df["Rank"] = range(1, len(df) + 1)

    fig = go.Figure()

    if df["Plate"].nunique() > 1:
        from pyprideap.viz.theme import pride_color_discrete

        plates = sorted(df["Plate"].unique())
        colors = pride_color_discrete(len(plates))
        plate_colors = {p: colors[i] for i, p in enumerate(plates)}
        for plate in plates:
            sub = df[df["Plate"] == plate]
            fig.add_trace(
                go.Scatter(
                    x=sub["Rank"],
                    y=sub["NormScale"],
                    mode="markers",
                    marker=dict(size=7, color=plate_colors[plate]),
                    name=plate,
                    text=sub["Sample"],
                    hovertemplate="%{text}<br>NormScale: %{y:.4f}<extra></extra>",
                )
            )
    else:
        fig.add_trace(
            go.Scatter(
                x=df["Rank"],
                y=df["NormScale"],
                mode="markers",
                marker=dict(size=7, color="#5bc0be"),
                name="Samples",
                text=df["Sample"],
                hovertemplate="%{text}<br>NormScale: %{y:.4f}<extra></extra>",
            )
        )

    # Threshold lines
    thresholds = [
        (0.4, "red", "dash", "0.4 (fail)"),
        (0.8, "orange", "dot", "0.8 (warn)"),
        (1.0, "#2ecc71", "solid", "1.0 (ideal)"),
        (1.2, "orange", "dot", "1.2 (warn)"),
        (2.5, "red", "dash", "2.5 (fail)"),
    ]
    for val, color, dash, label in thresholds:
        fig.add_hline(
            y=val, line_dash=dash, line_color=color, line_width=1.5, annotation_text=label, annotation_position="right"
        )

    n_legend_items = df["Plate"].nunique()
    if n_legend_items > 15:
        # Too many plates for horizontal legend — use scrollable vertical legend on the right
        legend_cfg = dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            font=dict(size=9),
        )
        margin_cfg = dict(r=140)
    else:
        legend_cfg = dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5)
        margin_cfg = dict(b=100)

    fig.update_layout(
        title=data.title,
        xaxis_title="Sample Rank (sorted by NormScale)",
        yaxis_title="HybControlNormScale",
        legend=legend_cfg,
        margin=margin_cfg,
    )
    return fig


def render_lod_comparison(data: LodComparisonData) -> Figure:
    """Scatter plots comparing pairs of LOD sources with dropdown selector."""
    go, px = _import_plotly()
    import numpy as np

    fig = go.Figure()

    # Add traces for each pair; only the first pair is visible by default
    trace_ranges: list[tuple[int, int]] = []  # (start_idx, end_idx) per pair
    for pair_idx, pair in enumerate(data.pairs):
        visible = pair_idx == 0
        panels = pair["panels"]
        unique_panels = sorted(set(panels)) if panels else [""]
        colors = px.colors.qualitative.Set2
        panel_colors = {p: colors[i % len(colors)] for i, p in enumerate(unique_panels)}

        start = len(fig.data)
        for panel in unique_panels:
            mask = [i for i, p in enumerate(panels) if p == panel]
            fig.add_trace(
                go.Scatter(
                    x=[pair["values_x"][i] for i in mask],
                    y=[pair["values_y"][i] for i in mask],
                    mode="markers",
                    marker=dict(size=5, color=panel_colors[panel], opacity=0.7),
                    name=panel if panel else "Protein",
                    text=[pair["assay_ids"][i] for i in mask],
                    hovertemplate=(
                        f"%{{text}}<br>{pair['name_x']}: %{{x:.3f}}<br>{pair['name_y']}: %{{y:.3f}}<extra></extra>"
                    ),
                    visible=visible,
                    showlegend=visible,
                )
            )
        end = len(fig.data)
        trace_ranges.append((start, end))

        # Add identity line for the first visible pair
        if visible:
            all_vals = pair["values_x"] + pair["values_y"]
            vmin = min(all_vals)
            vmax = max(all_vals)
            margin = (vmax - vmin) * 0.05
            fig.add_trace(
                go.Scatter(
                    x=[vmin - margin, vmax + margin],
                    y=[vmin - margin, vmax + margin],
                    mode="lines",
                    line=dict(dash="dash", color="gray", width=1),
                    showlegend=False,
                    hoverinfo="skip",
                    visible=True,
                )
            )
            identity_idx = len(fig.data) - 1

            # Compute correlation
            r = float(np.corrcoef(pair["values_x"], pair["values_y"])[0, 1])
            fig.add_annotation(
                x=0.02,
                y=0.98,
                xref="paper",
                yref="paper",
                text=f"r = {r:.3f}  (n = {len(pair['assay_ids'])})",
                showarrow=False,
                font=dict(size=13),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="#ccc",
                borderwidth=1,
            )

    # Build dropdown buttons if multiple pairs
    if len(data.pairs) > 1:
        buttons = []
        for pair_idx, pair in enumerate(data.pairs):
            label = f"{pair['name_x']} vs {pair['name_y']}"
            visibility = [False] * len(fig.data)
            start, end = trace_ranges[pair_idx]
            for t in range(start, end):
                visibility[t] = True

            # Compute identity line and annotation for this pair
            all_vals = pair["values_x"] + pair["values_y"]
            vmin, vmax = min(all_vals), max(all_vals)
            margin = (vmax - vmin) * 0.05
            r = float(np.corrcoef(pair["values_x"], pair["values_y"])[0, 1])

            # Also show the identity line trace
            visibility[identity_idx] = True

            buttons.append(
                dict(
                    label=label,
                    method="update",
                    args=[
                        {
                            "visible": visibility,
                            "x": ([None] * (len(fig.data) - 1) + [[vmin - margin, vmax + margin]]),
                            "y": ([None] * (len(fig.data) - 1) + [[vmin - margin, vmax + margin]]),
                        },
                        {
                            "xaxis.title.text": f"{pair['name_x']} ({data.unit})",
                            "yaxis.title.text": f"{pair['name_y']} ({data.unit})",
                            "annotations": [
                                dict(
                                    x=0.02,
                                    y=0.98,
                                    xref="paper",
                                    yref="paper",
                                    text=f"r = {r:.3f}  (n = {len(pair['assay_ids'])})",
                                    showarrow=False,
                                    font=dict(size=13),
                                    bgcolor="rgba(255,255,255,0.8)",
                                    bordercolor="#ccc",
                                    borderwidth=1,
                                )
                            ],
                        },
                    ],
                )
            )

        fig.update_layout(
            updatemenus=[
                dict(
                    type="dropdown",
                    direction="down",
                    x=0.0,
                    xanchor="left",
                    y=1.15,
                    yanchor="top",
                    buttons=buttons,
                    active=0,
                )
            ],
        )

    first = data.pairs[0]
    fig.update_layout(
        title=data.title,
        xaxis_title=f"{first['name_x']} ({data.unit})",
        yaxis_title=f"{first['name_y']} ({data.unit})",
    )
    return fig


def render_outlier_map(data: OutlierMapData) -> Figure:
    """Heatmap of MAD-based outlier flags (samples × analytes).

    For large datasets, only analytes with at least one outlier are shown
    to reduce file size significantly.
    """
    go, _ = _import_plotly()
    import numpy as np

    z = np.array(data.matrix, dtype=float)
    analyte_ids = list(data.analyte_ids)
    sample_ids = list(data.sample_ids)

    # Filter to only analytes with at least one outlier for large datasets
    n_analytes = z.shape[1] if z.ndim == 2 else 0
    if n_analytes > 500:
        col_has_outlier = z.sum(axis=0) > 0
        if col_has_outlier.any():
            z = z[:, col_has_outlier]
            analyte_ids = [a for a, keep in zip(analyte_ids, col_has_outlier) if keep]
            n_filtered = n_analytes - len(analyte_ids)
            title_suffix = f" ({len(analyte_ids)} analytes with outliers shown, {n_filtered} clean analytes hidden)"
        else:
            title_suffix = " (no outliers detected)"
    else:
        title_suffix = ""

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=analyte_ids,
            y=sample_ids,
            colorscale=[[0, "#f0f0f0"], [1, "#e74c3c"]],
            zmin=0,
            zmax=1,
            showscale=False,
            hovertemplate=("Sample: %{y}<br>Analyte: %{x}<br>Outlier: %{z:.0f}<extra></extra>"),
        )
    )

    fig.update_layout(
        title=data.title + title_suffix,
        xaxis_title="Analytes",
        yaxis_title="Samples",
        xaxis=dict(showticklabels=len(analyte_ids) <= 50),
        yaxis=dict(autorange="reversed"),
    )
    return fig


def render_row_check(data: RowCheckData) -> Figure:
    """RowCheck QC summary as a donut chart with PASS/FLAG counts."""
    go, _ = _import_plotly()

    labels = ["PASS", "FLAG"]
    values = [data.n_pass, data.n_flag]
    colors = ["#2ecc71", "#e74c3c"]

    # Filter out zero values
    filtered = [(lb, v, c) for lb, v, c in zip(labels, values, colors) if v > 0]
    if not filtered:
        filtered = [("PASS", 0, "#2ecc71")]

    fig = go.Figure(
        data=go.Pie(
            labels=[f[0] for f in filtered],
            values=[f[1] for f in filtered],
            marker=dict(colors=[f[2] for f in filtered]),
            hole=0.4,
            textinfo="label+value+percent",
            hovertemplate="%{label}: %{value} samples (%{percent})<extra></extra>",
        )
    )

    total = data.n_pass + data.n_flag
    fig.update_layout(
        title=data.title,
        annotations=[
            dict(
                text=f"{total}<br>samples",
                x=0.5,
                y=0.5,
                font_size=16,
                showarrow=False,
            )
        ],
    )
    return fig


def render_col_check(data: ColCheckData) -> Figure:
    """ColCheck as a strip plot of calibrator QC ratios with acceptance thresholds."""
    go, _ = _import_plotly()

    if data.qc_ratios:
        import numpy as np

        # Using analyte IDs directly as categorical x can compress points to near-invisible
        # when there are thousands of analytes. Plot against rank instead and show analyte
        # IDs in hover.
        n = len(data.qc_ratios)
        order = np.argsort(np.asarray(data.qc_ratios, dtype=float), kind="mergesort").tolist()
        rank = [i + 1 for i in range(n)]

        ordered_rank = [rank[i] for i in range(n)]
        ordered_ids = [data.analyte_ids[i] for i in order]
        ordered_ratios = [data.qc_ratios[i] for i in order]
        ordered_flags = [data.col_check_flags[i] for i in order]

        fig = go.Figure()

        # PASS points
        pass_idx = [i for i, f in enumerate(ordered_flags) if f == "PASS"]
        if pass_idx:
            fig.add_trace(
                go.Scatter(
                    x=[ordered_rank[i] for i in pass_idx],
                    y=[ordered_ratios[i] for i in pass_idx],
                    mode="markers",
                    marker=dict(size=3, color="#2ecc71", opacity=0.5),
                    name=f"PASS ({data.n_pass})",
                    text=[ordered_ids[i] for i in pass_idx],
                    hovertemplate="%{text}<br>QC Ratio: %{y:.3f}<extra></extra>",
                )
            )

        # FLAG points
        flag_idx = [i for i, f in enumerate(ordered_flags) if f != "PASS"]
        if flag_idx:
            fig.add_trace(
                go.Scatter(
                    x=[ordered_rank[i] for i in flag_idx],
                    y=[ordered_ratios[i] for i in flag_idx],
                    mode="markers",
                    marker=dict(size=4, color="#e74c3c", opacity=0.75),
                    name=f"FLAG ({data.n_flag})",
                    text=[ordered_ids[i] for i in flag_idx],
                    hovertemplate="%{text}<br>QC Ratio: %{y:.3f}<extra></extra>",
                )
            )
        # Acceptance thresholds
        fig.add_hline(
            y=0.8,
            line_dash="dash",
            line_color="#e67e22",
            line_width=2,
            annotation_text="0.8",
            annotation_position="bottom left",
        )
        fig.add_hline(
            y=1.2,
            line_dash="dash",
            line_color="#e67e22",
            line_width=2,
            annotation_text="1.2",
            annotation_position="top left",
        )
        fig.add_hline(
            y=1.0,
            line_dash="dot",
            line_color="#27ae60",
            line_width=1,
            annotation_text="ideal (1.0)",
            annotation_position="bottom right",
        )

        fig.update_layout(
            title=data.title,
            xaxis_title="Analyte rank (sorted by QC ratio)",
            yaxis_title="Calibrator QC Ratio",
            xaxis=dict(showticklabels=False),
            legend=dict(orientation="h", yanchor="bottom", y=-0.15),
            height=450,
        )
    else:
        # Fallback: simple text summary if no ratio data
        fig = go.Figure()
        fig.add_annotation(
            text=f"PASS: {data.n_pass} | FLAG: {data.n_flag}",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=20),
        )
        fig.update_layout(title=data.title)
    return fig


def render_control_analytes(data: ControlAnalyteData) -> Figure:
    """Bar chart of control analyte counts by category."""
    go, _ = _import_plotly()

    categories = list(data.category_counts.keys())
    counts = list(data.category_counts.values())

    _CATEGORY_COLORS = {
        "HybControlElution": "#3498db",
        "Spuriomer": "#e74c3c",
        "NonBiotin": "#f39c12",
        "NonHuman": "#9b59b6",
        "NonCleavable": "#1abc9c",
    }
    colors = [_CATEGORY_COLORS.get(c, "#95a5a6") for c in categories]

    fig = go.Figure(
        data=go.Bar(
            x=categories,
            y=counts,
            marker_color=colors,
            text=counts,
            textposition="outside",
            hovertemplate="%{x}: %{y} analytes<extra></extra>",
        )
    )

    fig.update_layout(
        title=data.title,
        xaxis_title="Control Category",
        yaxis_title="Number of Analytes",
        annotations=[
            dict(
                text=f"Total: {data.total_controls} control / {data.total_analytes} total analytes",
                xref="paper",
                yref="paper",
                x=0.98,
                y=0.98,
                showarrow=False,
                font=dict(size=12, color="#555"),
                bgcolor="rgba(255,255,255,0.8)",
            )
        ],
    )
    return fig


def render_norm_scale_boxplot(data: NormScaleBoxplotData) -> Figure:
    """Boxplots of normalization scale factors grouped by a variable.

    Equivalent to the ``data.qc`` plots in SomaDataIO's ``preProcessAdat()``.
    Shows one subplot per normalization scale column, with red threshold lines
    at 0.4 and 2.5.
    """
    go, _ = _import_plotly()
    from plotly.subplots import make_subplots

    n_cols = len(data.norm_scale_columns)
    if n_cols == 0:
        fig = go.Figure()
        fig.update_layout(title=data.title)
        return fig

    fig = make_subplots(
        rows=n_cols,
        cols=1,
        subplot_titles=data.norm_scale_columns,
        vertical_spacing=max(0.05, 0.3 / n_cols),
    )

    groups = sorted(set(data.groups))
    from pyprideap.viz.theme import pride_color_discrete

    colors = pride_color_discrete(len(groups))
    group_colors = {g: colors[i] for i, g in enumerate(groups)}

    for row_idx, col_name in enumerate(data.norm_scale_columns, start=1):
        vals = data.values.get(col_name, [])
        for group in groups:
            group_vals = [
                v
                for v, g in zip(vals, data.groups)
                if g == group and v is not None and not (isinstance(v, float) and v != v)
            ]
            if not group_vals:
                continue
            fig.add_trace(
                go.Box(
                    x=[group] * len(group_vals),
                    y=group_vals,
                    name=group,
                    marker_color=group_colors[group],
                    showlegend=(row_idx == 1),
                ),
                row=row_idx,
                col=1,
            )

        # Add threshold lines
        fig.add_hline(y=0.4, line_dash="dash", line_color="red", line_width=1.5, row=row_idx, col=1)
        fig.add_hline(y=2.5, line_dash="dash", line_color="red", line_width=1.5, row=row_idx, col=1)
        fig.update_yaxes(title_text="Scale Factor", range=[0, 2.8], row=row_idx, col=1)

    fig.update_layout(
        title=data.title,
        height=max(400, 300 * n_cols),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15),
    )
    return fig


def render_iqr_median_qc(data: IqrMedianQcData) -> Figure:
    """IQR vs Median QC scatter plot, faceted by panel.

    Mirrors ``olink_qc_plot()`` from OlinkAnalyze: each point is a sample,
    with dashed lines at ±n SD thresholds for both IQR and median.
    Outlier samples are highlighted in red.
    """
    go, _ = _import_plotly()
    from plotly.subplots import make_subplots

    panels = sorted(set(data.panels))
    n_panels = max(len(panels), 1)
    n_cols = min(n_panels, 3)
    n_rows = (n_panels + n_cols - 1) // n_cols

    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=panels,
        horizontal_spacing=0.08,
        vertical_spacing=0.12,
    )

    for p_idx, panel in enumerate(panels):
        row = p_idx // n_cols + 1
        col = p_idx % n_cols + 1

        # Get data for this panel
        mask = [i for i, p in enumerate(data.panels) if p == panel]
        med_vals = [data.median_values[i] for i in mask]
        iqr_vals = [data.iqr_values[i] for i in mask]
        outlier_flags = [data.is_outlier[i] for i in mask]
        qc_vals = [data.qc_status[i] for i in mask]
        sids = [data.sample_ids[i] for i in mask]

        # Plot non-outlier points (colored by QC status)
        for qc_label, color in [("Pass", "#2ecc71"), ("Warning", "#f39c12")]:
            idx = [i for i in range(len(mask)) if not outlier_flags[i] and qc_vals[i] == qc_label]
            if not idx:
                continue
            fig.add_trace(
                go.Scatter(
                    x=[med_vals[i] for i in idx],
                    y=[iqr_vals[i] for i in idx],
                    mode="markers",
                    marker=dict(size=7, color=color, opacity=0.8),
                    name=qc_label,
                    text=[sids[i] for i in idx],
                    hovertemplate="%{text}<br>Median: %{x:.2f}<br>IQR: %{y:.2f}<extra></extra>",
                    showlegend=(p_idx == 0),
                    legendgroup=qc_label,
                ),
                row=row,
                col=col,
            )

        # Plot outlier points
        outlier_idx = [i for i in range(len(mask)) if outlier_flags[i]]
        if outlier_idx:
            fig.add_trace(
                go.Scatter(
                    x=[med_vals[i] for i in outlier_idx],
                    y=[iqr_vals[i] for i in outlier_idx],
                    mode="markers+text",
                    marker=dict(size=9, color="#e74c3c", symbol="x"),
                    text=[sids[i] for i in outlier_idx],
                    textposition="top center",
                    textfont=dict(size=8),
                    name="Outlier",
                    hovertemplate="%{text}<br>Median: %{x:.2f}<br>IQR: %{y:.2f}<extra></extra>",
                    showlegend=(p_idx == 0),
                    legendgroup="Outlier",
                ),
                row=row,
                col=col,
            )

        # Add threshold lines
        if panel in data.median_low:
            fig.add_vline(x=data.median_low[panel], line_dash="dash", line_color="grey", line_width=1, row=row, col=col)
            fig.add_vline(
                x=data.median_high[panel], line_dash="dash", line_color="grey", line_width=1, row=row, col=col
            )
        if panel in data.iqr_low:
            fig.add_hline(y=data.iqr_low[panel], line_dash="dash", line_color="grey", line_width=1, row=row, col=col)
            fig.add_hline(y=data.iqr_high[panel], line_dash="dash", line_color="grey", line_width=1, row=row, col=col)

        fig.update_xaxes(title_text="Sample Median", row=row, col=col)
        fig.update_yaxes(title_text="IQR", row=row, col=col)

    fig.update_layout(
        title=f"{data.title} ({data.n_outlier_samples} outlier samples / {data.n_total_samples} total)",
        height=max(400, 350 * n_rows),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15),
    )
    return fig


def render_uniprot_duplicates(data: UniProtDuplicateData) -> Figure:
    """Stacked bar showing fraction of assays that are unique vs replicate."""
    go, _ = _import_plotly()

    n_unique = data.n_unique_proteins
    n_total = data.n_total_assays
    n_replicate = max(0, n_total - n_unique)

    if n_total <= 0:
        fig = go.Figure()
        fig.update_layout(title=data.title)
        return fig

    pct_unique = n_unique / n_total * 100.0
    pct_replicate = n_replicate / n_total * 100.0

    fig = go.Figure()
    fig.add_bar(
        x=["Assays"],
        y=[pct_unique],
        name="Unique proteins",
        marker_color="#2ecc71",
        text=[f"{pct_unique:.1f}%"],
        textposition="inside",
        hovertemplate=(f"Unique proteins:<br>{n_unique} / {n_total} assays<br>%{{y:.1f}}% of assays<extra></extra>"),
    )
    fig.add_bar(
        x=["Assays"],
        y=[pct_replicate],
        name="Replicate assays",
        marker_color="#e74c3c",
        text=[f"{pct_replicate:.1f}%"] if pct_replicate > 0 else [""],
        textposition="inside",
        hovertemplate=(
            "Replicate assays (same UniProt):<br>"
            f"{n_replicate} / {n_total} assays<br>"
            "%{y:.1f}% of assays<extra></extra>"
        ),
    )

    fig.update_layout(
        title=(f"{data.title} — {n_unique} unique proteins, {n_total} assays ({n_replicate} replicate assays)"),
        barmode="stack",
        yaxis_title="% of Assays",
        yaxis=dict(range=[0, 100], ticksuffix="%"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
    )
    return fig


def render_volcano(data: VolcanoData) -> Figure:
    """Volcano plot: fold change (x) vs -log10 adjusted p-value (y)."""
    import numpy as np

    go, _ = _import_plotly()
    from pyprideap.viz.theme import PRIDE_COLORS, set_plot_theme

    _DIR_COLORS = {
        "up": PRIDE_COLORS["error"],  # red
        "down": PRIDE_COLORS["accent"],  # blue
        "ns": "#d5d8dc",  # light gray
    }
    _DIR_NAMES = {"up": "Up-regulated", "down": "Down-regulated", "ns": "Not significant"}

    # Count per direction for legend labels
    dir_counts: dict[str, int] = {"up": 0, "down": 0, "ns": 0}
    for d in data.direction:
        dir_counts[d] += 1

    fig = go.Figure()

    # Draw non-significant first (background), then significant on top
    for direction in ["ns", "down", "up"]:
        mask = [i for i, d in enumerate(data.direction) if d == direction]
        if not mask:
            continue

        is_sig = direction != "ns"
        label = f"{_DIR_NAMES[direction]} ({dir_counts[direction]})"

        fig.add_trace(
            go.Scatter(
                x=[data.fold_change[i] for i in mask],
                y=[data.neg_log10_pval[i] for i in mask],
                mode="markers",
                marker=dict(
                    size=5 if not is_sig else 7,
                    color=_DIR_COLORS[direction],
                    opacity=0.35 if not is_sig else 0.85,
                    line=dict(width=0.5, color="white") if is_sig else dict(width=0),
                ),
                name=label,
                text=[data.assay_names[i] for i in mask],
                customdata=[[data.protein_ids[i], data.fold_change[i], data.neg_log10_pval[i]] for i in mask],
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Protein: %{customdata[0]}<br>"
                    "Fold Change: %{customdata[1]:.3f}<br>"
                    "-log<sub>10</sub>(adj p): %{customdata[2]:.2f}"
                    "<extra></extra>"
                ),
            )
        )

    # Threshold reference lines
    fc_max = max((abs(fc) for fc in data.fold_change), default=2.0)
    x_range = max(fc_max * 1.15, 1.5)
    p_line = -np.log10(0.05)

    fig.add_hline(
        y=p_line,
        line_dash="dot",
        line_color=PRIDE_COLORS["text_muted"],
        line_width=1,
        opacity=0.6,
        annotation_text="p = 0.05",
        annotation_position="top right",
        annotation_font_size=10,
        annotation_font_color=PRIDE_COLORS["text_muted"],
    )
    for fc_val in [-1, 1]:
        fig.add_vline(
            x=fc_val,
            line_dash="dot",
            line_color=PRIDE_COLORS["text_muted"],
            line_width=1,
            opacity=0.6,
        )

    set_plot_theme(fig)

    # Add method annotation if available
    annotations = []
    if data.method:
        annotations.append(
            dict(
                text=f"Method: {data.method}",
                xref="paper",
                yref="paper",
                x=0.0,
                y=1.02,
                xanchor="left",
                yanchor="bottom",
                showarrow=False,
                font=dict(size=11, color=PRIDE_COLORS["text_muted"]),
            )
        )

    fig.update_layout(
        title=dict(text=data.title, x=0.5, xanchor="center"),
        xaxis_title="Fold Change",
        yaxis_title="-log<sub>10</sub>(adjusted p-value)",
        xaxis=dict(
            range=[-x_range, x_range],
            zeroline=True,
            zerolinecolor=PRIDE_COLORS["border"],
            zerolinewidth=1,
        ),
        yaxis=dict(rangemode="tozero"),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.15,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(l=60, r=30, t=60, b=80),
        annotations=annotations,
    )
    return fig


def render_bridgeability(data: BridgeabilityData) -> Figure:
    """Render a 4-panel bridgeability diagnostic plot.

    Panels:
    1. Range difference distribution (histogram by recommendation)
    2. R² vs KS statistic scatter (colored by recommendation)
    3. Bridging recommendation summary (bar chart)
    4. Range diff vs R² scatter (colored by recommendation, size=KS stat)
    """
    go, _px = _import_plotly()
    from plotly.subplots import make_subplots

    _REC_COLORS = {
        "MedianCentering": "#2ca02c",
        "QuantileSmoothing": "#ff7f0e",
        "NotBridgeable": "#d62728",
    }

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "Range Difference Distribution",
            "R² vs KS Statistic",
            "Bridging Recommendation Summary",
            "Range Diff vs R²",
        ],
        horizontal_spacing=0.12,
        vertical_spacing=0.18,
    )

    import numpy as np

    # Panel 1: Range diff histogram by recommendation
    for rec in ["MedianCentering", "QuantileSmoothing", "NotBridgeable"]:
        vals = [
            rd
            for rd, r in zip(data.range_diffs, data.recommendations)
            if r == rec and not (isinstance(rd, float) and np.isnan(rd))
        ]
        if vals:
            fig.add_trace(
                go.Histogram(
                    x=vals,
                    name=rec,
                    marker_color=_REC_COLORS[rec],
                    opacity=0.7,
                    legendgroup=rec,
                    showlegend=True,
                ),
                row=1,
                col=1,
            )
    fig.update_xaxes(title_text="Range Difference", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)

    # Panel 2: R² vs KS scatter
    for rec in ["MedianCentering", "QuantileSmoothing", "NotBridgeable"]:
        mask = [i for i, r in enumerate(data.recommendations) if r == rec]
        if not mask:
            continue
        r2_vals = [data.r2_values[i] for i in mask]
        ks_vals = [data.ks_stats[i] for i in mask]
        pids = [data.protein_ids[i] for i in mask]
        fig.add_trace(
            go.Scatter(
                x=ks_vals,
                y=r2_vals,
                mode="markers",
                marker=dict(size=6, color=_REC_COLORS[rec], opacity=0.7),
                name=rec,
                legendgroup=rec,
                showlegend=False,
                text=pids,
                hovertemplate="%{text}<br>KS: %{x:.3f}<br>R²: %{y:.3f}<extra></extra>",
            ),
            row=1,
            col=2,
        )
    # KS threshold line
    fig.add_vline(x=0.2, line_dash="dash", line_color="gray", row=1, col=2)
    fig.update_xaxes(title_text="KS Statistic", row=1, col=2)
    fig.update_yaxes(title_text="R²", row=1, col=2)

    # Panel 3: Summary bar chart
    categories = ["MedianCentering", "QuantileSmoothing", "NotBridgeable"]
    counts = [data.n_median_centering, data.n_quantile_smoothing, data.n_not_bridgeable]
    colors = [_REC_COLORS[c] for c in categories]
    fig.add_trace(
        go.Bar(
            x=categories,
            y=counts,
            marker_color=colors,
            showlegend=False,
            text=counts,
            textposition="auto",
        ),
        row=2,
        col=1,
    )
    fig.update_xaxes(title_text="Recommendation", row=2, col=1)
    fig.update_yaxes(title_text="Number of Assays", row=2, col=1)

    # Panel 4: Range diff vs R² scatter
    for rec in ["MedianCentering", "QuantileSmoothing", "NotBridgeable"]:
        mask = [i for i, r in enumerate(data.recommendations) if r == rec]
        if not mask:
            continue
        rd_vals = [data.range_diffs[i] for i in mask]
        r2_vals = [data.r2_values[i] for i in mask]
        ks_vals = [data.ks_stats[i] for i in mask]
        pids = [data.protein_ids[i] for i in mask]
        # Size proportional to KS stat (min 4, max 14)
        sizes = [max(4, min(14, kv * 30)) if not np.isnan(kv) else 4 for kv in ks_vals]
        fig.add_trace(
            go.Scatter(
                x=rd_vals,
                y=r2_vals,
                mode="markers",
                marker=dict(size=sizes, color=_REC_COLORS[rec], opacity=0.7),
                name=rec,
                legendgroup=rec,
                showlegend=False,
                text=pids,
                hovertemplate="%{text}<br>Range Diff: %{x:.3f}<br>R²: %{y:.3f}<extra></extra>",
            ),
            row=2,
            col=2,
        )
    # Threshold lines
    fig.add_vline(x=1.0, line_dash="dash", line_color="gray", row=2, col=2)
    fig.add_hline(y=0.8, line_dash="dash", line_color="gray", row=2, col=2)
    fig.update_xaxes(title_text="Range Difference", row=2, col=2)
    fig.update_yaxes(title_text="R²", row=2, col=2)

    fig.update_layout(
        title=f"{data.title}: {data.product1_name} vs {data.product2_name}",
        height=800,
        legend=dict(orientation="h", yanchor="bottom", y=-0.12),
    )
    return fig
