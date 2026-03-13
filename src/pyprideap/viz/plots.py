"""Standalone plotting functions for pyprideap.

These functions produce Plotly figures for common exploratory plots that are
not part of the automated QC report but useful for interactive analysis.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from pyprideap.core import AffinityDataset, Platform

if TYPE_CHECKING:
    from plotly.graph_objects import Figure


def _import_plotly():
    try:
        import plotly.graph_objects as go

        return go
    except ImportError:
        raise ImportError(
            "Plotly is required for plotting. Install with: pip install pyprideap[plots]"
        ) from None


def boxplot(
    dataset: AffinityDataset,
    proteins: list[str] | None = None,
    group_by: str | None = None,
    max_proteins: int = 20,
) -> Figure:
    """Per-protein boxplot of expression values.

    Parameters
    ----------
    dataset : AffinityDataset
        The dataset to plot.
    proteins : list of str, optional
        Protein/assay IDs to include (column names from expression).
        If *None*, the most variable proteins (by std) are selected.
    group_by : str, optional
        Column name in ``dataset.samples`` to group samples by (e.g.
        SampleType, Site, Treatment).  Each group gets a separate colour.
    max_proteins : int
        Maximum number of proteins to display when *proteins* is None.
    """
    go = _import_plotly()

    numeric = dataset.expression.apply(pd.to_numeric, errors="coerce")

    # Select proteins
    if proteins is not None:
        cols = [c for c in proteins if c in numeric.columns]
    else:
        stds = numeric.std().dropna().nlargest(max_proteins)
        cols = stds.index.tolist()

    if not cols:
        raise ValueError("No valid proteins to plot")

    # Build assay name lookup
    id_col = "OlinkID" if "OlinkID" in dataset.features.columns else dataset.features.columns[0]
    assay_map: dict[str, str] = {}
    if "Assay" in dataset.features.columns:
        assay_map = dict(zip(dataset.features[id_col].astype(str), dataset.features["Assay"].astype(str)))

    is_somascan = dataset.platform == Platform.SOMASCAN
    ylabel = "log10(RFU)" if is_somascan else "NPX"

    fig = go.Figure()

    if group_by and group_by in dataset.samples.columns:
        groups = dataset.samples[group_by].unique()
        from pyprideap.viz.theme import pride_color_discrete

        colors = pride_color_discrete(len(groups))
        for gi, grp in enumerate(groups):
            mask = dataset.samples[group_by] == grp
            for col in cols:
                vals = numeric.loc[mask, col].dropna()
                label = assay_map.get(str(col), str(col))
                fig.add_trace(go.Box(
                    y=vals,
                    x=[label] * len(vals),
                    name=str(grp),
                    marker_color=colors[gi],
                    legendgroup=str(grp),
                    showlegend=col == cols[0],
                ))
        fig.update_layout(boxmode="group")
    else:
        for col in cols:
            vals = numeric[col].dropna()
            label = assay_map.get(str(col), str(col))
            fig.add_trace(go.Box(y=vals, name=label))

    fig.update_layout(
        title="Per-Protein Expression Distribution",
        yaxis_title=ylabel,
        xaxis_title="Protein",
        showlegend=group_by is not None,
    )
    return fig
