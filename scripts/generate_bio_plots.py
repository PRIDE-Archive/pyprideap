#!/usr/bin/env python3
"""Generate biologically meaningful plots from actual PRIDE Affinity data.

Loads the Olink NPX datasets listed in ``_DATASETS`` from pride_data/ and
produces plots describing protein coverage, expression dynamics, and protein
reuse across those datasets.

NPX is a relative, study-specific log2 scale, so values pooled across datasets
show typical dynamic range only; they are not comparable absolute abundances.

Usage:
    python scripts/generate_bio_plots.py [--output-dir reports/global]
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from _html import save_plot
from plotly.subplots import make_subplots

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

import pyprideap
from pyprideap.viz.theme import PRIDE_COLORS, pride_color_discrete, set_plot_theme

logger = logging.getLogger(__name__)

# Fixed seed so subsampled violin plots are reproducible between runs
_RNG = np.random.default_rng(42)


def _new_figure(*args: object, **kwargs: object) -> go.Figure:
    """Create a themed figure; per-plot layout set afterwards overrides the theme."""
    return set_plot_theme(go.Figure(*args, **kwargs))


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

_DATASETS: dict[str, str] = {
    "PAD000001": "PAD000001_olink_npx.csv",
    "PAD000002": "PAD000002_npx.parquet",
    "PAD000004": "PAD000004_Example_NPX.parquet",
    "PAD000005": "PAD000005_raw_data.npx.csv",
    "PAD000006": "PAD000006_VB-3207.npx.csv",
    "PAD000008": "PAD000008_440sample_NPX_CHART_long.npx.csv",
    # PAD000009_Extended_NPX.npx.csv holds the same samples × assays as this file;
    # listing both would count every PAD000009 protein twice.
    "PAD000009": "PAD000009_NPX.npx.csv",
    "PAD000016": "PAD000016_GAHT_and_pregnancy.npx.csv",
    "PAD000019": "PAD000019_Olink5K_npx.parquet",
    "PAD000022": "PAD000022_Olink3K_npx.csv",
    "PAD000030": "PAD000030_Detectability.npx.csv",
}


def load_all_datasets(data_dir: Path) -> dict[str, pyprideap.AffinityDataset]:
    """Load all available datasets, skipping failures."""
    datasets = {}
    for name, filename in _DATASETS.items():
        path = data_dir / filename
        if not path.exists():
            logger.warning("Skipping %s: file not found", path)
            continue
        try:
            ds = pyprideap.read(str(path))
            datasets[name] = ds
            logger.info(
                "Loaded %s: %d samples × %d features [%s]",
                name,
                len(ds.samples),
                len(ds.features),
                ds.platform.value,
            )
        except Exception as exc:
            logger.warning("Failed to load %s: %s", name, exc)
    return datasets


# ---------------------------------------------------------------------------
# 1. Dataset Landscape — bubble chart (samples × proteins × platform)
# ---------------------------------------------------------------------------


def plot_dataset_overview(datasets: dict[str, pyprideap.AffinityDataset], output_dir: Path) -> Path:
    """Bubble chart: x=samples, y=features, size=% non-missing values, color=platform."""
    names, n_samples, n_features, platforms, completeness = [], [], [], [], []

    for name, ds in datasets.items():
        expr = ds.expression
        total = expr.size
        rate = float(expr.count().sum() / total * 100) if total > 0 else 0.0

        names.append(name)
        n_samples.append(len(ds.samples))
        n_features.append(len(ds.features))
        platforms.append(ds.platform.value)
        completeness.append(rate)

    platform_set = sorted(set(platforms))
    colors = pride_color_discrete(len(platform_set))
    color_map = dict(zip(platform_set, colors))

    fig = _new_figure()
    for plat in platform_set:
        mask = [i for i, p in enumerate(platforms) if p == plat]
        fig.add_trace(
            go.Scatter(
                x=[n_samples[i] for i in mask],
                y=[n_features[i] for i in mask],
                mode="markers+text",
                marker=dict(
                    size=[max(15, completeness[i] / 3) for i in mask],
                    color=color_map[plat],
                    opacity=0.8,
                    line=dict(color="white", width=1.5),
                ),
                text=[names[i] for i in mask],
                customdata=[completeness[i] for i in mask],
                textposition="top center",
                textfont=dict(size=9),
                name=plat.replace("_", " ").title(),
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Samples: %{x}<br>"
                    "Proteins/Assays: %{y}<br>"
                    "Non-missing values: %{customdata:.1f}%<br>"
                    "<extra>%{fullData.name}</extra>"
                ),
            )
        )

    fig.update_layout(
        title="Dataset Landscape: Samples × Proteins per Project",
        xaxis_title="Number of Samples",
        yaxis_title="Number of Proteins / Assays",
        height=550,
        xaxis=dict(type="log"),
        yaxis=dict(type="log"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return save_plot(fig, output_dir / "dataset_overview.html", "Dataset Landscape")


# ---------------------------------------------------------------------------
# 2. Protein Reuse across datasets
# ---------------------------------------------------------------------------


def _uniprot_ids(ds: pyprideap.AffinityDataset) -> pd.Series:
    """UniProt ID per feature (aligned with expression columns); missing IDs are NaN."""
    if "UniProt" not in ds.features.columns:
        return pd.Series(np.nan, index=ds.features.index, dtype=object)
    ids = ds.features["UniProt"].astype("string").str.strip()
    return ids.mask(ids.isin(["", "nan", "NA"])).astype(object)


def plot_protein_overlap(datasets: dict[str, pyprideap.AffinityDataset], output_dir: Path) -> Path | None:
    """Show how many UniProt IDs are unique vs shared across datasets."""
    protein_sets = {name: set(_uniprot_ids(ds).dropna()) for name, ds in datasets.items()}
    protein_sets = {name: prots for name, prots in protein_sets.items() if prots}

    if len(protein_sets) < 2:
        logger.warning("Not enough datasets with UniProt IDs for overlap plot")
        return None

    # Count how many datasets each protein appears in
    protein_counts: Counter[str] = Counter(p for prots in protein_sets.values() for p in prots)

    # Bin: how many proteins appear in exactly 1, 2, 3, ... N datasets
    max_ds = len(protein_sets)
    bins = list(range(1, max_ds + 1))
    hist = Counter(protein_counts.values())
    bin_counts = [hist.get(b, 0) for b in bins]

    fig = _new_figure(
        go.Bar(
            x=bins,
            y=bin_counts,
            marker_color=pride_color_discrete(max_ds),
            text=bin_counts,
            textposition="outside",
            hovertemplate="<b>Found in %{x} dataset(s)</b><br>Proteins: %{y}<extra></extra>",
        )
    )
    fig.add_annotation(
        text=f"Total unique proteins: <b>{len(protein_counts):,}</b> across {max_ds} datasets",
        xref="paper",
        yref="paper",
        x=0.98,
        y=0.95,
        showarrow=False,
        font=dict(size=13, color=PRIDE_COLORS["primary"]),
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor=PRIDE_COLORS["border"],
        borderwidth=1,
        borderpad=6,
    )
    fig.update_layout(
        title="Protein Reuse Across Datasets",
        xaxis_title="Number of Datasets a Protein Appears In",
        yaxis_title="Number of Unique Proteins (UniProt)",
        height=450,
        xaxis=dict(dtick=1),
    )

    return save_plot(fig, output_dir / "protein_overlap.html", "Protein Overlap Across Datasets")


# ---------------------------------------------------------------------------
# 3. NPX Dynamic Range by panel (violin)
# ---------------------------------------------------------------------------


def plot_expression_by_panel(datasets: dict[str, pyprideap.AffinityDataset], output_dir: Path) -> Path | None:
    """Violin plot of NPX distributions per Olink panel, pooled across datasets."""
    panel_values: dict[str, list[float]] = defaultdict(list)

    for ds in datasets.values():
        if "Panel" not in ds.features.columns:
            continue
        panels = ds.features["Panel"].astype("string")
        for col_idx, panel in enumerate(panels):
            if pd.isna(panel) or panel.strip() in ("", "nan", "Unknown"):
                continue
            # Sample up to 200 values per assay so large cohorts do not dominate
            vals = ds.expression.iloc[:, col_idx].dropna().to_numpy()
            if len(vals) > 200:
                vals = _RNG.choice(vals, 200, replace=False)
            # Merge "_II" panel variants into their base panel
            base = panel.replace("_II", "").replace(" II", "").strip()
            panel_values[base].extend(vals.tolist())

    if not panel_values:
        logger.warning("No panel data found")
        return None

    sorted_panels = sorted(panel_values, key=lambda p: np.median(panel_values[p]))
    colors = pride_color_discrete(len(sorted_panels))
    fig = _new_figure()

    for panel, color in zip(sorted_panels, colors):
        vals = np.asarray(panel_values[panel])
        # Subsample for rendering performance
        if len(vals) > 5000:
            vals = _RNG.choice(vals, 5000, replace=False)
        fig.add_trace(
            go.Violin(
                y=vals,
                name=panel,
                box_visible=True,
                meanline_visible=True,
                fillcolor=color,
                line_color=color,
                opacity=0.75,
                points=False,
                hoverinfo="name+y",
            )
        )

    fig.update_layout(
        title="NPX Distribution by Olink Panel (pooled across datasets)",
        yaxis_title="NPX (relative log\u2082 scale)",
        height=550,
        showlegend=False,
        xaxis_tickangle=-25,
    )

    return save_plot(fig, output_dir / "expression_by_panel.html", "Expression by Panel")


# ---------------------------------------------------------------------------
# 4. Top 40 shared proteins
# ---------------------------------------------------------------------------


def plot_top_shared_proteins(
    datasets: dict[str, pyprideap.AffinityDataset], output_dir: Path, top_n: int = 40
) -> Path | None:
    """Proteins measured in the most datasets, with their pooled median NPX."""
    protein_datasets: dict[str, set[str]] = defaultdict(set)
    protein_values: dict[str, list[float]] = defaultdict(list)
    protein_assay: dict[str, str] = {}

    for name, ds in datasets.items():
        assays = ds.features["Assay"] if "Assay" in ds.features.columns else None
        for col_idx, uniprot in enumerate(_uniprot_ids(ds)):
            if pd.isna(uniprot):
                continue
            # A protein can be measured by several assays in one dataset (e.g. IL6, TNF
            # in each Explore 3072 panel); it still counts as one dataset.
            protein_datasets[uniprot].add(name)
            if assays is not None and not pd.isna(assays.iloc[col_idx]):
                protein_assay.setdefault(uniprot, str(assays.iloc[col_idx]))
            vals = ds.expression.iloc[:, col_idx].dropna().to_numpy()
            protein_values[uniprot].extend(vals[:100].tolist())

    if not protein_datasets:
        logger.warning("No UniProt annotations found for shared-protein plot")
        return None

    top = sorted(protein_datasets, key=lambda u: (-len(protein_datasets[u]), u))[:top_n]
    top.reverse()  # largest at the top of a horizontal bar chart

    labels = [f"{protein_assay.get(u, u)} ({u})" for u in top]
    counts = [len(protein_datasets[u]) for u in top]
    medians = [float(np.median(protein_values[u])) if protein_values[u] else None for u in top]

    fig = set_plot_theme(
        make_subplots(
            cols=2,
            shared_yaxes=True,
            column_widths=[0.5, 0.5],
            subplot_titles=["Datasets Containing Protein", "Pooled Median NPX"],
        )
    )
    fig.add_trace(
        go.Bar(
            y=labels,
            x=counts,
            orientation="h",
            marker_color=pride_color_discrete(1)[0],
            name="# Datasets",
            text=counts,
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Datasets: %{x}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            y=labels,
            x=medians,
            orientation="h",
            marker=dict(
                color=medians,
                colorscale=[[0, "#aed6f1"], [0.5, "#5dade2"], [1, "#e74c3c"]],
                showscale=True,
                colorbar=dict(title="NPX", x=1.02, len=0.5),
            ),
            name="Median NPX",
            hovertemplate="<b>%{y}</b><br>Median NPX: %{x:.2f}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    fig.update_xaxes(range=[0, len(datasets) + 1], dtick=1 if len(datasets) <= 15 else None, row=1, col=1)
    fig.update_layout(
        title=f"Top {len(top)} Most Frequently Measured Proteins",
        height=max(600, len(labels) * 20 + 120),
        showlegend=False,
        margin=dict(l=260, r=80, t=80, b=50),
    )

    return save_plot(fig, output_dir / "top_shared_proteins.html", "Top Shared Proteins")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--output-dir",
        "-o",
        default="reports/global",
        help="Output directory (default: reports/global)",
    )
    parser.add_argument(
        "--data-dir",
        "-d",
        default="pride_data",
        help="Directory with raw data files (default: pride_data)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    logger.info("Loading datasets from %s ...", data_dir)
    datasets = load_all_datasets(data_dir)
    logger.info("Loaded %d datasets", len(datasets))
    if not datasets:
        sys.exit(f"No datasets could be loaded from {data_dir}; nothing to plot.")

    logger.info("Generating biological plots...")
    results = [
        plot_dataset_overview(datasets, output_dir),
        plot_protein_overlap(datasets, output_dir),
        plot_expression_by_panel(datasets, output_dir),
        plot_top_shared_proteins(datasets, output_dir),
    ]
    generated = [p for p in results if p is not None]

    print(f"\nGenerated {len(generated)} plots in {output_dir}/:")
    for p in generated:
        print(f"  - {p.name}")


if __name__ == "__main__":
    main()
