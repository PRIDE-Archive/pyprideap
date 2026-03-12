from __future__ import annotations

from pathlib import Path

import pandas as pd

from pyprideap.core import AffinityDataset, Platform

_SAMPLE_COLS = {"SampleID", "PlateID", "WellID", "SampleType", "SampleQC", "PlateQC"}
_FEATURE_COLS = {"OlinkID", "UniProt", "Assay", "Panel", "LOD", "MissingFreq"}
_REQUIRED_COLS = {"SampleID", "OlinkID", "NPX"}


def read_olink_csv(path: str | Path) -> AffinityDataset:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path, sep=None, engine="python")
    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path.name}: {sorted(missing)}")

    sample_cols = [c for c in df.columns if c in _SAMPLE_COLS]
    samples = df[sample_cols].drop_duplicates(subset=["SampleID"]).reset_index(drop=True)

    feature_cols = [c for c in df.columns if c in _FEATURE_COLS]
    # Drop LOD from per-assay features since it varies per plate/sample
    feature_cols_no_lod = [c for c in feature_cols if c != "LOD"]
    features = df[feature_cols_no_lod].drop_duplicates(subset=["OlinkID"]).reset_index(drop=True)

    sample_order = samples["SampleID"].values

    expression = df.pivot_table(
        index="SampleID",
        columns="OlinkID",
        values="NPX",
        aggfunc="first",
    )
    expression = expression.reindex(sample_order).reset_index(drop=True)

    # Align features to match expression column order (pivot_table sorts columns)
    features = features.set_index("OlinkID").reindex(expression.columns).reset_index()

    metadata: dict[str, object] = {"source_file": str(path)}

    # Build per-sample × per-assay LOD matrix if LOD column exists
    if "LOD" in df.columns:
        lod_matrix = df.pivot_table(
            index="SampleID",
            columns="OlinkID",
            values="LOD",
            aggfunc="first",
        )
        lod_matrix = lod_matrix.reindex(sample_order).reset_index(drop=True)
        metadata["lod_matrix"] = lod_matrix

    platform = Platform.OLINK_EXPLORE

    return AffinityDataset(
        platform=platform,
        samples=samples,
        features=features,
        expression=expression,
        metadata=metadata,
    )
