from __future__ import annotations

from pathlib import Path

import pandas as pd

from pyprideap.core import AffinityDataset, Platform

_SAMPLE_COLS = {"SampleID", "PlateID", "WellID", "SampleType", "SampleQC", "PlateQC"}
_FEATURE_COLS = {"OlinkID", "UniProt", "Assay", "Panel", "LOD"}
_REQUIRED_COLS = {"SampleID", "OlinkID", "NPX"}


def read_olink_xlsx(path: str | Path) -> AffinityDataset:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_excel(path)
    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path.name}: {sorted(missing)}")

    sample_cols = [c for c in df.columns if c in _SAMPLE_COLS]
    samples = df[sample_cols].drop_duplicates(subset=["SampleID"]).reset_index(drop=True)

    feature_cols = [c for c in df.columns if c in _FEATURE_COLS]
    feature_cols_no_lod = [c for c in feature_cols if c != "LOD"]
    features = df[feature_cols_no_lod].drop_duplicates(subset=["OlinkID"]).reset_index(drop=True)

    sample_order = samples["SampleID"].values

    expression = df.pivot_table(index="SampleID", columns="OlinkID", values="NPX", aggfunc="first")
    expression = expression.reindex(sample_order).reset_index(drop=True)

    metadata: dict[str, object] = {"source_file": str(path)}

    if "LOD" in df.columns:
        lod_matrix = df.pivot_table(
            index="SampleID",
            columns="OlinkID",
            values="LOD",
            aggfunc="first",
        )
        lod_matrix = lod_matrix.reindex(sample_order).reset_index(drop=True)
        metadata["lod_matrix"] = lod_matrix

    return AffinityDataset(
        platform=Platform.OLINK_EXPLORE,
        samples=samples,
        features=features,
        expression=expression,
        metadata=metadata,
    )
