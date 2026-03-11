from __future__ import annotations

from pathlib import Path

import pandas as pd

from pyprideap.core import AffinityDataset, Platform

_SAMPLE_COLS = {"SampleID", "SampleType", "WellID", "PlateID", "SampleQC", "DataAnalysisRefID"}
_FEATURE_COLS = {"OlinkID", "UniProt", "Assay", "Panel", "Block", "Normalization"}
_REQUIRED_COLS = {"SampleID", "OlinkID", "NPX"}


def read_olink_parquet(path: str | Path) -> AffinityDataset:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_parquet(path)
    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path.name}: {sorted(missing)}")

    sample_cols = [c for c in df.columns if c in _SAMPLE_COLS]
    samples = df[sample_cols].drop_duplicates(subset=["SampleID"]).reset_index(drop=True)

    feature_cols = [c for c in df.columns if c in _FEATURE_COLS]
    features = df[feature_cols].drop_duplicates(subset=["OlinkID"]).reset_index(drop=True)

    expression = df.pivot_table(
        index="SampleID",
        columns="OlinkID",
        values="NPX",
        aggfunc="first",
    )
    expression = expression.reindex(samples["SampleID"].values).reset_index(drop=True)

    return AffinityDataset(
        platform=Platform.OLINK_EXPLORE_HT,
        samples=samples,
        features=features,
        expression=expression,
        metadata={"source_file": str(path)},
    )
