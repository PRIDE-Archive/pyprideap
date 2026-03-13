from __future__ import annotations

from pathlib import Path

import pandas as pd

from pyprideap.core import AffinityDataset
from pyprideap.io.readers.olink_csv import _detect_olink_platform, _detect_sample_key, _warn_data_quality

_SAMPLE_COLS = {"SampleID", "SampleName", "SampleType", "WellID", "PlateID", "SampleQC", "DataAnalysisRefID"}
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

    sample_key = _detect_sample_key(df, source=path.name)

    sample_cols = [c for c in df.columns if c in _SAMPLE_COLS]
    samples = df[sample_cols].drop_duplicates(subset=[sample_key]).reset_index(drop=True)

    feature_cols = [c for c in df.columns if c in _FEATURE_COLS]
    features = df[feature_cols].drop_duplicates(subset=["OlinkID"]).reset_index(drop=True)

    sample_order = samples[sample_key].values

    expression = df.pivot_table(
        index=sample_key,
        columns="OlinkID",
        values="NPX",
        aggfunc="first",
    )
    expression = expression.reindex(sample_order).reset_index(drop=True)

    # Align features to match expression column order (pivot_table sorts columns)
    features = features.set_index("OlinkID").reindex(expression.columns).reset_index()

    dataset = AffinityDataset(
        platform=_detect_olink_platform(features["OlinkID"]),
        samples=samples,
        features=features,
        expression=expression,
        metadata={"source_file": str(path)},
    )
    _warn_data_quality(dataset, source=path.name)
    return dataset
