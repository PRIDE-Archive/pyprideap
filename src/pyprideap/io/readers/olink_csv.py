from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

from pyprideap.core import AffinityDataset, Platform

_SAMPLE_COLS = {"SampleID", "SampleName", "PlateID", "WellID", "SampleType", "SampleQC", "PlateQC"}
_FEATURE_COLS = {"OlinkID", "UniProt", "Assay", "Panel", "LOD", "MissingFreq"}
_REQUIRED_COLS = {"SampleID", "OlinkID", "NPX"}

# OlinkID prefix → Platform mapping
_OLINK_ID_PREFIX_MAP = {
    "OID0": Platform.OLINK_TARGET,
    "OID1": Platform.OLINK_TARGET,
    "OID2": Platform.OLINK_EXPLORE,
    "OID3": Platform.OLINK_EXPLORE,
    "OID4": Platform.OLINK_EXPLORE_HT,
    "OID5": Platform.OLINK_REVEAL,
}


def _detect_sample_key(df: pd.DataFrame, *, source: str = "") -> str:
    """Choose the best column to identify samples in a long-format Olink file.

    Returns ``"SampleID"`` unless it appears to be an assay index (same
    cardinality as ``OlinkID``), in which case ``"SampleName"`` is used.
    """
    sample_key = "SampleID"
    if "SampleName" in df.columns:
        n_sid = df["SampleID"].nunique()
        n_olink = df["OlinkID"].nunique()
        n_sname = df["SampleName"].nunique()
        if n_sid == n_olink and n_sname != n_olink:
            sample_key = "SampleName"
            warnings.warn(
                f"{source}: SampleID has the same cardinality as OlinkID "
                f"({n_sid}), which suggests it indexes assays rather than "
                f"samples. Using SampleName ({n_sname} unique) as sample "
                f"identifier instead.",
                UserWarning,
                stacklevel=3,
            )
    return sample_key


def _warn_data_quality(dataset: AffinityDataset, *, source: str = "") -> None:
    """Emit warnings for common data quality issues after reading."""
    n_samples = len(dataset.samples)
    n_features = len(dataset.features)

    # High NaN fraction
    nan_frac = float(dataset.expression.isna().mean().mean())
    if nan_frac > 0.5:
        warnings.warn(
            f"{source}: Expression matrix is {nan_frac:.0%} NaN "
            f"({n_samples} samples × {n_features} features). "
            f"The data may have been pivoted incorrectly or is very sparse.",
            UserWarning,
            stacklevel=3,
        )

    # Suspicious square matrix (samples == features)
    if n_samples == n_features and n_samples > 10:
        warnings.warn(
            f"{source}: Expression matrix is square "
            f"({n_samples} samples = {n_features} features), which is unusual "
            f"for affinity proteomics data. Verify that sample and feature "
            f"identifiers were parsed correctly.",
            UserWarning,
            stacklevel=3,
        )

    # Very few non-NaN values per sample
    non_nan_per_sample = dataset.expression.notna().sum(axis=1)
    median_non_nan = float(non_nan_per_sample.median())
    if n_features > 0 and median_non_nan / n_features < 0.1:
        warnings.warn(
            f"{source}: Samples have very few measured values "
            f"(median {median_non_nan:.0f} of {n_features} features). "
            f"This may indicate a parsing issue.",
            UserWarning,
            stacklevel=3,
        )


def _detect_olink_platform(olink_ids: pd.Series) -> Platform:
    """Detect Olink platform from OlinkID prefixes."""
    prefixes = olink_ids.astype(str).str[:4]
    counts = prefixes.map(_OLINK_ID_PREFIX_MAP).value_counts()
    if counts.empty:
        return Platform.OLINK_EXPLORE
    return Platform(counts.index[0])


def read_olink_csv(path: str | Path) -> AffinityDataset:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path, sep=None, engine="python")
    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path.name}: {sorted(missing)}")

    sample_key = _detect_sample_key(df, source=path.name)

    sample_cols = [c for c in df.columns if c in _SAMPLE_COLS]
    samples = df[sample_cols].drop_duplicates(subset=[sample_key]).reset_index(drop=True)

    feature_cols = [c for c in df.columns if c in _FEATURE_COLS]
    # Drop LOD from per-assay features since it varies per plate/sample
    feature_cols_no_lod = [c for c in feature_cols if c != "LOD"]
    features = df[feature_cols_no_lod].drop_duplicates(subset=["OlinkID"]).reset_index(drop=True)

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

    metadata: dict[str, object] = {"source_file": str(path)}

    # Build per-sample × per-assay LOD matrix if LOD column exists
    if "LOD" in df.columns:
        lod_matrix = df.pivot_table(
            index=sample_key,
            columns="OlinkID",
            values="LOD",
            aggfunc="first",
        )
        lod_matrix = lod_matrix.reindex(sample_order).reset_index(drop=True)
        metadata["lod_matrix"] = lod_matrix

    platform = _detect_olink_platform(features["OlinkID"])

    dataset = AffinityDataset(
        platform=platform,
        samples=samples,
        features=features,
        expression=expression,
        metadata=metadata,
    )
    _warn_data_quality(dataset, source=path.name)
    return dataset
