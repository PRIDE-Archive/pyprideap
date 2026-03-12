"""Sample filtering utilities for AffinityDataset."""

from __future__ import annotations

from pyprideap.core import AffinityDataset

# Known control sample type values (case-insensitive matching)
_CONTROL_SAMPLE_TYPES = {
    "control",
    "sample control",
    "negative",
    "negative control",
    "negative_control",
    "neg",
    "pos",
    "positive control",
    "positive_control",
    "calibrator",
    "reference",
    "standard",
    "qc",
    "buffer",
    "plate control",
}


def filter_controls(dataset: AffinityDataset) -> AffinityDataset:
    """Remove control samples based on the SampleType column.

    Returns a new AffinityDataset with control samples removed from
    samples, expression, and metadata preserved.

    If SampleType column is not present, returns the dataset unchanged.
    """
    if "SampleType" not in dataset.samples.columns:
        return dataset

    is_control = dataset.samples["SampleType"].astype(str).str.lower().str.strip().isin(_CONTROL_SAMPLE_TYPES)

    if not is_control.any():
        return dataset

    keep_mask = ~is_control
    samples = dataset.samples[keep_mask].reset_index(drop=True)
    expression = dataset.expression[keep_mask].reset_index(drop=True)

    metadata = dict(dataset.metadata)
    if "lod_matrix" in metadata:
        import pandas as pd

        lod_df = metadata["lod_matrix"]
        if isinstance(lod_df, pd.DataFrame):
            metadata["lod_matrix"] = lod_df[keep_mask].reset_index(drop=True)

    return AffinityDataset(
        platform=dataset.platform,
        samples=samples,
        features=dataset.features,
        expression=expression,
        metadata=metadata,
    )


def filter_qc(
    dataset: AffinityDataset,
    *,
    keep: tuple[str, ...] = ("PASS", "WARN"),
    qc_column: str = "SampleQC",
) -> AffinityDataset:
    """Keep only samples with QC status in *keep*.

    Returns a new AffinityDataset with non-passing samples removed.
    If the QC column is not present, returns the dataset unchanged.
    """
    if qc_column not in dataset.samples.columns:
        return dataset

    keep_set = {v.upper() for v in keep}
    keep_mask = dataset.samples[qc_column].astype(str).str.upper().isin(keep_set)

    samples = dataset.samples[keep_mask].reset_index(drop=True)
    expression = dataset.expression[keep_mask].reset_index(drop=True)

    metadata = dict(dataset.metadata)
    if "lod_matrix" in metadata:
        import pandas as pd

        lod_df = metadata["lod_matrix"]
        if isinstance(lod_df, pd.DataFrame):
            metadata["lod_matrix"] = lod_df[keep_mask].reset_index(drop=True)

    return AffinityDataset(
        platform=dataset.platform,
        samples=samples,
        features=dataset.features,
        expression=expression,
        metadata=metadata,
    )
