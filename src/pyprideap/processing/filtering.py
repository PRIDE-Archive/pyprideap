"""Sample filtering utilities for AffinityDataset."""

from __future__ import annotations

import logging

from pyprideap.core import AffinityDataset

logger = logging.getLogger(__name__)

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
        logger.debug("filter_controls: no SampleType column, returning unchanged")
        return dataset

    is_control = dataset.samples["SampleType"].astype(str).str.lower().str.strip().isin(_CONTROL_SAMPLE_TYPES)

    if not is_control.any():
        logger.debug("filter_controls: no control samples found in %d samples", len(dataset.samples))
        return dataset

    control_types = dataset.samples.loc[is_control, "SampleType"].astype(str).str.lower().str.strip().value_counts()
    logger.debug(
        "filter_controls: removing %d control samples from %d total: %s",
        is_control.sum(),
        len(dataset.samples),
        ", ".join(f"{t}={c}" for t, c in control_types.items()),
    )

    keep_mask = ~is_control
    samples = dataset.samples[keep_mask].reset_index(drop=True)
    expression = dataset.expression[keep_mask].reset_index(drop=True)

    metadata = dict(dataset.metadata)
    import pandas as pd

    for key in ("lod_matrix", "count_matrix", "ext_count"):
        df = metadata.get(key)
        if isinstance(df, pd.DataFrame):
            metadata[key] = df[keep_mask].reset_index(drop=True)

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
        logger.debug("filter_qc: no %s column, returning unchanged", qc_column)
        return dataset

    keep_set = {v.upper() for v in keep}
    qc_values = dataset.samples[qc_column].astype(str).str.upper()
    keep_mask = qc_values.isin(keep_set)

    qc_breakdown = qc_values.value_counts()
    logger.debug(
        "filter_qc: %d/%d samples pass (keep=%s), breakdown: %s",
        keep_mask.sum(),
        len(dataset.samples),
        keep_set,
        ", ".join(f"{s}={c}" for s, c in qc_breakdown.items()),
    )

    samples = dataset.samples[keep_mask].reset_index(drop=True)
    expression = dataset.expression[keep_mask].reset_index(drop=True)

    metadata = dict(dataset.metadata)
    import pandas as pd

    for key in ("lod_matrix", "count_matrix", "ext_count"):
        df = metadata.get(key)
        if isinstance(df, pd.DataFrame):
            metadata[key] = df[keep_mask].reset_index(drop=True)

    return AffinityDataset(
        platform=dataset.platform,
        samples=samples,
        features=dataset.features,
        expression=expression,
        metadata=metadata,
    )
