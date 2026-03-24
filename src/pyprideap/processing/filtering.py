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


def get_unique_samples(
    dataset: AffinityDataset,
    *,
    exclude_controls: bool = False,
) -> list[str]:
    """Return sorted unique sample identifiers from a dataset.

    Args:
        dataset: The AffinityDataset to extract samples from.
        exclude_controls: If True, remove control/QC samples
            before collecting unique identifiers (default: False).

    Returns:
        Sorted list of unique sample identifier strings.
    """
    samples = dataset.samples

    if exclude_controls and "SampleType" in samples.columns:
        is_control = samples["SampleType"].astype(str).str.lower().str.strip().isin(_CONTROL_SAMPLE_TYPES)
        samples = samples[~is_control]
        logger.debug(
            "get_unique_samples: excluded %d control samples",
            int(is_control.sum()),
        )

    # Resolve the best sample identifier column.
    # SampleID may actually be an assay index in some Olink files (its
    # cardinality equals the number of features).  When that happens, skip
    # it and prefer SampleName instead.  Also accept the SomaScan-style
    # ``SampleId`` (lowercase 'd').
    id_col: str | None = None
    for candidate in ("SampleID", "SampleId", "SampleName"):
        if candidate not in samples.columns:
            continue
        # Guard against assay-indexed columns: if the number of unique
        # values equals the number of features, it is likely an assay key.
        n_unique = samples[candidate].nunique()
        n_features = len(dataset.features)
        if n_unique == n_features and n_features > 0 and n_unique != len(samples):
            logger.debug(
                "get_unique_samples: skipping %s (nunique=%d matches feature count)",
                candidate,
                n_unique,
            )
            continue
        id_col = candidate
        break

    if id_col is None:
        logger.debug("get_unique_samples: no suitable sample ID column found, using row index")
        ids = [str(i) for i in samples.index]
        return sorted(set(ids))

    raw = samples[id_col].dropna().astype(str).str.strip()
    unique = sorted(set(raw) - {""})
    logger.debug("get_unique_samples: %d unique samples (column=%s)", len(unique), id_col)
    return unique


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
