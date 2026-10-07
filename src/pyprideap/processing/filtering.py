"""Sample filtering utilities for AffinityDataset."""

from __future__ import annotations

import logging
from typing import cast

import pandas as pd

from pyprideap.core import AffinityDataset

logger = logging.getLogger(__name__)

# Known control sample type values, in normalized form (see normalize_sample_type):
# lower case, with "_" / "-" folded to spaces, so vendor spellings such as
# Olink "PLATE_CONTROL" / "SAMPLE_CONTROL" and SDRF "plate control" all match.
_CONTROL_SAMPLE_TYPES = frozenset(
    {
        "control",
        "sample control",
        "plate control",
        "negative",
        "negative control",
        "neg",
        "pos",
        "positive",
        "positive control",
        "calibrator",
        "calibrator control",
        "reference",
        "standard",
        "qc",
        "buffer",
        "buffer control",
        "blank",
    }
)

# Sample x assay matrices in metadata that must be subset together with samples
_PER_SAMPLE_METADATA = ("lod_matrix", "count_matrix", "ext_count", "assay_qc_matrix", "sample_qc_matrix")

# SDRF characteristics[sample type] after read_sdrf() shortens the column name
_SDRF_SAMPLE_TYPE_COLUMN = "sample type"


def normalize_sample_type(values: pd.Series) -> pd.Series:
    """Normalize sample type labels for matching: lower case, ``_``/``-`` → space, single spaces."""
    return values.astype(str).str.lower().str.replace(r"[_\-\s]+", " ", regex=True).str.strip()


def control_sample_mask(samples: pd.DataFrame) -> pd.Series:
    """Boolean mask of control (non-biological) samples.

    Uses the vendor ``SampleType`` column and, when an SDRF has been merged,
    its ``sample type`` column; a sample is a control if either says so.
    """
    mask = pd.Series(False, index=samples.index)
    for col in ("SampleType", _SDRF_SAMPLE_TYPE_COLUMN):
        if col in samples.columns:
            mask |= normalize_sample_type(samples[col]).isin(_CONTROL_SAMPLE_TYPES)
    return cast(pd.Series, mask)


def filter_controls(dataset: AffinityDataset) -> AffinityDataset:
    """Remove control samples based on the SampleType (or SDRF sample type) column.

    Returns a new AffinityDataset with control samples removed from
    samples, expression, and metadata preserved.

    If neither column is present, returns the dataset unchanged.
    """
    if "SampleType" not in dataset.samples.columns and _SDRF_SAMPLE_TYPE_COLUMN not in dataset.samples.columns:
        logger.debug("filter_controls: no SampleType column, returning unchanged")
        return dataset

    is_control = control_sample_mask(dataset.samples)

    if not is_control.any():
        logger.debug("filter_controls: no control samples found in %d samples", len(dataset.samples))
        return dataset

    type_col = "SampleType" if "SampleType" in dataset.samples.columns else _SDRF_SAMPLE_TYPE_COLUMN
    control_types = normalize_sample_type(dataset.samples.loc[is_control, type_col]).value_counts()
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
    for key in _PER_SAMPLE_METADATA:
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

    if exclude_controls:
        # Same matching as filter_controls (vendor spellings such as NEGATIVE_CONTROL,
        # and an SDRF "sample type" column when merged)
        is_control = control_sample_mask(samples)
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
    for key in _PER_SAMPLE_METADATA:
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
