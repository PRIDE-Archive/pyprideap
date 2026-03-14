"""SomaScan QC flag handling (RowCheck / ColCheck).

In the SomaDataIO R package:

- **RowCheck**: sample-level QC flag based on normalization scale factors.
  ``PASS`` when all normalization scales are within [0.4, 2.5], otherwise
  ``FLAG``.
- **ColCheck**: feature-level QC flag based on calibrator QC ratios.
  ``PASS`` when the QC ratio is within [0.8, 1.2], otherwise ``FLAG``.

This module provides:
- Parsing and evaluation of RowCheck / ColCheck flags
- Filtering functions that use these flags
- A function to compute RowCheck from HybControlNormScale when missing
"""

from __future__ import annotations

from dataclasses import replace

import pandas as pd

from pyprideap.core import AffinityDataset

# SomaDataIO thresholds
_ROW_CHECK_LOW = 0.4
_ROW_CHECK_HIGH = 2.5
_COL_CHECK_LOW = 0.8
_COL_CHECK_HIGH = 1.2


def add_row_check(
    dataset: AffinityDataset,
    *,
    low: float = _ROW_CHECK_LOW,
    high: float = _ROW_CHECK_HIGH,
) -> AffinityDataset:
    """Compute and add a ``RowCheck`` column to sample metadata.

    Evaluates all normalization scale columns (matching pattern
    ``NormScale`` or ``Med.Scale.*``) against the acceptance range
    [low, high].  A sample passes only when **all** scale values
    fall within the range.

    If ``RowCheck`` already exists, it is left unchanged.  If no
    normalization scale columns are found, ``RowCheck`` is set to
    ``"PASS"`` for all samples.

    Parameters
    ----------
    dataset:
        SomaScan AffinityDataset.
    low:
        Lower acceptance threshold (default 0.4).
    high:
        Upper acceptance threshold (default 2.5).

    Returns
    -------
    AffinityDataset
        Copy with ``RowCheck`` column added to samples.
    """
    if "RowCheck" in dataset.samples.columns:
        return dataset

    samples = dataset.samples.copy()

    # Find normalization scale columns
    norm_cols = [c for c in samples.columns if "normscale" in c.lower() or c.startswith("Med.Scale.")]

    if not norm_cols:
        samples["RowCheck"] = "PASS"
        return replace(dataset, samples=samples)

    # Evaluate: PASS only when ALL norm scales are in [low, high]
    all_pass = pd.Series(True, index=samples.index)
    for col in norm_cols:
        vals = pd.to_numeric(samples[col], errors="coerce")
        in_range = vals.between(low, high) | vals.isna()
        all_pass = all_pass & in_range

    samples["RowCheck"] = all_pass.map({True: "PASS", False: "FLAG"})
    return replace(dataset, samples=samples)


def filter_by_row_check(
    dataset: AffinityDataset,
    *,
    low: float = _ROW_CHECK_LOW,
    high: float = _ROW_CHECK_HIGH,
) -> AffinityDataset:
    """Remove samples that do not pass RowCheck.

    If ``RowCheck`` is not present, it is computed from normalization
    scale columns first.

    Parameters
    ----------
    dataset:
        SomaScan AffinityDataset.
    low:
        Lower acceptance threshold (default 0.4).
    high:
        Upper acceptance threshold (default 2.5).

    Returns
    -------
    AffinityDataset
        Copy with FLAG samples removed.
    """
    ds = add_row_check(dataset, low=low, high=high)

    keep_mask = ds.samples["RowCheck"] == "PASS"
    if keep_mask.all():
        return ds

    samples = ds.samples[keep_mask].reset_index(drop=True)
    expression = ds.expression[keep_mask].reset_index(drop=True)

    metadata = dict(ds.metadata)
    if "lod_matrix" in metadata:
        lod_df = metadata["lod_matrix"]
        if isinstance(lod_df, pd.DataFrame):
            metadata["lod_matrix"] = lod_df[keep_mask].reset_index(drop=True)

    return AffinityDataset(
        platform=ds.platform,
        samples=samples,
        features=ds.features,
        expression=expression,
        metadata=metadata,
    )


def filter_by_col_check(
    dataset: AffinityDataset,
) -> AffinityDataset:
    """Remove analytes that do not pass ColCheck.

    ``ColCheck`` is expected in the features table. Analytes with
    ``ColCheck == "FLAG"`` are removed from both expression and features.

    If ``ColCheck`` is not present, the dataset is returned unchanged.

    Returns
    -------
    AffinityDataset
        Copy with flagged analytes removed.
    """
    if "ColCheck" not in dataset.features.columns:
        return dataset

    keep_mask = dataset.features["ColCheck"] != "FLAG"
    if keep_mask.all():
        return dataset

    features = dataset.features[keep_mask].reset_index(drop=True)

    # Map features to expression columns by SeqId (or first ID column)
    id_col = "SeqId" if "SeqId" in dataset.features.columns else dataset.features.columns[0]
    keep_ids = set(dataset.features.loc[keep_mask, id_col].astype(str))
    keep_cols = [c for c in dataset.expression.columns if str(c) in keep_ids]

    expression = dataset.expression[keep_cols].reset_index(drop=True)

    return AffinityDataset(
        platform=dataset.platform,
        samples=dataset.samples,
        features=features,
        expression=expression,
        metadata=dict(dataset.metadata),
    )


def get_col_check_summary(
    dataset: AffinityDataset,
) -> dict[str, int]:
    """Summarize ColCheck flags.

    Returns
    -------
    dict
        Counts of PASS and FLAG analytes, e.g. ``{"PASS": 5000, "FLAG": 42}``.
    """
    if "ColCheck" not in dataset.features.columns:
        return {"PASS": len(dataset.features), "FLAG": 0}

    counts = dataset.features["ColCheck"].value_counts()
    return {
        "PASS": int(counts.get("PASS", 0)),
        "FLAG": int(counts.get("FLAG", 0)),
    }


def get_row_check_summary(
    dataset: AffinityDataset,
) -> dict[str, int]:
    """Summarize RowCheck flags.

    If ``RowCheck`` is not present, all samples are counted as PASS.

    Returns
    -------
    dict
        Counts of PASS and FLAG samples.
    """
    if "RowCheck" not in dataset.samples.columns:
        return {"PASS": len(dataset.samples), "FLAG": 0}

    counts = dataset.samples["RowCheck"].value_counts()
    return {
        "PASS": int(counts.get("PASS", 0)),
        "FLAG": int(counts.get("FLAG", 0)),
    }
