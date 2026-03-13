"""Normalization methods for affinity proteomics datasets.

Provides platform-appropriate normalization:

**Olink (NPX, log2 scale)** — additive shifts (equivalent to multiplicative
in linear space):
  - ``bridge_normalize`` — per-protein median shift using bridge samples
  - ``subset_normalize`` — shift using a reference protein subset
  - ``reference_median_normalize`` — shift to match reference medians

**SomaScan (RFU, linear scale)** — multiplicative scaling (equivalent to
``lift_adat`` / ``scaleAnalytes`` in SomaDataIO):
  - ``scale_analytes`` — per-analyte multiplicative scaling with named scalars
  - ``lift_somascan`` — cross-version calibration (5k ↔ 7k ↔ 11k)

Every public function returns a **new** :class:`AffinityDataset` — input
objects are never mutated.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from pyprideap.core import AffinityDataset

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def bridge_normalize(
    dataset1: AffinityDataset,
    dataset2: AffinityDataset,
    bridge_samples: list[str],
) -> AffinityDataset:
    """Adjust *dataset2* NPX values using overlapping bridge samples.

    For every protein present in both datasets the per-protein adjustment is::

        adjustment = median(bridge in dataset1) − median(bridge in dataset2)

    The adjustment is then added to **all** samples in *dataset2*.

    Parameters
    ----------
    dataset1:
        Reference dataset whose scale is kept unchanged.
    dataset2:
        Dataset to adjust.
    bridge_samples:
        Sample identifiers that appear in **both** datasets.

    Returns
    -------
    AffinityDataset
        A copy of *dataset2* with adjusted expression values.

    Raises
    ------
    ValueError
        If no bridge samples are found in both datasets or there are no
        overlapping proteins.
    """
    # Validate bridge samples exist in both datasets
    ds1_samples = set(dataset1.expression.index)
    ds2_samples = set(dataset2.expression.index)
    valid_bridge = [s for s in bridge_samples if s in ds1_samples and s in ds2_samples]
    if not valid_bridge:
        raise ValueError(f"No bridge samples found in both datasets. Requested: {bridge_samples}")

    # Validate overlapping proteins
    overlapping_proteins = dataset1.expression.columns.intersection(dataset2.expression.columns)
    if overlapping_proteins.empty:
        raise ValueError("No overlapping proteins between the two datasets.")

    # Compute per-protein adjustment on overlapping proteins
    bridge_ds1 = dataset1.expression.loc[valid_bridge, overlapping_proteins]
    bridge_ds2 = dataset2.expression.loc[valid_bridge, overlapping_proteins]
    adjustment = bridge_ds1.median(axis=0) - bridge_ds2.median(axis=0)

    # Apply adjustment to dataset2 (only overlapping columns are shifted;
    # non-overlapping columns in dataset2 are left as-is)
    adjusted_expression = dataset2.expression.copy()
    adjusted_expression[overlapping_proteins] = adjusted_expression[overlapping_proteins].add(adjustment, axis=1)

    return replace(dataset2, expression=adjusted_expression)


def subset_normalize(
    dataset1: AffinityDataset,
    dataset2: AffinityDataset,
    reference_proteins: list[str],
) -> AffinityDataset:
    """Adjust *dataset2* using a reference subset of proteins.

    For every protein the per-protein adjustment is the difference between the
    overall median of the reference proteins in *dataset1* and the overall
    median of the reference proteins in *dataset2* (computed across all
    samples).

    Parameters
    ----------
    dataset1:
        Reference dataset whose scale is kept unchanged.
    dataset2:
        Dataset to adjust.
    reference_proteins:
        Protein identifiers used to compute the adjustment.

    Returns
    -------
    AffinityDataset
        A copy of *dataset2* with adjusted expression values.

    Raises
    ------
    ValueError
        If none of the reference proteins are found in both datasets.
    """
    ref_in_ds1 = [p for p in reference_proteins if p in dataset1.expression.columns]
    ref_in_ds2 = [p for p in reference_proteins if p in dataset2.expression.columns]
    valid_ref = sorted(set(ref_in_ds1) & set(ref_in_ds2))
    if not valid_ref:
        raise ValueError(f"None of the reference proteins were found in both datasets. Requested: {reference_proteins}")

    # Per-protein adjustment: median across all samples in ds1 vs ds2
    median_ds1 = dataset1.expression[valid_ref].median(axis=0)
    median_ds2 = dataset2.expression[valid_ref].median(axis=0)
    adjustment = median_ds1 - median_ds2

    adjusted_expression = dataset2.expression.copy()
    adjusted_expression[valid_ref] = adjusted_expression[valid_ref].add(adjustment, axis=1)

    return replace(dataset2, expression=adjusted_expression)


def reference_median_normalize(
    dataset: AffinityDataset,
    reference_medians: dict[str, float] | pd.Series,
) -> AffinityDataset:
    """Shift each protein so its median matches the provided reference median.

    For every protein::

        adjustment = reference_median − current_median

    Parameters
    ----------
    dataset:
        Dataset to adjust.
    reference_medians:
        Mapping of protein identifier to desired median NPX value.

    Returns
    -------
    AffinityDataset
        A copy of *dataset* with adjusted expression values.
    """
    if isinstance(reference_medians, dict):
        reference_medians = pd.Series(reference_medians)

    proteins_to_adjust = dataset.expression.columns.intersection(reference_medians.index)
    if proteins_to_adjust.empty:
        raise ValueError("None of the reference median proteins were found in the dataset.")

    current_medians = dataset.expression[proteins_to_adjust].median(axis=0)
    adjustment = reference_medians[proteins_to_adjust] - current_medians

    adjusted_expression = dataset.expression.copy()
    adjusted_expression[proteins_to_adjust] = adjusted_expression[proteins_to_adjust].add(adjustment, axis=1)

    return replace(dataset, expression=adjusted_expression)


def select_bridge_samples(
    dataset: AffinityDataset,
    n: int = 8,
) -> list[str]:
    """Select optimal bridge samples for cross-dataset normalization.

    Each sample is scored by::

        score = detectability × IQR_coverage

    where *detectability* is the fraction of non-NaN values and
    *IQR_coverage* is the inter-quartile range of the sample's NPX values.
    Samples with ``SampleQC == "PASS"`` (if the column exists) are preferred.

    Parameters
    ----------
    dataset:
        The dataset from which to select bridge samples.
    n:
        Number of bridge samples to return (default 8).

    Returns
    -------
    list[str]
        Up to *n* sample identifiers, ordered by descending score.
    """
    expr = dataset.expression

    # Detectability: fraction of non-NaN values per sample
    detectability = expr.notna().mean(axis=1)

    # IQR coverage per sample
    q1 = expr.quantile(0.25, axis=1)
    q3 = expr.quantile(0.75, axis=1)
    iqr = q3 - q1

    score = detectability * iqr

    # Filter to QC PASS if SampleQC column exists in the samples frame
    if "SampleQC" in dataset.samples.columns:
        # Align by index
        pass_mask = dataset.samples["SampleQC"].eq("PASS")
        # Restrict to PASS samples that are also in expression index
        pass_ids = pass_mask[pass_mask].index.intersection(score.index)
        if not pass_ids.empty:
            score = score.loc[pass_ids]

    return list(score.sort_values(ascending=False).head(n).index)


def assess_bridgeability(
    dataset1: AffinityDataset,
    dataset2: AffinityDataset,
) -> pd.DataFrame:
    """Compare assay performance across two datasets.

    For each overlapping protein the function computes:

    * **correlation** – Pearson correlation of matched samples (NaN if fewer
      than 3 matched samples).
    * **median_diff** – median(ds1) − median(ds2).
    * **detection_rate_1 / detection_rate_2** – fraction of non-NaN values.
    * **bridgeable** – ``True`` when correlation > 0.7 **and** both detection
      rates > 0.5.

    Parameters
    ----------
    dataset1, dataset2:
        Datasets to compare.

    Returns
    -------
    pd.DataFrame
        One row per overlapping protein.

    Raises
    ------
    ValueError
        If there are no overlapping proteins.
    """
    overlapping_proteins = dataset1.expression.columns.intersection(dataset2.expression.columns)
    if overlapping_proteins.empty:
        raise ValueError("No overlapping proteins between the two datasets.")

    common_samples = dataset1.expression.index.intersection(dataset2.expression.index)

    records: list[dict] = []
    for protein in overlapping_proteins:
        vals1 = dataset1.expression[protein]
        vals2 = dataset2.expression[protein]

        detection_rate_1 = float(vals1.notna().mean())
        detection_rate_2 = float(vals2.notna().mean())
        median_diff = float(vals1.median() - vals2.median())

        # Correlation requires matched samples
        if len(common_samples) >= 3:
            paired = pd.DataFrame({"a": vals1.reindex(common_samples), "b": vals2.reindex(common_samples)}).dropna()
            correlation = float(paired["a"].corr(paired["b"])) if len(paired) >= 3 else np.nan
        else:
            correlation = np.nan

        bridgeable = (
            not np.isnan(correlation) and correlation > 0.7 and detection_rate_1 > 0.5 and detection_rate_2 > 0.5
        )

        records.append(
            {
                "protein_id": protein,
                "correlation": correlation,
                "median_diff": median_diff,
                "detection_rate_1": detection_rate_1,
                "detection_rate_2": detection_rate_2,
                "bridgeable": bridgeable,
            }
        )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# SomaScan-specific normalization (multiplicative, linear RFU scale)
# ---------------------------------------------------------------------------


def scale_analytes(
    dataset: AffinityDataset,
    scalars: dict[str, float] | pd.Series,
) -> AffinityDataset:
    """Apply per-analyte multiplicative scalars to RFU values.

    Equivalent to ``scaleAnalytes()`` in SomaDataIO.  Each analyte's
    RFU vector is multiplied by its corresponding scalar::

        RFU_scaled = RFU × scalar

    This is the fundamental operation for SomaScan cross-version
    calibration (lifting) and custom normalization.

    Parameters
    ----------
    dataset:
        Dataset to scale (typically SomaScan with RFU values).
    scalars:
        Mapping of analyte identifier (expression column name or SeqId)
        to scalar value.

    Returns
    -------
    AffinityDataset
        A copy of *dataset* with scaled expression values.

    Raises
    ------
    ValueError
        If none of the scalar keys match expression columns.
    """
    if isinstance(scalars, dict):
        scalars = pd.Series(scalars)

    matched = dataset.expression.columns.intersection(scalars.index)
    if matched.empty:
        raise ValueError(
            "None of the scalar keys match expression columns. "
            f"First few scalar keys: {list(scalars.index[:5])}, "
            f"first few columns: {list(dataset.expression.columns[:5])}"
        )

    scaled_expression = dataset.expression.copy()
    scaled_expression[matched] = scaled_expression[matched].multiply(scalars[matched], axis=1)

    return replace(dataset, expression=scaled_expression)


def lift_somascan(
    dataset: AffinityDataset,
    scalars: dict[str, float] | pd.Series,
    target_version: str | None = None,
) -> AffinityDataset:
    """Calibrate SomaScan RFU values across assay versions.

    Equivalent to ``lift_adat()`` in SomaDataIO.  Applies per-analyte
    multiplicative scalars derived from matched reference populations::

        scalar_i = median(version_A)_i / median(version_B)_i

    Analytes without a scalar in the mapping are left unchanged (scalar
    defaults to 1.0), matching SomaDataIO behaviour for new analytes
    not present in older versions.

    Parameters
    ----------
    dataset:
        SomaScan dataset to lift.
    scalars:
        Per-analyte lifting scalars (SeqId → scalar).
    target_version:
        Optional label for the target signal space (e.g. ``"7k"``).
        Stored in ``metadata["SignalSpace"]`` if provided.

    Returns
    -------
    AffinityDataset
        A copy with lifted RFU values.
    """
    if isinstance(scalars, dict):
        scalars = pd.Series(scalars)

    # Default missing scalars to 1.0 (no transformation)
    full_scalars = pd.Series(1.0, index=dataset.expression.columns)
    matched = full_scalars.index.intersection(scalars.index)
    full_scalars[matched] = scalars[matched]

    scaled_expression = dataset.expression.multiply(full_scalars, axis=1)
    # Round to 1 decimal, matching SomaDataIO convention
    scaled_expression = scaled_expression.round(1)

    new_metadata = {**dataset.metadata}
    if target_version is not None:
        new_metadata["SignalSpace"] = target_version

    return replace(dataset, expression=scaled_expression, metadata=new_metadata)
