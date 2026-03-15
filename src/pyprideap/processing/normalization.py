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

import logging
from dataclasses import dataclass, replace
from typing import cast

import numpy as np
import pandas as pd

from pyprideap.core import AffinityDataset

logger = logging.getLogger(__name__)


def _resolve_common_samples(
    dataset1: AffinityDataset,
    dataset2: AffinityDataset,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Index]:
    """Return aligned expression frames and common sample index.

    Matches samples by SampleID / SampleId / SampleName when available,
    falling back to positional integer index otherwise.
    """
    _id_col = None
    for col in ("SampleID", "SampleId", "SampleName"):
        if col in dataset1.samples.columns and col in dataset2.samples.columns:
            _id_col = col
            break

    if _id_col is not None:
        common_ids = set(dataset1.samples[_id_col]) & set(dataset2.samples[_id_col])
        idx1 = dataset1.samples.index[dataset1.samples[_id_col].isin(common_ids)]
        idx2 = dataset2.samples.index[dataset2.samples[_id_col].isin(common_ids)]
        aligned1 = dataset1.expression.loc[idx1].copy()
        aligned1.index = dataset1.samples.loc[idx1, _id_col].values
        aligned2 = dataset2.expression.loc[idx2].copy()
        aligned2.index = dataset2.samples.loc[idx2, _id_col].values
        common_samples = aligned1.index.intersection(aligned2.index)
    else:
        aligned1 = dataset1.expression
        aligned2 = dataset2.expression
        common_samples = aligned1.index.intersection(aligned2.index)
    return aligned1, aligned2, common_samples


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

    # Resolve bridge samples by sample ID column (not DataFrame index),
    # since select_bridge_samples returns SampleID/SampleName strings.
    def _resolve_bridge_mask(ds: AffinityDataset, bridge: list[str]) -> pd.Index:
        for col in ("SampleID", "SampleId", "SampleName"):
            if col in ds.samples.columns:
                ids = ds.samples[col].astype(str)
                mask = ids.isin(bridge)
                if mask.any():
                    result: pd.Index = pd.Index(ds.samples.index[mask])
                    return result
        # Fallback: try matching against the expression index directly
        idx_match = ds.expression.index.isin(bridge)
        if idx_match.any():
            fallback: pd.Index = pd.Index(ds.expression.index[idx_match])
            return fallback
        return pd.Index([])

    ds1_bridge_idx = _resolve_bridge_mask(dataset1, bridge_samples)
    ds2_bridge_idx = _resolve_bridge_mask(dataset2, bridge_samples)
    logger.debug(
        "bridge_normalize: %d bridge samples requested, matched %d in ds1, %d in ds2",
        len(bridge_samples),
        len(ds1_bridge_idx),
        len(ds2_bridge_idx),
    )
    if ds1_bridge_idx.empty or ds2_bridge_idx.empty:
        raise ValueError(f"No bridge samples found in both datasets. Requested: {bridge_samples}")

    # Validate overlapping proteins
    overlapping_proteins = dataset1.expression.columns.intersection(dataset2.expression.columns)
    if overlapping_proteins.empty:
        raise ValueError("No overlapping proteins between the two datasets.")

    # Compute per-protein adjustment on overlapping proteins
    bridge_ds1 = dataset1.expression.loc[ds1_bridge_idx, overlapping_proteins]
    bridge_ds2 = dataset2.expression.loc[ds2_bridge_idx, overlapping_proteins]
    adjustment = bridge_ds1.median(axis=0) - bridge_ds2.median(axis=0)
    logger.debug(
        "bridge_normalize: %d overlapping proteins, median adjustment range [%.3f, %.3f]",
        len(overlapping_proteins),
        float(adjustment.min()),
        float(adjustment.max()),
    )

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

    For each reference protein the adjustment is the difference between its
    median in *dataset1* and its median in *dataset2*.  Only reference
    proteins are shifted; non-reference proteins are left unchanged.

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
    logger.debug(
        "subset_normalize: %d reference proteins requested, %d found in both datasets",
        len(reference_proteins),
        len(valid_ref),
    )
    if not valid_ref:
        raise ValueError(f"None of the reference proteins were found in both datasets. Requested: {reference_proteins}")

    # Per-protein adjustment applied only to the reference proteins;
    # non-reference proteins in dataset2 are left unchanged.
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
    logger.debug(
        "reference_median_normalize: adjusting %d proteins, median shift range [%.3f, %.3f]",
        len(proteins_to_adjust),
        float(adjustment.min()),
        float(adjustment.max()),
    )

    adjusted_expression = dataset.expression.copy()
    adjusted_expression[proteins_to_adjust] = adjusted_expression[proteins_to_adjust].add(adjustment, axis=1)

    return replace(dataset, expression=adjusted_expression)


def select_bridge_samples(
    dataset: AffinityDataset,
    n: int = 8,
    *,
    sample_missing_freq: float = 0.5,
    exclude_qc_outliers: bool = True,
    iqr_outlier_def: float = 3.0,
    median_outlier_def: float = 3.0,
) -> list[str]:
    """Select optimal bridge samples for cross-dataset normalization.

    Enhanced to mirror ``olink_bridgeselector()`` from OlinkAnalyze:

    1. Remove control samples (CONTROL_SAMPLE pattern)
    2. Remove QC outliers (IQR vs Median, ±n SD per panel)
    3. Keep only QC PASS samples
    4. Filter by ``sample_missing_freq`` (max fraction below LOD per sample)
    5. Select *n* evenly-spaced samples across the MeanNPX range

    Parameters
    ----------
    dataset:
        The dataset from which to select bridge samples.
    n:
        Number of bridge samples to return (default 8).
    sample_missing_freq:
        Maximum allowed fraction of assays below LOD per sample (default 0.5).
        Samples exceeding this are excluded.
    exclude_qc_outliers:
        Whether to exclude IQR/Median QC outliers (default True).
    iqr_outlier_def:
        SD multiplier for IQR outlier threshold (default 3.0).
    median_outlier_def:
        SD multiplier for median outlier threshold (default 3.0).

    Returns
    -------
    list[str]
        Up to *n* sample identifiers selected for bridge normalization.

    Raises
    ------
    ValueError
        If fewer than *n* samples satisfy the selection criteria.
    """
    expr = dataset.expression.apply(pd.to_numeric, errors="coerce")

    # Build sample index → sample ID mapping
    sid_col = "SampleID"
    for col in ("SampleID", "SampleId", "SampleName"):
        if col in dataset.samples.columns:
            sid_col = col
            break

    if sid_col in dataset.samples.columns:
        sample_ids = dataset.samples[sid_col].astype(str)
    else:
        sample_ids = pd.Series([f"S{i}" for i in range(len(dataset.samples))])

    # 1. Exclude control samples
    keep = pd.Series(True, index=dataset.samples.index)
    if sid_col in dataset.samples.columns:
        keep &= ~sample_ids.str.contains("CONTROL_SAMPLE", case=False, na=False)
    if "SampleType" in dataset.samples.columns:
        from pyprideap.processing.filtering import _CONTROL_SAMPLE_TYPES

        keep &= ~dataset.samples["SampleType"].astype(str).str.lower().str.strip().isin(_CONTROL_SAMPLE_TYPES)

    # 2. Exclude QC outliers
    if exclude_qc_outliers:
        try:
            from pyprideap.processing.olink.outliers import compute_iqr_median_outliers

            result = compute_iqr_median_outliers(
                dataset,
                iqr_outlier_def=iqr_outlier_def,
                median_outlier_def=median_outlier_def,
            )
            outlier_ids = set(result.outlier_sample_ids)
            keep &= ~sample_ids.isin(outlier_ids)
        except Exception:
            pass  # If outlier detection fails, skip this filter

    # 3. Keep only QC PASS samples
    for qc_col in ("SampleQC", "QC_Warning"):
        if qc_col in dataset.samples.columns:
            qc_vals = dataset.samples[qc_col].astype(str).str.upper()
            keep &= qc_vals == "PASS"
            break

    # 4. Filter by missing frequency (below-LOD rate per sample)
    from pyprideap.processing.lod import get_lod_values

    lod = get_lod_values(dataset)
    if lod is not None:
        from pyprideap.processing.lod import _above_lod_matrix

        above_lod, has_lod = _above_lod_matrix(expr, lod)
        n_valid = (expr.notna() & has_lod).sum(axis=1)
        n_below = n_valid - (above_lod & has_lod).sum(axis=1)
        pct_below = n_below / n_valid.clip(lower=1)
        keep &= pct_below < sample_missing_freq

    # Apply filter
    valid_idx = keep[keep].index
    if len(valid_idx) == 0:
        # Fallback: just use QC PASS without outlier/LOD filtering
        keep = pd.Series(True, index=dataset.samples.index)
        if "SampleQC" in dataset.samples.columns:
            keep &= dataset.samples["SampleQC"].astype(str).str.upper() == "PASS"
        valid_idx = keep[keep].index

    if len(valid_idx) < n:
        raise ValueError(
            f"Only {len(valid_idx)} samples satisfy the selection criteria. "
            f"Increase sample_missing_freq or decrease n (requested {n})."
        )

    # 5. Compute MeanNPX and select evenly-spaced samples across the range
    valid_expr = expr.loc[valid_idx]
    mean_npx = valid_expr.mean(axis=1)

    # Sort by MeanNPX descending and select evenly spaced indices
    sorted_idx = mean_npx.sort_values(ascending=False).index
    total = len(sorted_idx)

    if total == n:
        selected_idx = sorted_idx
    else:
        # Evenly space through the range (matching OlinkAnalyze's approach)
        positions = np.linspace(0, total - 1, n + 2, dtype=int)[1:-1]
        selected_idx = sorted_idx[positions]

    selected: list[str] = sample_ids.loc[selected_idx].tolist()
    logger.debug(
        "select_bridge_samples: %d candidates after filtering, selected %d bridge samples",
        len(valid_idx),
        len(selected),
    )
    return selected


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

    aligned1, aligned2, common_samples = _resolve_common_samples(dataset1, dataset2)
    logger.debug(
        "assess_bridgeability: %d overlapping proteins, %d common samples",
        len(overlapping_proteins),
        len(common_samples),
    )

    records: list[dict] = []
    for protein in overlapping_proteins:
        vals1 = dataset1.expression[protein]
        vals2 = dataset2.expression[protein]

        detection_rate_1 = float(vals1.notna().mean())
        detection_rate_2 = float(vals2.notna().mean())
        median_diff = float(vals1.median() - vals2.median())

        # Correlation requires matched samples
        if len(common_samples) >= 3:
            paired = pd.DataFrame(
                {
                    "a": aligned1[protein].reindex(common_samples),
                    "b": aligned2[protein].reindex(common_samples),
                }
            ).dropna()
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


def assess_cross_product_bridgeability(
    dataset1: AffinityDataset,
    dataset2: AffinityDataset,
    *,
    iqr_multiplier: float = 3.0,
    min_datapoint_count: int = 10,
    median_count_threshold: int = 150,
) -> pd.DataFrame:
    """Assess per-assay bridgeability between two Olink products.

    Mirrors OlinkAnalyze's ``olink_normalization_bridgeable()``.
    For each overlapping assay, computes:

    * **range_diff** — absolute difference of 90th–10th percentile ranges
    * **low_cnt** — True if either dataset has median count < 150
    * **r2** — squared Pearson correlation of matched bridge samples
    * **ks_stat** — KS test statistic comparing the two distributions

    Bridging recommendation:
    * ``range_diff > 1 AND low_cnt AND r² < 0.8`` → NotBridgeable
    * ``IsBridgeable AND ks_stat ≤ 0.2`` → MedianCentering
    * ``IsBridgeable AND ks_stat > 0.2`` → QuantileSmoothing

    Parameters
    ----------
    dataset1, dataset2:
        Olink datasets from different products (e.g. Explore 3072 vs HT).
        Should contain only bridge sample data.
    iqr_multiplier:
        IQR multiplier for outlier removal (default 3.0).
    min_datapoint_count:
        Minimum count threshold for filtering data points (default 10).
    median_count_threshold:
        Threshold for low median count detection (default 150).

    Returns
    -------
    pd.DataFrame
        One row per overlapping assay with columns: ``protein_id``,
        ``range_diff``, ``low_cnt``, ``r2``, ``ks_stat``,
        ``is_bridgeable``, ``bridging_recommendation``.
    """
    from scipy import stats as sp_stats

    from pyprideap.processing.olink.outliers import is_iqr_outlier

    overlapping = dataset1.expression.columns.intersection(dataset2.expression.columns)
    if overlapping.empty:
        raise ValueError("No overlapping proteins between the two datasets.")

    aligned1, aligned2, common_samples = _resolve_common_samples(dataset1, dataset2)

    numeric1 = dataset1.expression[overlapping].apply(pd.to_numeric, errors="coerce")
    numeric2 = dataset2.expression[overlapping].apply(pd.to_numeric, errors="coerce")
    aligned_numeric1 = aligned1[overlapping].apply(pd.to_numeric, errors="coerce")
    aligned_numeric2 = aligned2[overlapping].apply(pd.to_numeric, errors="coerce")

    records: list[dict] = []
    for protein in overlapping:
        v1 = numeric1[protein].dropna()
        v2 = numeric2[protein].dropna()

        if len(v1) < 3 or len(v2) < 3:
            records.append(
                {
                    "protein_id": protein,
                    "range_diff": np.nan,
                    "low_cnt": True,
                    "r2": np.nan,
                    "ks_stat": np.nan,
                    "is_bridgeable": False,
                    "bridging_recommendation": "NotBridgeable",
                }
            )
            continue

        # Remove outliers per assay
        not_outlier1 = ~is_iqr_outlier(v1, iqr_multiplier=iqr_multiplier)
        not_outlier2 = ~is_iqr_outlier(v2, iqr_multiplier=iqr_multiplier)
        v1_clean = v1[not_outlier1]
        v2_clean = v2[not_outlier2]

        if len(v1_clean) < 3 or len(v2_clean) < 3:
            records.append(
                {
                    "protein_id": protein,
                    "range_diff": np.nan,
                    "low_cnt": True,
                    "r2": np.nan,
                    "ks_stat": np.nan,
                    "is_bridgeable": False,
                    "bridging_recommendation": "NotBridgeable",
                }
            )
            continue

        # Range difference (90th - 10th percentile)
        range1 = float(np.percentile(v1_clean, 90) - np.percentile(v1_clean, 10))
        range2 = float(np.percentile(v2_clean, 90) - np.percentile(v2_clean, 10))
        range_diff = abs(range1 - range2)

        # Low count check
        count_matrix1_raw = dataset1.metadata.get("count_matrix")
        count_matrix2_raw = dataset2.metadata.get("count_matrix")
        low_cnt = True  # default if no count data
        if count_matrix1_raw is not None and count_matrix2_raw is not None:
            cm1 = cast(pd.DataFrame, count_matrix1_raw)
            cm2 = cast(pd.DataFrame, count_matrix2_raw)
            if protein in cm1.columns and protein in cm2.columns:
                med_cnt1 = float(pd.to_numeric(cm1[protein], errors="coerce").median())
                med_cnt2 = float(pd.to_numeric(cm2[protein], errors="coerce").median())
                low_cnt = med_cnt1 < median_count_threshold or med_cnt2 < median_count_threshold

        # Correlation (r²) on matched samples
        r2 = np.nan
        if len(common_samples) >= 3:
            paired = pd.DataFrame(
                {
                    "a": aligned_numeric1[protein].reindex(common_samples),
                    "b": aligned_numeric2[protein].reindex(common_samples),
                }
            ).dropna()
            if len(paired) >= 3:
                corr = float(paired["a"].corr(paired["b"]))
                r2 = corr**2 if not np.isnan(corr) else np.nan

        # KS test statistic
        ks_result = sp_stats.ks_2samp(v1_clean.values, v2_clean.values)
        ks_stat = float(ks_result.statistic)

        # Bridgeability decision (OlinkAnalyze logic)
        is_bridgeable = not (range_diff > 1 and low_cnt and (np.isnan(r2) or r2 < 0.8))

        if not is_bridgeable:
            recommendation = "NotBridgeable"
        elif ks_stat <= 0.2:
            recommendation = "MedianCentering"
        else:
            recommendation = "QuantileSmoothing"

        records.append(
            {
                "protein_id": protein,
                "range_diff": round(range_diff, 4),
                "low_cnt": low_cnt,
                "r2": round(r2, 4) if not np.isnan(r2) else np.nan,
                "ks_stat": round(ks_stat, 4),
                "is_bridgeable": is_bridgeable,
                "bridging_recommendation": recommendation,
            }
        )

    return pd.DataFrame(records)


_QS_KNOT_PROBS = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)

# Minimum bridge samples per product pair (OlinkAnalyze convention)
_MIN_BRIDGE_SAMPLES: dict[tuple[str, str], int] = {
    ("E3072", "HT"): 40,
    ("HT", "E3072"): 40,
    ("E3072", "Reveal"): 32,
    ("Reveal", "E3072"): 32,
    ("HT", "Reveal"): 24,
    ("Reveal", "HT"): 24,
}
_DEFAULT_MIN_BRIDGE = 24


def quantile_smooth_normalize(
    reference: AffinityDataset,
    target: AffinityDataset,
    bridge_samples: list[str] | pd.Index,
    *,
    min_bridge_samples: int | None = None,
    product_pair: tuple[str, str] | None = None,
) -> AffinityDataset:
    """Quantile smoothing (QS) normalization between two Olink products.

    Mirrors OlinkAnalyze's ``olink_normalization_qs()``.

    For each overlapping assay, maps the *target* NPX distribution to the
    *reference* distribution using:

    1. Build ECDF of reference bridge sample values
    2. Map target bridge sample values through the inverse ECDF
    3. Fit a natural spline regression using 7 quantile knots
       (5%, 10%, 25%, 50%, 75%, 90%, 95%)
    4. Predict normalized NPX for all target samples

    Parameters
    ----------
    reference:
        Reference dataset (e.g. Olink Explore HT).
    target:
        Non-reference dataset to normalize (e.g. Explore 3072).
    bridge_samples:
        Sample IDs present in both datasets used for mapping.
    min_bridge_samples:
        Minimum bridge samples required per assay. If None, determined
        from *product_pair* or defaults to 24.
    product_pair:
        Tuple like ``("E3072", "HT")`` to auto-select min bridge samples.

    Returns
    -------
    AffinityDataset
        Copy of *target* with QS-normalized expression values.
        Assays with insufficient bridge samples are left unchanged.
    """
    from scipy.interpolate import make_interp_spline

    if min_bridge_samples is None:
        if product_pair is not None:
            min_bridge_samples = _MIN_BRIDGE_SAMPLES.get(product_pair, _DEFAULT_MIN_BRIDGE)
        else:
            min_bridge_samples = _DEFAULT_MIN_BRIDGE

    bridge_idx = pd.Index(bridge_samples)
    overlapping = reference.expression.columns.intersection(target.expression.columns)
    if overlapping.empty:
        raise ValueError("No overlapping proteins between reference and target.")

    ref_numeric = reference.expression[overlapping].apply(pd.to_numeric, errors="coerce")
    tgt_numeric = target.expression[overlapping].apply(pd.to_numeric, errors="coerce")
    normalized = tgt_numeric.copy()

    # Bridge sample indices in each dataset
    ref_bridge = ref_numeric.index.intersection(bridge_idx)
    tgt_bridge = tgt_numeric.index.intersection(bridge_idx)
    common_bridge = ref_bridge.intersection(tgt_bridge)
    logger.debug(
        "quantile_smooth_normalize: %d overlapping assays, %d common bridge samples, min_required=%d",
        len(overlapping),
        len(common_bridge),
        min_bridge_samples,
    )

    n_skipped = 0
    for protein in overlapping:
        # Get bridge sample values for this assay
        ref_vals = ref_numeric.loc[common_bridge, protein].dropna()
        tgt_vals = tgt_numeric.loc[common_bridge, protein].dropna()

        # Intersect to matched samples
        matched = ref_vals.index.intersection(tgt_vals.index)
        if len(matched) < min_bridge_samples:
            n_skipped += 1
            continue

        ref_bridge_vals = ref_vals.loc[matched].values
        tgt_bridge_vals = tgt_vals.loc[matched].values

        # Step 1: ECDF of target bridge values
        sorted_tgt = np.sort(tgt_bridge_vals)
        ecdf_probs = np.arange(1, len(sorted_tgt) + 1) / len(sorted_tgt)

        # Step 2: Map to reference quantiles at same probabilities
        ref_quantiles = np.quantile(ref_bridge_vals, ecdf_probs)

        # Step 3: Fit spline — knots at 7 quantile positions of target bridge
        # Use unique sorted target values as x, mapped reference quantiles as y
        unique_tgt = np.unique(sorted_tgt)
        # Map: for each unique target value, find the corresponding ref quantile
        mapped_ref = np.interp(unique_tgt, sorted_tgt, ref_quantiles)

        if len(unique_tgt) < 4:
            # Too few unique values for spline; fall back to linear interp
            all_tgt = tgt_numeric[protein].values
            normalized[protein] = np.interp(all_tgt, unique_tgt, mapped_ref)
            continue

        # Natural cubic spline through the mapped points
        try:
            # Use degree min(3, n_unique - 1) to avoid overfitting
            k = min(3, len(unique_tgt) - 1)
            spline = make_interp_spline(unique_tgt, mapped_ref, k=k)
            all_tgt = tgt_numeric[protein].values
            predicted = spline(all_tgt)
            # Clip extreme extrapolations to data range
            ref_min, ref_max = float(ref_bridge_vals.min()), float(ref_bridge_vals.max())
            margin = (ref_max - ref_min) * 0.1
            predicted = np.clip(predicted, ref_min - margin, ref_max + margin)
            normalized[protein] = predicted
        except (ValueError, np.linalg.LinAlgError):
            # Fallback: linear interpolation
            all_tgt = tgt_numeric[protein].values
            normalized[protein] = np.interp(all_tgt, unique_tgt, mapped_ref)

    logger.debug(
        "quantile_smooth_normalize: normalized %d assays, skipped %d (insufficient bridge samples)",
        len(overlapping) - n_skipped,
        n_skipped,
    )

    # Preserve NaN positions from original
    nan_mask = tgt_numeric.isna()
    normalized[nan_mask] = np.nan

    # Build result dataset
    result_expression = target.expression.copy()
    result_expression[overlapping] = normalized

    return replace(
        target,
        expression=result_expression,
        metadata={
            **target.metadata,
            "qs_normalization": {
                "n_assays_normalized": int(len(overlapping) - n_skipped),
                "n_assays_skipped": n_skipped,
                "n_bridge_samples": len(common_bridge),
                "min_bridge_required": min_bridge_samples,
            },
        },
    )


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


# ---------------------------------------------------------------------------
# SomaScan version mapping (matches SomaDataIO R/utils-lift.R)
# ---------------------------------------------------------------------------

# Map commercial assay version names to internal size labels
_VER_TO_SIZE: dict[str, str] = {
    "V3": "1.1k",
    "v3": "1.1k",
    "v3.0": "1.1k",
    "V3.2": "1.3k",
    "v3.2": "1.3k",
    "V4": "5k",
    "v4": "5k",
    "v4.0": "5k",
    "V4.1": "7k",
    "v4.1": "7k",
    "V5": "11k",
    "v5": "11k",
    "v5.0": "11k",
}

_SIZE_TO_VER: dict[str, str] = {
    "1.1k": "v3.0",
    "1.3k": "v3.2",
    "5k": "v4.0",
    "7k": "v4.1",
    "11k": "v5.0",
}

# Valid lifting bridges
_VALID_BRIDGES = frozenset(
    {
        "11k_to_7k",
        "11k_to_5k",
        "7k_to_11k",
        "7k_to_5k",
        "5k_to_11k",
        "5k_to_7k",
    }
)


def _resolve_somascan_version(dataset: AffinityDataset) -> str | None:
    """Resolve the SomaScan assay size label (e.g. '5k', '7k', '11k').

    Checks ``metadata["SignalSpace"]`` first, then ``metadata["AssayVersion"]``.
    Returns None if the version cannot be determined.
    """
    for key in ("SignalSpace", "AssayVersion"):
        val = dataset.metadata.get(key)
        if val is not None:
            val_str = str(val).strip()
            # Direct size label
            if val_str.lower() in _SIZE_TO_VER:
                return val_str.lower()
            # Version string
            size = _VER_TO_SIZE.get(val_str)
            if size is not None:
                return size
    return None


def validate_lift_requirements(
    dataset: AffinityDataset,
    bridge: str,
) -> list[str]:
    """Validate that a SomaScan dataset meets lifting requirements.

    Checks performed (matching SomaDataIO ``lift_adat()``):
    1. Bridge direction is valid
    2. Data has been ANML normalized (ProcessSteps contains "ANML")
    3. Current signal space matches the 'from' side of the bridge
    4. Sample matrix is supported (plasma or serum)

    Parameters
    ----------
    dataset:
        SomaScan AffinityDataset.
    bridge:
        Bridge direction, e.g. ``"5k_to_11k"``.

    Returns
    -------
    list[str]
        List of validation error messages. Empty if all checks pass.
    """
    errors: list[str] = []

    # 1. Valid bridge
    if bridge not in _VALID_BRIDGES:
        errors.append(f"Invalid bridge '{bridge}'. Valid options: {sorted(_VALID_BRIDGES)}")
        return errors

    from_space = bridge.split("_to_")[0]
    to_space = bridge.split("_to_")[1]

    # 2. ANML normalization check
    process_steps = str(dataset.metadata.get("ProcessSteps", ""))
    if "anml" not in process_steps.lower():
        errors.append("ANML normalized SomaScan data is required for lifting. ProcessSteps does not contain 'ANML'.")

    # 3. Signal space check
    current_space = _resolve_somascan_version(dataset)
    if current_space is not None and current_space != from_space:
        errors.append(
            f"Bridge '{bridge}' expects data in {from_space} space, but dataset appears to be in {current_space} space."
        )
    if current_space == to_space:
        errors.append(f"Data already appears to be in {to_space} space.")

    # 4. Matrix check
    matrix = str(dataset.metadata.get("StudyMatrix", "")).lower()
    if matrix and not any(m in matrix for m in ("plasma", "serum")):
        errors.append(
            f"Unsupported sample matrix: '{dataset.metadata.get('StudyMatrix')}'. "
            f"Lifting is only supported for 'EDTA Plasma' or 'Serum'."
        )

    return errors


def lift_somascan(
    dataset: AffinityDataset,
    scalars: dict[str, float] | pd.Series,
    target_version: str | None = None,
    *,
    bridge: str | None = None,
    validate: bool = True,
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
    bridge:
        Optional bridge direction string (e.g. ``"5k_to_11k"``).
        When provided, ``validate_lift_requirements`` is run first
        and ``target_version`` is inferred if not explicitly set.
    validate:
        Whether to run validation checks when ``bridge`` is provided.
        Set to False to skip validation (not recommended).

    Returns
    -------
    AffinityDataset
        A copy with lifted RFU values.

    Raises
    ------
    ValueError
        If validation fails when ``bridge`` is provided and ``validate=True``.
    """
    if isinstance(scalars, dict):
        scalars = pd.Series(scalars)

    # Validate if bridge is provided
    if bridge is not None and validate:
        errors = validate_lift_requirements(dataset, bridge)
        if errors:
            raise ValueError("Lifting validation failed:\n" + "\n".join(f"  - {e}" for e in errors))
        # Infer target_version from bridge
        if target_version is None:
            to_space = bridge.split("_to_")[1]
            target_version = _SIZE_TO_VER.get(to_space, to_space)

    # Default missing scalars to 1.0 (no transformation)
    full_scalars = pd.Series(1.0, index=dataset.expression.columns)
    matched = full_scalars.index.intersection(scalars.index)
    full_scalars[matched] = scalars[matched]
    logger.debug(
        "lift_somascan: %d/%d analytes have scalars, bridge=%s, target_version=%s",
        len(matched),
        len(dataset.expression.columns),
        bridge,
        target_version,
    )

    scaled_expression = dataset.expression.multiply(full_scalars, axis=1)
    # Round to 1 decimal, matching SomaDataIO convention
    scaled_expression = scaled_expression.round(1)

    new_metadata = {**dataset.metadata}
    if target_version is not None:
        new_metadata["SignalSpace"] = target_version

    # Record lift in ProcessSteps
    if bridge is not None:
        from_space = bridge.split("_to_")[0]
        to_space = bridge.split("_to_")[1]
        step = f"Lifting Bridge ({from_space} -> {to_space})"
        existing = str(new_metadata.get("ProcessSteps", ""))
        new_metadata["ProcessSteps"] = f"{existing}, {step}" if existing else step

    return replace(dataset, expression=scaled_expression, metadata=new_metadata)


def lins_ccc(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Lin's Concordance Correlation Coefficient (CCC).

    Measures agreement between two measurements, accounting for both
    correlation and deviation from the line of identity.

    Formula::

        CCC = (2 × ρ × σx × σy) / [(μx − μy)² + σx² + σy²]

    where ρ is the Pearson correlation coefficient.

    References
    ----------
    Lin, Lawrence I-Kuei. 1989. A Concordance Correlation Coefficient
    to Evaluate Reproducibility. Biometrics. 45:255-268.

    Parameters
    ----------
    x, y:
        1-D arrays of matched measurements.

    Returns
    -------
    float
        CCC value in [-1, 1]. Values near 1 indicate strong agreement.
    """
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]

    if len(x_clean) < 3:
        return float("nan")

    mx = np.mean(x_clean)
    my = np.mean(y_clean)
    sx = np.std(x_clean, ddof=1)
    sy = np.std(y_clean, ddof=1)

    if sx == 0 or sy == 0:
        return float("nan")

    rho = float(np.corrcoef(x_clean, y_clean)[0, 1])
    numerator = 2.0 * rho * sx * sy
    denominator = (mx - my) ** 2 + sx**2 + sy**2

    if denominator == 0:
        return float("nan")

    return float(numerator / denominator)


def assess_lift_quality(
    original: AffinityDataset,
    lifted: AffinityDataset,
) -> pd.DataFrame:
    """Assess lifting quality using Lin's CCC per analyte.

    Compares RFU values before and after lifting to measure how well
    the linear transformation preserves the measurement agreement.
    Equivalent to ``getSomaScanLiftCCC()`` in SomaDataIO but computed
    from the actual data rather than a lookup table.

    Parameters
    ----------
    original:
        Dataset before lifting.
    lifted:
        Dataset after lifting.

    Returns
    -------
    pd.DataFrame
        One row per overlapping analyte with columns:
        ``analyte``, ``ccc``, ``pearson_r``, ``median_original``,
        ``median_lifted``, ``scalar`` (lifted/original ratio).
    """
    common = original.expression.columns.intersection(lifted.expression.columns)
    if common.empty:
        raise ValueError("No overlapping analytes between original and lifted datasets")

    records: list[dict] = []
    for col in common:
        orig_vals = pd.to_numeric(original.expression[col], errors="coerce").values
        lift_vals = pd.to_numeric(lifted.expression[col], errors="coerce").values

        ccc = lins_ccc(orig_vals, lift_vals)
        mask = ~(np.isnan(orig_vals) | np.isnan(lift_vals))
        if mask.sum() >= 3:
            pearson_r = float(np.corrcoef(orig_vals[mask], lift_vals[mask])[0, 1])
        else:
            pearson_r = float("nan")

        med_orig = float(np.nanmedian(orig_vals))
        med_lift = float(np.nanmedian(lift_vals))
        scalar = med_lift / med_orig if med_orig != 0 else float("nan")

        records.append(
            {
                "analyte": str(col),
                "ccc": round(ccc, 4),
                "pearson_r": round(pearson_r, 4),
                "median_original": round(med_orig, 2),
                "median_lifted": round(med_lift, 2),
                "scalar": round(scalar, 4),
            }
        )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Multi-project normalization chaining (mirrors olink_normalization_n)
# ---------------------------------------------------------------------------


@dataclass
class NormalizationStep:
    """Single step in a multi-project normalization schema.

    Parameters
    ----------
    order:
        Sequential identifier (1-based). Order 1 is the global reference.
    name:
        Unique name for this project.
    dataset:
        The AffinityDataset for this project.
    bridge_samples:
        Dict with keys ``"ref"`` and ``"target"`` mapping to lists of
        sample IDs.  For bridge normalization both lists must have the
        same length (paired). For subset normalization they can differ.
        ``None`` for the reference project (order=1).
    normalization_type:
        ``"bridge"`` or ``"subset"``.  ``None`` for the reference project.
    normalize_to:
        List of order values (ints) referencing which project(s) to
        normalize to.  ``None`` for the reference project.
    """

    order: int
    name: str
    dataset: AffinityDataset
    bridge_samples: dict[str, list[str]] | None = None
    normalization_type: str | None = None
    normalize_to: list[int] | None = None


def normalize_n(
    steps: list[NormalizationStep],
) -> dict[str, AffinityDataset]:
    """Chain bridge/subset normalization across N projects.

    Mirrors OlinkAnalyze's ``olink_normalization_n()``.

    The first step (order=1) is the global reference and is not
    modified.  Each subsequent step is normalized against the project(s)
    specified by its ``normalize_to`` field.  When ``normalize_to``
    references multiple projects, their datasets are concatenated before
    computing the adjustment.

    Parameters
    ----------
    steps:
        List of :class:`NormalizationStep` objects, one per project.
        Must include exactly one step with ``order=1`` (the reference).

    Returns
    -------
    dict[str, AffinityDataset]
        Mapping of project name → normalized dataset.

    Raises
    ------
    ValueError
        If schema validation fails.
    """
    _validate_norm_schema(steps)
    logger.debug("normalize_n: %d steps, reference='%s'", len(steps), next(s.name for s in steps if s.order == 1))

    sorted_steps = sorted(steps, key=lambda s: s.order)
    ref_step = sorted_steps[0]

    results: dict[str, AffinityDataset] = {ref_step.name: ref_step.dataset}

    for step in sorted_steps[1:]:
        if step.normalize_to is None or step.normalization_type is None:
            raise ValueError(f"Step '{step.name}' (order={step.order}) must have normalize_to and normalization_type.")

        # Build reference dataset by concatenating target projects
        ref_names = [s.name for s in sorted_steps if s.order in step.normalize_to]
        if len(ref_names) == 1:
            ref_ds = results[ref_names[0]]
        else:
            # Concatenate expression from multiple reference projects
            ref_datasets = [results[n] for n in ref_names]
            concat_expr = pd.concat(
                [d.expression for d in ref_datasets],
                axis=0,
            )
            concat_samples = pd.concat(
                [d.samples for d in ref_datasets],
                axis=0,
                ignore_index=True,
            )
            ref_ds = replace(
                ref_datasets[0],
                expression=concat_expr,
                samples=concat_samples,
            )

        # Normalize
        logger.debug(
            "normalize_n: step '%s' (order=%d), type=%s, normalizing to %s",
            step.name,
            step.order,
            step.normalization_type,
            ref_names,
        )
        norm_type = step.normalization_type.lower()
        if norm_type == "bridge":
            if step.bridge_samples is None:
                raise ValueError(f"Step '{step.name}' uses bridge normalization but bridge_samples is None.")
            normalized = bridge_normalize(
                ref_ds,
                step.dataset,
                step.bridge_samples["ref"],
            )
        elif norm_type == "subset":
            if step.bridge_samples is None:
                raise ValueError(f"Step '{step.name}' uses subset normalization but bridge_samples is None.")
            normalized = subset_normalize(
                ref_ds,
                step.dataset,
                step.bridge_samples["ref"],
            )
        else:
            raise ValueError(f"Unknown normalization_type '{step.normalization_type}'. Must be 'bridge' or 'subset'.")

        results[step.name] = normalized

    return results


def _validate_norm_schema(steps: list[NormalizationStep]) -> None:
    """Validate multi-project normalization schema."""
    if not steps:
        raise ValueError("At least one step is required.")

    orders = [s.order for s in steps]
    names = [s.name for s in steps]

    if len(set(orders)) != len(orders):
        raise ValueError("Step orders must be unique.")
    if len(set(names)) != len(names):
        raise ValueError("Step names must be unique.")
    if 1 not in orders:
        raise ValueError("Schema must include a reference step with order=1.")
    if sorted(orders) != list(range(1, len(orders) + 1)):
        raise ValueError("Step orders must be a contiguous sequence starting from 1.")

    # Validate normalize_to references
    order_set = set(orders)
    for step in steps:
        if step.order == 1:
            continue
        if step.normalize_to is None:
            raise ValueError(f"Step '{step.name}' (order={step.order}) must specify normalize_to.")
        for ref_order in step.normalize_to:
            if ref_order not in order_set:
                raise ValueError(f"Step '{step.name}' references order={ref_order} which does not exist.")
            if ref_order == step.order:
                raise ValueError(f"Step '{step.name}' cannot normalize to itself.")
            if ref_order >= step.order:
                raise ValueError(
                    f"Step '{step.name}' (order={step.order}) cannot normalize to a later step (order={ref_order})."
                )


# ---------------------------------------------------------------------------
# Post-normalization formatting (mirrors olink_normalization_format)
# ---------------------------------------------------------------------------


def format_normalized(
    reference: AffinityDataset,
    target: AffinityDataset,
    normalized_target: AffinityDataset,
    *,
    remove_controls: bool = True,
    add_non_overlapping: bool = True,
) -> AffinityDataset:
    """Format a normalized Olink dataset for downstream analysis.

    Mirrors OlinkAnalyze's ``olink_normalization_format()``:

    1. **Remove external controls** — strips NEGATIVE_CONTROL and
       PLATE_CONTROL samples from the normalized target dataset.
    2. **Add non-overlapping assays** — assays present in either the
       reference or target but not both are included without adjustment
       (their original NPX values are kept).

    Parameters
    ----------
    reference:
        The reference dataset (unchanged by normalization).
    target:
        The original (pre-normalization) target dataset.
    normalized_target:
        The normalized target dataset.
    remove_controls:
        Whether to remove NEGATIVE_CONTROL and PLATE_CONTROL samples.
    add_non_overlapping:
        Whether to add non-overlapping assays from both datasets.

    Returns
    -------
    AffinityDataset
        Formatted dataset ready for downstream analysis.
    """
    result = normalized_target

    # Step 1: Remove control samples
    if remove_controls:
        result = _remove_external_controls(result)

    # Step 2: Add non-overlapping assays
    if add_non_overlapping:
        result = _add_non_overlapping_assays(reference, target, result)

    return result


def _remove_external_controls(dataset: AffinityDataset) -> AffinityDataset:
    """Remove NEGATIVE_CONTROL and PLATE_CONTROL samples."""
    # Check SampleType column first
    sample_type_col = None
    for col in ("SampleType", "Sample_Type", "sample_type"):
        if col in dataset.samples.columns:
            sample_type_col = col
            break

    if sample_type_col is not None:
        control_types = {"NEGATIVE_CONTROL", "PLATE_CONTROL"}
        keep_mask = ~dataset.samples[sample_type_col].astype(str).isin(control_types)
    else:
        # Fall back to regex matching on SampleID
        sid_col = "SampleID"
        for col in ("SampleID", "SampleId", "SampleName"):
            if col in dataset.samples.columns:
                sid_col = col
                break

        if sid_col not in dataset.samples.columns:
            return dataset

        control_patterns = (
            "NEGATIVE",
            "NEG_CTRL",
            "PLATE_CONTROL",
            "IPC",
            "Neg_Ctrl",
        )
        sid_str = dataset.samples[sid_col].astype(str)
        keep_mask = ~sid_str.str.contains(
            "|".join(control_patterns),
            case=False,
            na=False,
        )

    n_removed = int((~keep_mask).sum())
    if n_removed == 0:
        return dataset

    return replace(
        dataset,
        samples=dataset.samples[keep_mask].reset_index(drop=True),
        expression=dataset.expression[keep_mask].reset_index(drop=True),
    )


def _add_non_overlapping_assays(
    reference: AffinityDataset,
    original_target: AffinityDataset,
    normalized_target: AffinityDataset,
) -> AffinityDataset:
    """Add non-overlapping assays from reference and original target.

    Assays present in only one dataset are added to the normalized
    result with their original NPX values (no adjustment).
    """
    norm_cols = set(normalized_target.expression.columns)
    orig_tgt_cols = set(original_target.expression.columns)

    # Find non-overlapping assays
    tgt_only = orig_tgt_cols - norm_cols

    if not tgt_only:
        return normalized_target

    result_expr = normalized_target.expression.copy()

    # Add target-only assays (from the original, unnormalized target)
    for col in sorted(tgt_only):
        if col in original_target.expression.columns:
            result_expr[col] = original_target.expression[col].reindex(result_expr.index)

    # Update features table if available
    new_features = normalized_target.features
    if new_features is not None and not new_features.empty:
        id_col = "OlinkID" if "OlinkID" in new_features.columns else new_features.columns[0]
        existing_ids = set(new_features[id_col])
        new_rows = []

        if original_target.features is not None:
            tgt_id_col = (
                "OlinkID" if "OlinkID" in original_target.features.columns else original_target.features.columns[0]
            )
            for _, row in original_target.features.iterrows():
                if row[tgt_id_col] in tgt_only and row[tgt_id_col] not in existing_ids:
                    new_rows.append(row)
                    existing_ids.add(row[tgt_id_col])

        if new_rows:
            new_features = pd.concat(
                [new_features, pd.DataFrame(new_rows)],
                ignore_index=True,
            )

    return replace(
        normalized_target,
        expression=result_expr,
        features=new_features,
    )
