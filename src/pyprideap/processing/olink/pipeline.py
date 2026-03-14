"""Unified Olink preprocessing pipeline.

Mirrors OlinkAnalyze's preprocessing workflow with togglable steps:

1. Filter controls → remove NEGATIVE_CONTROL, PLATE_CONTROL, etc.
2. QC outlier detection → IQR vs Median per-panel outlier removal
3. Filter by QC status → remove WARN/FAIL samples
4. LOD filtering → remove assays below LOD threshold
5. UniProt duplicate handling → flag/remove duplicate mappings
6. Preprocessing for dimensionality reduction → zero-variance removal,
   missingness filtering, median imputation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from pyprideap.core import AffinityDataset

logger = logging.getLogger(__name__)


@dataclass
class OlinkPreprocessingReport:
    """Summary of actions taken during Olink preprocessing."""

    steps: list[str] = field(default_factory=list)
    n_controls_removed: int = 0
    n_qc_outliers_removed: int = 0
    n_qc_warning_removed: int = 0
    n_assays_below_lod: int = 0
    n_assays_zero_variance: int = 0
    n_assays_high_missingness: int = 0
    n_assays_imputed: int = 0
    n_uniprot_duplicates: int = 0
    final_samples: int = 0
    final_features: int = 0

    def summary(self) -> str:
        lines = ["Olink Preprocessing Report", "=" * 40]
        for step in self.steps:
            lines.append(f"  - {step}")
        lines.append(f"Final: {self.final_samples} samples x {self.final_features} features")
        return "\n".join(lines)


def preprocess_olink(
    dataset: AffinityDataset,
    *,
    filter_controls: bool = True,
    filter_qc_outliers: bool = True,
    filter_qc_warning: bool = False,
    filter_lod: bool = False,
    lod_detection_rate: float = 0.5,
    remove_uniprot_duplicates: bool = False,
    prep_for_dimred: bool = False,
    missingness_cutoff: float | None = None,
    iqr_outlier_def: float = 3.0,
    median_outlier_def: float = 3.0,
) -> tuple[AffinityDataset, OlinkPreprocessingReport]:
    """Preprocess an Olink dataset for analysis.

    Parameters
    ----------
    dataset:
        Raw Olink AffinityDataset with NPX values.
    filter_controls:
        Remove control samples (NEGATIVE_CONTROL, PLATE_CONTROL, etc.).
    filter_qc_outliers:
        Remove samples flagged as outliers by IQR vs Median QC.
    filter_qc_warning:
        Remove samples with SampleQC == WARN or FAIL.
    filter_lod:
        Remove assays where fewer than *lod_detection_rate* fraction
        of samples have NPX above LOD.
    lod_detection_rate:
        Minimum fraction of samples above LOD to keep an assay (default 0.5).
    remove_uniprot_duplicates:
        Remove assays with multiple UniProt mappings.
    prep_for_dimred:
        Apply structured preprocessing for PCA/t-SNE: remove zero-variance
        assays, drop high-missingness assays, impute remaining by median.
    missingness_cutoff:
        Maximum fraction of missing values per assay. If None, uses
        0.10 for large datasets (>88 samples) or 0.05 for small.
    iqr_outlier_def:
        Number of SDs for IQR outlier threshold (default 3.0).
    median_outlier_def:
        Number of SDs for median outlier threshold (default 3.0).

    Returns
    -------
    tuple[AffinityDataset, OlinkPreprocessingReport]
        Preprocessed dataset and report of actions taken.
    """
    report = OlinkPreprocessingReport()
    ds = dataset

    # Step 1: Remove control samples
    if filter_controls:
        ds, n_removed = _filter_control_samples(ds)
        report.n_controls_removed = n_removed
        if n_removed > 0:
            report.steps.append(f"Removed {n_removed} control samples")

    # Step 2: Remove QC outliers (IQR vs Median)
    if filter_qc_outliers:
        ds, n_removed = _filter_qc_outliers(ds, iqr_outlier_def, median_outlier_def)
        report.n_qc_outliers_removed = n_removed
        if n_removed > 0:
            report.steps.append(f"Removed {n_removed} IQR/Median QC outlier samples")

    # Step 3: Filter by QC status
    if filter_qc_warning:
        ds, n_removed = _filter_qc_warning(ds)
        report.n_qc_warning_removed = n_removed
        if n_removed > 0:
            report.steps.append(f"Removed {n_removed} WARN/FAIL QC samples")

    # Step 4: LOD filtering (assays)
    if filter_lod:
        ds, n_removed = _filter_by_lod(ds, lod_detection_rate)
        report.n_assays_below_lod = n_removed
        if n_removed > 0:
            report.steps.append(f"Removed {n_removed} assays below LOD ({lod_detection_rate:.0%} threshold)")

    # Step 5: UniProt duplicate handling
    if remove_uniprot_duplicates:
        ds, n_removed = _remove_uniprot_duplicates(ds)
        report.n_uniprot_duplicates = n_removed
        if n_removed > 0:
            report.steps.append(f"Removed {n_removed} assays with duplicate UniProt mappings")

    # Step 6: Preprocessing for dimensionality reduction
    if prep_for_dimred:
        ds, dimred_report = _prep_for_dimred(ds, missingness_cutoff)
        report.n_assays_zero_variance = dimred_report["zero_variance"]
        report.n_assays_high_missingness = dimred_report["high_missingness"]
        report.n_assays_imputed = dimred_report["imputed"]
        if dimred_report["zero_variance"] > 0:
            report.steps.append(f"Removed {dimred_report['zero_variance']} zero-variance assays")
        if dimred_report["high_missingness"] > 0:
            cutoff = dimred_report["cutoff"]
            report.steps.append(f"Removed {dimred_report['high_missingness']} assays with >{cutoff:.0%} missingness")
        if dimred_report["imputed"] > 0:
            report.steps.append(f"Imputed {dimred_report['imputed']} assays by median")

    report.final_samples = len(ds.samples)
    report.final_features = len(ds.features) if ds.features is not None else ds.expression.shape[1]
    return ds, report


def _filter_control_samples(
    dataset: AffinityDataset,
) -> tuple[AffinityDataset, int]:
    """Remove control samples (NEGATIVE_CONTROL, PLATE_CONTROL, etc.)."""
    from pyprideap.processing.filtering import filter_controls

    filtered = filter_controls(dataset)
    n_removed = len(dataset.samples) - len(filtered.samples)
    return filtered, n_removed


def _filter_qc_outliers(
    dataset: AffinityDataset,
    iqr_outlier_def: float,
    median_outlier_def: float,
) -> tuple[AffinityDataset, int]:
    """Remove samples flagged as outliers by IQR vs Median QC."""
    from pyprideap.processing.olink.outliers import compute_iqr_median_outliers

    result = compute_iqr_median_outliers(
        dataset,
        iqr_outlier_def=iqr_outlier_def,
        median_outlier_def=median_outlier_def,
    )

    outlier_ids = set(result.outlier_sample_ids)
    if not outlier_ids:
        return dataset, 0

    # Find sample ID column
    sid_col = "SampleID"
    for col in ("SampleID", "SampleId", "SampleName"):
        if col in dataset.samples.columns:
            sid_col = col
            break

    if sid_col in dataset.samples.columns:
        keep_mask = ~dataset.samples[sid_col].astype(str).isin(outlier_ids)
    else:
        keep_mask = pd.Series(True, index=dataset.samples.index)

    n_removed = int((~keep_mask).sum())
    if n_removed == 0:
        return dataset, 0

    return replace(
        dataset,
        samples=dataset.samples[keep_mask].reset_index(drop=True),
        expression=dataset.expression[keep_mask].reset_index(drop=True),
    ), n_removed


def _filter_qc_warning(
    dataset: AffinityDataset,
) -> tuple[AffinityDataset, int]:
    """Remove samples with SampleQC == WARN or FAIL."""
    qc_col = None
    for col in ("SampleQC", "QC_Warning"):
        if col in dataset.samples.columns:
            qc_col = col
            break

    if qc_col is None:
        return dataset, 0

    qc_vals = dataset.samples[qc_col].astype(str).str.upper()
    keep_mask = ~qc_vals.isin({"WARN", "WARNING", "FAIL"})
    n_removed = int((~keep_mask).sum())

    if n_removed == 0:
        return dataset, 0

    return replace(
        dataset,
        samples=dataset.samples[keep_mask].reset_index(drop=True),
        expression=dataset.expression[keep_mask].reset_index(drop=True),
    ), n_removed


def _filter_by_lod(
    dataset: AffinityDataset,
    detection_rate: float,
) -> tuple[AffinityDataset, int]:
    """Remove assays where fewer than detection_rate of samples are above LOD."""
    from pyprideap.processing.lod import get_lod_values

    lod = get_lod_values(dataset)
    if lod is None:
        return dataset, 0

    numeric = dataset.expression.apply(pd.to_numeric, errors="coerce")

    if isinstance(lod, pd.DataFrame):
        above = numeric.gt(lod.reindex(columns=numeric.columns))
    else:
        above = numeric.gt(lod.reindex(numeric.columns), axis=1)

    rate_per_assay = above.mean(axis=0)
    keep_cols = rate_per_assay[rate_per_assay >= detection_rate].index
    drop_cols = rate_per_assay.index.difference(keep_cols)
    n_removed = len(drop_cols)

    if n_removed == 0:
        return dataset, 0

    # Filter features table
    id_col = "OlinkID" if "OlinkID" in dataset.features.columns else dataset.features.columns[0]
    features = dataset.features[dataset.features[id_col].isin(keep_cols)]

    return replace(
        dataset,
        expression=dataset.expression[keep_cols],
        features=features.reset_index(drop=True),
    ), n_removed


def _remove_uniprot_duplicates(
    dataset: AffinityDataset,
) -> tuple[AffinityDataset, int]:
    """Remove assays with multiple UniProt mappings."""
    from pyprideap.processing.olink.uniprot import detect_uniprot_duplicates

    result = detect_uniprot_duplicates(dataset)
    if result.n_affected_assays == 0:
        return dataset, 0

    dup_olink_ids = set(result.duplicates.keys())
    id_col = "OlinkID" if "OlinkID" in dataset.features.columns else dataset.features.columns[0]

    keep_mask = ~dataset.features[id_col].isin(dup_olink_ids)
    keep_cols = dataset.features[keep_mask][id_col]
    expr_keep = [c for c in dataset.expression.columns if c in set(keep_cols)]

    return replace(
        dataset,
        expression=dataset.expression[expr_keep],
        features=dataset.features[keep_mask].reset_index(drop=True),
    ), result.n_affected_assays


def _prep_for_dimred(
    dataset: AffinityDataset,
    missingness_cutoff: float | None,
) -> tuple[AffinityDataset, dict]:
    """Structured preprocessing for dimensionality reduction.

    Mirrors OlinkAnalyze's ``npxProcessing_forDimRed()``:
    1. Remove zero-variance assays
    2. Drop assays with >10% missingness (>5% if <=88 samples)
    3. Impute remaining missing values by assay median
    """
    numeric = dataset.expression.apply(pd.to_numeric, errors="coerce")
    n_samples = len(numeric)

    # Determine missingness cutoff
    if missingness_cutoff is None:
        missingness_cutoff = 0.05 if n_samples <= 88 else 0.10

    # Step 1: Remove zero-variance assays
    variances = numeric.var()
    zero_var = variances.isna() | (variances == 0)
    n_zero_var = int(zero_var.sum())
    if n_zero_var > 0:
        numeric = numeric.loc[:, ~zero_var]

    # Step 2: Drop high-missingness assays
    miss_rate = numeric.isna().mean(axis=0)
    high_miss = miss_rate > missingness_cutoff
    n_high_miss = int(high_miss.sum())
    if n_high_miss > 0:
        numeric = numeric.loc[:, ~high_miss]

    # Step 3: Impute remaining by assay median
    remaining_miss = numeric.isna().any()
    n_imputed = int(remaining_miss.sum())
    if n_imputed > 0:
        medians = numeric.median()
        numeric = numeric.fillna(medians)

    # Rebuild dataset with filtered columns
    keep_cols = numeric.columns
    id_col = "OlinkID" if "OlinkID" in dataset.features.columns else dataset.features.columns[0]
    features = dataset.features[dataset.features[id_col].isin(keep_cols)]

    result = replace(
        dataset,
        expression=numeric,
        features=features.reset_index(drop=True),
    )

    return result, {
        "zero_variance": n_zero_var,
        "high_missingness": n_high_miss,
        "imputed": n_imputed,
        "cutoff": missingness_cutoff,
    }
