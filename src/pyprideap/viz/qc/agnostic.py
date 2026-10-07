"""Technology-agnostic QC metrics for affinity proteomics.

These summaries are computed from the submitted expression matrix, on study
samples only (control samples removed). They do not use LOD, vendor CV tables,
or platform-specific acceptance rules, and they do not re-normalise vendor
outputs.

MPI and relative spread divide by the median, so they are computed on a linear
scale (Olink 2^NPX, SomaScan RFU as deposited), the same convention as the CV
plots: on log2 NPX, which is centred near 0 and often negative, a ratio to the
median depends on the arbitrary NPX reference rather than on precision.
Rank concordance uses ranks, which do not depend on the scale.

When SDRF (or sample metadata) distinguishes biological groups such as cases
and controls, MPI is calculated within each group so disease or treatment
differences are not treated as measurement noise.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import cast

import numpy as np
import pandas as pd

from pyprideap.core import AffinityDataset
from pyprideap.io.readers.sdrf import resolve_biological_groups

logger = logging.getLogger(__name__)

# scipy.stats.median_abs_deviation default scale='normal'
_MAD_TO_SD = 1.482602218505602
_MIN_SAMPLES_MPI = 3
_MIN_SAMPLES_DYNAMIC_RANGE = 4
_MIN_OVERLAP_RANK = 10
_DEFAULT_MAX_PAIRS = 500


@dataclass
class MpiData:
    """Per-protein Measurement Precision Index (robust 1/CV)."""

    values: list[float]
    feature_ids: list[str]
    median: float
    has_case_control: bool
    group_column: str = ""
    group_counts: dict[str, int] = field(default_factory=dict)
    n_proteins: int = 0
    title: str = "Measurement Precision Index"
    xlabel: str = "MPI"
    ylabel: str = "Number of proteins"


@dataclass
class DynamicRangeData:
    """Per-protein relative spread: IQR / median on a linear scale (study samples)."""

    values: list[float]
    feature_ids: list[str]
    median: float
    has_case_control: bool
    group_column: str = ""
    group_counts: dict[str, int] = field(default_factory=dict)
    n_proteins: int = 0
    title: str = "Relative Spread"
    xlabel: str = "Relative IQR (IQR / median, linear scale)"
    ylabel: str = "Number of proteins"


@dataclass
class RankConcordanceData:
    """Pairwise Spearman correlation of protein ranks between samples."""

    values: list[float]
    median: float
    has_case_control: bool
    group_column: str = ""
    group_counts: dict[str, int] = field(default_factory=dict)
    n_pairs: int = 0
    title: str = "Rank Concordance"
    xlabel: str = "Spearman correlation"
    ylabel: str = "Number of sample pairs"


def _numeric_expression(dataset: AffinityDataset) -> pd.DataFrame:
    return cast(pd.DataFrame, dataset.expression.apply(pd.to_numeric, errors="coerce"))


def _study_mask(dataset: AffinityDataset) -> np.ndarray:
    from pyprideap.processing.filtering import control_sample_mask

    return ~control_sample_mask(dataset.samples).to_numpy()


def _linear_study_values(dataset: AffinityDataset) -> pd.DataFrame:
    """Study-sample matrix on a linear scale (Olink 2^NPX, SomaScan RFU as deposited)."""
    from pyprideap.viz.qc.compute import _linear_study_samples

    return _linear_study_samples(dataset).expression


def _study_groups(dataset: AffinityDataset, groups: pd.Series | None) -> pd.Series | None:
    """Subset *groups* (aligned with all samples) to the study samples, re-indexed 0..n-1."""
    if groups is None:
        return None
    mask = _study_mask(dataset)
    if len(groups) == len(mask):
        groups = groups[mask]
    return groups.reset_index(drop=True)


def _clean_group_labels(labels: pd.Series) -> pd.Series:
    cleaned = labels.astype(str).str.strip()
    invalid = cleaned.str.lower().isin({"not available", "not applicable", "nan", "", "na", "none"})
    cleaned = cleaned.mask(invalid | labels.isna())
    return cleaned


def _mpi_from_matrix(numeric: pd.DataFrame) -> pd.Series:
    """Vectorised MPI = 1 / (1.4826 × MAD / |median|) per protein."""
    n = numeric.notna().sum(axis=0)
    median = numeric.median(axis=0)
    mad = (numeric - median).abs().median(axis=0)
    robust_cv = (mad * _MAD_TO_SD) / median.abs()
    mpi = 1.0 / robust_cv.replace(0, np.nan)
    mpi = mpi.replace([np.inf, -np.inf], np.nan)
    mpi = mpi.where((n >= _MIN_SAMPLES_MPI) & median.abs().gt(0) & mad.gt(0))
    return cast(pd.Series, mpi.dropna())


def compute_mpi(
    dataset: AffinityDataset,
    groups: pd.Series | None = None,
) -> MpiData | None:
    """Measurement Precision Index per protein (study samples, linear scale).

    MPI = |median| / (1.4826 x MAD), i.e. the inverse of a robust CV. When
    *groups* (aligned with ``dataset.samples``) has at least two levels with
    >=3 study samples, MPI is the mean of within-group MPI values so biological
    differences are not treated as noise.
    """
    numeric = _linear_study_values(dataset)
    groups = _study_groups(dataset, groups)
    if numeric.shape[0] < _MIN_SAMPLES_MPI or numeric.shape[1] < 1:
        return None

    grouped = False
    group_column = ""
    group_counts: dict[str, int] = {}
    mpi: pd.Series | None = None

    if groups is not None:
        labels = _clean_group_labels(groups)
        labels.index = numeric.index
        usable = labels.dropna()
        counts = usable.value_counts()
        eligible = counts[counts >= _MIN_SAMPLES_MPI]
        if len(eligible) >= 2:
            parts: list[pd.Series] = []
            for name in eligible.index:
                idx = labels.index[labels == name]
                parts.append(_mpi_from_matrix(numeric.loc[idx]))
            if parts:
                mpi = pd.concat(parts, axis=1).mean(axis=1).dropna()
                grouped = True
                group_counts = {str(k): int(v) for k, v in eligible.items()}
                logger.debug("MPI computed within groups %s", group_counts)

    if mpi is None or mpi.empty:
        mpi = _mpi_from_matrix(numeric)
        grouped = False
        group_counts = {}

    if mpi.empty:
        return None

    return MpiData(
        values=[round(float(v), 4) for v in mpi.tolist()],
        feature_ids=[str(i) for i in mpi.index],
        median=round(float(mpi.median()), 4),
        has_case_control=grouped,
        group_column=group_column,
        group_counts=group_counts,
        n_proteins=len(mpi),
    )


def compute_dynamic_range(
    dataset: AffinityDataset,
    *,
    has_case_control: bool = False,
    group_column: str = "",
    group_counts: dict[str, int] | None = None,
) -> DynamicRangeData | None:
    """Relative spread per protein: IQR / median of study samples on a linear scale."""
    numeric = _linear_study_values(dataset)
    if numeric.shape[0] < _MIN_SAMPLES_DYNAMIC_RANGE or numeric.shape[1] < 1:
        return None

    n = numeric.notna().sum(axis=0)
    median = numeric.median(axis=0)
    iqr = numeric.quantile(0.75) - numeric.quantile(0.25)
    values = iqr / median.abs()
    values = values.replace([np.inf, -np.inf], np.nan)
    values = values.where((n >= _MIN_SAMPLES_DYNAMIC_RANGE) & median.abs().gt(1e-6))
    values = values.dropna()
    if values.empty:
        return None

    return DynamicRangeData(
        values=[round(float(v), 4) for v in values.tolist()],
        feature_ids=[str(i) for i in values.index],
        median=round(float(values.median()), 4),
        has_case_control=has_case_control,
        group_column=group_column,
        group_counts=group_counts or {},
        n_proteins=len(values),
    )


def compute_rank_concordance(
    dataset: AffinityDataset,
    *,
    max_pairs: int = _DEFAULT_MAX_PAIRS,
    has_case_control: bool = False,
    group_column: str = "",
    group_counts: dict[str, int] | None = None,
) -> RankConcordanceData | None:
    """Spearman correlation of protein ranks between study-sample pairs."""
    numeric = _numeric_expression(dataset).loc[_study_mask(dataset)].reset_index(drop=True)
    n_samples = numeric.shape[0]
    if n_samples < 2 or numeric.shape[1] < _MIN_OVERLAP_RANK:
        return None

    ranked = numeric.rank(axis=1, method="average")
    n_possible = n_samples * (n_samples - 1) // 2
    rng = np.random.default_rng(42)

    if not ranked.isna().any().any() and n_possible <= max_pairs:
        corr = np.corrcoef(ranked.to_numpy(dtype=float))
        iu = np.triu_indices(n_samples, k=1)
        pair_vals = corr[iu]
        pair_vals = pair_vals[np.isfinite(pair_vals)]
    else:
        pairs = [(i, j) for i in range(n_samples) for j in range(i + 1, n_samples)]
        if len(pairs) > max_pairs:
            chosen = rng.choice(len(pairs), size=max_pairs, replace=False)
            pairs = [pairs[int(k)] for k in chosen]
        pair_list: list[float] = []
        values = ranked.to_numpy(dtype=float)
        for i, j in pairs:
            mask = np.isfinite(values[i]) & np.isfinite(values[j])
            if int(mask.sum()) < _MIN_OVERLAP_RANK:
                continue
            corr = np.corrcoef(values[i, mask], values[j, mask])[0, 1]
            if np.isfinite(corr):
                pair_list.append(float(corr))
        pair_vals = np.asarray(pair_list, dtype=float)

    if pair_vals.size == 0:
        return None

    return RankConcordanceData(
        values=[round(float(v), 4) for v in pair_vals.tolist()],
        median=round(float(np.median(pair_vals)), 4),
        has_case_control=has_case_control,
        group_column=group_column,
        group_counts=group_counts or {},
        n_pairs=int(pair_vals.size),
    )


def compute_agnostic_qc(dataset: AffinityDataset) -> dict[str, object]:
    """Compute MPI, relative spread and rank concordance for a dataset (study samples).

    Biological groups are resolved on study samples only, so control samples
    (whose IDs often contain words such as "control") cannot create groups.
    """
    from pyprideap.processing.filtering import filter_controls

    resolved = resolve_biological_groups(filter_controls(dataset))
    groups: pd.Series | None = None
    group_column = ""
    group_counts: dict[str, int] = {}
    if resolved is not None:
        groups, group_column = resolved

    mpi = compute_mpi(dataset, groups=groups)
    has_cc = bool(mpi is not None and mpi.has_case_control)
    if mpi is not None:
        mpi.group_column = group_column
        group_counts = mpi.group_counts

    results: dict[str, object] = {}
    if mpi is not None:
        results["mpi"] = mpi
    dr = compute_dynamic_range(
        dataset,
        has_case_control=has_cc,
        group_column=group_column,
        group_counts=group_counts,
    )
    if dr is not None:
        results["dynamic_range"] = dr
    rc = compute_rank_concordance(
        dataset,
        has_case_control=has_cc,
        group_column=group_column,
        group_counts=group_counts,
    )
    if rc is not None:
        results["rank_concordance"] = rc

    logger.debug(
        "Agnostic QC: mpi=%s dynamic_range=%s rank_concordance=%s case/control=%s",
        mpi is not None,
        dr is not None,
        rc is not None,
        has_cc,
    )
    return results
