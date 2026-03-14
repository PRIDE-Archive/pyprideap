"""Olink IQR-based outlier detection.

Mirrors the QC outlier detection from OlinkAnalyze's ``olink_qc_plot()``
and ``olink_outlier_detection_utils.R``.

Two complementary methods:

1. **IQR vs Median QC** (``compute_iqr_median_outliers``):
   Per panel, computes IQR and median NPX per sample. A sample is an outlier
   if its IQR or median falls outside ``mean ± n × SD`` (default n=3).
   This is the primary Olink sample-level QC plot.

2. **IQR-based value outlier** (``is_iqr_outlier``):
   For a given grouping, a value is an outlier if it falls outside
   ``median ± IQR × multiplier``. Used in bridgeability assessment
   and cross-product normalization.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pyprideap.core import AffinityDataset


@dataclass
class IqrMedianOutlierResult:
    """Per-sample IQR vs Median outlier detection results.

    One row per sample per panel, matching OlinkAnalyze's ``olink_qc_plot()``.
    """

    sample_ids: list[str]
    panels: list[str]
    iqr_values: list[float]
    median_values: list[float]
    is_outlier: list[bool]
    qc_status: list[str]  # "Pass" or "Warning"
    # Per-panel thresholds
    iqr_low: dict[str, float] = field(default_factory=dict)
    iqr_high: dict[str, float] = field(default_factory=dict)
    median_low: dict[str, float] = field(default_factory=dict)
    median_high: dict[str, float] = field(default_factory=dict)

    @property
    def n_outliers(self) -> int:
        return sum(self.is_outlier)

    @property
    def n_samples(self) -> int:
        return len(set(self.sample_ids))

    @property
    def outlier_sample_ids(self) -> list[str]:
        """Unique sample IDs flagged as outlier in any panel."""
        return sorted({s for s, o in zip(self.sample_ids, self.is_outlier) if o})


def compute_iqr_median_outliers(
    dataset: AffinityDataset,
    *,
    iqr_outlier_def: float = 3.0,
    median_outlier_def: float = 3.0,
    panel_column: str = "Panel",
) -> IqrMedianOutlierResult:
    """Compute IQR vs Median outlier detection per sample per panel.

    Mirrors ``olink_qc_plot()`` from OlinkAnalyze:

    1. For each panel, compute per-sample IQR and median of NPX values
    2. Compute mean and SD of IQR/median across all samples in that panel
    3. Flag sample as outlier if IQR or median falls outside
       ``mean ± n × SD``

    Parameters
    ----------
    dataset:
        Olink AffinityDataset with NPX expression values.
    iqr_outlier_def:
        Number of SDs from mean IQR that defines an outlier (default 3).
    median_outlier_def:
        Number of SDs from mean median that defines an outlier (default 3).
    panel_column:
        Column name in features DataFrame containing panel labels.

    Returns
    -------
    IqrMedianOutlierResult
        Per-sample, per-panel outlier flags and thresholds.
    """
    numeric = dataset.expression.apply(pd.to_numeric, errors="coerce")

    # Build per-protein panel mapping
    protein_to_panel: dict[str, str] = {}
    if panel_column in dataset.features.columns:
        id_col = "OlinkID" if "OlinkID" in dataset.features.columns else dataset.features.columns[0]
        protein_to_panel = dict(
            zip(
                dataset.features[id_col].astype(str),
                dataset.features[panel_column].astype(str),
            )
        )

    # If no panel info, use a single "All" panel
    panels_available = set(protein_to_panel.values()) if protein_to_panel else set()
    if not panels_available:
        panels_available = {"All"}
        protein_to_panel = {str(c): "All" for c in numeric.columns}

    # Get sample IDs
    sid_col = "SampleID"
    if sid_col not in dataset.samples.columns:
        for col in ("SampleId", "SampleName"):
            if col in dataset.samples.columns:
                sid_col = col
                break

    sample_ids_series = (
        dataset.samples[sid_col].astype(str)
        if sid_col in dataset.samples.columns
        else pd.Series([f"S{i}" for i in range(len(dataset.samples))])
    )

    # Get QC status
    qc_col = None
    for col in ("QC_Warning", "SampleQC"):
        if col in dataset.samples.columns:
            qc_col = col
            break

    # Compute per-sample per-panel IQR and median
    all_sample_ids: list[str] = []
    all_panels: list[str] = []
    all_iqr: list[float] = []
    all_median: list[float] = []
    all_outlier: list[bool] = []
    all_qc: list[str] = []
    panel_iqr_low: dict[str, float] = {}
    panel_iqr_high: dict[str, float] = {}
    panel_median_low: dict[str, float] = {}
    panel_median_high: dict[str, float] = {}

    for panel in sorted(panels_available):
        # Get columns for this panel
        panel_cols = [c for c in numeric.columns if protein_to_panel.get(str(c), "All") == panel]
        if not panel_cols:
            continue

        panel_data = numeric[panel_cols]

        # Per-sample IQR and median
        sample_iqrs = panel_data.apply(
            lambda row: (
                float(row.dropna().quantile(0.75) - row.dropna().quantile(0.25)) if row.notna().sum() > 1 else np.nan
            ),
            axis=1,
        )
        sample_medians = panel_data.median(axis=1)

        # Compute panel-level stats
        mean_iqr = float(sample_iqrs.mean())
        sd_iqr = float(sample_iqrs.std())
        mean_median = float(sample_medians.mean())
        sd_median = float(sample_medians.std())

        iqr_lo = mean_iqr - iqr_outlier_def * sd_iqr
        iqr_hi = mean_iqr + iqr_outlier_def * sd_iqr
        med_lo = mean_median - median_outlier_def * sd_median
        med_hi = mean_median + median_outlier_def * sd_median

        panel_iqr_low[panel] = round(iqr_lo, 4)
        panel_iqr_high[panel] = round(iqr_hi, 4)
        panel_median_low[panel] = round(med_lo, 4)
        panel_median_high[panel] = round(med_hi, 4)

        for idx in range(len(panel_data)):
            iqr_val = float(sample_iqrs.iloc[idx])
            med_val = float(sample_medians.iloc[idx])
            sid = str(sample_ids_series.iloc[idx])

            # Outlier: outside either threshold (matching OlinkAnalyze logic)
            is_outlier_flag = not (med_lo < med_val < med_hi and iqr_lo < iqr_val < iqr_hi)

            # QC status consolidation per sample
            qc_status = "Pass"
            if qc_col is not None:
                qc_val = str(dataset.samples.iloc[idx][qc_col]).upper()
                if qc_val in ("WARN", "WARNING", "FAIL"):
                    qc_status = "Warning"

            all_sample_ids.append(sid)
            all_panels.append(panel)
            all_iqr.append(round(iqr_val, 4))
            all_median.append(round(med_val, 4))
            all_outlier.append(is_outlier_flag)
            all_qc.append(qc_status)

    return IqrMedianOutlierResult(
        sample_ids=all_sample_ids,
        panels=all_panels,
        iqr_values=all_iqr,
        median_values=all_median,
        is_outlier=all_outlier,
        qc_status=all_qc,
        iqr_low=panel_iqr_low,
        iqr_high=panel_iqr_high,
        median_low=panel_median_low,
        median_high=panel_median_high,
    )


def is_iqr_outlier(
    values: pd.Series,
    *,
    iqr_multiplier: float = 3.0,
) -> pd.Series:
    """Flag values outside ``median ± IQR × multiplier``.

    Mirrors ``olink_median_iqr_outlier()`` from OlinkAnalyze.

    Parameters
    ----------
    values:
        Numeric values to check.
    iqr_multiplier:
        Multiplier for IQR (default 3.0).

    Returns
    -------
    pd.Series
        Boolean series, True for outlier values.
    """
    med = values.median()
    iqr = values.quantile(0.75) - values.quantile(0.25)
    threshold = iqr * iqr_multiplier
    return (values < (med - threshold)) | (values > (med + threshold))
