"""Unified SomaScan preprocessing pipeline.

Equivalent to ``preProcessAdat()`` in the SomaDataIO R package.

Steps (each independently togglable):
1. Filter features → remove non-human, non-protein control analytes
2. Filter control samples → keep only ``SampleType == "Sample"``
3. Filter by RowCheck → remove samples failing normalization QC
4. Filter outliers → MAD-based outlier detection and removal
5. Log10 transform → ``log10(RFU)``
6. Center and scale → per-analyte Z-score
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from pyprideap.core import AffinityDataset

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingReport:
    """Summary of actions taken during preprocessing."""

    steps: list[str] = field(default_factory=list)
    n_features_removed: int = 0
    n_controls_removed: int = 0
    n_buffer_removed: int = 0
    n_calibrator_removed: int = 0
    n_qc_removed: int = 0
    n_rowcheck_removed: int = 0
    n_outliers_removed: int = 0
    n_colcheck_flagged: int = 0
    final_samples: int = 0
    final_features: int = 0

    def summary(self) -> str:
        lines = ["SomaScan Preprocessing Report", "=" * 40]
        for step in self.steps:
            lines.append(f"  - {step}")
        lines.append(f"Final: {self.final_samples} samples × {self.final_features} features")
        return "\n".join(lines)


def preprocess_somascan(
    dataset: AffinityDataset,
    *,
    filter_features: bool = True,
    filter_controls: bool = True,
    filter_rowcheck: bool = True,
    filter_outliers: bool = False,
    log10: bool = False,
    center_scale: bool = False,
    fc_crit: float = 5.0,
    outlier_flags: float = 0.05,
) -> tuple[AffinityDataset, PreprocessingReport]:
    """Preprocess a SomaScan dataset for analysis.

    Mirrors ``preProcessAdat()`` from the SomaDataIO R package.

    Parameters
    ----------
    dataset:
        Raw SomaScan AffinityDataset with RFU values.
    filter_features:
        Remove non-human, non-protein control analytes
        (HybControlElution, Spuriomers, NonBiotin, NonHuman,
        NonCleavable). Also removes analytes where
        ``Type != "Protein"`` or ``Organism != "Human"`` if those
        columns are present in the features table.
    filter_controls:
        Remove Buffer, Calibrator, and QC samples — keep only
        ``SampleType == "Sample"``.
    filter_rowcheck:
        Remove samples that fail normalization acceptance criteria
        (RowCheck == FLAG, i.e. norm scale outside [0.4, 2.5]).
    filter_outliers:
        Remove samples identified as MAD-based outliers (≥5% of
        analytes exceed 6×MAD and 5× fold-change). Recommended
        for plasma/serum; not appropriate for tissue/cell lysate.
    log10:
        Apply log10 transformation to RFU values.
    center_scale:
        Z-score normalize per analyte (center then scale).
        Recommended to use with ``log10=True``.
    fc_crit:
        Fold-change criterion for outlier detection (default 5).
    outlier_flags:
        Fraction of analyte outliers to flag a sample (default 0.05).

    Returns
    -------
    tuple[AffinityDataset, PreprocessingReport]
        The processed dataset and a report summarizing what was done.
    """
    from pyprideap.processing.somascan.controls import remove_control_analytes
    from pyprideap.processing.somascan.outliers import (
        calc_outlier_map,
        get_outlier_ids,
    )
    from pyprideap.processing.somascan.qc_flags import (
        add_row_check,
        filter_by_row_check,
    )

    report = PreprocessingReport()
    ds = dataset

    # 1. Filter features (control analytes + non-human/non-protein)
    if filter_features:
        n_before = len(ds.expression.columns)

        # Remove control analyte SeqIds
        ds = remove_control_analytes(ds)

        # Also filter by Type/Organism if available
        if "Type" in ds.features.columns and "Organism" in ds.features.columns:
            keep_mask = (ds.features["Type"].str.strip() == "Protein") & (
                ds.features["Organism"].str.strip() == "Human"
            )
            # Also exclude "Internal Use Only" targets
            if "TargetFullName" in ds.features.columns:
                keep_mask = keep_mask & ~ds.features["TargetFullName"].str.startswith("Internal Use Only", na=False)

            if not keep_mask.all():
                keep_indices = keep_mask[keep_mask].index.tolist()
                keep_cols = [ds.expression.columns[i] for i in keep_indices if i < len(ds.expression.columns)]
                ds = AffinityDataset(
                    platform=ds.platform,
                    samples=ds.samples,
                    features=ds.features[keep_mask].reset_index(drop=True),
                    expression=ds.expression[keep_cols].reset_index(drop=True),
                    metadata=dict(ds.metadata),
                )

        n_removed = n_before - len(ds.expression.columns)
        report.n_features_removed = n_removed

        # Report ColCheck flags
        if "ColCheck" in ds.features.columns:
            n_flagged = int((ds.features["ColCheck"] == "FLAG").sum())
            report.n_colcheck_flagged = n_flagged
            if n_flagged > 0:
                report.steps.append(f"{n_flagged} human proteins flagged in ColCheck (did not pass QC ratio 0.8–1.2)")

        if n_removed > 0:
            report.steps.append(f"{n_removed} non-human/control features removed")

    # 2. Filter control samples
    if filter_controls and "SampleType" in ds.samples.columns:
        st = ds.samples["SampleType"].astype(str).str.strip()
        n_buffer = int((st == "Buffer").sum())
        n_calibrator = int((st == "Calibrator").sum())
        n_qc = int((st == "QC").sum())

        keep_mask = st == "Sample"
        if not keep_mask.all():
            ds = AffinityDataset(
                platform=ds.platform,
                samples=ds.samples[keep_mask].reset_index(drop=True),
                features=ds.features,
                expression=ds.expression[keep_mask].reset_index(drop=True),
                metadata=dict(ds.metadata),
            )

        report.n_buffer_removed = n_buffer
        report.n_calibrator_removed = n_calibrator
        report.n_qc_removed = n_qc
        report.n_controls_removed = n_buffer + n_calibrator + n_qc
        parts = []
        if n_buffer > 0:
            parts.append(f"{n_buffer} buffer")
        if n_calibrator > 0:
            parts.append(f"{n_calibrator} calibrator")
        if n_qc > 0:
            parts.append(f"{n_qc} QC")
        if parts:
            report.steps.append(f"Removed {', '.join(parts)} samples")

    # 3. Filter by RowCheck
    if filter_rowcheck:
        ds = add_row_check(ds)
        n_before = len(ds.samples)
        ds = filter_by_row_check(ds)
        n_removed = n_before - len(ds.samples)
        report.n_rowcheck_removed = n_removed
        if n_removed > 0:
            report.steps.append(f"{n_removed} samples removed (RowCheck FLAG: norm scale outside [0.4, 2.5])")

    # 4. Filter outliers
    if filter_outliers:
        n_before = len(ds.samples)
        outlier_map = calc_outlier_map(ds, fc_crit=fc_crit)
        flagged_ids = get_outlier_ids(outlier_map, flags=outlier_flags)

        if flagged_ids:
            keep_mask = pd.Series(~ds.samples.index.isin(flagged_ids), index=ds.samples.index)
            ds = AffinityDataset(
                platform=ds.platform,
                samples=ds.samples[keep_mask].reset_index(drop=True),
                features=ds.features,
                expression=ds.expression[keep_mask].reset_index(drop=True),
                metadata=dict(ds.metadata),
            )

        n_removed = n_before - len(ds.samples)
        report.n_outliers_removed = n_removed
        if n_removed > 0:
            report.steps.append(
                f"{n_removed} outlier samples removed (≥{outlier_flags:.0%} analytes exceeded 6×MAD & {fc_crit}× FC)"
            )

    # 5. Log10 transform
    if log10:
        numeric = ds.expression.apply(pd.to_numeric, errors="coerce")
        # Clip to avoid log(0)
        numeric = numeric.clip(lower=1e-10)
        log_expr = np.log10(numeric)
        ds = replace(ds, expression=log_expr)
        report.steps.append("Applied log10 transformation")

    # 6. Center and scale (Z-score per analyte)
    if center_scale:
        numeric = ds.expression.apply(pd.to_numeric, errors="coerce")
        means = numeric.mean()
        centered = numeric - means
        stds = centered.std().replace(0, 1)
        z_scored = centered / stds
        ds = replace(ds, expression=z_scored)
        report.steps.append("Applied center-scale (Z-score) transformation")

    report.final_samples = len(ds.samples)
    report.final_features = len(ds.expression.columns)

    return ds, report
