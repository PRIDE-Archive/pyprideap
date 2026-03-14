"""MAD-based outlier detection for SomaScan data.

Equivalent to ``calcOutlierMap()`` and ``getOutlierIds()`` from the
SomaDataIO R package.

Outlier criteria (applied per analyte):
    A sample is an outlier for a given analyte when **both**:
    1. ``|x - median(x)| > 6 × MAD(x)``   (statistical criterion)
    2. ``fold_change(x, median) > fc_crit`` (fold-change criterion, default 5×)

A sample is flagged overall when ≥ ``flags`` fraction of analytes are
outliers (default 5%).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pyprideap.core import AffinityDataset


@dataclass
class OutlierMap:
    """Result of MAD-based outlier detection.

    Attributes
    ----------
    matrix:
        Boolean DataFrame (samples × analytes) where True = outlier.
    fc_crit:
        Fold-change criterion used.
    title:
        Human-readable description of the criteria.
    sample_order:
        Row indices in the order used for the map.
    """

    matrix: pd.DataFrame
    fc_crit: float = 5.0
    title: str = ""
    sample_order: list[int] = field(default_factory=list)

    @property
    def n_outliers_per_sample(self) -> pd.Series:
        """Number of outlier analytes per sample."""
        return self.matrix.sum(axis=1)

    @property
    def n_outliers_per_analyte(self) -> pd.Series:
        """Number of outlier samples per analyte."""
        return self.matrix.sum(axis=0)

    @property
    def outlier_fraction_per_sample(self) -> pd.Series:
        """Fraction of analytes that are outliers per sample."""
        n_analytes = self.matrix.shape[1]
        if n_analytes == 0:
            return pd.Series(dtype=float)
        return self.matrix.sum(axis=1) / n_analytes


def _get_outliers_per_analyte(
    values: np.ndarray,
    fc_crit: float = 5.0,
) -> np.ndarray:
    """Identify outlier indices for a single analyte vector.

    Mirrors ``.getOutliers()`` from SomaDataIO/R/utils.R:
        stat_bool = |x - median(x)| > 6 * MAD(x, constant=1)
        fold_bool = (x/median > fc_crit) | (median/x > fc_crit)
        outlier = stat_bool & fold_bool

    Parameters
    ----------
    values:
        1-D array of RFU values for one analyte across samples.
    fc_crit:
        Fold-change criterion (default 5).

    Returns
    -------
    Boolean array of same length, True = outlier.
    """
    med = np.nanmedian(values)
    if med == 0 or np.isnan(med):
        return np.zeros(len(values), dtype=bool)

    # MAD with constant=1 (no normality scaling)
    mad = np.nanmedian(np.abs(values - med))
    if mad == 0:
        # All identical values — no outliers by MAD
        stat_bool = np.zeros(len(values), dtype=bool)
    else:
        stat_bool = np.abs(values - med) > 6.0 * mad

    # Fold-change criterion
    with np.errstate(divide="ignore", invalid="ignore"):
        fc_up = values / med > fc_crit
        fc_down = med / values > fc_crit
    fold_bool = fc_up | fc_down

    result: np.ndarray = stat_bool & fold_bool
    return result


def calc_outlier_map(
    dataset: AffinityDataset,
    *,
    fc_crit: float = 5.0,
    order_by: str | None = None,
) -> OutlierMap:
    """Compute a MAD-based outlier map for SomaScan data.

    Equivalent to ``calcOutlierMap()`` in SomaDataIO.

    Parameters
    ----------
    dataset:
        SomaScan AffinityDataset (expression values should be RFU).
    fc_crit:
        Fold-change criterion. A sample must exceed this fold-change
        from the analyte median AND the 6×MAD criterion to be flagged.
    order_by:
        How to order columns in the map:
        - ``None``: as-is in the dataset
        - ``"signal"``: by median signal (ascending)
        - ``"frequency"``: rows ordered by outlier frequency

    Returns
    -------
    OutlierMap
        Boolean matrix with outlier flags and metadata.
    """
    numeric = dataset.expression.apply(pd.to_numeric, errors="coerce")
    mat = np.zeros(numeric.shape, dtype=bool)

    for j in range(numeric.shape[1]):
        col_values = numeric.iloc[:, j].values.astype(float)
        mat[:, j] = _get_outliers_per_analyte(col_values, fc_crit)

    outlier_df = pd.DataFrame(
        mat,
        index=numeric.index,
        columns=numeric.columns,
    )

    # Column ordering
    if order_by == "signal":
        medians = numeric.median().sort_values()
        outlier_df = outlier_df[medians.index]

    sample_order = list(range(len(outlier_df)))

    # Row ordering by frequency
    if order_by == "frequency":
        freq = outlier_df.sum(axis=1).sort_values()
        outlier_df = outlier_df.loc[freq.index]
        sample_order = freq.index.tolist()

    title = f"Outlier Map: |x - median(x)| > 6 × MAD(x) & FC > {fc_crit}x"

    return OutlierMap(
        matrix=outlier_df,
        fc_crit=fc_crit,
        title=title,
        sample_order=sample_order,
    )


def get_outlier_ids(
    outlier_map: OutlierMap,
    *,
    flags: float = 0.05,
) -> list[int]:
    """Return indices of samples flagged as outliers.

    A sample is flagged when ≥ ``flags`` fraction of analytes are
    outliers (default 5%, matching SomaDataIO).

    Parameters
    ----------
    outlier_map:
        Result from :func:`calc_outlier_map`.
    flags:
        Fraction threshold (0–1). Samples with at least this fraction
        of analyte outliers are flagged.

    Returns
    -------
    list[int]
        Row indices of flagged samples.
    """
    if not 0.0 <= flags <= 1.0:
        raise ValueError(f"flags must be between 0 and 1, got {flags}")

    n_analytes = outlier_map.matrix.shape[1]
    if n_analytes == 0:
        return []

    threshold = n_analytes * flags
    row_sums = outlier_map.matrix.sum(axis=1)
    flagged: list[int] = row_sums[row_sums >= threshold].index.tolist()
    return flagged
