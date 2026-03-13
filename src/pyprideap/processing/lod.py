"""LOD (Limit of Detection) computation and analysis.

Supports both Olink and SomaScan platforms with platform-appropriate
methods.

**Olink LOD sources** (following OlinkAnalyze R package):

* **NCLOD** — computed from the project's own negative control samples:
  ``LOD = median(NC_NPX) + max(0.2, 3 * SD(NC_NPX))``
  When PlateID is available, a per-plate intensity-normalization
  adjustment is applied so the LOD becomes plate-specific.
  OlinkAnalyze recommends this method when ≥10 negative controls are
  available.

* **FixedLOD** — pre-computed LOD from an **external CSV** provided by
  Olink (downloaded from the Document Download Center at olink.com).
  These values are specific to the *Data Analysis Reference ID* and
  reagent lot, not project-specific.  The CSV contains columns
  ``OlinkID``, ``DataAnalysisRefID``, ``LODNPX``, ``LODCount``,
  ``LODMethod``.  OlinkAnalyze recommends this when <10 negative
  controls are available.

* **Reported LOD** — LOD values already present in the NPX data file
  (the ``LOD`` column).  These are read by the readers and stored as a
  sample × assay ``lod_matrix`` in ``dataset.metadata``.  This is
  *not* the same as FixedLOD.

**SomaScan LOD** (following SomaDataIO R package):

* **eLOD** — estimated LOD from buffer samples using a MAD-based
  formula robust to outliers:
  ``eLOD = median(buffer_RFU) + 3.3 * 1.4826 * MAD(buffer_RFU)``
  The 1.4826 factor converts MAD to SD-equivalent for normal
  distributions; the 3.3 multiplier targets ~95% detection probability.
  Best suited for non-core matrices (cell lysate, CSF); use carefully
  for plasma/serum where background signal behaviour differs.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd

from pyprideap.core import AffinityDataset, Platform
from pyprideap.processing.filtering import _CONTROL_SAMPLE_TYPES

_MIN_CONTROLS_FOR_LOD = 10
_MIN_STD_FLOOR = 0.2
_SOMA_ELOD_K = 3.3  # multiplier for ~95% detection probability
_MAD_TO_SD = 1.4826  # MAD → SD conversion factor for normal distributions
_NEGATIVE_CONTROL_TYPES = frozenset(
    {
        "negative",
        "negative control",
        "negative_control",
        "neg",
    }
)


class LodMethod(Enum):
    """LOD source/method.

    * ``NCLOD`` — computed from negative controls in the project data
      (Olink: median + max(0.2, 3×SD)).
    * ``FIXED`` — from an external Olink FixedLOD CSV (reagent-lot specific).
    * ``REPORTED`` — LOD values already present in the NPX data file.
    * ``SOMA_ELOD`` — SomaScan estimated LOD from buffer samples
      (median + 3.3 × 1.4826 × MAD).
    """

    NCLOD = "NCLOD"
    FIXED = "FIXED"
    REPORTED = "REPORTED"
    SOMA_ELOD = "SOMA_ELOD"


@dataclass
class LodStats:
    """LOD analysis results."""

    lod_source: str
    n_assays_with_lod: int
    n_assays_total: int
    above_lod_rate: float
    above_lod_per_sample: dict[str, float]
    above_lod_per_panel: dict[str, float]

    def summary(self) -> str:
        lines = [
            f"LOD source: {self.lod_source}",
            f"Assays with LOD: {self.n_assays_with_lod}/{self.n_assays_total}",
            f"Overall above-LOD rate: {self.above_lod_rate:.1%}",
        ]
        if self.above_lod_per_panel:
            lines.append("Per-panel above-LOD rates:")
            for panel, rate in sorted(self.above_lod_per_panel.items()):
                lines.append(f"  {panel}: {rate:.1%}")
        return "\n".join(lines)


def _find_negative_controls(dataset: AffinityDataset) -> pd.Series:
    """Return a boolean mask over ``dataset.samples`` for negative controls.

    Raises ``ValueError`` when no suitable controls are found.
    """
    if "SampleType" not in dataset.samples.columns:
        raise ValueError("SampleType column required to identify negative controls")

    st = dataset.samples["SampleType"].astype(str).str.lower().str.strip()

    is_negative = st.isin(_NEGATIVE_CONTROL_TYPES)
    if is_negative.any():
        return is_negative

    # Fallback: use any control-like sample
    is_control = st.isin(_CONTROL_SAMPLE_TYPES)
    if is_control.any():
        return is_control

    raise ValueError("No negative control samples found in SampleType column")


def _nclod_base(dataset: AffinityDataset) -> pd.Series:
    """Compute a single (pooled) NCLOD per assay from negative controls.

    Formula (OlinkAnalyze NCLOD, high-count path):
    ``LOD = median(NC_NPX) + max(0.2, 3 * SD(NC_NPX))``

    Returns a Series indexed by expression column name.
    """
    control_mask = _find_negative_controls(dataset)
    if control_mask.sum() < _MIN_CONTROLS_FOR_LOD:
        raise ValueError(
            f"Need at least {_MIN_CONTROLS_FOR_LOD} negative control samples "
            f"for LOD calculation, found {control_mask.sum()}"
        )

    numeric_expr = dataset.expression[control_mask].apply(pd.to_numeric, errors="coerce")

    medians = numeric_expr.median()
    stds = numeric_expr.std()
    result: pd.Series = medians + np.maximum(_MIN_STD_FLOOR, 3 * stds)
    return result


# ---------------------------------------------------------------------------
# SomaScan eLOD
# ---------------------------------------------------------------------------

_BUFFER_SAMPLE_TYPES = frozenset(
    {
        "buffer",
        "buf",
    }
)


def _find_buffer_samples(dataset: AffinityDataset) -> pd.Series:
    """Return a boolean mask over ``dataset.samples`` for buffer samples.

    Raises ``ValueError`` when no buffer samples are found.
    """
    if "SampleType" not in dataset.samples.columns:
        raise ValueError("SampleType column required to identify buffer samples")

    st = dataset.samples["SampleType"].astype(str).str.lower().str.strip()
    is_buffer = st.isin(_BUFFER_SAMPLE_TYPES)
    if not is_buffer.any():
        raise ValueError(
            "No buffer samples found in SampleType column. "
            "SomaScan eLOD requires buffer samples (SampleType = 'Buffer')."
        )
    return is_buffer


def compute_soma_elod(dataset: AffinityDataset) -> pd.Series:
    """Compute SomaScan estimated LOD (eLOD) from buffer samples.

    Uses the SomaDataIO formula (``calc_eLOD``):

    ``eLOD = median(buffer_RFU) + 3.3 × 1.4826 × MAD(buffer_RFU)``

    where MAD is the median absolute deviation.  The 1.4826 factor
    converts MAD to a SD-equivalent under normality; the 3.3 multiplier
    targets ~95% detection probability.

    Returns a Series indexed by expression column name (one eLOD per assay).

    Raises
    ------
    ValueError
        If no buffer samples are found in the dataset.
    """
    buffer_mask = _find_buffer_samples(dataset)
    numeric_expr = dataset.expression[buffer_mask].apply(pd.to_numeric, errors="coerce")

    medians = numeric_expr.median()
    # scipy.stats.median_abs_deviation uses scale=1.4826 by default,
    # but we compute manually to avoid the scipy dependency.
    mads = (numeric_expr.subtract(medians, axis=1)).abs().median()

    result: pd.Series = medians + _SOMA_ELOD_K * _MAD_TO_SD * mads
    return result


def _intensity_adjustment(
    dataset: AffinityDataset,
) -> pd.Series | None:
    """Per-plate intensity-normalisation adjustment factors.

    For intensity-normalised Olink data the LOD must be adjusted per
    plate.  The adjustment factor for each plate is the median NPX of
    all *non-control* samples on that plate (OlinkAnalyze convention).

    Returns a Series indexed like ``dataset.samples`` with one
    adjustment value per sample, or *None* when PlateID is unavailable.
    """
    if "PlateID" not in dataset.samples.columns:
        return None

    st = dataset.samples.get("SampleType")
    if st is not None:
        is_ext_ctrl = st.astype(str).str.lower().str.strip().isin(_CONTROL_SAMPLE_TYPES)
    else:
        is_ext_ctrl = pd.Series(False, index=dataset.samples.index)

    numeric = dataset.expression.apply(pd.to_numeric, errors="coerce")
    plates = dataset.samples["PlateID"]

    # Global median across all non-control samples (reference)
    global_median = numeric[~is_ext_ctrl].median(axis=1).median()

    # Per-plate median of non-control sample medians
    adjustments = pd.Series(0.0, index=dataset.samples.index, dtype=float)
    for plate_id, grp_idx in plates.groupby(plates).groups.items():
        plate_mask = dataset.samples.index.isin(grp_idx) & ~is_ext_ctrl
        if plate_mask.sum() == 0:
            continue
        plate_median = numeric[plate_mask].median(axis=1).median()
        # Adjustment = plate_median - global_median
        adjustments.loc[grp_idx] = plate_median - global_median

    return adjustments


def compute_nclod(
    dataset: AffinityDataset,
    *,
    plate_adjusted: bool = True,
) -> pd.DataFrame | pd.Series:
    """Compute NCLOD from negative controls (OlinkAnalyze method).

    When *plate_adjusted* is True and PlateID is available, the base
    NCLOD is adjusted per plate using the intensity-normalisation
    factor, returning a sample × assay DataFrame.  Otherwise a single
    Series (one LOD per assay) is returned.

    Formula:
        base_LOD  = median(NC_NPX) + max(0.2, 3 * SD(NC_NPX))
        plate_LOD = base_LOD + plate_adjustment
    where plate_adjustment = plate_median - global_median.
    A plate with higher signal has a higher noise floor, so the LOD increases.
    """
    base = _nclod_base(dataset)

    if not plate_adjusted:
        return base

    adj = _intensity_adjustment(dataset)
    if adj is None:
        return base

    # Broadcast: for each sample row, LOD = base + adjustment[row]
    lod_matrix = pd.DataFrame(
        np.tile(np.asarray(base), (len(dataset.samples), 1)),
        columns=base.index,
        index=dataset.expression.index,
    )
    adjusted: pd.DataFrame = lod_matrix.add(np.asarray(adj), axis=0)
    return adjusted


# Keep the old name as an alias for backwards compatibility
def compute_lod_from_controls(dataset: AffinityDataset) -> pd.DataFrame | pd.Series:
    """Calculate LOD per assay from negative control samples.

    This is an alias for :func:`compute_nclod` with plate adjustment
    enabled.  See that function for full documentation.
    """
    return compute_nclod(dataset, plate_adjusted=True)


def get_reported_lod(dataset: AffinityDataset) -> pd.DataFrame | pd.Series | None:
    """Extract the LOD values reported in the NPX data file.

    These are the LOD values present in the ``LOD`` column of the
    original NPX CSV/XLSX file.  They are **not** the same as
    OlinkAnalyze FixedLOD (which comes from an external reference CSV).

    Returns a DataFrame (sample × assay) when plate-specific LOD is
    available in ``metadata["lod_matrix"]``, a Series indexed by
    expression column when only a single LOD per assay is stored in the
    features table, or None if no LOD is available.
    """
    if "lod_matrix" in dataset.metadata:
        lod_df = dataset.metadata["lod_matrix"]
        if isinstance(lod_df, pd.DataFrame) and not lod_df.empty:
            return lod_df

    # Legacy fallback: single LOD per assay from features table
    if "LOD" not in dataset.features.columns:
        return None

    lod_series = pd.to_numeric(dataset.features["LOD"], errors="coerce")
    if lod_series.isna().all():
        return None

    if "OlinkID" in dataset.features.columns:
        lod_map = dict(zip(dataset.features["OlinkID"], lod_series))
        return pd.Series({col: lod_map.get(col, np.nan) for col in dataset.expression.columns})

    return pd.Series(lod_series.values, index=dataset.expression.columns[: len(lod_series)])


_CONFIGS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "configs"

# Mapping from Platform to bundled FixedLOD CSV filename.
_BUNDLED_FIXED_LOD: dict[str, str] = {
    "olink_explore": "Explore-3072-Fixed-LOD-2024-12-19.csv",
    "olink_explore_ht": "Explore-HT-Fixed-LOD.csv",
    "olink_reveal": "Reveal-Fixed-LOD.csv",
}


def get_bundled_fixed_lod_path(platform: Platform | str) -> Path | None:
    """Return the path to the bundled FixedLOD CSV for a given platform.

    Returns None if no bundled file exists for the platform.
    """
    key = platform.value if isinstance(platform, Platform) else str(platform).lower()
    filename = _BUNDLED_FIXED_LOD.get(key)
    if filename is None:
        return None
    path = _CONFIGS_DIR / filename
    return path if path.exists() else None


def load_fixed_lod(
    dataset: AffinityDataset,
    lod_file_path: str | Path | None = None,
) -> pd.Series:
    """Load FixedLOD from an external Olink reference CSV.

    The CSV is downloaded from the Olink Document Download Center and
    contains reagent-lot-specific LOD values.  Expected columns
    (semicolon-delimited): ``OlinkID``, ``DataAnalysisRefID``,
    ``LODNPX``, ``LODCount``, ``LODMethod``.

    The function joins the CSV to the dataset on ``OlinkID`` (and
    ``DataAnalysisRefID`` when available in the dataset features).

    When *lod_file_path* is None, the function looks for a bundled
    FixedLOD CSV matching the dataset platform under ``configs/``.

    Args:
        dataset: The AffinityDataset to apply LOD to.
        lod_file_path: Path to the Olink FixedLOD CSV file.  If None,
            uses the bundled file for the dataset's platform.

    Returns:
        A Series indexed by expression column name with one LOD per assay.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If required columns are missing or no bundled file
            is available for the platform.
    """
    if lod_file_path is None:
        bundled = get_bundled_fixed_lod_path(dataset.platform)
        if bundled is None:
            raise ValueError(
                f"No bundled FixedLOD file for platform {dataset.platform.value}. Provide lod_file_path explicitly."
            )
        path = bundled
    else:
        path = Path(lod_file_path)

    if not path.exists():
        raise FileNotFoundError(f"FixedLOD file not found: {path}")

    lod_df = pd.read_csv(path, sep=None, engine="python")

    required = {"OlinkID", "LODNPX"}
    missing = required - set(lod_df.columns)
    if missing:
        raise ValueError(f"FixedLOD CSV missing required columns: {sorted(missing)}")

    # Join on OlinkID + DataAnalysisRefID when available
    join_cols = ["OlinkID"]
    if "DataAnalysisRefID" in lod_df.columns and "DataAnalysisRefID" in dataset.features.columns:
        join_cols.append("DataAnalysisRefID")

    if len(join_cols) > 1:
        feat_keys = dataset.features[join_cols].copy()
        merged = feat_keys.merge(
            lod_df[join_cols + ["LODNPX"]],
            on=join_cols,
            how="left",
        )
        lod_map = dict(zip(dataset.features["OlinkID"], merged["LODNPX"]))
    else:
        # Without DataAnalysisRefID, use OlinkID only (first match wins)
        lod_dedup = lod_df.drop_duplicates(subset=["OlinkID"], keep="first")
        lod_map = dict(zip(lod_dedup["OlinkID"], lod_dedup["LODNPX"]))

    lod_series = pd.Series({col: lod_map.get(col, np.nan) for col in dataset.expression.columns})
    return lod_series


# Backwards-compat alias
get_fixed_lod = load_fixed_lod


def get_lod_values(
    dataset: AffinityDataset,
    method: LodMethod | str = LodMethod.REPORTED,
    *,
    lod_file_path: str | Path | None = None,
) -> pd.DataFrame | pd.Series | None:
    """Return LOD values for *dataset* using the specified method.

    Args:
        dataset: The AffinityDataset to query.
        method: Which LOD source to use:
            - ``LodMethod.REPORTED`` (default) — LOD from the NPX data file.
            - ``LodMethod.FIXED`` — FixedLOD from an external Olink CSV
              (requires *lod_file_path*).
            - ``LodMethod.NCLOD`` — computed from negative controls.
            - ``LodMethod.SOMA_ELOD`` — SomaScan eLOD from buffer samples.
            A string ``"REPORTED"`` / ``"FIXED"`` / ``"NCLOD"`` /
            ``"SOMA_ELOD"`` is also accepted.
        lod_file_path: Path to the Olink FixedLOD CSV file.  Required
            when *method* is ``FIXED``.

    Returns:
        A DataFrame (sample × assay) or Series (one per assay), or None
        if the requested method cannot produce LOD values.
    """
    if isinstance(method, str):
        method = LodMethod(method.upper())

    if method is LodMethod.NCLOD:
        try:
            return compute_nclod(dataset, plate_adjusted=True)
        except ValueError:
            return None

    if method is LodMethod.FIXED:
        return load_fixed_lod(dataset, lod_file_path)

    if method is LodMethod.SOMA_ELOD:
        try:
            return compute_soma_elod(dataset)
        except ValueError:
            return None

    # LodMethod.REPORTED
    return get_reported_lod(dataset)


def _above_lod_matrix(
    expr_numeric: pd.DataFrame,
    lod: pd.DataFrame | pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (above_lod_bool, has_lod_bool) aligned to *expr_numeric*.

    *lod* may be a DataFrame (sample × assay, plate-specific) or a
    Series (one value per assay).
    """
    if isinstance(lod, pd.DataFrame):
        lod_aligned = lod.reindex(columns=expr_numeric.columns)
        above = expr_numeric.gt(lod_aligned)
        has_lod = lod_aligned.notna()
    else:
        lod_aligned = lod.reindex(expr_numeric.columns)
        above = expr_numeric.gt(lod_aligned, axis=1)
        has_lod = pd.DataFrame(
            np.tile(np.asarray(lod_aligned.notna()), (len(expr_numeric), 1)),
            columns=expr_numeric.columns,
            index=expr_numeric.index,
        )
    return above, has_lod


def compute_lod_stats(
    dataset: AffinityDataset,
    lod: pd.DataFrame | pd.Series | None = None,
) -> LodStats:
    """Compute LOD-related statistics for the dataset.

    Args:
        dataset: The AffinityDataset to analyze.
        lod: Pre-computed LOD values — a DataFrame (sample × assay) for
             plate-specific LOD, or a Series (one per assay). If None,
             attempts to extract from the dataset, then falls back to
             computing from controls.

    Returns:
        LodStats with detection rates overall, per-sample, and per-panel.
    """
    source = "provided"

    if lod is None:
        lod = get_reported_lod(dataset)
        if lod is not None:
            source = "reported"

    if lod is None:
        try:
            lod = compute_nclod(dataset, plate_adjusted=True)
            source = "nclod"
        except ValueError:
            return LodStats(
                lod_source="not_available",
                n_assays_with_lod=0,
                n_assays_total=len(dataset.expression.columns),
                above_lod_rate=0.0,
                above_lod_per_sample={},
                above_lod_per_panel={},
            )

    n_assays_total = len(dataset.expression.columns)

    # Compute above-LOD matrix
    expr_numeric = dataset.expression.apply(pd.to_numeric, errors="coerce")
    above_lod, has_lod = _above_lod_matrix(expr_numeric, lod)

    # Count assays that have any LOD value
    n_assays_with_lod = int(has_lod.any(axis=0).sum())

    above_lod_masked = above_lod & has_lod
    expr_valid = expr_numeric.notna() & has_lod

    total_valid = int(expr_valid.sum().sum())
    total_above = int(above_lod_masked.sum().sum())
    above_lod_rate = total_above / total_valid if total_valid > 0 else 0.0

    # Per-sample rates
    per_sample_above = above_lod_masked.sum(axis=1)
    per_sample_total = expr_valid.sum(axis=1)
    per_sample_rate = (per_sample_above / per_sample_total).fillna(0.0)

    sample_ids = dataset.samples.get("SampleID", pd.Series(range(len(dataset.samples))))
    above_lod_per_sample = dict(zip(sample_ids.astype(str), per_sample_rate.round(4)))

    # Per-panel rates
    above_lod_per_panel: dict[str, float] = {}
    if "Panel" in dataset.features.columns and "OlinkID" in dataset.features.columns:
        panel_map = dict(zip(dataset.features["OlinkID"], dataset.features["Panel"]))
        has_lod_per_col = has_lod.any(axis=0)
        for panel_name in dataset.features["Panel"].dropna().unique():
            panel_cols = [
                c
                for c in dataset.expression.columns
                if panel_map.get(c) == panel_name and has_lod_per_col.get(c, False)
            ]
            if panel_cols:
                panel_above = int(above_lod_masked[panel_cols].sum().sum())
                panel_total = int(expr_valid[panel_cols].sum().sum())
                above_lod_per_panel[str(panel_name)] = panel_above / panel_total if panel_total > 0 else 0.0

    return LodStats(
        lod_source=source,
        n_assays_with_lod=n_assays_with_lod,
        n_assays_total=n_assays_total,
        above_lod_rate=above_lod_rate,
        above_lod_per_sample=above_lod_per_sample,
        above_lod_per_panel=above_lod_per_panel,
    )


def get_valid_proteins(
    dataset: AffinityDataset,
    lod: pd.DataFrame | pd.Series | None = None,
) -> list[str]:
    """Return protein/assay IDs that pass QC and are above LOD.

    A protein is valid if at least one non-control sample has:
    - SampleQC in (PASS, WARN) — or no QC column present
    - NPX > LOD for that assay (using plate-specific LOD when available)

    Args:
        dataset: The AffinityDataset to analyze.
        lod: Pre-computed LOD values. If None, resolved automatically.

    Returns:
        Sorted list of valid protein/assay identifiers (expression column names).
    """
    from pyprideap.processing.filtering import filter_controls, filter_qc

    # Filter to biological samples with acceptable QC
    ds = filter_controls(dataset)
    ds = filter_qc(ds)

    if ds.expression.empty:
        return []

    # Resolve LOD
    if lod is None:
        lod = get_lod_values(dataset)
    if lod is None:
        try:
            lod = compute_lod_from_controls(dataset)
        except ValueError:
            expr_numeric = ds.expression.apply(pd.to_numeric, errors="coerce")
            return sorted(expr_numeric.columns[expr_numeric.notna().any()].tolist())

    expr_numeric = ds.expression.apply(pd.to_numeric, errors="coerce")
    above_lod, _ = _above_lod_matrix(expr_numeric, lod)
    valid_cols = above_lod.any(axis=0)

    return sorted(valid_cols[valid_cols].index.tolist())
