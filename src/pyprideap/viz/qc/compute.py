from __future__ import annotations

import logging
import re
from collections.abc import Hashable
from dataclasses import dataclass, field, replace
from html import escape as html_escape
from typing import cast

import numpy as np
import pandas as pd

from pyprideap.core import AffinityDataset, Platform
from pyprideap.processing.lod import resolve_lod_with_source
from pyprideap.viz.qc.agnostic import compute_agnostic_qc

logger = logging.getLogger(__name__)


@dataclass
class DistributionData:
    """Per-sample NPX/RFU distribution curves."""

    sample_ids: list[str]
    sample_values: list[list[float]]
    xlabel: str
    ylabel: str = "Number of Proteins"
    title: str = "Expression Distribution"
    platform: str = ""


@dataclass
class QcLodSummaryData:
    """QC status crossed with LOD: stacked bar."""

    categories: list[str]
    counts: list[int]
    title: str = "QC and LOD Summary"


@dataclass
class LodAnalysisData:
    assay_ids: list[str]
    above_lod_pct: list[float]
    panel: list[str]
    title: str = "LOD Analysis: % Samples Above LOD"
    unit: str = "NPX"  # "NPX" for Olink, "RFU" for SomaScan


@dataclass
class PcaData:
    pc1: list[float]
    pc2: list[float]
    variance_explained: list[float]
    labels: list[str]
    groups: list[str]
    title: str = "PCA"


@dataclass
class UmapData:
    x: list[float]
    y: list[float]
    labels: list[str]
    groups: list[str]
    title: str = "UMAP"


@dataclass
class HeatmapData:
    """Clustered protein expression heatmap."""

    values: list[list[float]]  # expression matrix (samples x proteins)
    sample_labels: list[str]
    protein_labels: list[str]
    sample_order: list[int]  # row indices after clustering
    protein_order: list[int]  # col indices after clustering
    title: str = "Expression Heatmap"


@dataclass
class CorrelationData:
    matrix: list[list[float | None]]
    labels: list[str]
    title: str = "Sample Correlation"


@dataclass
class DataCompletenessData:
    """Per-sample and per-protein data completeness based on LOD."""

    sample_ids: list[str]
    above_lod_rate: list[float]  # per-sample fraction above LOD (0-1)
    below_lod_rate: list[float]  # per-sample fraction below LOD (0-1)
    protein_ids: list[str] = field(default_factory=list)  # per-protein identifiers
    missing_freq: list[float] = field(default_factory=list)  # per-protein fraction below LOD (0-1)
    title: str = "Sample Completeness"


@dataclass
class CvDistributionData:
    feature_ids: list[str]
    cv_values: list[float]
    dilution: list[str] = field(default_factory=list)
    title: str = "CV Distribution"


@dataclass
class VolcanoData:
    """Volcano plot data from differential expression results."""

    protein_ids: list[str]
    assay_names: list[str]
    fold_change: list[float]
    neg_log10_pval: list[float]
    significant: list[bool]
    direction: list[str]  # "up", "down", "ns"
    title: str = "Volcano Plot"
    method: str = ""  # e.g. "Welch t-test", "Linear model (adjusted for age, sex)"


@dataclass
class LodComparisonData:
    """Pairwise LOD comparison scatter data."""

    pairs: list[dict]
    """Each dict has keys: name_x, name_y, assay_ids, values_x, values_y, panels."""
    title: str = "LOD Comparison"
    unit: str = "NPX"  # "NPX" for Olink, "RFU" for SomaScan


@dataclass
class PlateCvData:
    """Per-plate intra CV and overall inter-plate CV distributions."""

    intra_cv: list[float]  # per-analyte CV within each plate (long format)
    intra_plate_label: list[str]  # plate id per intra entry
    inter_cv: list[float]  # per-analyte CV of plate medians
    feature_ids: list[str]  # analyte ids for inter_cv
    plate_ids: list[str] = field(default_factory=list)
    title: str = "Plate CV Distribution"


@dataclass
class NormScaleData:
    """SomaScan normalisation scale factors per sample."""

    sample_ids: list[str]
    values: list[float]  # HybControlNormScale values
    plate_ids: list[str] = field(default_factory=list)
    title: str = "Hybridization Control Normalization Scale"


@dataclass
class OutlierMapData:
    """SomaScan MAD-based outlier map for heatmap visualization."""

    sample_ids: list[str]
    analyte_ids: list[str]
    matrix: list[list[bool]]  # samples × analytes, True = outlier
    outlier_count_per_sample: list[int]
    outlier_fraction_per_sample: list[float]
    fc_crit: float = 5.0
    title: str = "Outlier Map: |x - median| > 6×MAD & FC > 5×"


@dataclass
class RowCheckData:
    """SomaScan RowCheck QC summary."""

    n_pass: int
    n_flag: int
    flagged_sample_ids: list[str] = field(default_factory=list)
    norm_scale_values: list[float] = field(default_factory=list)  # for flagged samples
    title: str = "RowCheck QC Summary"


@dataclass
class ColCheckData:
    """SomaScan ColCheck QC summary with calibrator QC ratio values."""

    n_pass: int
    n_flag: int
    flagged_analyte_ids: list[str] = field(default_factory=list)
    qc_ratios: list[float] = field(default_factory=list)
    analyte_ids: list[str] = field(default_factory=list)
    col_check_flags: list[str] = field(default_factory=list)
    title: str = "Calibrator QC Ratio"


@dataclass
class ControlAnalyteData:
    """SomaScan control analyte classification summary."""

    category_counts: dict[str, int]  # e.g. {"HybControlElution": 12, ...}
    total_controls: int
    total_analytes: int
    title: str = "Control Analyte Classification"


@dataclass
class NormScaleBoxplotData:
    """SomaScan normalization scale factors grouped by a categorical variable."""

    groups: list[str]  # group labels (one per sample)
    norm_scale_columns: list[str]  # normalization scale column names
    values: dict[str, list[float]]  # column_name → list of values
    title: str = "Normalization Scale Factors"


@dataclass
class IqrMedianQcData:
    """Olink IQR vs Median QC plot data (per sample per panel).

    Mirrors ``olink_qc_plot()`` from OlinkAnalyze.
    """

    sample_ids: list[str]
    panels: list[str]
    iqr_values: list[float]
    median_values: list[float]
    is_outlier: list[bool]
    qc_status: list[str]  # "Pass" or "Warning"
    # Per-panel thresholds for drawing boundary lines
    iqr_low: dict[str, float] = field(default_factory=dict)
    iqr_high: dict[str, float] = field(default_factory=dict)
    median_low: dict[str, float] = field(default_factory=dict)
    median_high: dict[str, float] = field(default_factory=dict)
    n_outlier_samples: int = 0
    n_total_samples: int = 0
    title: str = "IQR vs Median QC"


@dataclass
class UniProtDuplicateData:
    """Summary of proteins with multiple assays (UniProt → assay mapping).

    Represents the inverse of assay→UniProt: for each UniProt that has more
    than one assay, we store the list of assay IDs. So n_unique_proteins is
    the number of distinct proteins, n_total_assays is the total number of
    assays, and duplicates[uniprot] = list of assay IDs targeting that protein.
    """

    n_unique_proteins: int
    n_total_assays: int
    duplicates: dict[str, list[str]] = field(default_factory=dict)
    """UniProt → list of assay IDs, only for proteins with >1 assay."""
    title: str = "UniProt Duplicate Detection"


@dataclass
class BridgeabilityData:
    """Cross-product bridgeability diagnostic data for 4-panel plot.

    Mirrors the bridgeability assessment from OlinkAnalyze's
    ``olink_normalization_bridgeable()``.
    """

    protein_ids: list[str]
    range_diffs: list[float]
    r2_values: list[float]
    ks_stats: list[float]
    low_cnts: list[bool]
    recommendations: list[str]  # "MedianCentering", "QuantileSmoothing", "NotBridgeable"
    n_bridgeable: int = 0
    n_not_bridgeable: int = 0
    n_median_centering: int = 0
    n_quantile_smoothing: int = 0
    product1_name: str = "Product 1"
    product2_name: str = "Product 2"
    title: str = "Cross-Product Bridgeability Diagnostic"


# Keep old name for backwards compat in tests
QcSummaryData = QcLodSummaryData


# ---------------------------------------------------------------------------
# Compute functions
# ---------------------------------------------------------------------------


def _sample_id_col(dataset: AffinityDataset) -> str:
    """Pick the best column for labelling samples.

    Prefers ``SampleID`` (Olink) or ``SampleId`` (SomaScan) but falls
    back to ``SampleName`` when ``SampleID`` values are not unique across
    samples (e.g. when the column is actually an assay index).
    """
    for col in ("SampleID", "SampleId"):
        if col in dataset.samples.columns:
            if dataset.samples[col].nunique() == len(dataset.samples):
                return col
    # Samples run on several plates (bridging) carry a unique SampleRun label
    if "SampleRun" in dataset.samples.columns and dataset.samples["SampleRun"].nunique() == len(dataset.samples):
        return "SampleRun"
    # SampleID exists but is not unique — try SampleName if fully populated
    if "SampleName" in dataset.samples.columns:
        non_empty = dataset.samples["SampleName"].astype(str).str.strip().replace({"": pd.NA}).dropna()
        if len(non_empty) == len(dataset.samples):
            return "SampleName"
    # Fall back to whichever ID column exists (even if not fully unique)
    for col in ("SampleID", "SampleId"):
        if col in dataset.samples.columns:
            return col
    return "SampleID"


def _sample_ids(dataset: AffinityDataset) -> list[str]:
    col = _sample_id_col(dataset)
    if col in dataset.samples.columns:
        result: list[str] = dataset.samples[col].astype(str).tolist()
        return result
    return [f"S{i}" for i in range(len(dataset.samples))]


def compute_distribution(dataset: AffinityDataset) -> DistributionData:
    """Per-sample NPX/RFU value lists for overlaid density curves.

    Values are rounded to 2 decimal places to reduce HTML output size
    without visible loss of quality in the histogram plots.
    """
    logger.debug("Computing distribution...")
    numeric = dataset.expression.apply(pd.to_numeric, errors="coerce")

    is_somascan = dataset.platform == Platform.SOMASCAN
    sample_ids = _sample_ids(dataset)
    sample_values: list[list[float]] = []

    for idx in range(len(numeric)):
        row = numeric.iloc[idx].dropna().values
        if is_somascan:
            row = np.log10(row[row > 0])
        sample_values.append(np.round(row, 2).tolist())

    if is_somascan:
        xlabel = "log10(RFU)"
        title = "RFU Distribution (log10)"
    else:
        xlabel = "NPX Value"
        title = "NPX Distribution"

    return DistributionData(
        sample_ids=sample_ids,
        sample_values=sample_values,
        xlabel=xlabel,
        title=title,
        platform=dataset.platform.value,
    )


def _measurement_qc_flags(dataset: AffinityDataset, numeric: pd.DataFrame) -> pd.DataFrame | None:
    """Sample QC flag of every measurement (samples x assays), upper-case PASS/WARN/FAIL.

    Olink assigns sample QC per sample and panel block, so the per-measurement
    flags are used when the reader kept them; otherwise each sample's SampleQC
    value is applied to all its measurements.
    """
    matrix = dataset.metadata.get("sample_qc_matrix")
    if isinstance(matrix, pd.DataFrame) and matrix.shape == numeric.shape:
        flags = matrix.copy()
        flags.columns = numeric.columns
        flags.index = numeric.index
    elif "SampleQC" in dataset.samples.columns:
        per_sample = dataset.samples["SampleQC"].reset_index(drop=True)
        flags = pd.DataFrame(
            np.repeat(per_sample.to_numpy(dtype=object)[:, None], numeric.shape[1], axis=1),
            index=numeric.index,
            columns=numeric.columns,
        )
    else:
        return None
    text = flags.astype("string").apply(lambda c: c.str.strip().str.upper())
    return cast(pd.DataFrame, text.replace({"WARNING": "WARN"}))


def compute_qc_summary(dataset: AffinityDataset) -> QcLodSummaryData | None:
    """QC status × LOD stacked bar. Falls back to simple QC counts if no LOD."""
    # Some Olink exports (and some PAD uploads) have no sample QC flags. In that
    # case we can still compute the overall % above/below LOD, but we can't
    # stratify by PASS/WARN/FAIL.

    from pyprideap.processing.lod import _above_lod_matrix

    # Same LOD source as the completeness / LOD analysis plots
    lod = _resolve_lod(dataset)
    numeric = dataset.expression.apply(pd.to_numeric, errors="coerce")

    if lod is not None and (isinstance(lod, pd.DataFrame) or len(lod) > 0):
        above_lod, has_lod = _above_lod_matrix(numeric, lod)

        unit = "RFU" if dataset.platform == Platform.SOMASCAN else "NPX"
        categories: list[str] = []
        counts: list[int] = []

        qc = _measurement_qc_flags(dataset, numeric)
        if qc is not None:
            valid = numeric.notna() & has_lod
            above_all = above_lod & valid
            # All PASS / WARN / FAIL categories are reported, including empty ones,
            # so the plot shows explicitly when there are no WARN or FAIL measurements
            for qc_val in ["PASS", "WARN", "FAIL"]:
                in_flag = qc.isin([qc_val]).astype(bool)
                above = int((above_all & in_flag).to_numpy().sum())
                below = int((valid & in_flag).to_numpy().sum()) - above
                categories += [f"{qc_val} & {unit} > LOD", f"{qc_val} & {unit} ≤ LOD"]
                counts += [above, below]
                # Olink leaves the value empty for failed measurements: count them too
                no_value = int((numeric.isna() & in_flag).to_numpy().sum())
                if no_value:
                    categories.append(f"{qc_val} & no value")
                    counts.append(no_value)
            # Measurements without a recognised flag are shown rather than dropped
            unflagged = valid & ~qc.isin(["PASS", "WARN", "FAIL"]).fillna(False).astype(bool)
            if unflagged.to_numpy().any():
                above = int((above_all & unflagged).to_numpy().sum())
                categories += [f"No QC flag & {unit} > LOD", f"No QC flag & {unit} ≤ LOD"]
                counts += [above, int(unflagged.to_numpy().sum()) - above]
        else:
            # No QC flags available: show overall above/below LOD split
            valid = numeric.notna() & has_lod
            above = int((above_lod & valid).sum().sum())
            below = int(valid.sum().sum()) - above
            if above > 0:
                categories.append(f"{unit} > LOD")
                counts.append(above)
            if below > 0:
                categories.append(f"{unit} ≤ LOD")
                counts.append(below)

        if categories and sum(counts) > 0:
            return QcLodSummaryData(categories=categories, counts=counts)

    # Fallback: simple QC counts when no LOD is available but SampleQC exists
    if "SampleQC" in dataset.samples.columns:
        vc = dataset.samples["SampleQC"].value_counts()
        return QcLodSummaryData(categories=vc.index.tolist(), counts=vc.values.tolist())

    return None


def compute_lod_analysis(dataset: AffinityDataset) -> LodAnalysisData | None:
    from pyprideap.processing.lod import _above_lod_matrix

    lod = _resolve_lod(dataset)
    if lod is None:
        return None

    numeric = dataset.expression.apply(pd.to_numeric, errors="coerce")
    above_lod, has_lod = _above_lod_matrix(numeric, lod)

    assay_ids = []
    above_lod_pct = []
    panels = []

    id_col = "OlinkID" if "OlinkID" in dataset.features.columns else dataset.features.columns[0]
    id_to_panel: dict[str, str] = {}
    if "Panel" in dataset.features.columns:
        id_to_panel = dict(zip(dataset.features[id_col].astype(str), dataset.features["Panel"].astype(str)))

    for col in numeric.columns:
        # Skip assays with no LOD for any sample
        if not has_lod[col].any():
            continue
        vals_valid = numeric[col].notna() & has_lod[col]
        n_valid = int(vals_valid.sum())
        if n_valid == 0:
            pct = 0.0
        else:
            pct = float(above_lod.loc[vals_valid, col].sum() / n_valid * 100)

        assay_ids.append(str(col))
        above_lod_pct.append(round(pct, 2))
        panels.append(id_to_panel.get(str(col), ""))

    if not assay_ids:
        return None

    unit = "RFU" if dataset.platform == Platform.SOMASCAN else "NPX"
    return LodAnalysisData(assay_ids=assay_ids, above_lod_pct=above_lod_pct, panel=panels, unit=unit)


def compute_pca(dataset: AffinityDataset, n_components: int = 2) -> PcaData | None:
    logger.debug("Computing PCA...")
    try:
        from sklearn.decomposition import PCA
        from sklearn.impute import SimpleImputer
    except ImportError:
        return None

    numeric = dataset.expression.apply(pd.to_numeric, errors="coerce")
    if numeric.shape[0] < 2 or numeric.shape[1] < 2:
        return None

    imputer = SimpleImputer(strategy="median")
    imputed = imputer.fit_transform(numeric)

    n_comp = min(n_components, *imputed.shape)
    pca = PCA(n_components=n_comp)
    transformed = pca.fit_transform(imputed)
    logger.debug(
        "PCA: expression matrix %s, variance explained=%s",
        numeric.shape,
        [round(float(v), 4) for v in pca.explained_variance_ratio_],
    )

    labels = _sample_ids(dataset)

    # Use SampleQC for color if all SampleType values are the same
    groups: list[str]
    if "SampleType" in dataset.samples.columns:
        types = dataset.samples["SampleType"].unique()
        if len(types) == 1 and "SampleQC" in dataset.samples.columns:
            groups = dataset.samples["SampleQC"].astype(str).tolist()
        else:
            groups = dataset.samples["SampleType"].astype(str).tolist()
    else:
        groups = [""] * len(labels)

    return PcaData(
        pc1=np.round(transformed[:, 0], 4).tolist(),
        pc2=np.round(transformed[:, 1], 4).tolist() if n_comp >= 2 else [0.0] * len(labels),
        variance_explained=[round(float(v), 4) for v in pca.explained_variance_ratio_],
        labels=labels,
        groups=groups,
    )


def compute_tsne(dataset: AffinityDataset) -> UmapData | None:
    """Non-linear dimensionality reduction via t-SNE (scikit-learn).

    Returns *None* when scikit-learn is not available or the dataset is too
    small (fewer than 4 samples).
    """
    logger.debug("Computing t-SNE...")
    try:
        from sklearn.impute import SimpleImputer
        from sklearn.manifold import TSNE
    except ImportError:
        return None

    numeric = dataset.expression.apply(pd.to_numeric, errors="coerce")
    if numeric.shape[0] < 4 or numeric.shape[1] < 2:
        return None

    imputer = SimpleImputer(strategy="median")
    imputed = imputer.fit_transform(numeric)

    perplexity = min(30.0, max(2.0, (imputed.shape[0] - 1) / 3.0))
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    transformed = tsne.fit_transform(imputed)

    labels = _sample_ids(dataset)

    # Use SampleQC for color if all SampleType values are the same
    groups: list[str]
    if "SampleType" in dataset.samples.columns:
        types = dataset.samples["SampleType"].unique()
        if len(types) == 1 and "SampleQC" in dataset.samples.columns:
            groups = dataset.samples["SampleQC"].astype(str).tolist()
        else:
            groups = dataset.samples["SampleType"].astype(str).tolist()
    else:
        groups = [""] * len(labels)

    return UmapData(
        x=np.round(transformed[:, 0], 4).tolist(),
        y=np.round(transformed[:, 1], 4).tolist(),
        labels=labels,
        groups=groups,
        title="t-SNE",
    )


# Keep old name for backwards compatibility
compute_umap = compute_tsne


def compute_heatmap(
    dataset: AffinityDataset,
    max_proteins: int = 200,
    max_samples: int = 100,
) -> HeatmapData | None:
    """Clustered expression heatmap (samples x proteins).

    Selects the most variable proteins (by std) up to *max_proteins*, and
    subsamples rows if needed.  Hierarchical clustering is applied to both
    axes when scipy is available; otherwise the original order is kept.
    """
    logger.debug("Computing heatmap...")
    numeric = dataset.expression.apply(pd.to_numeric, errors="coerce")
    if numeric.shape[0] < 2 or numeric.shape[1] < 2:
        return None

    # Subsample samples if too many
    if numeric.shape[0] > max_samples:
        numeric = numeric.sample(n=max_samples, random_state=42)

    # Select most variable proteins
    stds = numeric.std()
    stds = stds.replace([np.inf, -np.inf], np.nan).dropna()
    if len(stds) == 0:
        return None
    top = stds.nlargest(min(max_proteins, len(stds))).index
    numeric = numeric[top]
    logger.debug("Heatmap: selected %d most variable proteins from %d total", len(top), len(stds))

    # Fill NaN with column median for clustering
    filled = numeric.fillna(numeric.median())

    # Z-score normalise per protein (column) for visualisation
    col_mean = filled.mean()
    col_std = filled.std().replace(0, 1)
    z = (filled - col_mean) / col_std

    # Cluster rows and columns
    sample_order = list(range(z.shape[0]))
    protein_order = list(range(z.shape[1]))
    try:
        from scipy.cluster.hierarchy import leaves_list, linkage

        if z.shape[0] > 2:
            sample_order = leaves_list(linkage(z.values, method="ward")).tolist()
        if z.shape[1] > 2:
            protein_order = leaves_list(linkage(z.values.T, method="ward")).tolist()
    except ImportError:
        pass

    # Build labels
    sample_labels: list[str]
    id_col = _sample_id_col(dataset)
    if id_col in dataset.samples.columns:
        sample_labels = dataset.samples.loc[numeric.index, id_col].astype(str).tolist()
    else:
        sample_labels = [f"S{i}" for i in range(len(numeric))]

    protein_labels = [str(c) for c in numeric.columns]

    return HeatmapData(
        values=np.round(z.values, 3).tolist(),
        sample_labels=sample_labels,
        protein_labels=protein_labels,
        sample_order=sample_order,
        protein_order=protein_order,
    )


def compute_correlation(dataset: AffinityDataset, max_samples: int = 50) -> CorrelationData:
    logger.debug("Computing correlation matrix...")
    numeric = dataset.expression.apply(pd.to_numeric, errors="coerce")

    if numeric.shape[0] > max_samples:
        numeric = numeric.sample(n=max_samples, random_state=42)

    # Compute correlation (samples x samples)
    corr = numeric.T.corr()

    # Reorder samples by similarity (hierarchical clustering on correlation distance)
    # If scipy isn't available, fall back to a metadata-based ordering so the plot is stable.
    order = list(range(len(corr)))
    try:
        from scipy.cluster.hierarchy import leaves_list, linkage
        from scipy.spatial.distance import squareform

        # Convert correlation to a distance matrix in [0, 2]; NaNs -> max distance
        dist = 1.0 - corr.astype(float)
        dist = dist.fillna(1.0)

        # Ensure symmetry and zero diagonal for squareform
        dist = dist.copy()
        diag_vals = dist.values.copy()
        np.fill_diagonal(diag_vals, 0.0)
        dist = pd.DataFrame(diag_vals, index=dist.index, columns=dist.columns)
        dist = (dist + dist.T) / 2.0

        if dist.shape[0] > 2:
            condensed = squareform(dist.values, checks=False)
            order = leaves_list(linkage(condensed, method="average")).tolist()
    except ImportError:
        sort_cols = [c for c in ("SampleType", "PlateID", "PlateId") if c in dataset.samples.columns]
        if sort_cols:
            ordered_idx = dataset.samples.loc[numeric.index].sort_values(sort_cols).index
            numeric = numeric.loc[ordered_idx]
            corr = numeric.T.corr()
            order = list(range(len(corr)))

    if order and len(order) == len(corr):
        corr = corr.iloc[order, order]

    id_col = _sample_id_col(dataset)
    if id_col in dataset.samples.columns:
        labels = dataset.samples.loc[numeric.index, id_col].astype(str).tolist()
    else:
        labels = [f"S{i}" for i in range(len(numeric))]

    if order and len(order) == len(labels):
        labels = [labels[i] for i in order]

    return CorrelationData(
        matrix=[[None if np.isnan(v) else round(v, 3) for v in row] for row in corr.values],
        labels=labels,
    )


def _resolve_lod(dataset: AffinityDataset) -> pd.DataFrame | pd.Series | None:
    """LOD from :func:`resolve_lod_with_source`, without the source name."""
    return resolve_lod_with_source(dataset)[0]


def compute_data_completeness(dataset: AffinityDataset) -> DataCompletenessData | None:
    """Per-sample completeness (above/below LOD) and per-protein missing frequency.

    LOD resolution priority: Reported LOD → NCLOD → FixedLOD.
    Per-protein missing frequency uses Olink's MissingFreq column when
    available (fraction of samples with NPX below LOD), otherwise
    computed from the LOD matrix.

    Control samples (negative controls, plate controls, buffer, calibrator, QC)
    are excluded so the plot only shows biological samples.

    Returns None when no LOD source is available.
    """
    from pyprideap.processing.filtering import filter_controls
    from pyprideap.processing.lod import _above_lod_matrix

    # Filter out control samples — only show biological samples
    ds = filter_controls(dataset)

    numeric = ds.expression.apply(pd.to_numeric, errors="coerce")
    sample_ids = _sample_ids(ds)

    # Try to get per-protein missing frequency from Olink's MissingFreq column
    # MissingFreq = fraction of samples with NPX below LOD
    olink_missing_freq = None
    if "MissingFreq" in ds.features.columns:
        mf = pd.to_numeric(ds.features["MissingFreq"], errors="coerce")
        if mf.notna().any():
            olink_missing_freq = mf

    # Resolve LOD from all available sources (use original dataset for negative controls)
    lod = _resolve_lod(dataset)
    has_lod_data = lod is not None and (isinstance(lod, pd.DataFrame) or len(lod) > 0)

    if not has_lod_data and olink_missing_freq is None:
        return None

    # Per-sample above/below LOD rates
    above_lod_rate: list[float] = []
    below_lod_rate: list[float] = []
    protein_ids: list[str] = []
    missing_freq_values: list[float] = []

    if has_lod_data:
        assert lod is not None  # narrowing for mypy
        above_lod, has_lod = _above_lod_matrix(numeric, lod)

        for idx in range(len(numeric)):
            row_has_lod = has_lod.iloc[idx]
            row_valid = numeric.iloc[idx].notna() & row_has_lod
            n_valid = int(row_valid.sum())
            if n_valid > 0:
                n_above = int((above_lod.iloc[idx] & row_valid).sum())
                above_lod_rate.append(round(n_above / n_valid, 4))
                below_lod_rate.append(round((n_valid - n_above) / n_valid, 4))
            else:
                above_lod_rate.append(0.0)
                below_lod_rate.append(0.0)

        # Per-protein missing frequency from LOD matrix
        for col in numeric.columns:
            valid = numeric[col].notna() & has_lod[col]
            n_valid = int(valid.sum())
            if n_valid > 0:
                frac_below = 1.0 - float(above_lod.loc[valid, col].sum() / n_valid)
            else:
                frac_below = 0.0
            protein_ids.append(str(col))
            missing_freq_values.append(round(frac_below, 4))

    # Override per-protein missing frequency with MissingFreq column if available
    if olink_missing_freq is not None:
        protein_ids = [str(c) for c in numeric.columns]
        missing_freq_values = [round(float(v), 4) if pd.notna(v) else 0.0 for v in olink_missing_freq]
        # If we didn't have LOD data for per-sample rates, compute approximate
        # overall rates from MissingFreq
        if not has_lod_data:
            overall_below = float(olink_missing_freq.mean())
            above_lod_rate = [round(1.0 - overall_below, 4)] * len(sample_ids)
            below_lod_rate = [round(overall_below, 4)] * len(sample_ids)

    return DataCompletenessData(
        sample_ids=sample_ids,
        above_lod_rate=above_lod_rate,
        below_lod_rate=below_lod_rate,
        protein_ids=protein_ids,
        missing_freq=missing_freq_values,
    )


def _linear_study_samples(dataset: AffinityDataset) -> AffinityDataset:
    """Study samples only, with values on a linear scale for CV = SD / mean.

    Control samples are dropped. Olink NPX is log2, so it is converted to
    2^NPX (SD / mean is meaningless on log data); SomaScan RFU is already
    linear and is used as deposited.
    """
    from pyprideap.processing.filtering import filter_controls

    ds = filter_controls(dataset)
    numeric = ds.expression.apply(pd.to_numeric, errors="coerce")
    if ds.platform != Platform.SOMASCAN:
        numeric = np.power(2, numeric)
    return replace(ds, expression=numeric)


def compute_cv_distribution(dataset: AffinityDataset) -> CvDistributionData | None:
    """Per-analyte CV (SD / mean) across study samples, on a linear scale.

    See :func:`_linear_study_samples` for the transformation; the same
    definition is used by :func:`compute_plate_cv`.
    """
    numeric = _linear_study_samples(dataset).expression

    means = numeric.mean()
    stds = numeric.std()
    cv = stds / means
    cv = cv.replace([np.inf, -np.inf], np.nan).dropna()

    if cv.empty:
        return None

    feature_ids = cv.index.tolist()
    dilution = (
        dataset.features["Dilution"].astype(str).tolist()
        if "Dilution" in dataset.features.columns and len(dataset.features) == len(numeric.columns)
        else []
    )

    return CvDistributionData(
        feature_ids=feature_ids,
        cv_values=cv.tolist(),
        dilution=dilution,
    )


def compute_plate_cv(dataset: AffinityDataset) -> PlateCvData | None:
    """Compute intra-plate and inter-plate CV.

    Uses the same definition as :func:`compute_cv_distribution`: study samples
    only, Olink NPX converted to linear scale (2^NPX), SomaScan RFU as deposited.

    Intra-plate CV: for each plate, CV = SD / mean per analyte across samples.
    Returned in long format (one entry per analyte per plate).

    Inter-plate CV: for each analyte, CV of plate medians across plates.
    One value per analyte.

    Only applicable when a plate column (SomaScan ``PlateId``, Olink
    ``PlateID``) exists with >= 2 plates.
    """
    plate_col = next((c for c in ("PlateId", "PlateID") if c in dataset.samples.columns), None)
    if plate_col is None:
        return None

    ds = _linear_study_samples(dataset)
    plates = ds.samples[plate_col]
    unique_plates = sorted(plates.dropna().unique(), key=str)
    if len(unique_plates) < 2:
        return None

    numeric = ds.expression

    # --- Intra-plate CV (long format) ---
    intra_cv: list[float] = []
    intra_plate_label: list[str] = []

    plate_medians: dict[str, pd.Series] = {}

    for plate_id in unique_plates:
        mask = plates == plate_id
        plate_data = numeric.loc[mask]
        if plate_data.shape[0] < 3:
            continue
        means = plate_data.mean()
        stds = plate_data.std()
        cv = (stds / means).replace([np.inf, -np.inf], np.nan).dropna()

        intra_cv.extend(cv.tolist())
        intra_plate_label.extend([str(plate_id)] * len(cv))

        plate_medians[str(plate_id)] = plate_data.median()

    if not intra_cv or len(plate_medians) < 2:
        return None

    # --- Inter-plate CV: CV of plate medians per analyte ---
    median_df = pd.DataFrame(plate_medians)
    inter_mean = median_df.mean(axis=1)
    inter_std = median_df.std(axis=1)
    inter_cv_series = (inter_std / inter_mean).replace([np.inf, -np.inf], np.nan).dropna()

    return PlateCvData(
        intra_cv=intra_cv,
        intra_plate_label=intra_plate_label,
        inter_cv=inter_cv_series.tolist(),
        feature_ids=inter_cv_series.index.astype(str).tolist(),
        plate_ids=[str(p) for p in unique_plates],
    )


def compute_norm_scale(dataset: AffinityDataset) -> NormScaleData | None:
    """Extract HybControlNormScale from SomaScan sample metadata.

    This is a standard SomaScan QC metric: values near 1.0 indicate good
    hybridization, while values outside 0.4–2.5 flag potential issues.
    """
    if "HybControlNormScale" not in dataset.samples.columns:
        return None

    vals = pd.to_numeric(dataset.samples["HybControlNormScale"], errors="coerce")
    if vals.notna().sum() == 0:
        return None

    sample_ids = _sample_ids(dataset)
    plate_ids = dataset.samples["PlateId"].astype(str).tolist() if "PlateId" in dataset.samples.columns else []

    return NormScaleData(
        sample_ids=sample_ids,
        values=vals.tolist(),
        plate_ids=plate_ids,
    )


def compute_lod_comparison(dataset: AffinityDataset) -> LodComparisonData | None:
    """Compute pairwise LOD comparisons across all available LOD sources."""
    from pyprideap.processing.lod import (
        compute_nclod,
        compute_soma_elod,
        get_reported_lod,
        load_fixed_lod,
    )

    # Collect available LOD sources as per-assay Series
    sources: dict[str, pd.Series] = {}

    # 1. Reported LOD
    reported = get_reported_lod(dataset)
    if reported is not None:
        if isinstance(reported, pd.DataFrame):
            sources["Reported LOD"] = reported.median(axis=0)
        else:
            sources["Reported LOD"] = reported

    # 2. NCLOD (from negative controls — both platforms)
    try:
        nclod = compute_nclod(dataset, plate_adjusted=False)
        if isinstance(nclod, pd.DataFrame):
            sources["NCLOD"] = nclod.median(axis=0)
        else:
            sources["NCLOD"] = nclod
    except (ValueError, KeyError):
        pass

    # 3. Platform-specific LOD sources
    if dataset.platform == Platform.SOMASCAN:
        # SomaScan eLOD from buffer samples
        try:
            sources["eLOD"] = compute_soma_elod(dataset)
        except (ValueError, KeyError):
            pass
    else:
        # Olink FixedLOD from bundled config
        try:
            fixed = load_fixed_lod(dataset)
            sources["FixedLOD"] = fixed
        except (ValueError, FileNotFoundError):
            pass

    if len(sources) < 2:
        return None

    # Build panel map for coloring
    id_col = "OlinkID" if "OlinkID" in dataset.features.columns else dataset.features.columns[0]
    panel_map: dict[str, str] = {}
    if "Panel" in dataset.features.columns:
        panel_map = dict(zip(dataset.features[id_col].astype(str), dataset.features["Panel"].astype(str)))

    # Generate all pairs
    names = list(sources.keys())
    pairs: list[dict] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            sx = sources[names[i]]
            sy = sources[names[j]]
            common = sx.dropna().index.intersection(sy.dropna().index)
            if len(common) < 2:
                continue
            pairs.append(
                {
                    "name_x": names[i],
                    "name_y": names[j],
                    "assay_ids": [str(c) for c in common],
                    "values_x": np.round(np.asarray(sx.reindex(common)), 4).tolist(),
                    "values_y": np.round(np.asarray(sy.reindex(common)), 4).tolist(),
                    "panels": [panel_map.get(str(c), "") for c in common],
                }
            )

    if not pairs:
        return None

    unit = "RFU" if dataset.platform == Platform.SOMASCAN else "NPX"
    return LodComparisonData(pairs=pairs, unit=unit)


def compute_volcano(
    test_results: pd.DataFrame,
    fc_threshold: float = 1.0,
    p_threshold: float = 0.05,
) -> VolcanoData | None:
    """Build volcano plot data from differential expression results.

    Parameters
    ----------
    test_results : DataFrame
        Output of :func:`pyprideap.differential.ttest` or similar, with columns
        ``protein_id``, ``estimate`` (fold change), ``adj_p_value``, and
        optionally ``assay``.
    fc_threshold : float
        Absolute fold-change threshold for significance colouring.
    p_threshold : float
        Adjusted p-value threshold for significance colouring.
    """
    required = {"protein_id", "estimate", "adj_p_value"}
    if not required.issubset(test_results.columns):
        return None

    df = test_results.dropna(subset=["estimate", "adj_p_value"]).copy()
    if df.empty:
        return None

    neg_log10 = -np.log10(df["adj_p_value"].clip(lower=1e-300))

    directions: list[str] = []
    sig: list[bool] = []
    for _, row in df.iterrows():
        is_sig = row["adj_p_value"] < p_threshold and abs(row["estimate"]) >= fc_threshold
        sig.append(is_sig)
        if is_sig and row["estimate"] > 0:
            directions.append("up")
        elif is_sig and row["estimate"] < 0:
            directions.append("down")
        else:
            directions.append("ns")

    assay_col = "assay" if "assay" in df.columns else "protein_id"
    return VolcanoData(
        protein_ids=df["protein_id"].astype(str).tolist(),
        assay_names=df[assay_col].astype(str).tolist(),
        fold_change=np.round(df["estimate"].values, 4).tolist(),
        neg_log10_pval=np.round(neg_log10.values, 4).tolist(),
        significant=sig,
        direction=directions,
    )


def compute_outlier_map(
    dataset: AffinityDataset,
    *,
    fc_crit: float = 5.0,
    max_analytes: int = 500,
) -> OutlierMapData | None:
    """Compute MAD-based outlier map for SomaScan QC visualization.

    Equivalent to ``calcOutlierMap()`` in SomaDataIO.  Returns an outlier
    boolean matrix suitable for heatmap rendering.

    Only applicable to SomaScan datasets.
    """
    if dataset.platform != Platform.SOMASCAN:
        return None

    from pyprideap.processing.somascan.outliers import calc_outlier_map

    omap = calc_outlier_map(dataset, fc_crit=fc_crit)

    # Subsample analytes for visualization if too many
    mat = omap.matrix
    if mat.shape[1] > max_analytes:
        # Keep analytes with most outliers
        outlier_counts = mat.sum(axis=0)
        top_cols = outlier_counts.nlargest(max_analytes).index
        mat = mat[top_cols]

    sample_ids = _sample_ids(dataset)
    analyte_ids = [str(c) for c in mat.columns]

    return OutlierMapData(
        sample_ids=sample_ids,
        analyte_ids=analyte_ids,
        matrix=[row.tolist() for row in mat.values],
        outlier_count_per_sample=omap.n_outliers_per_sample.tolist(),
        outlier_fraction_per_sample=omap.outlier_fraction_per_sample.round(4).tolist(),
        fc_crit=fc_crit,
        title=omap.title,
    )


def compute_row_check(dataset: AffinityDataset) -> RowCheckData | None:
    """Compute RowCheck QC summary for SomaScan data.

    Returns None for non-SomaScan datasets or when no normalization
    scale columns are present.
    """
    if dataset.platform != Platform.SOMASCAN:
        return None

    from pyprideap.processing.somascan.qc_flags import add_row_check, get_row_check_summary

    ds = add_row_check(dataset)
    summary = get_row_check_summary(ds)

    flagged_mask = ds.samples["RowCheck"] == "FLAG"
    sample_ids = _sample_ids(ds)
    flagged_ids = [sample_ids[i] for i, flagged in enumerate(flagged_mask) if flagged]

    # Get norm scale values for flagged samples
    norm_vals: list[float] = []
    if "HybControlNormScale" in ds.samples.columns and flagged_mask.any():
        vals = pd.to_numeric(ds.samples.loc[flagged_mask, "HybControlNormScale"], errors="coerce")
        norm_vals = vals.tolist()

    return RowCheckData(
        n_pass=summary["PASS"],
        n_flag=summary["FLAG"],
        flagged_sample_ids=flagged_ids,
        norm_scale_values=norm_vals,
    )


def compute_col_check(dataset: AffinityDataset) -> ColCheckData | None:
    """Compute ColCheck QC summary with calibrator QC ratio values for SomaScan data."""
    if dataset.platform != Platform.SOMASCAN:
        return None

    from pyprideap.processing.somascan.qc_flags import get_col_check_summary

    # Try to find a calibrator QC ratio column in the feature metadata.
    # Common ADATs provide one (or more) columns named CalQcRatio*.
    ratio_col = None
    for c in dataset.features.columns:
        cs = str(c)
        if cs.startswith("CalQcRatio"):
            ratio_col = c
            break
    if ratio_col is None:
        for c in dataset.features.columns:
            cs = str(c).lower()
            if "calqcratio" in cs or ("qc" in cs and "ratio" in cs and "cal" in cs):
                ratio_col = c
                break

    # Fallback: some ADAT exports store calibrator QC ratios as multiple Cal_* columns
    # (e.g. Cal_P0029868) with values centered at 1.0 and used to derive ColCheck.
    cal_ratio_cols: list[str] = []
    if ratio_col is None:
        for c in dataset.features.columns:
            cs = str(c)
            if cs.startswith("Cal_") and cs != "CalReference":
                cal_ratio_cols.append(cs)

    has_flags = "ColCheck" in dataset.features.columns
    has_ratios = ratio_col is not None or len(cal_ratio_cols) > 0
    if not has_flags and not has_ratios:
        return None

    # Summary (PASS/FLAG counts). If ColCheck isn't present but ratios are,
    # compute flags on the fly using SomaDataIO thresholds [0.8, 1.2].
    if has_flags:
        summary = get_col_check_summary(dataset)
        flags_series = dataset.features["ColCheck"].astype(str)
    else:
        if ratio_col is not None:
            ratios_tmp = pd.to_numeric(dataset.features[ratio_col], errors="coerce")
        else:
            # Use median across calibrator ratio columns when no explicit ratio column exists
            cal_df = dataset.features[cal_ratio_cols].apply(pd.to_numeric, errors="coerce")
            ratios_tmp = cal_df.median(axis=1, skipna=True)
        in_range = ratios_tmp.between(0.8, 1.2)
        flags_series = in_range.map({True: "PASS", False: "FLAG"}).where(ratios_tmp.notna(), other="PASS")
        counts = flags_series.value_counts()
        summary = {"PASS": int(counts.get("PASS", 0)), "FLAG": int(counts.get("FLAG", 0))}

    id_col = "SeqId" if "SeqId" in dataset.features.columns else dataset.features.columns[0]
    flagged_ids: list[str] = []
    if summary["FLAG"] > 0:
        flag_mask = flags_series == "FLAG"
        flagged_ids = dataset.features.loc[flag_mask, id_col].astype(str).tolist()

    # Extract CalQcRatio values for the scatter/strip plot
    qc_ratios: list[float] = []
    analyte_ids: list[str] = []
    col_check_flags: list[str] = []

    ids = dataset.features[id_col].astype(str)
    ratios = None
    if ratio_col is not None:
        ratios = pd.to_numeric(dataset.features[ratio_col], errors="coerce")
    elif cal_ratio_cols:
        cal_df = dataset.features[cal_ratio_cols].apply(pd.to_numeric, errors="coerce")
        ratios = cal_df.median(axis=1, skipna=True)

    if ratios is not None:
        valid = ratios.notna()
        qc_ratios = ratios[valid].round(4).tolist()
        analyte_ids = ids[valid].tolist()
        col_check_flags = flags_series[valid].astype(str).tolist()

    return ColCheckData(
        n_pass=summary["PASS"],
        n_flag=summary["FLAG"],
        flagged_analyte_ids=flagged_ids,
        qc_ratios=qc_ratios,
        analyte_ids=analyte_ids,
        col_check_flags=col_check_flags,
    )


def compute_control_analytes(dataset: AffinityDataset) -> ControlAnalyteData | None:
    """Classify and count control analytes in SomaScan data."""
    if dataset.platform != Platform.SOMASCAN:
        return None

    from pyprideap.processing.somascan.controls import (
        CONTROL_ANALYTE_TYPES,
        classify_control_analytes,
    )

    classified = classify_control_analytes(dataset)
    if not classified:
        return None

    category_counts: dict[str, int] = {}
    for cat_type in CONTROL_ANALYTE_TYPES:
        count = sum(1 for v in classified.values() if v == cat_type)
        if count > 0:
            category_counts[cat_type.value] = count

    return ControlAnalyteData(
        category_counts=category_counts,
        total_controls=len(classified),
        total_analytes=len(dataset.expression.columns),
    )


def compute_norm_scale_boxplot(
    dataset: AffinityDataset,
    group_by: str | None = None,
) -> NormScaleBoxplotData | None:
    """Compute normalization scale factors grouped by a variable.

    Equivalent to the ``data.qc`` plots in SomaDataIO's ``preProcessAdat()``.
    Shows boxplots of all NormScale / Med.Scale.* columns grouped by a
    categorical variable (e.g. Sex, PlateId).

    Only applicable to SomaScan datasets.
    """
    if dataset.platform != Platform.SOMASCAN:
        return None

    # Find normalization scale columns
    norm_cols = [c for c in dataset.samples.columns if "normscale" in c.lower() or c.startswith("Med.Scale.")]
    if not norm_cols:
        return None

    # Determine grouping variable
    if group_by and group_by in dataset.samples.columns:
        groups = dataset.samples[group_by].astype(str).tolist()
    elif "PlateId" in dataset.samples.columns:
        groups = dataset.samples["PlateId"].astype(str).tolist()
    else:
        groups = ["All"] * len(dataset.samples)

    values: dict[str, list[float]] = {}
    for col in norm_cols:
        vals = pd.to_numeric(dataset.samples[col], errors="coerce")
        values[col] = vals.tolist()

    return NormScaleBoxplotData(
        groups=groups,
        norm_scale_columns=norm_cols,
        values=values,
    )


def compute_iqr_median_qc(
    dataset: AffinityDataset,
    *,
    iqr_outlier_def: float = 3.0,
    median_outlier_def: float = 3.0,
) -> IqrMedianQcData | None:
    """Compute IQR vs Median QC data for Olink datasets.

    Mirrors ``olink_qc_plot()`` from OlinkAnalyze: per panel, computes IQR
    and median NPX per sample, then flags samples outside ±n SD.

    Returns None for SomaScan datasets or when Panel column is absent.
    """
    if dataset.platform == Platform.SOMASCAN:
        return None

    from pyprideap.processing.olink.outliers import compute_iqr_median_outliers

    result = compute_iqr_median_outliers(
        dataset,
        iqr_outlier_def=iqr_outlier_def,
        median_outlier_def=median_outlier_def,
    )

    return IqrMedianQcData(
        sample_ids=result.sample_ids,
        panels=result.panels,
        iqr_values=result.iqr_values,
        median_values=result.median_values,
        is_outlier=result.is_outlier,
        qc_status=result.qc_status,
        iqr_low=result.iqr_low,
        iqr_high=result.iqr_high,
        median_low=result.median_low,
        median_high=result.median_high,
        n_outlier_samples=len(result.outlier_sample_ids),
        n_total_samples=result.n_samples,
    )


def compute_uniprot_duplicates(dataset: AffinityDataset) -> UniProtDuplicateData | None:
    """Summarise proteins with multiple assays (UniProt → assays).

    Groups by UniProt and lists assay IDs per protein. So we get unique protein
    count, total assay count, and which proteins are targeted by more than one
    assay (e.g. SomaScan replicate aptamers, or Olink panels overlapping).
    Returns None when UniProt or assay ID column is absent.
    """
    if "UniProt" not in dataset.features.columns:
        return None

    if dataset.platform == Platform.SOMASCAN:
        assay_id_col = "Name" if "Name" in dataset.features.columns else "SeqId"
        if assay_id_col not in dataset.features.columns:
            return None
    else:
        assay_id_col = "OlinkID"
        if assay_id_col not in dataset.features.columns:
            return None

    features = dataset.features
    # Group by UniProt: for each protein, list of assay IDs
    grouped = (
        features[[assay_id_col, "UniProt"]]
        .dropna(subset=["UniProt"])
        .drop_duplicates()
        .groupby("UniProt", sort=False)[assay_id_col]
        .apply(list)
        .to_dict()
    )

    n_unique_proteins = len(grouped)
    n_total_assays = len(features)  # or len(dataset.expression.columns)
    # Proteins with more than one assay
    duplicates = {up: assays for up, assays in grouped.items() if len(assays) > 1}

    return UniProtDuplicateData(
        n_unique_proteins=n_unique_proteins,
        n_total_assays=n_total_assays,
        duplicates=duplicates,
    )


def compute_bridgeability(
    dataset1: AffinityDataset,
    dataset2: AffinityDataset,
    *,
    iqr_multiplier: float = 3.0,
    product1_name: str | None = None,
    product2_name: str | None = None,
) -> BridgeabilityData | None:
    """Compute cross-product bridgeability diagnostics for visualization.

    Wraps ``assess_cross_product_bridgeability`` and packages the result
    into a :class:`BridgeabilityData` suitable for the 4-panel plot.

    Returns None if there are no overlapping proteins.
    """
    from pyprideap.processing.normalization import assess_cross_product_bridgeability

    try:
        df = assess_cross_product_bridgeability(
            dataset1,
            dataset2,
            iqr_multiplier=iqr_multiplier,
        )
    except ValueError:
        return None

    if df.empty:
        return None

    name1 = product1_name or str(getattr(dataset1.platform, "value", "Product 1"))
    name2 = product2_name or str(getattr(dataset2.platform, "value", "Product 2"))

    recs = df["bridging_recommendation"].value_counts()

    return BridgeabilityData(
        protein_ids=df["protein_id"].tolist(),
        range_diffs=df["range_diff"].tolist(),
        r2_values=df["r2"].tolist(),
        ks_stats=df["ks_stat"].tolist(),
        low_cnts=df["low_cnt"].tolist(),
        recommendations=df["bridging_recommendation"].tolist(),
        n_bridgeable=int((df["is_bridgeable"]).sum()),
        n_not_bridgeable=int(recs.get("NotBridgeable", 0)),
        n_median_centering=int(recs.get("MedianCentering", 0)),
        n_quantile_smoothing=int(recs.get("QuantileSmoothing", 0)),
        product1_name=name1,
        product2_name=name2,
    )


# ---------------------------------------------------------------------------
# Technical QC: replicate-control CV, plate/batch effect, vendor QC flags
# ---------------------------------------------------------------------------

# Replicate controls of one material that are not used for normalization or
# calibration, so their CV is an independent estimate of technical precision.
# Olink PLATE_CONTROLs feed plate normalization and SomaScan Calibrators feed
# calibration; both would understate the CV and are not used.
_TECHNICAL_REPLICATE_TYPES = {
    "olink": ("sample control", "Sample controls"),
    "somascan": ("qc", "QC samples"),
}
_MIN_REPLICATES = 3


@dataclass
class ReplicateCvData:
    """Technical CV across replicate controls next to the CV across study samples."""

    control_label: str
    n_replicates: int
    technical_cv: list[float]
    study_cv: list[float]
    n_assays_total: int
    lod_filtered: bool  # technical_cv excludes assays below LOD in the controls (where LOD is known)
    title: str = "Technical vs Study-Sample CV"


def compute_replicate_cv(dataset: AffinityDataset) -> ReplicateCvData | None:
    """CV of each assay across replicate control samples (Olink sample controls, SomaScan QC samples).

    Values are linearized the same way as :func:`compute_cv_distribution`
    (Olink 2^NPX, SomaScan RFU as deposited). When an LOD is available only
    assays whose median control value is above LOD are kept, because CVs of
    assays at noise level are not meaningful.
    """
    from pyprideap.processing.filtering import normalize_sample_type

    if "SampleType" not in dataset.samples.columns:
        return None
    family = "somascan" if dataset.platform == Platform.SOMASCAN else "olink"
    control_type, label = _TECHNICAL_REPLICATE_TYPES[family]
    is_rep = (normalize_sample_type(dataset.samples["SampleType"]) == control_type).to_numpy()
    n_rep = int(is_rep.sum())
    if n_rep < _MIN_REPLICATES:
        return None

    numeric = dataset.expression.apply(pd.to_numeric, errors="coerce")
    controls = numeric.loc[is_rep]
    linear = np.power(2, controls) if family == "olink" else controls
    cv = (linear.std() / linear.mean()).replace([np.inf, -np.inf], np.nan)

    lod = _resolve_lod(dataset)
    lod_filtered = lod is not None
    if lod is not None:
        if isinstance(lod, pd.DataFrame):
            lod_ctrl = lod.reindex(columns=numeric.columns).loc[is_rep].median()
        else:
            lod_ctrl = pd.Series(lod).reindex(numeric.columns)
        # Assays without a known LOD (e.g. missing from the FixedLOD reference) are kept
        lod_filtered = bool(lod_ctrl.notna().any())
        detected = (controls.median() > lod_ctrl) | lod_ctrl.isna()
        cv = cv[detected]
    cv = cv.dropna()
    if cv.empty:
        return None

    study = compute_cv_distribution(dataset)
    return ReplicateCvData(
        control_label=label,
        n_replicates=n_rep,
        technical_cv=cv.round(4).tolist(),
        study_cv=study.cv_values if study is not None else [],
        n_assays_total=int(numeric.shape[1]),
        lod_filtered=lod_filtered,
    )


@dataclass
class BatchEffectData:
    """Association of the main expression structure with plate, and per-plate signal."""

    pc_labels: list[str]
    variance_explained: list[float]  # fraction of total variance per PC
    plate_r2: list[float]  # fraction of each PC's variance explained by plate (eta squared)
    plate_ids: list[str]
    plate_sample_medians: list[list[float]]  # per plate: per-sample median signal
    value_label: str
    title: str = "Plate Effect on Principal Components"
    # Same quantities after plate median centring (each assay's per-plate median moved
    # to its overall median), to show residual plate structure
    corrected_sample_medians: list[list[float]] = field(default_factory=list)
    corrected_plate_r2: list[float] = field(default_factory=list)


def _plate_column(samples: pd.DataFrame) -> str | None:
    return next((c for c in ("PlateId", "PlateID") if c in samples.columns), None)


def _plate_pca_r2(numeric: pd.DataFrame, plates: pd.Series, n_components: int) -> tuple[list[float], list[float]]:
    """Variance explained by the top PCs, and the share of each PC explained by plate (eta squared).

    Missing values are median-imputed per assay before the PCA.
    """
    filled = numeric.fillna(numeric.median())
    centered = filled.to_numpy(dtype=float) - filled.to_numpy(dtype=float).mean(axis=0)
    n_comp = min(n_components, centered.shape[0] - 1, centered.shape[1])
    try:
        from sklearn.decomposition import PCA

        pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=0)
        scores = pca.fit_transform(centered)
        var_ratio = pca.explained_variance_ratio_
    except ImportError:
        u, sv, _ = np.linalg.svd(centered, full_matrices=False)
        scores = u[:, :n_comp] * sv[:n_comp]
        var_ratio = (sv**2 / np.sum(sv**2))[:n_comp]

    codes = pd.Categorical(plates).codes
    r2 = []
    for k in range(n_comp):
        pc = scores[:, k]
        total = float(np.sum((pc - pc.mean()) ** 2))
        between = sum(
            float((codes == g).sum()) * float(pc[codes == g].mean() - pc.mean()) ** 2 for g in np.unique(codes)
        )
        r2.append(round(between / total, 4) if total > 0 else 0.0)
    return [float(v) for v in var_ratio], r2


def compute_batch_effect(dataset: AffinityDataset, n_components: int = 5) -> BatchEffectData | None:
    """How much of each top principal component is explained by plate (study samples only).

    PCA uses Olink NPX or SomaScan log10 RFU, assays with at most 50% missing
    values (median-imputed). For each PC, eta squared = between-plate sum of
    squares / total sum of squares of the PC scores.
    """
    from pyprideap.processing.filtering import filter_controls

    plate_col = _plate_column(dataset.samples)
    if plate_col is None:
        return None
    ds = filter_controls(dataset)
    plates = ds.samples[plate_col].astype("string")
    counts = plates.value_counts()
    keep_plates = counts[counts >= 3].index
    mask = plates.isin(keep_plates).to_numpy()
    if len(keep_plates) < 2 or mask.sum() < 6:
        return None

    numeric = ds.expression.apply(pd.to_numeric, errors="coerce").loc[mask]
    plates = plates[mask].reset_index(drop=True)
    if dataset.platform == Platform.SOMASCAN:
        numeric = np.log10(numeric.where(numeric > 0))
        value_label = "Median log10 RFU"
    else:
        value_label = "Median NPX"
    numeric = numeric.loc[:, numeric.isna().mean() <= 0.5].reset_index(drop=True)
    if numeric.shape[1] < 2:
        return None

    var_ratio, r2 = _plate_pca_r2(numeric, plates, n_components)

    # Plate median centring: per assay, shift each plate's median to the overall median
    plate_medians = numeric.groupby(plates.to_numpy()).transform("median")
    corrected = numeric - plate_medians + numeric.median()
    _, corrected_r2 = _plate_pca_r2(corrected, plates, n_components)

    sample_medians = numeric.median(axis=1)
    plate_ids = sorted(plates.unique(), key=str)
    return BatchEffectData(
        pc_labels=[f"PC{k + 1}" for k in range(len(r2))],
        variance_explained=[round(float(v), 4) for v in var_ratio],
        plate_r2=r2,
        plate_ids=[str(p) for p in plate_ids],
        plate_sample_medians=[sample_medians[plates == p].round(4).tolist() for p in plate_ids],
        value_label=value_label,
        corrected_sample_medians=[corrected.median(axis=1)[plates == p].round(4).tolist() for p in plate_ids],
        corrected_plate_r2=corrected_r2,
    )


_QC_STATUS_ORDER = ("PASS", "WARN", "FAIL", "NA")


@dataclass
class QcFlagData:
    """Vendor per-measurement QC flags summarised per panel (Olink AssayQC / SampleQC)."""

    rows: list[str]  # e.g. "Explore_HT · Assay QC"
    status_pct: dict[str, list[float]]  # status -> % of measurements per row
    flagged_assay_pct: float | None  # % of measurements with assay QC WARN/FAIL (study samples)
    sample_worst: dict[str, int]  # per-sample worst sample-QC status across blocks
    title: str = "Vendor QC Flags"


def _status_counts(matrix: pd.DataFrame) -> dict[str, int]:
    # Flatten via numpy (DataFrame.stack(future_stack=...) needs pandas >= 2.1)
    vals = pd.Series(matrix.to_numpy().ravel(), dtype="object").fillna("NA").astype(str)
    vals = vals.where(vals.isin(_QC_STATUS_ORDER[:3]), "NA")
    counts = vals.value_counts()
    return {s: int(counts.get(s, 0)) for s in _QC_STATUS_ORDER}


def compute_qc_flags(dataset: AffinityDataset) -> QcFlagData | None:
    """Summarise Olink per-measurement assay and sample QC flags per panel (study samples only)."""
    from pyprideap.processing.filtering import filter_controls

    ds = filter_controls(dataset)
    matrices = {
        name: m
        for name, key in (("Assay QC", "assay_qc_matrix"), ("Sample QC", "sample_qc_matrix"))
        if isinstance(m := ds.metadata.get(key), pd.DataFrame) and m.shape == ds.expression.shape
    }
    if not matrices:
        return None

    panels = (
        ds.features["Panel"].astype(str).to_numpy()
        if "Panel" in ds.features.columns and len(ds.features) == ds.expression.shape[1]
        else np.array(["All assays"] * ds.expression.shape[1])
    )
    rows: list[str] = []
    status_pct: dict[str, list[float]] = {s: [] for s in _QC_STATUS_ORDER}
    for panel in sorted(set(panels)):
        cols = panels == panel
        for name, m in matrices.items():
            counts = _status_counts(m.loc[:, cols])
            total = sum(counts.values())
            if total == 0:
                continue
            rows.append(f"{panel} · {name}")
            for st in _QC_STATUS_ORDER:
                status_pct[st].append(round(100 * counts[st] / total, 2))

    flagged = None
    if "Assay QC" in matrices:
        c = _status_counts(matrices["Assay QC"])
        assessed = c["PASS"] + c["WARN"] + c["FAIL"]
        flagged = round(100 * (c["WARN"] + c["FAIL"]) / assessed, 2) if assessed else None

    worst: dict[str, int] = {}
    if "Sample QC" in matrices:
        rank = {"PASS": 0, "WARN": 1, "FAIL": 2}
        per_sample = matrices["Sample QC"].apply(
            lambda r: max((rank.get(str(v), -1) for v in r.dropna()), default=-1), axis=1
        )
        names: dict[Hashable, str] = {0: "PASS", 1: "WARN", 2: "FAIL", -1: "NA"}
        worst = {names[k]: int(v) for k, v in per_sample.value_counts().sort_index().items()}

    return QcFlagData(rows=rows, status_pct=status_pct, flagged_assay_pct=flagged, sample_worst=worst)


# ---------------------------------------------------------------------------
# Pre-analytical indicators, SomaScan dilution breakdown, sex consistency
# ---------------------------------------------------------------------------

# Marker proteins (gene symbols). Erythrocyte proteins rise with in-vitro hemolysis;
# platelet alpha-granule / surface proteins rise with platelet activation during
# collection or with serum vs plasma handling. These are indicators, not diagnoses.
_PREANALYTICAL_MARKERS = {
    "hemolysis": ("HBA1", "HBA2", "HBB", "HBD", "HBQ1", "CA1", "CA2", "PRDX2", "BLVRB"),
    "platelet": ("PF4", "PPBP", "SELP", "GP1BA", "ITGA2B"),
}
# Male-specific: Y-chromosome genes and prostate-specific KLK3 (PSA). Female-higher: PZP.
# LHB/FSHB/CGA are not used: their levels track menopause rather than sex.
_SEX_MARKERS = {
    "male": ("KLK3", "EIF1AY", "DDX3Y", "RPS4Y1", "KDM5D", "UTY", "NLGN4Y", "ZFY", "USP9Y"),
    "female": ("PZP",),
}
_MIN_MARKERS = 2
_ROBUST_Z_OUTLIER = 3.0


def _gene_symbols(dataset: AffinityDataset) -> pd.Series:
    """Gene symbol(s) per feature, aligned with expression columns (Olink Assay, SomaScan EntrezGeneSymbol)."""
    for col in ("EntrezGeneSymbol", "Assay"):
        if col in dataset.features.columns and len(dataset.features) == dataset.expression.shape[1]:
            return dataset.features[col].astype("string").fillna("")
    return pd.Series([""] * dataset.expression.shape[1], dtype="string")


def _log_study_matrix(dataset: AffinityDataset) -> tuple[AffinityDataset, pd.DataFrame]:
    """Study samples and their log-scale matrix (Olink NPX, SomaScan log10 RFU)."""
    from pyprideap.processing.filtering import filter_controls

    ds = filter_controls(dataset)
    numeric = ds.expression.apply(pd.to_numeric, errors="coerce")
    if ds.platform == Platform.SOMASCAN:
        numeric = np.log10(numeric.where(numeric > 0))
    return ds, numeric


def _marker_columns(symbols: pd.Series, markers: tuple[str, ...]) -> dict[str, int]:
    """Map each marker gene to the first feature column whose symbol list contains only that gene."""
    found: dict[str, int] = {}
    for idx, sym in enumerate(symbols):
        genes = [g for g in str(sym).replace(",", "|").split("|") if g.strip()]
        # Multi-target reagents (e.g. "HBA1|HBA2") are accepted when all their genes are markers
        if genes and all(g.strip() in markers for g in genes):
            for g in genes:
                found.setdefault(g.strip(), idx)
    return found


def _marker_score(numeric: pd.DataFrame, columns: list[int]) -> pd.Series:
    """Mean of per-marker z-scores across samples (NaN-tolerant)."""
    sub = numeric.iloc[:, sorted(set(columns))]
    z = (sub - sub.mean()) / sub.std(ddof=0).replace(0, np.nan)
    return cast(pd.Series, z.mean(axis=1))


def _robust_z(values: pd.Series) -> pd.Series:
    med = values.median()
    mad = (values - med).abs().median() * _MAD_TO_SD_FACTOR
    return (values - med) / mad if mad and mad > 0 else values * 0.0


_MAD_TO_SD_FACTOR = 1.4826


@dataclass
class PreanalyticalData:
    """Per-sample hemolysis and platelet-activation indicator scores (study samples)."""

    sample_ids: list[str]
    scores: dict[str, list[float]]  # indicator -> per-sample score (mean marker z-score)
    markers: dict[str, list[str]]  # indicator -> marker genes used
    outliers: dict[str, list[str]]  # indicator -> sample ids with robust z > threshold
    threshold: float = _ROBUST_Z_OUTLIER
    title: str = "Pre-analytical Indicators"


def compute_preanalytical(dataset: AffinityDataset) -> PreanalyticalData | None:
    """Hemolysis and platelet-activation indicator scores from marker proteins.

    Each indicator is the mean z-score of its marker proteins present on the
    panel (at least two required). Samples whose score is more than 3 robust
    SDs (median/MAD) above the median are listed as outliers.
    """
    ds, numeric = _log_study_matrix(dataset)
    if numeric.shape[0] < 5:
        return None
    symbols = _gene_symbols(ds)
    scores: dict[str, list[float]] = {}
    markers: dict[str, list[str]] = {}
    outliers: dict[str, list[str]] = {}
    sample_ids = _sample_ids(ds)
    for name, genes in _PREANALYTICAL_MARKERS.items():
        cols = _marker_columns(symbols, genes)
        # Count distinct reagents: a multi-target reagent (e.g. "HBA1|HBA2") is one measurement
        if len(set(cols.values())) < _MIN_MARKERS:
            continue
        score = _marker_score(numeric, list(cols.values()))
        rz = _robust_z(score)
        scores[name] = score.round(4).tolist()
        markers[name] = sorted(cols)
        outliers[name] = [sid for sid, z in zip(sample_ids, rz) if pd.notna(z) and z > _ROBUST_Z_OUTLIER]
    if not scores:
        return None
    return PreanalyticalData(sample_ids=sample_ids, scores=scores, markers=markers, outliers=outliers)


@dataclass
class DilutionQcData:
    """SomaScan QC broken down by dilution bin."""

    dilutions: list[str]  # e.g. ["20%", "0.5%", "0.005%"]
    n_assays: list[int]
    study_cv: list[list[float]]  # per dilution: per-assay CV across study samples
    technical_cv: list[list[float]]  # per dilution: per-assay CV across QC samples (may be empty)
    above_lod_pct: list[float | None]  # per dilution: % of assays above LOD in >50% of study samples
    norm_scale: list[list[float]]  # per dilution: per-sample NormScale_<dilution>, if present
    title: str = "QC by Dilution"


def _dilution_label(value: str) -> str:
    return f"{value}%"


def compute_dilution_qc(dataset: AffinityDataset) -> DilutionQcData | None:
    """CV, technical CV, LOD detectability and normalization scale per SomaScan dilution bin."""
    if dataset.platform != Platform.SOMASCAN or "Dilution" not in dataset.features.columns:
        return None
    if len(dataset.features) != dataset.expression.shape[1]:
        return None
    dil = dataset.features["Dilution"].astype(str).str.strip().to_numpy()
    bins = sorted({d for d in dil if d not in ("", "0", "nan")}, key=lambda d: -float(d))
    if len(bins) < 2:
        return None

    study = compute_cv_distribution(dataset)
    study_cv = pd.Series(study.cv_values, index=study.feature_ids) if study else pd.Series(dtype=float)
    rep = compute_replicate_cv(dataset)
    columns = dataset.expression.columns
    rep_cv = pd.Series(dtype=float)
    if rep is not None:
        # Recompute unfiltered per-assay technical CV with feature ids for grouping
        from pyprideap.processing.filtering import normalize_sample_type

        is_qc = (normalize_sample_type(dataset.samples["SampleType"]) == "qc").to_numpy()
        qc = dataset.expression.apply(pd.to_numeric, errors="coerce").loc[is_qc]
        rep_cv = (qc.std() / qc.mean()).replace([np.inf, -np.inf], np.nan)

    # Share of study samples above LOD per assay. The LOD is resolved on the full
    # dataset because SomaScan eLOD is estimated from buffer samples.
    from pyprideap.processing.filtering import control_sample_mask
    from pyprideap.processing.lod import _above_lod_matrix

    above = None
    lod = _resolve_lod(dataset)
    if lod is not None:
        is_study = ~control_sample_mask(dataset.samples).to_numpy()
        numeric_all = dataset.expression.apply(pd.to_numeric, errors="coerce")
        above_m, has_lod = _above_lod_matrix(numeric_all, lod)
        valid = (numeric_all.notna() & has_lod).loc[is_study]
        n_valid = valid.sum()
        above = ((above_m & has_lod).loc[is_study].sum() / n_valid.where(n_valid > 0)) * 100
        above.index = above.index.astype(str)

    out = DilutionQcData([], [], [], [], [], [])
    for b in bins:
        cols = columns[dil == b]
        out.dilutions.append(_dilution_label(b))
        out.n_assays.append(len(cols))
        out.study_cv.append(study_cv.reindex(cols).dropna().round(4).tolist())
        out.technical_cv.append(rep_cv.reindex(cols).dropna().round(4).tolist())
        if above is not None:
            vals = above.reindex([str(c) for c in cols]).dropna()
            out.above_lod_pct.append(round(float((vals > 50).mean() * 100), 1) if len(vals) else None)
        else:
            out.above_lod_pct.append(None)
        ns_col = "NormScale_" + b.replace(".", "_")
        if ns_col in dataset.samples.columns:
            out.norm_scale.append(pd.to_numeric(dataset.samples[ns_col], errors="coerce").dropna().round(4).tolist())
        else:
            out.norm_scale.append([])
    return out


@dataclass
class SexCheckData:
    """Protein-based sex score per sample, optionally compared with annotated sex."""

    sample_ids: list[str]
    score: list[float]  # high = male-like
    predicted: list[str]  # "male" / "female"
    annotated: list[str]  # annotated sex or "" when not available
    markers: dict[str, list[str]]  # markers used in the score ("male" / "female" direction)
    mismatches: list[str]  # sample ids where prediction and annotation disagree
    threshold: float
    # Candidate markers on the panel and how well each separates the sexes:
    # AUC in its expected direction when annotation is available, else NaN
    marker_auc: dict[str, float] = field(default_factory=dict)
    title: str = "Sex Consistency"


def _annotated_sex(samples: pd.DataFrame) -> list[str]:
    for col in ("sex", "Sex", "gender", "Gender"):
        if col in samples.columns:
            vals = samples[col].astype("string").str.strip().str.lower().fillna("")
            return [v if v in ("male", "female") else "" for v in vals]
    return [""] * len(samples)


_SEX_MIN_ASHMAN_D = 3.0  # separation of the two mixture components
_SEX_MIN_GROUP_FRACTION = 0.05
_SEX_MIN_BIC_GAIN = 10.0


def _two_group_split(values: np.ndarray) -> float | None:
    """Threshold between two clearly separated groups, or None when the data are not bimodal.

    Fits 1- and 2-component Gaussian mixtures; requires the 2-component model to
    improve BIC by > 10, Ashman's D >= 3 and the smaller component to hold >= 5%
    of samples. The threshold is where the two weighted components cross.
    """
    try:
        from sklearn.mixture import GaussianMixture
    except ImportError:
        return None
    x = values.reshape(-1, 1)
    one = GaussianMixture(1, random_state=0).fit(x)
    two = GaussianMixture(2, random_state=0, n_init=3).fit(x)
    if one.bic(x) - two.bic(x) < _SEX_MIN_BIC_GAIN or two.weights_.min() < _SEX_MIN_GROUP_FRACTION:
        return None
    mu = two.means_.ravel()
    sd = np.sqrt(two.covariances_.ravel())
    ashman_d = np.sqrt(2) * abs(mu[0] - mu[1]) / np.sqrt(sd[0] ** 2 + sd[1] ** 2)
    if ashman_d < _SEX_MIN_ASHMAN_D:
        return None
    grid = np.linspace(mu.min(), mu.max(), 512).reshape(-1, 1)
    labels = two.predict(grid)
    change = np.nonzero(np.diff(labels))[0]
    return float(grid[change[0] + 1, 0]) if len(change) else float(mu.mean())


# Minimum AUC (annotated groups, expected direction) for a marker to enter the score.
# Reagent specificity differs between platforms and versions: e.g. on SomaScan v4
# (PAD000003) KLK3 separates sexes with AUC 0.98 but the EIF1AY and NLGN4Y reagents
# do not (0.49, 0.61), and averaging them in drowns the KLK3 signal.
_SEX_MIN_MARKER_AUC = 0.8


def _auc(values: pd.Series, positive: np.ndarray) -> float:
    """Mann-Whitney AUC: probability that a positive sample has a higher value than a negative one."""
    v = values.to_numpy(dtype=float)
    ok = ~np.isnan(v)
    v, pos = v[ok], positive[ok]
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(v).rank().to_numpy()
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _youden_threshold(high: np.ndarray, low: np.ndarray) -> float:
    """Cut that maximises sensitivity + specificity for separating *high* from *low*."""
    candidates = np.unique(np.concatenate([high, low]))
    return float(max(candidates, key=lambda t: (high > t).mean() + (low <= t).mean()))


def _sex_mismatches(
    ids: list[str], score: np.ndarray, annotated: np.ndarray, male: np.ndarray, female: np.ndarray
) -> list[str]:
    """Samples whose score is an outlier for their annotated sex and typical of the other sex."""

    def robust_z(v: float, ref: np.ndarray) -> float:
        med = float(np.median(ref))
        mad = float(np.median(np.abs(ref - med))) * _MAD_TO_SD_FACTOR
        return (v - med) / mad if mad > 0 else 0.0

    m_lo, m_hi = np.percentile(male, [5, 95])
    f_lo, f_hi = np.percentile(female, [5, 95])
    out = []
    for sid, a, v in zip(ids, annotated, score):
        if np.isnan(v):
            continue
        if a == "female" and robust_z(v, female) > _ROBUST_Z_OUTLIER and m_lo <= v <= m_hi:
            out.append(sid)
        elif a == "male" and robust_z(v, male) < -_ROBUST_Z_OUTLIER and f_lo <= v <= f_hi:
            out.append(sid)
    return out


def compute_sex_check(dataset: AffinityDataset) -> SexCheckData | None:
    """Protein-based sex score, compared with annotated sex when a ``sex`` column is present.

    Candidate markers are Y-linked genes and KLK3 (higher in males) and PZP
    (higher in females). Only markers that actually separate the sexes in this
    dataset enter the score (mean z of male markers minus mean z of female markers):

    * With annotated sex (>= 3 per group): markers with AUC >= 0.8 for the
      annotated groups. The plotted threshold maximises sensitivity +
      specificity (Youden). A sample is reported as a mismatch only when its
      score is an outlier for its own annotated group (robust z > 3 towards the
      other sex) and lies within the other group's central 90%. Wide normal
      variation (e.g. low PSA in some men) is therefore not flagged.
    * Without annotation: male markers that are individually bimodal (see
      :func:`_two_group_split`); samples are split only when the combined
      score is bimodal too.

    Returns None when no marker qualifies, so uninformative panels and
    single-sex cohorts produce no plot rather than false sample swaps.
    """
    ds, numeric = _log_study_matrix(dataset)
    if numeric.shape[0] < 10:
        return None
    symbols = _gene_symbols(ds)
    candidates = {
        **{g: (i, "male") for g, i in _marker_columns(symbols, _SEX_MARKERS["male"]).items()},
        **{g: (i, "female") for g, i in _marker_columns(symbols, _SEX_MARKERS["female"]).items()},
    }
    if not any(direction == "male" for _, direction in candidates.values()):
        return None

    annotated = _annotated_sex(ds.samples)
    ann = np.array(annotated)
    has_annotation = (ann == "male").sum() >= 3 and (ann == "female").sum() >= 3

    marker_auc: dict[str, float] = {}
    used: dict[str, list[int]] = {"male": [], "female": []}
    for gene, (idx, direction) in sorted(candidates.items()):
        values = numeric.iloc[:, idx]
        if has_annotation:
            labelled = ann != ""
            auc = _auc(values[labelled], ann[labelled] == direction)
            marker_auc[gene] = round(auc, 3)
            if auc >= _SEX_MIN_MARKER_AUC:
                used[direction].append(idx)
        else:
            marker_auc[gene] = float("nan")
            if direction == "male" and _two_group_split(values.dropna().to_numpy()) is not None:
                used["male"].append(idx)
    if not used["male"] and not used["female"]:
        return None

    score = pd.Series(0.0, index=numeric.index)
    if used["male"]:
        score = score + _marker_score(numeric, used["male"])
    if used["female"]:
        score = score - _marker_score(numeric, used["female"])
    valid = score.notna().to_numpy()
    if valid.sum() < 10:
        return None

    if has_annotation:
        male_scores = score[(ann == "male") & valid].to_numpy()
        female_scores = score[(ann == "female") & valid].to_numpy()
        threshold = _youden_threshold(male_scores, female_scores)
    else:
        split = _two_group_split(score[valid].to_numpy())
        if split is None:
            return None
        threshold = split

    predicted = ["male" if (pd.notna(v) and v > threshold) else ("female" if pd.notna(v) else "") for v in score]
    ids = _sample_ids(ds)
    mismatches = _sex_mismatches(ids, score.to_numpy(), ann, male_scores, female_scores) if has_annotation else []
    names = {idx: gene for gene, (idx, _) in candidates.items()}
    return SexCheckData(
        sample_ids=ids,
        score=score.round(4).tolist(),
        predicted=predicted,
        annotated=annotated,
        markers={d: sorted(names[i] for i in used[d]) for d in ("male", "female")},
        mismatches=mismatches,
        threshold=round(float(threshold), 4),
        marker_auc=marker_auc,
    )


# ---------------------------------------------------------------------------
# Bridging samples and reanalysis readiness
# ---------------------------------------------------------------------------


@dataclass
class BridgeAgreementData:
    """Agreement of samples measured on more than one plate (bridging / replicate runs)."""

    n_samples: int
    plates: list[str]
    median_offset: float  # median over sample-assays of (later plate - reference plate)
    offset_iqr: tuple[float, float]
    assay_offsets: list[float]  # per-assay median offset
    median_correlation: float  # per-sample correlation between its runs
    value_label: str  # "NPX" or "log10 RFU"


def compute_bridge_agreement(dataset: AffinityDataset) -> BridgeAgreementData | None:
    """Plate-to-plate offset for samples measured on several plates.

    Each bridging sample's values on a plate are compared with the same sample
    on the first plate where it appears (Olink NPX, SomaScan log10 RFU). After
    a successful bridge normalisation the offsets should be centred on 0.
    """
    plate_col = _plate_column(dataset.samples)
    id_col = next((c for c in ("SampleID", "SampleId") if c in dataset.samples.columns), None)
    if plate_col is None or id_col is None:
        return None
    ids = dataset.samples[id_col].astype(str)
    plates = dataset.samples[plate_col].astype(str)
    multi = ids.groupby(ids).transform(lambda s: plates[s.index].nunique()) > 1
    if not multi.any():
        return None

    numeric = dataset.expression.apply(pd.to_numeric, errors="coerce")
    if dataset.platform == Platform.SOMASCAN:
        numeric = np.log10(numeric.where(numeric > 0))
        value_label = "log10 RFU"
    else:
        value_label = "NPX"

    diffs: list[pd.Series] = []
    correlations: list[float] = []
    for _sid, rows in ids[multi].groupby(ids[multi]):
        idx = list(rows.index)
        order = sorted(idx, key=lambda i: plates[i])
        ref = numeric.loc[order[0]]
        for other in order[1:]:
            d = (numeric.loc[other] - ref).dropna()
            if len(d):
                diffs.append(d)
            ok = ref.notna() & numeric.loc[other].notna()
            if ok.sum() >= 3:
                correlations.append(float(np.corrcoef(ref[ok], numeric.loc[other][ok])[0, 1]))
    if not diffs:
        return None
    all_diffs = pd.concat(diffs)
    per_assay = pd.concat(diffs, axis=1).median(axis=1).dropna()
    return BridgeAgreementData(
        n_samples=int(ids[multi].nunique()),
        plates=sorted(plates[multi].unique()),
        median_offset=round(float(all_diffs.median()), 3),
        offset_iqr=(round(float(all_diffs.quantile(0.25)), 3), round(float(all_diffs.quantile(0.75)), 3)),
        assay_offsets=per_assay.round(3).tolist(),
        median_correlation=round(float(np.median(correlations)), 3) if correlations else float("nan"),
        value_label=value_label,
    )


# Median plate offset of bridging samples above which the card notes that the
# deposited values are not bridge-normalised (Olink NPX / SomaScan log10 RFU)
_BRIDGE_OFFSET_NOTE = 0.2


@dataclass
class ReadinessItem:
    category: str
    status: str  # "available" | "partial" | "missing" | "n/a"
    detail: str
    impact: str = ""  # what cannot be checked or reproduced when not available


@dataclass
class ReadinessData:
    """Which metadata needed to interpret or reanalyse the data is present."""

    items: list[ReadinessItem]
    title: str = "Reanalysis Readiness"

    @property
    def n_available(self) -> int:
        return sum(1 for i in self.items if i.status == "available")

    @property
    def n_applicable(self) -> int:
        return sum(1 for i in self.items if i.status != "n/a")


def _first_column(frame: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    lower = {c.lower(): c for c in frame.columns}
    return next((lower[n] for n in names if n in lower), None)


def _non_empty_values(series: pd.Series) -> pd.Series:
    vals = series.astype(str).str.strip()
    return vals[~vals.str.lower().isin({"", "nan", "none", "not available", "not applicable", "na"})]


def compute_readiness(dataset: AffinityDataset, results: dict[str, object] | None = None) -> ReadinessData:
    """Check the metadata needed to interpret the QC and reproduce the analyses.

    Each item reports whether the information is available in the data file or
    a merged SDRF, and what cannot be checked without it. There is no overall
    pass/fail: the card states what is known and what is not.
    """
    from pyprideap.io.readers.sdrf import resolve_biological_groups
    from pyprideap.processing.filtering import filter_controls

    results = results or {}
    samples = dataset.samples
    items: list[ReadinessItem] = []

    # 1. Sample types (controls)
    st_col = _first_column(samples, ("sampletype", "sample type"))
    if st_col is not None and len(_non_empty_values(samples[st_col])):
        counts = _non_empty_values(samples[st_col]).value_counts()
        detail = ", ".join(f"{html_escape(str(k))}: {v}" for k, v in counts.head(6).items())
        items.append(ReadinessItem("Sample types (controls)", "available", detail))
    else:
        items.append(
            ReadinessItem(
                "Sample types (controls)",
                "missing",
                "No SampleType column and no SDRF sample type",
                "Control samples cannot be separated from study samples: technical CV and "
                "negative-control LOD are unavailable, and controls enter every plot.",
            )
        )

    # 2. Plates / batches
    plate_col = _plate_column(samples)
    n_plates = int(samples[plate_col].nunique()) if plate_col else 0
    if plate_col:
        items.append(ReadinessItem("Plates / batches", "available", f"{n_plates} plate(s)"))
    else:
        items.append(
            ReadinessItem("Plates / batches", "missing", "No plate column", "Batch (plate) effects cannot be assessed.")
        )

    # 3. Limit of detection
    _lod, lod_source = resolve_lod_with_source(dataset)
    if lod_source:
        items.append(ReadinessItem("Limit of detection", "available", lod_source))
    else:
        items.append(
            ReadinessItem(
                "Limit of detection",
                "missing",
                "No LOD column, negative controls, buffers or reference LOD",
                "Detectability cannot be assessed; values near the noise floor cannot be told apart.",
            )
        )

    # 4. Vendor QC flags
    flag_labels = {"assay_qc_matrix": "assay QC per measurement", "sample_qc_matrix": "sample QC per measurement"}
    flags = [c for c in ("SampleQC", "QC_Warning", "RowCheck", "HybControlNormScale") if c in samples.columns] + [
        label for key, label in flag_labels.items() if key in dataset.metadata
    ]
    if "ColCheck" in dataset.features.columns:
        flags.append("ColCheck")
    if flags:
        items.append(ReadinessItem("Vendor QC flags", "available", ", ".join(sorted(set(flags)))))
    else:
        items.append(
            ReadinessItem(
                "Vendor QC flags",
                "missing",
                "No sample or assay QC flags",
                "Measurements the vendor software flagged as failed cannot be identified.",
            )
        )

    # 5. SDRF linkage
    merge = dataset.metadata.get("sdrf_merge")
    if isinstance(merge, dict):
        matched, total = merge.get("matched", 0), merge.get("total", len(samples))
        status = "available" if matched == total else ("partial" if matched else "missing")
        detail = f"SDRF provided; {matched} of {total} samples matched"
        if merge.get("error"):
            detail += f" (merge failed: {html_escape(str(merge['error']))})"
        elif not matched:
            detail += " (sample names in the SDRF and the data file differ)"
        items.append(
            ReadinessItem(
                "SDRF linked to samples",
                status,
                detail,
                "" if status == "available" else "Unmatched samples have no annotation (groups, sex, age).",
            )
        )
    else:
        items.append(
            ReadinessItem(
                "SDRF linked to samples",
                "missing",
                "No SDRF provided or found",
                "Sample annotation (groups, covariates, replicates) is not available to the QC.",
            )
        )

    # 6. Study design groups
    resolved = resolve_biological_groups(filter_controls(dataset))
    if resolved is not None:
        labels, column = resolved
        counts = _non_empty_values(labels).value_counts()
        factor_columns = dataset.metadata.get("sdrf_factor_columns")
        is_factor = isinstance(factor_columns, list) and column in factor_columns
        base_name = re.sub(r" \d+$", "", column)
        label = f"factor value[{base_name}]" if is_factor else column
        detail = f"{html_escape(label)}: " + ", ".join(f"{html_escape(str(k))} {v}" for k, v in counts.items())
        status = "partial" if column == "sample identifier" else "available"
        impact = "Groups inferred from sample names, not from annotation." if status == "partial" else ""
        items.append(ReadinessItem("Study groups", status, detail, impact))
    else:
        no_factor = bool(dataset.metadata.get("sdrf_without_factor_values"))
        items.append(
            ReadinessItem(
                "Study groups",
                "missing",
                "The SDRF has no factor value[...] column, so the study design is not declared"
                if no_factor
                else "No grouping column (e.g. disease, factor value) with 2–10 groups",
                "The comparisons reported in the publication cannot be related to the samples; "
                "within-group precision and volcano plots are not computed.",
            )
        )

    # 7. Replicates / bridging
    bridge = results.get("bridge_agreement")
    rep_col = _first_column(samples, ("comment[technical replicate]", "technical replicate"))
    if isinstance(bridge, BridgeAgreementData):
        offset_note = (
            f"The same samples differ by {bridge.median_offset:+.2f} {bridge.value_label} between plates: "
            "the deposited values are not bridge-normalised, so plate and any groups confined to one "
            "plate are confounded."
            if abs(bridge.median_offset) > _BRIDGE_OFFSET_NOTE
            else ""
        )
        items.append(
            ReadinessItem(
                "Bridging / replicate samples",
                "available",
                f"{bridge.n_samples} samples measured on {len(bridge.plates)} plates; "
                f"median plate offset {bridge.median_offset:+.2f} {bridge.value_label}, "
                f"median correlation between runs {bridge.median_correlation:.3f}",
                offset_note,
            )
        )
    elif rep_col is not None and samples[rep_col].nunique() > 1:
        items.append(ReadinessItem("Bridging / replicate samples", "available", f"SDRF {html_escape(rep_col)}"))
    elif n_plates > 1:
        items.append(
            ReadinessItem(
                "Bridging / replicate samples",
                "missing",
                f"{n_plates} plates, no sample identifiable on more than one",
                "Whether plates were bridged, and how well, cannot be verified.",
            )
        )
    else:
        items.append(ReadinessItem("Bridging / replicate samples", "n/a", "Single plate"))

    # 8. Normalisation applied to the deposited values
    norm: list[str] = []
    for col in samples.columns:
        if "normalization" in col.lower() or "normalisation" in col.lower():
            vals = _non_empty_values(samples[col]).unique()
            norm.extend(f"{col}: {v}" for v in vals[:3])
    if "Normalization" in dataset.features.columns:
        norm.extend(f"Normalization: {v}" for v in _non_empty_values(dataset.features["Normalization"]).unique()[:3])
    for key in ("ProcessSteps", "NormalizationAlgorithm"):
        if dataset.metadata.get(key):
            norm.append(f"{key}: {dataset.metadata[key]}")
    if norm:
        items.append(ReadinessItem("Normalisation recorded", "available", html_escape("; ".join(norm[:4]))))
    else:
        items.append(
            ReadinessItem(
                "Normalisation recorded",
                "missing",
                "No normalisation method in the file or SDRF",
                "Which normalisation (e.g. bridging, intensity, ANML) produced the values is unknown.",
            )
        )

    # 9. Covariates
    have = [c for c in ("sex", "age") if _first_column(samples, (c, f"characteristics[{c}]")) is not None]
    if len(have) == 2:
        items.append(ReadinessItem("Sex and age", "available", "sex, age"))
    else:
        items.append(
            ReadinessItem(
                "Sex and age",
                "partial" if have else "missing",
                ", ".join(have) if have else "Neither annotated",
                "Common covariates cannot be checked (e.g. protein-predicted sex vs annotation, age effects).",
            )
        )

    # 10. Sample matrix
    matrix_col = _first_column(samples, ("sample matrix", "samplematrix", "organism part"))
    if matrix_col is not None and len(_non_empty_values(samples[matrix_col])):
        matrix_counts = _non_empty_values(samples[matrix_col]).value_counts()
        matrices = ", ".join(html_escape(str(k)) for k in matrix_counts.index[:3])
        items.append(ReadinessItem("Sample matrix", "available", matrices))
    else:
        items.append(
            ReadinessItem(
                "Sample matrix",
                "missing",
                "Not annotated",
                "Plasma, serum, CSF or other matrices cannot be distinguished when comparing datasets.",
            )
        )

    return ReadinessData(items=items)


def compute_all(dataset: AffinityDataset) -> dict[str, object]:
    """Compute all applicable QC plot data for the dataset."""
    logger.debug(
        "compute_all: expression matrix shape=%s, platform=%s", dataset.expression.shape, dataset.platform.value
    )
    results: dict[str, object] = {}
    results["distribution"] = compute_distribution(dataset)
    results["qc_summary"] = compute_qc_summary(dataset)
    results["lod_analysis"] = compute_lod_analysis(dataset)
    results["pca"] = compute_pca(dataset)
    results["umap"] = compute_tsne(dataset)
    results["heatmap"] = compute_heatmap(dataset)
    results["correlation"] = compute_correlation(dataset)
    results["data_completeness"] = compute_data_completeness(dataset)
    results["cv_distribution"] = compute_cv_distribution(dataset)
    results["plate_cv"] = compute_plate_cv(dataset)
    results["norm_scale"] = compute_norm_scale(dataset)
    results["lod_comparison"] = compute_lod_comparison(dataset)
    results["replicate_cv"] = compute_replicate_cv(dataset)
    results["batch_effect"] = compute_batch_effect(dataset)
    results["qc_flags"] = compute_qc_flags(dataset)
    results["preanalytical"] = compute_preanalytical(dataset)
    results["dilution_qc"] = compute_dilution_qc(dataset)
    results["sex_check"] = compute_sex_check(dataset)

    # SomaScan-specific QC
    if dataset.platform == Platform.SOMASCAN:
        results["outlier_map"] = compute_outlier_map(dataset)
        results["row_check"] = compute_row_check(dataset)
        results["col_check"] = compute_col_check(dataset)
        results["control_analytes"] = compute_control_analytes(dataset)
        results["norm_scale_boxplot"] = compute_norm_scale_boxplot(dataset)

    # Olink-specific QC
    if dataset.platform != Platform.SOMASCAN:
        results["iqr_median_qc"] = compute_iqr_median_qc(dataset)

    # UniProt duplicate detection (Olink and SomaScan when feature table has UniProt)
    results["uniprot_duplicates"] = compute_uniprot_duplicates(dataset)

    # Technology-agnostic QC (MPI, dynamic range, rank concordance)
    results.update(compute_agnostic_qc(dataset))

    results["bridge_agreement"] = compute_bridge_agreement(dataset)
    results["readiness"] = compute_readiness(dataset, results)

    available = {k: v for k, v in results.items() if v is not None}
    logger.debug("compute_all: %d/%d plots computed successfully", len(available), len(results))
    return available
