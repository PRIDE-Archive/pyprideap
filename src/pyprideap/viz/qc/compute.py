from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pyprideap.core import AffinityDataset, Platform


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
    title: str = "Data Completeness"


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


@dataclass
class LodComparisonData:
    """Pairwise LOD comparison scatter data."""

    pairs: list[dict]
    """Each dict has keys: name_x, name_y, assay_ids, values_x, values_y, panels."""
    title: str = "LOD Comparison"


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
    # SampleID exists but is not unique — try SampleName
    if "SampleName" in dataset.samples.columns:
        return "SampleName"
    # Last resort: return whichever ID column exists
    for col in ("SampleID", "SampleId"):
        if col in dataset.samples.columns:
            return col
    return "SampleID"


def _sample_ids(dataset: AffinityDataset) -> list[str]:
    col = _sample_id_col(dataset)
    if col in dataset.samples.columns:
        return dataset.samples[col].astype(str).tolist()
    return [f"S{i}" for i in range(len(dataset.samples))]


def compute_distribution(dataset: AffinityDataset) -> DistributionData:
    """Per-sample NPX/RFU value lists for overlaid density curves.

    Values are rounded to 2 decimal places to reduce HTML output size
    without visible loss of quality in the histogram plots.
    """
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


def compute_qc_summary(dataset: AffinityDataset) -> QcLodSummaryData | None:
    """QC status × LOD stacked bar. Falls back to simple QC counts if no LOD."""
    if "SampleQC" not in dataset.samples.columns:
        return None

    from pyprideap.processing.lod import _above_lod_matrix, get_lod_values

    lod = get_lod_values(dataset)
    numeric = dataset.expression.apply(pd.to_numeric, errors="coerce")

    if lod is not None and (isinstance(lod, pd.DataFrame) or len(lod) > 0):
        above_lod, has_lod = _above_lod_matrix(numeric, lod)

        categories: list[str] = []
        counts: list[int] = []

        for qc_val in ["PASS", "WARN", "FAIL"]:
            mask = dataset.samples["SampleQC"] == qc_val
            if mask.sum() == 0:
                continue

            above_subset = above_lod.loc[mask] & has_lod.loc[mask]
            valid_subset = numeric.loc[mask].notna() & has_lod.loc[mask]

            above = int(above_subset.sum().sum())
            below = int(valid_subset.sum().sum()) - above

            if above > 0:
                categories.append(f"{qc_val} & NPX > LOD")
                counts.append(above)
            if below > 0:
                categories.append(f"{qc_val} & NPX ≤ LOD")
                counts.append(below)

        if categories:
            return QcLodSummaryData(categories=categories, counts=counts)

    # Fallback: simple QC counts
    vc = dataset.samples["SampleQC"].value_counts()
    return QcLodSummaryData(categories=vc.index.tolist(), counts=vc.values.tolist())


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

    return LodAnalysisData(assay_ids=assay_ids, above_lod_pct=above_lod_pct, panel=panels)


def compute_pca(dataset: AffinityDataset, n_components: int = 2) -> PcaData | None:
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
    numeric = dataset.expression.apply(pd.to_numeric, errors="coerce")

    if numeric.shape[0] > max_samples:
        numeric = numeric.sample(n=max_samples, random_state=42)

    corr = numeric.T.corr()

    id_col = _sample_id_col(dataset)
    if id_col in dataset.samples.columns:
        labels = dataset.samples.loc[numeric.index, id_col].astype(str).tolist()
    else:
        labels = [f"S{i}" for i in range(len(numeric))]

    return CorrelationData(
        matrix=[[None if np.isnan(v) else round(v, 3) for v in row] for row in corr.values],
        labels=labels,
    )


def _resolve_lod(dataset: AffinityDataset) -> pd.DataFrame | pd.Series | None:
    """Try all LOD sources in priority order.

    Olink:    Reported → NCLOD → FixedLOD
    SomaScan: Reported → NCLOD → eLOD (buffer-based, MAD formula)
    """
    from pyprideap.processing.lod import (
        compute_nclod,
        compute_soma_elod,
        get_reported_lod,
        load_fixed_lod,
    )

    # 1. Reported LOD (from data file)
    lod = get_reported_lod(dataset)
    if lod is not None:
        return lod

    # 2. NCLOD (from negative controls)
    try:
        return compute_nclod(dataset, plate_adjusted=True)
    except (ValueError, KeyError):
        pass

    # 3. Platform-specific fallback
    if dataset.platform == Platform.SOMASCAN:
        # SomaScan eLOD from buffer samples
        try:
            return compute_soma_elod(dataset)
        except (ValueError, KeyError):
            pass
    else:
        # Olink FixedLOD from bundled config
        try:
            return load_fixed_lod(dataset)
        except (ValueError, FileNotFoundError):
            pass

    return None


def compute_data_completeness(dataset: AffinityDataset) -> DataCompletenessData | None:
    """Per-sample completeness (above/below LOD) and per-protein missing frequency.

    LOD resolution priority: Reported LOD → NCLOD → FixedLOD.
    Per-protein missing frequency uses Olink's MissingFreq column when
    available (fraction of samples with NPX below LOD), otherwise
    computed from the LOD matrix.

    Returns None when no LOD source is available.
    """
    from pyprideap.processing.lod import _above_lod_matrix

    numeric = dataset.expression.apply(pd.to_numeric, errors="coerce")
    sample_ids = _sample_ids(dataset)

    # Try to get per-protein missing frequency from Olink's MissingFreq column
    # MissingFreq = fraction of samples with NPX below LOD
    olink_missing_freq = None
    if "MissingFreq" in dataset.features.columns:
        mf = pd.to_numeric(dataset.features["MissingFreq"], errors="coerce")
        if mf.notna().any():
            olink_missing_freq = mf

    # Resolve LOD from all available sources
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
                above_lod_rate.append(1.0)
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


def compute_cv_distribution(dataset: AffinityDataset) -> CvDistributionData | None:
    numeric = dataset.expression.apply(pd.to_numeric, errors="coerce")

    # Olink NPX values are log2-scale — CV = SD/mean is meaningless on log data.
    # Convert to linear scale (2^NPX) before computing CV.
    # SomaScan RFU values are already linear.
    if dataset.platform != Platform.SOMASCAN:
        numeric = np.power(2, numeric)

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

    Intra-plate CV: for each plate, CV = SD / mean per analyte across samples.
    Returned in long format (one entry per analyte per plate).

    Inter-plate CV: for each analyte, CV of plate medians across plates.
    One value per analyte.

    Only applicable when PlateId column exists with >= 2 plates.
    """
    if "PlateId" not in dataset.samples.columns:
        return None

    plates = dataset.samples["PlateId"]
    unique_plates = sorted(plates.unique(), key=str)
    if len(unique_plates) < 2:
        return None

    numeric = dataset.expression.apply(pd.to_numeric, errors="coerce")

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

    return LodComparisonData(pairs=pairs)


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


def compute_all(dataset: AffinityDataset) -> dict[str, object]:
    """Compute all applicable QC plot data for the dataset."""
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
    return {k: v for k, v in results.items() if v is not None}
