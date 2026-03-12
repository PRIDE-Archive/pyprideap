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
class MissingFrequencyData:
    """Histogram of per-assay missing frequency."""

    missing_freq: list[float]
    title: str = "Missing Frequency"


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
class CorrelationData:
    matrix: list[list[float]]
    labels: list[str]
    title: str = "Sample Correlation"


@dataclass
class MissingValuesData:
    missing_rate_per_sample: list[float]
    missing_rate_per_feature: list[float]
    sample_ids: list[str]
    feature_ids: list[str]
    title: str = "Missing Values"


@dataclass
class CvDistributionData:
    feature_ids: list[str]
    cv_values: list[float]
    dilution: list[str] = field(default_factory=list)
    title: str = "CV Distribution"


@dataclass
class DetectionRateData:
    sample_ids: list[str]
    rates: list[float]
    title: str = "Detection Rate per Sample"


# Keep old name for backwards compat in tests
QcSummaryData = QcLodSummaryData


# ---------------------------------------------------------------------------
# Compute functions
# ---------------------------------------------------------------------------


def _sample_id_col(dataset: AffinityDataset) -> str:
    return "SampleID" if "SampleID" in dataset.samples.columns else "SampleId"


def _sample_ids(dataset: AffinityDataset) -> list[str]:
    col = _sample_id_col(dataset)
    if col in dataset.samples.columns:
        return dataset.samples[col].astype(str).tolist()
    return [f"S{i}" for i in range(len(dataset.samples))]


def compute_distribution(dataset: AffinityDataset) -> DistributionData:
    """Per-sample NPX/RFU value lists for overlaid density curves."""
    numeric = dataset.expression.apply(pd.to_numeric, errors="coerce")

    is_somascan = dataset.platform == Platform.SOMASCAN
    sample_ids = _sample_ids(dataset)
    sample_values: list[list[float]] = []

    for idx in range(len(numeric)):
        row = numeric.iloc[idx].dropna().values
        if is_somascan:
            row = np.log10(row[row > 0])
        sample_values.append(row.tolist())

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


def compute_missing_frequency(dataset: AffinityDataset) -> MissingFrequencyData:
    """Per-assay missing frequency.

    Uses the MissingFreq column from features metadata if available
    (provided by Olink in NPX CSV). Otherwise falls back to computing
    the NaN rate per feature from the expression matrix.
    """
    if "MissingFreq" in dataset.features.columns:
        freq = pd.to_numeric(dataset.features["MissingFreq"], errors="coerce").fillna(0).tolist()
    elif "MissingFreq" in dataset.metadata:
        freq = list(dataset.metadata["MissingFreq"])
    else:
        numeric = dataset.expression.apply(pd.to_numeric, errors="coerce")
        freq = numeric.isna().mean(axis=0).tolist()
    return MissingFrequencyData(missing_freq=freq)


def compute_qc_summary(dataset: AffinityDataset) -> QcLodSummaryData | None:
    """QC status × LOD stacked bar. Falls back to simple QC counts if no LOD."""
    if "SampleQC" not in dataset.samples.columns:
        return None

    from pyprideap.lod import _above_lod_matrix, get_lod_values

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
    from pyprideap.lod import _above_lod_matrix, compute_lod_from_controls, get_lod_values

    lod = get_lod_values(dataset)
    if lod is None:
        try:
            lod = compute_lod_from_controls(dataset)
        except (ValueError, KeyError):
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
        above_lod_pct.append(pct)
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
        pc1=transformed[:, 0].tolist(),
        pc2=transformed[:, 1].tolist() if n_comp >= 2 else [0.0] * len(labels),
        variance_explained=[float(v) for v in pca.explained_variance_ratio_],
        labels=labels,
        groups=groups,
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
        matrix=[[None if np.isnan(v) else v for v in row] for row in corr.values],
        labels=labels,
    )


def compute_missing_values(dataset: AffinityDataset) -> MissingValuesData:
    numeric = dataset.expression.apply(pd.to_numeric, errors="coerce")
    is_missing = numeric.isna()

    per_sample = is_missing.mean(axis=1).tolist()
    per_feature = is_missing.mean(axis=0).tolist()

    sample_ids = _sample_ids(dataset)
    feature_ids = numeric.columns.tolist()

    return MissingValuesData(
        missing_rate_per_sample=per_sample,
        missing_rate_per_feature=per_feature,
        sample_ids=sample_ids,
        feature_ids=feature_ids,
    )


def compute_cv_distribution(dataset: AffinityDataset) -> CvDistributionData | None:
    if dataset.platform != Platform.SOMASCAN:
        return None

    numeric = dataset.expression.apply(pd.to_numeric, errors="coerce")
    means = numeric.mean()
    stds = numeric.std()
    cv = stds / means
    cv = cv.replace([np.inf, -np.inf], np.nan).dropna()

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


def compute_detection_rate(dataset: AffinityDataset) -> DetectionRateData:
    numeric = dataset.expression.apply(pd.to_numeric, errors="coerce")
    rates = numeric.notna().mean(axis=1).tolist()

    return DetectionRateData(sample_ids=_sample_ids(dataset), rates=rates)


def compute_all(dataset: AffinityDataset) -> dict[str, object]:
    """Compute all applicable QC plot data for the dataset."""
    results: dict[str, object] = {}
    results["distribution"] = compute_distribution(dataset)
    results["missing_frequency"] = compute_missing_frequency(dataset)
    results["qc_summary"] = compute_qc_summary(dataset)
    results["lod_analysis"] = compute_lod_analysis(dataset)
    results["pca"] = compute_pca(dataset)
    results["correlation"] = compute_correlation(dataset)
    results["missing_values"] = compute_missing_values(dataset)
    results["cv_distribution"] = compute_cv_distribution(dataset)
    results["detection_rate"] = compute_detection_rate(dataset)
    return {k: v for k, v in results.items() if v is not None}
