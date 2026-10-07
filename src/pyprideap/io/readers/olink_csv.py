from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from pyprideap.core import AffinityDataset, Platform

logger = logging.getLogger(__name__)

_SAMPLE_COLS = {"SampleID", "SampleName", "SampleRun", "PlateID", "WellID", "SampleType", "SampleQC", "PlateQC"}
_FEATURE_COLS = {"OlinkID", "UniProt", "Assay", "Panel", "LOD", "MissingFreq", "Normalization"}
_REQUIRED_COLS = {"SampleID", "OlinkID", "NPX"}

# OlinkID prefix → Platform mapping
_OLINK_ID_PREFIX_MAP = {
    "OID0": Platform.OLINK_TARGET,
    "OID1": Platform.OLINK_TARGET,
    "OID2": Platform.OLINK_EXPLORE,
    "OID3": Platform.OLINK_EXPLORE,
    "OID4": Platform.OLINK_EXPLORE_HT,
    "OID5": Platform.OLINK_REVEAL,
}


# Alternative column names seen in deposited Olink long tables (e.g. de-identified
# exports); renamed to the standard names when the standard column is absent.
_OLINK_COLUMN_ALIASES = {
    "DeidentifiedSampleID": "SampleID",
    "Sample_Type": "SampleType",
    "DeidentifiedPlateID": "PlateID",
}


def _apply_olink_aliases(df: pd.DataFrame) -> pd.DataFrame:
    rename = {old: new for old, new in _OLINK_COLUMN_ALIASES.items() if old in df.columns and new not in df.columns}
    if rename:
        logger.debug("Renaming Olink columns: %s", rename)
        df = df.rename(columns=rename)
    return df


def _detect_sample_key(df: pd.DataFrame, *, source: str = "") -> str:
    """Choose the best column to identify samples in a long-format Olink file.

    Returns ``"SampleID"`` unless it appears to be an assay index (same
    cardinality as ``OlinkID``), in which case ``"SampleName"`` is used.
    """
    sample_key = "SampleID"
    if "SampleName" in df.columns:
        n_sid = df["SampleID"].nunique()
        n_olink = df["OlinkID"].nunique()
        n_sname = df["SampleName"].nunique()
        logger.debug("Sample key check: SampleID=%d, OlinkID=%d, SampleName=%d unique", n_sid, n_olink, n_sname)
        if n_sid == n_olink and n_sname != n_olink:
            sample_key = "SampleName"
            warnings.warn(
                f"{source}: SampleID has the same cardinality as OlinkID "
                f"({n_sid}), which suggests it indexes assays rather than "
                f"samples. Using SampleName ({n_sname} unique) as sample "
                f"identifier instead.",
                UserWarning,
                stacklevel=3,
            )
    logger.debug("Sample key selected: %s", sample_key)
    return sample_key


def _sample_run_key(df: pd.DataFrame, sample_key: str) -> tuple[pd.DataFrame, str]:
    """Give each plate run of a sample its own row (bridging / replicate samples).

    Multi-plate Olink studies often run the same samples on every plate to
    bridge them. Keyed by sample ID alone, those runs would be merged and one
    of them silently dropped. When a sample ID appears on more than one plate,
    rows are keyed by a ``SampleRun`` column ("<sample> @ <plate>") instead;
    ``SampleID`` keeps the original identifier.
    """
    if "PlateID" not in df.columns:
        return df, sample_key
    plates_per_sample = df.groupby(sample_key)["PlateID"].nunique()
    n_multi = int((plates_per_sample > 1).sum())
    if n_multi == 0:
        return df, sample_key
    logger.debug("%d samples measured on more than one plate; keying rows by SampleRun", n_multi)
    df = df.copy()
    df["SampleRun"] = df[sample_key].astype(str) + " @ " + df["PlateID"].astype(str)
    return df, "SampleRun"


def _warn_data_quality(dataset: AffinityDataset, *, source: str = "") -> None:
    """Emit warnings for common data quality issues after reading."""
    n_samples = len(dataset.samples)
    n_features = len(dataset.features)

    # High NaN fraction
    nan_frac = float(dataset.expression.isna().mean().mean())
    if nan_frac > 0.5:
        warnings.warn(
            f"{source}: Expression matrix is {nan_frac:.0%} NaN "
            f"({n_samples} samples × {n_features} features). "
            f"The data may have been pivoted incorrectly or is very sparse.",
            UserWarning,
            stacklevel=3,
        )

    # Suspicious square matrix (samples == features)
    if n_samples == n_features and n_samples > 10:
        warnings.warn(
            f"{source}: Expression matrix is square "
            f"({n_samples} samples = {n_features} features), which is unusual "
            f"for affinity proteomics data. Verify that sample and feature "
            f"identifiers were parsed correctly.",
            UserWarning,
            stacklevel=3,
        )

    # Very few non-NaN values per sample
    non_nan_per_sample = dataset.expression.notna().sum(axis=1)
    median_non_nan = float(non_nan_per_sample.median())
    if n_features > 0 and median_non_nan / n_features < 0.1:
        warnings.warn(
            f"{source}: Samples have very few measured values "
            f"(median {median_non_nan:.0f} of {n_features} features). "
            f"This may indicate a parsing issue.",
            UserWarning,
            stacklevel=3,
        )


def _detect_olink_platform(olink_ids: pd.Series) -> Platform:
    """Detect Olink platform from OlinkID prefixes."""
    prefixes = olink_ids.astype(str).str[:4]
    counts = prefixes.map(_OLINK_ID_PREFIX_MAP).value_counts()
    if counts.empty:
        return Platform.OLINK_EXPLORE
    return Platform(counts.index[0])


# Per-measurement QC flag columns in Olink exports, mapped to metadata matrix keys.
# Assay-level: AssayQC (Explore HT / Reveal parquet), Assay_Warning (Explore 3072 CSV).
# Sample-level, per measurement (it varies by panel/block): SampleQC, QC_Warning.
_QC_FLAG_COLUMNS = {
    "AssayQC": "assay_qc_matrix",
    "Assay_Warning": "assay_qc_matrix",
    "SampleQC": "sample_qc_matrix",
    "QC_Warning": "sample_qc_matrix",
}


def _qc_flag_matrices(df: pd.DataFrame, sample_key: str, sample_order: object, assays: pd.Index) -> dict[str, object]:
    """Pivot per-measurement Olink QC flags into sample x assay matrices of upper-case strings.

    Returns metadata entries ``assay_qc_matrix`` / ``sample_qc_matrix`` for the
    columns present; when two columns map to the same key the first listed wins.
    """
    matrices: dict[str, object] = {}
    present = [(col, key) for col, key in _QC_FLAG_COLUMNS.items() if col in df.columns]
    if not present:
        return matrices
    # Row and column position of every measurement, computed once. Placing values by
    # position is much faster than pivot_table on multi-million-row Explore HT exports.
    rows = pd.Index(pd.Series(sample_order)).get_indexer(df[sample_key])
    cols = pd.Index(assays).get_indexer(df["OlinkID"])
    placed = (rows >= 0) & (cols >= 0)
    for col, key in present:
        if key in matrices:
            continue
        # Flags take a handful of distinct values: normalise each once. Olink writes
        # WARN in current exports; accept the long form too, as filter_qc does.
        codes, uniques = pd.factorize(df[col])
        labels = pd.Series(uniques).astype("string").str.strip().str.upper().replace({"WARNING": "WARN"})
        values = np.where(codes >= 0, labels.reindex(codes).to_numpy(dtype=object), None)
        ok = placed & (codes >= 0)
        matrix = np.full((len(pd.Series(sample_order)), len(assays)), None, dtype=object)
        # First non-missing measurement wins (as pivot_table's "first"); dropping
        # later repeats leaves each position written exactly once
        r, c, v = rows[ok], cols[ok], values[ok]
        first = ~pd.Series(r * len(assays) + c).duplicated(keep="first").to_numpy()
        matrix[r[first], c[first]] = v[first]
        matrices[key] = pd.DataFrame(matrix, columns=assays).astype("string")
    return matrices


_DELIMITERS = (",", ";", "\t", "|")


def _sniff_delimiter(path: Path) -> str | None:
    """Delimiter of a delimited text file, from its header line (None if unclear)."""
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        header = f.readline()
    counts = {d: header.count(d) for d in _DELIMITERS}
    best = max(counts, key=lambda d: counts[d])
    return best if counts[best] > 0 else None


def _read_delimited(path: Path) -> pd.DataFrame:
    """Read a delimited Olink export.

    The delimiter is taken from the header line and the file is parsed with
    pandas' C engine, which is about 10x faster than delimiter sniffing with the
    Python engine on large exports (e.g. a 1.7 GB Explore HT CSV). Files the C
    engine cannot parse (e.g. ragged rows) fall back to the Python engine.
    """
    sep = _sniff_delimiter(path)
    if sep is not None:
        try:
            return pd.read_csv(path, sep=sep, low_memory=False)
        except (pd.errors.ParserError, UnicodeDecodeError) as exc:
            logger.debug("C engine could not parse %s (%s); using the Python engine", path.name, exc)
    return pd.read_csv(path, sep=None, engine="python")


def read_olink_csv(path: str | Path) -> AffinityDataset:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = _apply_olink_aliases(_read_delimited(path))
    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path.name}: {sorted(missing)}")

    sample_key = _detect_sample_key(df, source=path.name)
    df, sample_key = _sample_run_key(df, sample_key)

    # Numeric columns can arrive as text when a file was re-arranged by hand and
    # some rows are shifted; unparsable values become NaN instead of breaking LOD
    # comparisons downstream.
    for col in ("NPX", "LOD", "LODNPX", "PlateLOD"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    sample_cols = [c for c in df.columns if c in _SAMPLE_COLS]
    samples = df[sample_cols].drop_duplicates(subset=[sample_key]).reset_index(drop=True)

    feature_cols = [c for c in df.columns if c in _FEATURE_COLS]
    # Drop LOD from per-assay features since it varies per plate/sample
    feature_cols_no_lod = [c for c in feature_cols if c != "LOD"]
    features = df[feature_cols_no_lod].drop_duplicates(subset=["OlinkID"]).reset_index(drop=True)

    sample_order = samples[sample_key].values

    expression = df.pivot_table(
        index=sample_key,
        columns="OlinkID",
        values="NPX",
        aggfunc="first",
    )
    expression = expression.reindex(sample_order).reset_index(drop=True)
    logger.debug("Pivot shape: %d samples x %d features", expression.shape[0], expression.shape[1])

    # Align features to match expression column order (pivot_table sorts columns)
    features = features.set_index("OlinkID").reindex(expression.columns).reset_index()

    metadata: dict[str, object] = {"source_file": str(path)}

    # Build per-sample x per-assay LOD matrix. Newer Explore HT / Reveal exports
    # carry both LODNPX (LOD on the NPX scale) and LOD (count-based); NPX must be
    # compared with LODNPX.
    # (NPX Signature exports for Olink Target name it PlateLOD)
    lod_col = next((c for c in ("LODNPX", "LOD", "PlateLOD") if c in df.columns), None)
    if lod_col is not None:
        logger.debug("LOD column %s present, building LOD matrix", lod_col)
        lod_matrix = df.pivot_table(
            index=sample_key,
            columns="OlinkID",
            values=lod_col,
            aggfunc="first",
        )
        lod_matrix = lod_matrix.reindex(sample_order).reset_index(drop=True)
        metadata["lod_matrix"] = lod_matrix
    else:
        logger.debug("No LOD column in input")

    metadata.update(_qc_flag_matrices(df, sample_key, sample_order, expression.columns))

    platform = _detect_olink_platform(features["OlinkID"])

    dataset = AffinityDataset(
        platform=platform,
        samples=samples,
        features=features,
        expression=expression,
        metadata=metadata,
    )
    _warn_data_quality(dataset, source=path.name)
    return dataset
