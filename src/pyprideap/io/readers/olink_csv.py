from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import cast

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


def _positions(values: pd.Series, labels: pd.Index) -> np.ndarray:
    """Position of each value in *labels* (-1 when absent or missing).

    Categorical columns are mapped through their categories, so millions of rows
    are resolved with one lookup per distinct value. Values are compared as text
    when the labels are text (identifiers read as categories are text).
    """
    as_text = labels.inferred_type in ("string", "empty")
    if isinstance(values.dtype, pd.CategoricalDtype):
        categories = values.cat.categories
        lookup = labels.get_indexer(categories.astype(str) if as_text else categories)
        codes = values.cat.codes.to_numpy()
        return np.where(codes >= 0, lookup[codes], -1)
    if as_text:
        values = values.map(lambda v: v if pd.isna(v) else str(v))
    return np.asarray(labels.get_indexer(pd.Index(values)))


def _first_per_cell(rows: np.ndarray, cols: np.ndarray, n_cols: int) -> np.ndarray:
    """Mask keeping the first measurement of each (row, column) cell."""
    duplicated = pd.Series(rows.astype(np.int64) * n_cols + cols).duplicated(keep="first").to_numpy(dtype=bool)
    return cast(np.ndarray, np.logical_not(duplicated))


def _long_to_wide(df: pd.DataFrame, sample_key: str, sample_order: pd.Index, value_col: str) -> pd.DataFrame:
    """Samples x assays matrix of *value_col*, equivalent to
    ``pivot_table(index=sample_key, columns="OlinkID", aggfunc="first")`` reindexed to
    *sample_order*: first non-missing value per cell, assays sorted, assays without
    any value dropped. Built by position, which is much faster and lighter than
    pivot_table on multi-million-row exports.
    """
    values = pd.to_numeric(df[value_col], errors="coerce").to_numpy(dtype=float)
    assay_ids = df["OlinkID"]
    labels = pd.Index(
        sorted(assay_ids.cat.categories.astype(str))
        if isinstance(assay_ids.dtype, pd.CategoricalDtype)
        else sorted({str(v) for v in assay_ids.dropna().unique()})
    )
    rows = _positions(df[sample_key], sample_order)
    cols = _positions(assay_ids, labels)
    ok = (rows >= 0) & (cols >= 0) & ~np.isnan(values)
    r, c, v = rows[ok], cols[ok], values[ok]
    first = _first_per_cell(r, c, len(labels))
    matrix = np.full((len(sample_order), len(labels)), np.nan)
    matrix[r[first], c[first]] = v[first]
    keep = np.zeros(len(labels), dtype=bool)
    keep[np.unique(c)] = True
    wide = pd.DataFrame(matrix[:, keep], columns=pd.Index(labels[keep], name="OlinkID"))
    return cast(pd.DataFrame, wide)


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
    rows = _positions(df[sample_key], pd.Index(list(sample_order)))  # type: ignore[call-overload]
    cols = _positions(df["OlinkID"], pd.Index(assays))
    placed = (rows >= 0) & (cols >= 0)
    for col, key in present:
        if key in matrices:
            continue
        # Flags take a handful of distinct values: normalise each once. Olink writes
        # WARN in current exports; accept the long form too, as filter_qc does.
        codes, uniques = pd.factorize(df[col])
        labels = pd.Series(uniques).astype("string").str.strip().str.upper().replace({"WARNING": "WARN"})
        ok = placed & (codes >= 0)
        # Work with the small integer codes; text is looked up once per cell at the end
        coded = np.full((len(pd.Series(sample_order)), len(assays)), -1, dtype=np.int32)
        # First non-missing measurement wins (as pivot_table's "first"); dropping
        # later repeats leaves each position written exactly once
        r, c, v = rows[ok], cols[ok], codes[ok]
        first = _first_per_cell(r, c, len(assays))
        coded[r[first], c[first]] = v[first]
        lookup = np.array([*labels.astype(object).where(labels.notna(), None), None], dtype=object)
        matrices[key] = pd.DataFrame(lookup[coded], columns=assays).astype("string")
    return matrices


_DELIMITERS = (",", ";", "\t", "|")


def _sniff_delimiter(path: Path) -> str | None:
    """Delimiter of a delimited text file, from its header line (None if unclear)."""
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        header = f.readline()
    counts = {d: header.count(d) for d in _DELIMITERS}
    best = max(counts, key=lambda d: counts[d])
    return best if counts[best] > 0 else None


# Columns the reader uses; everything else in an export is skipped while parsing
# Per-measurement numbers, parsed as float. MissingFreq (per assay, sometimes
# written as "30%") is read as text and made numeric afterwards when it can be.
_NUMERIC_COLS = {"NPX", "LOD", "LODNPX", "PlateLOD"}
_USED_COLS = (
    _REQUIRED_COLS | _SAMPLE_COLS | _FEATURE_COLS | _NUMERIC_COLS | set(_QC_FLAG_COLUMNS) | set(_OLINK_COLUMN_ALIASES)
)
# pandas' default missing-value markers, so both parsers treat the same text as missing
_NA_VALUES = [
    "",
    "#N/A",
    "#N/A N/A",
    "#NA",
    "-1.#IND",
    "-1.#QNAN",
    "-NaN",
    "-nan",
    "1.#IND",
    "1.#QNAN",
    "<NA>",
    "N/A",
    "NA",
    "NULL",
    "NaN",
    "None",
    "n/a",
    "nan",
    "null",
]


# Files at least this large are streamed to bound peak memory
_STREAM_MIN_BYTES = 256 * 1024 * 1024


def _read_header(path: Path, sep: str) -> list[str]:
    import csv

    with open(path, encoding="utf-8-sig", errors="replace", newline="") as f:
        return next(csv.reader(f, delimiter=sep), [])


def _read_arrow(path: Path, sep: str, columns: list[str]) -> pd.DataFrame:
    """Read the file with pyarrow: only *columns*, text stored as categories.

    Repeated identifiers (sample, assay, plate, flags) are stored once per
    distinct value. Large files are streamed, keeping only a few blocks of raw
    text in memory at a time.
    """
    import pyarrow as pa
    import pyarrow.csv as pacsv

    text = pa.dictionary(pa.int32(), pa.string())
    convert = pacsv.ConvertOptions(
        include_columns=columns,
        column_types={c: (pa.float64() if c in _NUMERIC_COLS else text) for c in columns},
        null_values=_NA_VALUES,
        strings_can_be_null=True,
    )
    parse = pacsv.ParseOptions(delimiter=sep)
    if path.stat().st_size < _STREAM_MIN_BYTES:
        # Small files: one multithreaded read (streaming has ~1 s fixed overhead)
        table = pacsv.read_csv(path, parse_options=parse, convert_options=convert)
    else:
        read = pacsv.ReadOptions(block_size=1 << 24)
        with pacsv.open_csv(path, read_options=read, parse_options=parse, convert_options=convert) as reader:
            table = pa.Table.from_batches(list(reader), schema=reader.schema)
    return cast(pd.DataFrame, table.unify_dictionaries().to_pandas())


def _read_delimited(path: Path) -> pd.DataFrame:
    """Read a delimited Olink export, keeping only the columns the reader uses.

    The delimiter comes from the header line. The file is streamed with pyarrow
    with identifiers stored as categories: on a 1.7 GB Explore HT export this
    parses in ~5 s with a ~1.5 GB peak, against ~27 s and ~3.9 GB for pandas.
    Files pyarrow cannot parse (e.g. ragged rows, text in a numeric column) fall
    back to pandas.
    """
    sep = _sniff_delimiter(path)
    if sep is not None:
        header = _read_header(path, sep)
        columns = [c for c in header if c in _USED_COLS]
        if len(columns) == len(set(columns)) and _REQUIRED_COLS <= {_OLINK_COLUMN_ALIASES.get(c, c) for c in columns}:
            try:
                return _read_arrow(path, sep, columns)
            except Exception as exc:  # pyarrow raises several error types for malformed files
                logger.debug("pyarrow could not parse %s (%s); using pandas", path.name, exc)
        try:
            return pd.read_csv(path, sep=sep, low_memory=False, dtype={"SampleID": str})
        except (pd.errors.ParserError, UnicodeDecodeError) as exc:
            logger.debug("C engine could not parse %s (%s); using the Python engine", path.name, exc)
    return pd.read_csv(path, sep=None, engine="python", dtype={"SampleID": str})


def _plain(frame: pd.DataFrame) -> pd.DataFrame:
    """Small sample/feature tables: categorical columns back to plain values.

    MissingFreq becomes numeric when all its values are numbers (as pandas would
    infer); identifier columns stay text so leading zeros are kept.
    """
    for col in frame.columns:
        if isinstance(frame[col].dtype, pd.CategoricalDtype):
            values = frame[col].astype(object).where(frame[col].notna(), np.nan)
            if col == "MissingFreq":
                numeric = pd.to_numeric(values, errors="coerce")
                if not (numeric.isna() & values.notna()).any():
                    values = numeric
            frame[col] = values
    return frame


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
    samples = _plain(df[sample_cols].drop_duplicates(subset=[sample_key]).reset_index(drop=True))

    feature_cols = [c for c in df.columns if c in _FEATURE_COLS]
    # Drop LOD from per-assay features since it varies per plate/sample
    feature_cols_no_lod = [c for c in feature_cols if c != "LOD"]
    features = _plain(df[feature_cols_no_lod].drop_duplicates(subset=["OlinkID"]).reset_index(drop=True))

    # Samples whose key is missing are kept as empty rows, as pivot_table left them
    sample_order = pd.Index(samples[sample_key].astype(object).map(lambda v: v if pd.isna(v) else str(v)))

    expression = _long_to_wide(df, sample_key, sample_order, "NPX")
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
        lod_matrix = _long_to_wide(df, sample_key, sample_order, lod_col)
        metadata["lod_matrix"] = lod_matrix
    else:
        logger.debug("No LOD column in input")

    metadata.update(_qc_flag_matrices(df, sample_key, sample_order, expression.columns))
    del df

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
