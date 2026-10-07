"""SDRF (Sample and Data Relationship Format) reader and utilities.

Parses SDRF TSV files to extract sample-level metadata (characteristics
and factor values) that can be merged into an :class:`AffinityDataset`.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import cast

import pandas as pd

from pyprideap.core import AffinityDataset

logger = logging.getLogger(__name__)

# Columns that are not useful for grouping / differential expression
_SKIP_PATTERNS = {
    "organism",
    "organism part",
    "sample matrix",
    "biological replicate",
    "individual",
    "technical replicate",
}

# Accession prefixes like PAD000001-XB6 or PAD000003-r00001-C126_positive_7
_PAD_ID_RE = re.compile(r"^PAD\d{6}-(?:r\d+-)?", re.IGNORECASE)

# Prefer these SDRF / sample-metadata columns as case/control (or equivalent) groups.
_BIOLOGICAL_GROUP_PRIORITY = (
    "disease",
    "disease state",
    "phenotype",
    "condition",
    "pre-existing condition",
    "health status",
    "treatment",
    "group",
    "case control",
    "case/control",
    "status",
)

# Grouping columns that are real covariates but are not case/control for MPI.
_NOT_CASE_CONTROL = {
    "sex",
    "gender",
    "age",
    "sampleqc",
    "sample type",
    "sampletype",
    "sample matrix",
    "plateid",
    "plate id",
    "plate",
    "hybcontrolnormscale",
    "rowcheck",
    "technology type",
    "organism part",
}

# Fallback: parse binary labels out of sample IDs / source names when SDRF
# disease columns are "not available" (common in auto-annotated PAD SDRFs).
_ID_GROUP_PATTERNS = (
    re.compile(r"(?i)(?:^|[_-])(positive|negative)(?:[_-]|$)"),
    re.compile(r"(?i)(?:^|[_-])(case|control)(?:[_-]|$)"),
    re.compile(r"(?i)(?:^|[_-])(patient|healthy)(?:[_-]|$)"),
)

# Minimum / maximum number of unique values for a column to be
# considered useful for differential expression comparisons.
_MIN_GROUPS = 2
_MAX_GROUPS = 10
_MIN_SAMPLES_PER_GROUP = 3


def read_sdrf(path: str | Path) -> pd.DataFrame:
    """Read an SDRF TSV file and return a tidy DataFrame.

    Columns with the same base name (e.g. multiple
    ``characteristics[pre-existing condition]``) are kept as separate
    columns named ``pre-existing condition``,
    ``pre-existing condition 2``, etc.  Column names are shortened from
    the full SDRF syntax (e.g. ``disease`` instead of
    ``characteristics[disease]``).

    Returns
    -------
    pd.DataFrame
        Rows correspond to samples.  The ``source name`` column (if
        present) is preserved as-is for joining to expression data.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"SDRF file not found: {path}")

    df = pd.read_csv(path, sep="\t")
    logger.debug("SDRF loaded: %d rows x %d cols from %s", len(df), len(df.columns), path.name)

    # Build clean column names, disambiguating repeated base names
    rename: dict[str, str] = {}
    seen_counts: dict[str, int] = {}
    factor_columns: list[str] = []
    for col in df.columns:
        # Extract short name from characteristics[X] or factor value[X]
        m = re.match(r"(characteristics|factor value)\[(.+?)\]", col)
        if m:
            short = m.group(2)
        else:
            short = col

        # Strip pandas .N suffixes from duplicate raw column names
        base = re.sub(r"\.\d+$", "", short)

        count = seen_counts.get(base, 0) + 1
        seen_counts[base] = count
        if count == 1:
            rename[col] = base
        else:
            rename[col] = f"{base} {count}"
        if m and m.group(1) == "factor value":
            factor_columns.append(rename[col])

    renamed_df = pd.DataFrame(df.rename(columns=rename))
    # Factor value columns are the study's declared design variables; their
    # (shortened) names are kept so group detection can prefer them.
    renamed_df.attrs["factor_value_columns"] = factor_columns
    logger.debug("SDRF columns renamed: %d mappings applied", len(rename))
    return cast(pd.DataFrame, renamed_df)


def get_grouping_columns(sdrf: pd.DataFrame) -> list[str]:
    """Return SDRF columns suitable for differential expression grouping.

    A column is suitable if it has between :data:`_MIN_GROUPS` and
    :data:`_MAX_GROUPS` unique non-null values (excluding ``not available``),
    and at least :data:`_MIN_SAMPLES_PER_GROUP` samples per group.
    """
    candidates: list[str] = []
    skip = {"source name", "assay name", "technology type"}
    skip.update(f"comment[{x}]" for x in _SKIP_PATTERNS)
    skip.update(_SKIP_PATTERNS)

    for col in sdrf.columns:
        # Repeated SDRF columns are renamed "individual 2", "individual 3", ...
        if col.lower() in skip or _base_column_name(col) in skip:
            continue
        # Filter out "not available" / "not applicable" / NaN
        vals = sdrf[col].astype(str).str.strip().str.lower()
        mask = ~vals.isin({"not available", "not applicable", "nan", "", "na"})
        clean = sdrf.loc[mask, col]

        n_unique = clean.nunique()
        if n_unique < _MIN_GROUPS or n_unique > _MAX_GROUPS:
            continue

        # Check minimum samples per group
        counts = clean.value_counts()
        if counts.min() < _MIN_SAMPLES_PER_GROUP:
            continue

        candidates.append(col)

    logger.debug("Grouping columns found: %s", candidates)
    return candidates


def _loose_sample_id(value: object) -> str:
    """Sample ID for tolerant matching: no PAD prefix, case, separators or leading zeros."""
    return re.sub(r"[\s_\-.]+", "", normalize_sample_id(value).lower()).lstrip("0")


def normalize_sample_id(value: object) -> str:
    """Strip PAD accession prefixes so SDRF source names can join expression IDs.

    ``PAD000001-XB6`` → ``XB6``
    ``PAD000003-r00001-C126_positive_7`` → ``C126_positive_7``
    """
    return _PAD_ID_RE.sub("", str(value).strip())


def _base_column_name(name: str) -> str:
    """Column name without the " 2", " 3" suffix added for repeated SDRF columns."""
    return re.sub(r" \d+$", "", name.lower().strip())


def select_biological_group_column(frame: pd.DataFrame, factor_columns: list[str] | None = None) -> str | None:
    """Pick the best case/control-like column from SDRF or sample metadata.

    Uses :func:`get_grouping_columns` (2–10 groups, ≥3 samples each). SDRF
    ``factor value[...]`` columns (*factor_columns*) come first, as the study's
    declared design variables; otherwise disease / phenotype / treatment /
    group are preferred over other covariates, matching repeated columns by
    base name (``disease 2`` counts as ``disease``). Sex, plate, and QC flags
    are never treated as case/control.
    """
    candidates = get_grouping_columns(frame)
    if not candidates:
        return None

    for col in factor_columns or []:
        if col in candidates and _base_column_name(col) not in _NOT_CASE_CONTROL:
            return col

    for preferred in _BIOLOGICAL_GROUP_PRIORITY:
        for col in candidates:
            if _base_column_name(col) == preferred:
                return col
    return None


def infer_groups_from_sample_ids(sample_ids: list[str]) -> pd.Series | None:
    """Infer a binary group label from sample IDs when SDRF groups are missing.

    Looks for tokens such as ``positive``/``negative`` or ``case``/``control``
    in source names / SampleIDs. Returns a Series aligned to ``sample_ids``,
    or ``None`` if both groups are not present with enough samples.
    """
    ids = [str(s) for s in sample_ids]
    for pattern in _ID_GROUP_PATTERNS:
        labels: list[str | None] = []
        for sid in ids:
            match = pattern.search(sid)
            labels.append(match.group(1).lower() if match else None)
        series = pd.Series(labels, dtype="object")
        valid = series.dropna()
        if valid.nunique() < _MIN_GROUPS:
            continue
        counts = valid.value_counts()
        if counts.min() < _MIN_SAMPLES_PER_GROUP:
            continue
        logger.debug("Inferred groups from sample IDs via %s: %s", pattern.pattern, counts.to_dict())
        return series
    return None


def resolve_biological_groups(dataset: AffinityDataset) -> tuple[pd.Series, str] | None:
    """Resolve case/control-like labels aligned to ``dataset.samples`` rows.

    Search order:
    1. Biological grouping columns already in ``dataset.samples`` (SDRF merge
       or a Group column from the expression file).
    2. Tokens in SampleID / SampleName / source name (positive/negative, …).

    Returns
    -------
    tuple[pd.Series, str] | None
        ``(labels, column_name)`` where ``labels`` has one value per sample
        (NaN if unknown). ``None`` if no usable case/control split is found.
    """
    factor_columns = dataset.metadata.get("sdrf_factor_columns")
    column = select_biological_group_column(
        dataset.samples, factor_columns if isinstance(factor_columns, list) else None
    )
    if column is not None:
        labels = dataset.samples[column].astype("object")
        logger.debug("Biological groups from column %s", column)
        return labels, column

    id_values: list[str] = []
    for col in ("SampleID", "SampleId", "SampleName", "source name"):
        if col in dataset.samples.columns:
            id_values = dataset.samples[col].astype(str).tolist()
            break
    if not id_values:
        return None

    inferred = infer_groups_from_sample_ids(id_values)
    if inferred is None:
        return None
    inferred.index = dataset.samples.index
    return inferred, "sample identifier"


def merge_sdrf(
    dataset: AffinityDataset,
    sdrf: pd.DataFrame,
    *,
    sample_col: str | None = None,
    sdrf_col: str = "source name",
) -> AffinityDataset:
    """Merge SDRF metadata columns into the dataset's sample table.

    Parameters
    ----------
    dataset : AffinityDataset
        Dataset whose ``samples`` DataFrame will be enriched.
    sdrf : pd.DataFrame
        Parsed SDRF (output of :func:`read_sdrf`).
    sample_col : str | None
        Column in ``dataset.samples`` to join on.  Detected automatically
        if ``None`` (tries ``SampleId``, ``SampleID``, ``SampleName``).
    sdrf_col : str
        Column in the SDRF to join on (default: ``source name``).

    Returns
    -------
    AffinityDataset
        A new dataset with additional columns in ``samples``.
    """
    from dataclasses import replace

    if sdrf_col not in sdrf.columns:
        raise ValueError(f"SDRF column '{sdrf_col}' not found. Available: {list(sdrf.columns)}")

    # Auto-detect sample column: the candidate with the most distinct values
    # (some exports carry a constant SampleID and the real names in SampleName)
    if sample_col is None:
        present = [c for c in ("SampleId", "SampleID", "SampleName") if c in dataset.samples.columns]
        if present:
            sample_col = max(present, key=lambda c: (dataset.samples[c].nunique(), -present.index(c)))
        logger.debug("Auto-detected sample column: %s", sample_col)
    if sample_col is None:
        raise ValueError("Cannot detect sample ID column in dataset.samples. Specify sample_col explicitly.")

    # Only merge columns not already in the dataset
    existing = set(dataset.samples.columns)
    skip_keys = {sdrf_col, "source name", "assay name"}
    new_cols = [c for c in sdrf.columns if c not in existing and c not in skip_keys]
    if not new_cols:
        return dataset

    sdrf_join_col = sdrf_col
    if sdrf_join_col not in sdrf.columns and "assay name" in sdrf.columns:
        sdrf_join_col = "assay name"

    # An SDRF often describes several data files; use only the rows for this one
    source_file = dataset.metadata.get("source_file")
    if source_file and "comment[data file]" in sdrf.columns:
        own = sdrf["comment[data file]"].astype(str).str.strip() == Path(str(source_file)).name
        if own.any() and not own.all():
            logger.debug(
                "SDRF merge: using %d of %d rows for %s", int(own.sum()), len(sdrf), Path(str(source_file)).name
            )
            sdrf = sdrf[own]

    sdrf_subset = sdrf[[sdrf_join_col] + new_cols].copy()
    left = dataset.samples.copy()

    # Join keys, tried in order; the one matching the most samples wins:
    # exact text, PAD accession prefix removed ("PAD000001-XB6" -> "XB6"), and
    # additionally case, separators and leading zeros ignored
    # ("L14_NIST N4" ~ "L14_NIST_N4", "CE002303 201" ~ "CE002303-201").
    key_functions = (
        lambda v: str(v).strip(),
        normalize_sample_id,
        _loose_sample_id,
    )
    merged, matched = left, -1
    for key_fn in key_functions:
        lk, rk = left[sample_col].map(key_fn), sdrf_subset[sdrf_join_col].map(key_fn)
        # SDRF allows several rows per source name (e.g. one per assay); keep one
        # so the merge stays 1:1 and samples stay aligned with the expression matrix.
        right = sdrf_subset.assign(_prideap_key=rk).drop_duplicates(subset=["_prideap_key"], keep="first")
        attempt = left.assign(_prideap_key=lk).merge(right, on="_prideap_key", how="left")
        n = int(attempt[new_cols[0]].notna().sum())
        if n > matched:
            merged, matched = attempt.drop(columns=["_prideap_key"]), n
        if matched == len(left):
            break
    logger.debug("SDRF merge: %d of %d samples matched", matched, len(left))

    if sdrf_join_col != sample_col and sdrf_join_col in merged.columns:
        merged = merged.drop(columns=[sdrf_join_col])

    if len(merged) != len(left):
        logger.warning("SDRF merge would change the number of samples; SDRF not merged")
        return dataset

    unmatched = len(merged) - matched
    logger.debug(
        "SDRF merge: %d matched, %d unmatched rows (join: %s -> %s)",
        matched,
        unmatched,
        sample_col,
        sdrf_join_col,
    )

    metadata = dict(dataset.metadata)
    metadata["sdrf_merge"] = {"matched": int(matched), "total": int(len(merged))}
    factor_columns = [c for c in sdrf.attrs.get("factor_value_columns", []) if c in new_cols]
    if factor_columns:
        metadata["sdrf_factor_columns"] = factor_columns
    return replace(dataset, samples=merged, metadata=metadata)
