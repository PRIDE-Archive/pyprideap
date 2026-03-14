"""SDRF (Sample and Data Relationship Format) reader and utilities.

Parses SDRF TSV files to extract sample-level metadata (characteristics
and factor values) that can be merged into an :class:`AffinityDataset`.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from pyprideap.core import AffinityDataset

# Columns that are not useful for grouping / differential expression
_SKIP_PATTERNS = {
    "organism",
    "organism part",
    "sample matrix",
    "biological replicate",
    "individual",
    "technical replicate",
}

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

    # Build clean column names, disambiguating repeated base names
    rename: dict[str, str] = {}
    seen_counts: dict[str, int] = {}
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

    return pd.DataFrame(df.rename(columns=rename))


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
        if col.lower() in skip:
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

    return candidates


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

    # Auto-detect sample column
    if sample_col is None:
        for candidate in ("SampleId", "SampleID", "SampleName"):
            if candidate in dataset.samples.columns:
                sample_col = candidate
                break
    if sample_col is None:
        raise ValueError("Cannot detect sample ID column in dataset.samples. Specify sample_col explicitly.")

    # Only merge columns not already in the dataset
    existing = set(dataset.samples.columns)
    new_cols = [c for c in sdrf.columns if c not in existing and c != sdrf_col]
    if not new_cols:
        return dataset

    sdrf_subset = sdrf[[sdrf_col] + new_cols].copy()
    merged = dataset.samples.merge(
        sdrf_subset,
        left_on=sample_col,
        right_on=sdrf_col,
        how="left",
    )
    # Drop the join key from SDRF side if it differs from sample_col
    if sdrf_col != sample_col and sdrf_col in merged.columns:
        merged = merged.drop(columns=[sdrf_col])

    return replace(dataset, samples=merged)
