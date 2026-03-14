"""SomaScan control analyte classification.

Provides the canonical SeqId lists for each control analyte category,
matching the SomaDataIO R package (``getAnalytes(..., rm.controls=TRUE)``).

Control categories:
- **HybControlElution**: hybridization control elution probes
- **Spuriomer**: spurious aptamer sequences
- **NonBiotin**: non-biotinylated controls
- **NonHuman**: non-human protein targets
- **NonCleavable**: non-cleavable linker controls
"""

from __future__ import annotations

from enum import Enum

from pyprideap.core import AffinityDataset


class ControlAnalyteType(Enum):
    """SomaScan control analyte categories."""

    HYB_CONTROL_ELUTION = "HybControlElution"
    SPURIOMER = "Spuriomer"
    NON_BIOTIN = "NonBiotin"
    NON_HUMAN = "NonHuman"
    NON_CLEAVABLE = "NonCleavable"


# Canonical SeqId lists from SomaDataIO R package (getAnalytes.R)
_SEQ_HYB_CONTROL_ELUTION = frozenset(
    {
        "2171-12",
        "2178-55",
        "2194-91",
        "2229-54",
        "2249-25",
        "2273-34",
        "2288-7",
        "2305-52",
        "2312-13",
        "2359-65",
        "2430-52",
        "2513-7",
    }
)

_SEQ_SPURIOMER = frozenset(
    {
        "2052-1",
        "2053-2",
        "2054-3",
        "2055-4",
        "2056-5",
        "2057-6",
        "2058-7",
        "2060-9",
        "2061-10",
        "4666-193",
        "4666-194",
        "4666-195",
        "4666-199",
        "4666-200",
        "4666-202",
        "4666-205",
        "4666-206",
        "4666-212",
        "4666-213",
        "4666-214",
    }
)

_SEQ_NON_BIOTIN = frozenset(
    {
        "3525-1",
        "3525-2",
        "3525-3",
        "3525-4",
        "4666-218",
        "4666-219",
        "4666-220",
        "4666-222",
        "4666-223",
        "4666-224",
    }
)

_SEQ_NON_HUMAN = frozenset(
    {
        "16535-61",
        "3507-1",
        "3512-72",
        "3650-8",
        "3717-23",
        "3721-5",
        "3724-64",
        "3742-78",
        "3849-56",
        "4584-5",
        "8443-9",
        "8444-3",
        "8444-46",
        "8445-184",
        "8445-54",
        "8449-103",
        "8449-124",
        "8471-53",
        "8481-26",
        "8481-44",
        "8482-39",
        "8483-5",
    }
)

_SEQ_NON_CLEAVABLE = frozenset(
    {
        "4666-225",
        "4666-230",
        "4666-232",
        "4666-236",
    }
)

# Lookup by enum value
CONTROL_ANALYTE_TYPES: dict[ControlAnalyteType, frozenset[str]] = {
    ControlAnalyteType.HYB_CONTROL_ELUTION: _SEQ_HYB_CONTROL_ELUTION,
    ControlAnalyteType.SPURIOMER: _SEQ_SPURIOMER,
    ControlAnalyteType.NON_BIOTIN: _SEQ_NON_BIOTIN,
    ControlAnalyteType.NON_HUMAN: _SEQ_NON_HUMAN,
    ControlAnalyteType.NON_CLEAVABLE: _SEQ_NON_CLEAVABLE,
}

# Union of all control SeqIds
_ALL_CONTROL_SEQIDS = frozenset().union(*CONTROL_ANALYTE_TYPES.values())


def get_control_seqids(
    *categories: ControlAnalyteType,
) -> frozenset[str]:
    """Return the set of control SeqIds for the given categories.

    If no categories are specified, returns all control SeqIds.
    """
    if not categories:
        return _ALL_CONTROL_SEQIDS
    return frozenset().union(*(CONTROL_ANALYTE_TYPES[c] for c in categories))


def _extract_seqid(col_name: str) -> str:
    """Extract a bare SeqId from a column name.

    Handles formats like ``seq.1234.56`` → ``1234-56`` and
    bare ``1234-56`` or ``1234-56_7`` → ``1234-56``.
    """
    name = col_name.strip()
    if name.startswith("seq."):
        # seq.1234.56 → 1234-56
        parts = name[4:].split(".")
        if len(parts) >= 2:
            return f"{parts[0]}-{parts[1]}"
    # Strip optional version suffix: 1234-56_7 → 1234-56
    base = name.split("_")[0]
    return base


def is_control_analyte(
    seq_id: str,
    categories: tuple[ControlAnalyteType, ...] | None = None,
) -> bool:
    """Check if a SeqId belongs to a control analyte category."""
    bare = _extract_seqid(seq_id)
    pool = get_control_seqids(*categories) if categories else _ALL_CONTROL_SEQIDS
    return bare in pool


def classify_control_analytes(
    dataset: AffinityDataset,
) -> dict[str, ControlAnalyteType]:
    """Classify expression columns as control analytes.

    Returns a mapping of column name → ControlAnalyteType for all columns
    that match a known control SeqId. Columns that are not controls are
    omitted from the result.
    """
    result: dict[str, ControlAnalyteType] = {}
    for col in dataset.expression.columns:
        bare = _extract_seqid(str(col))
        for cat, seqids in CONTROL_ANALYTE_TYPES.items():
            if bare in seqids:
                result[str(col)] = cat
                break
    return result


def remove_control_analytes(
    dataset: AffinityDataset,
    categories: tuple[ControlAnalyteType, ...] | None = None,
) -> AffinityDataset:
    """Remove control analyte columns from the dataset.

    Parameters
    ----------
    dataset:
        SomaScan AffinityDataset.
    categories:
        Which control categories to remove. If None, removes all
        control analytes (HybControlElution, Spuriomer, NonBiotin,
        NonHuman, NonCleavable).

    Returns
    -------
    AffinityDataset
        A copy with control columns removed from expression and features.
    """
    control_seqids = get_control_seqids(*(categories or ()))

    # Identify columns to drop
    drop_cols = [col for col in dataset.expression.columns if _extract_seqid(str(col)) in control_seqids]

    if not drop_cols:
        return dataset

    keep_cols = [c for c in dataset.expression.columns if c not in drop_cols]
    expression = dataset.expression[keep_cols].reset_index(drop=True)

    # Filter features table to match
    features = dataset.features
    if "SeqId" in features.columns:
        keep_mask = ~features["SeqId"].apply(lambda s: _extract_seqid(str(s)) in control_seqids)
        features = features[keep_mask].reset_index(drop=True)
    elif "Name" in features.columns:
        keep_mask = ~features["Name"].apply(lambda s: _extract_seqid(str(s)) in control_seqids)
        features = features[keep_mask].reset_index(drop=True)
    else:
        # Best-effort: trim to match expression columns
        features = features.iloc[: len(keep_cols)].reset_index(drop=True)

    return AffinityDataset(
        platform=dataset.platform,
        samples=dataset.samples,
        features=features,
        expression=expression,
        metadata=dict(dataset.metadata),
    )
