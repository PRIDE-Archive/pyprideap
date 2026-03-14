"""UniProt duplicate detection for Olink datasets.

Mirrors ``npx_check_uniprot_dups.R`` from OlinkAnalyze:
- Detects OlinkIDs that map to multiple UniProt IDs
- Warns users about potential downstream ambiguity
"""

from __future__ import annotations

from dataclasses import dataclass

from pyprideap.core import AffinityDataset


@dataclass
class UniProtDuplicateInfo:
    """Summary of UniProt duplicate detection.

    Attributes
    ----------
    duplicates:
        Dict mapping OlinkID → list of UniProt IDs where len > 1.
    n_affected_assays:
        Number of OlinkIDs with multiple UniProt mappings.
    n_total_assays:
        Total number of OlinkIDs checked.
    """

    duplicates: dict[str, list[str]]
    n_affected_assays: int
    n_total_assays: int

    @property
    def has_duplicates(self) -> bool:
        return self.n_affected_assays > 0


def detect_uniprot_duplicates(
    dataset: AffinityDataset,
    *,
    olink_id_column: str = "OlinkID",
    uniprot_column: str = "UniProt",
) -> UniProtDuplicateInfo:
    """Detect OlinkIDs that map to multiple UniProt identifiers.

    Mirrors ``assay_identifiers()`` from OlinkAnalyze's
    ``npx_check_uniprot_dups.R``. In some Olink datasets, a single OlinkID
    can correspond to multiple UniProt IDs (e.g. multi-specific binders).
    This can cause issues in downstream pathway/enrichment analyses.

    Parameters
    ----------
    dataset:
        Olink AffinityDataset.
    olink_id_column:
        Column name for Olink assay identifier (default "OlinkID").
    uniprot_column:
        Column name for UniProt identifier (default "UniProt").

    Returns
    -------
    UniProtDuplicateInfo
        Summary with duplicate mappings.
    """
    features = dataset.features

    if olink_id_column not in features.columns or uniprot_column not in features.columns:
        return UniProtDuplicateInfo(
            duplicates={},
            n_affected_assays=0,
            n_total_assays=len(features),
        )

    # Group UniProt IDs per OlinkID
    grouped = (
        features[[olink_id_column, uniprot_column]]
        .dropna(subset=[uniprot_column])
        .drop_duplicates()
        .groupby(olink_id_column)[uniprot_column]
        .apply(list)
    )

    # Find duplicates (multiple UniProt per OlinkID)
    duplicates = {oid: uniprots for oid, uniprots in grouped.items() if len(uniprots) > 1}

    return UniProtDuplicateInfo(
        duplicates=duplicates,
        n_affected_assays=len(duplicates),
        n_total_assays=int(features[olink_id_column].nunique()),
    )
