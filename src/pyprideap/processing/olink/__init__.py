"""Olink-specific processing utilities.

Mirrors functionality from the OlinkAnalyze R package, including:
- IQR-based QC outlier detection (equivalent to olink_qc_plot / olink_bridgeselector)
- UniProt duplicate detection (equivalent to npx_check_uniprot_dups)
- Enhanced bridge sample selection with QC outlier + LOD filtering
- Preprocessing pipeline (equivalent to npxProcessing_forDimRed)
"""

from pyprideap.processing.olink.outliers import (
    IqrMedianOutlierResult,
    compute_iqr_median_outliers,
    is_iqr_outlier,
)
from pyprideap.processing.olink.pipeline import (
    OlinkPreprocessingReport,
    preprocess_olink,
)
from pyprideap.processing.olink.uniprot import (
    UniProtDuplicateInfo,
    detect_uniprot_duplicates,
)

__all__ = [
    "IqrMedianOutlierResult",
    "OlinkPreprocessingReport",
    "UniProtDuplicateInfo",
    "compute_iqr_median_outliers",
    "detect_uniprot_duplicates",
    "is_iqr_outlier",
    "preprocess_olink",
]
