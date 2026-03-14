"""SomaScan-specific processing utilities.

Mirrors functionality from the SomaDataIO R package, including:
- Control analyte classification (HybControlElution, Spuriomers, etc.)
- Outlier detection (MAD-based, equivalent to calcOutlierMap / getOutlierIds)
- QC flag handling (RowCheck / ColCheck)
- Unified preprocessing pipeline (preprocess_somascan)
"""

from pyprideap.processing.somascan.controls import (
    CONTROL_ANALYTE_TYPES,
    ControlAnalyteType,
    classify_control_analytes,
    get_control_seqids,
    is_control_analyte,
    remove_control_analytes,
)
from pyprideap.processing.somascan.outliers import (
    OutlierMap,
    calc_outlier_map,
    get_outlier_ids,
)
from pyprideap.processing.somascan.pipeline import preprocess_somascan
from pyprideap.processing.somascan.qc_flags import (
    add_row_check,
    filter_by_col_check,
    filter_by_row_check,
    get_col_check_summary,
    get_row_check_summary,
)

__all__ = [
    "CONTROL_ANALYTE_TYPES",
    "ControlAnalyteType",
    "OutlierMap",
    "add_row_check",
    "calc_outlier_map",
    "classify_control_analytes",
    "filter_by_col_check",
    "filter_by_row_check",
    "get_col_check_summary",
    "get_control_seqids",
    "get_outlier_ids",
    "get_row_check_summary",
    "is_control_analyte",
    "preprocess_somascan",
    "remove_control_analytes",
]
