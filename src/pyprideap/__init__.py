"""pyprideap — Python library for PRIDE Affinity Proteomics (PAD) archive data."""

from importlib.metadata import PackageNotFoundError, version

from pyprideap.api.pride import PrideClient
from pyprideap.core import AffinityDataset, Level, Platform, ValidationResult
from pyprideap.io.readers.registry import read
from pyprideap.io.validators import validate
from pyprideap.processing.filtering import filter_controls, filter_qc, get_unique_samples
from pyprideap.processing.lod import (
    LodMethod,
    LodStats,
    compute_lod_from_controls,
    compute_lod_stats,
    compute_nclod,
    get_bundled_fixed_lod_path,
    get_proteins_above_lod,
    get_reported_lod,
    get_valid_proteins,
    load_fixed_lod,
)
from pyprideap.processing.normalization import (
    assess_bridgeability,
    bridge_normalize,
    reference_median_normalize,
    select_bridge_samples,
    subset_normalize,
)
from pyprideap.stats.descriptive import DatasetStats, compute_stats
from pyprideap.stats.design import randomize_plates
from pyprideap.viz.plots import boxplot
from pyprideap.viz.qc.compute import compute_all as compute_qc
from pyprideap.viz.qc.compute import compute_volcano
from pyprideap.viz.qc.report import qc_report, qc_report_split
from pyprideap.viz.theme import (
    PRIDE_COLORS,
    pride_color_discrete,
    pride_color_gradient,
    pride_fill_discrete,
    set_plot_theme,
)

try:
    __version__ = version("pyprideap")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"


# Lazy imports for optional-dependency modules
def ttest(*args, **kwargs):
    from pyprideap.stats.differential import ttest as _ttest

    return _ttest(*args, **kwargs)


def wilcoxon(*args, **kwargs):
    from pyprideap.stats.differential import wilcoxon as _wilcoxon

    return _wilcoxon(*args, **kwargs)


def anova(*args, **kwargs):
    from pyprideap.stats.differential import anova as _anova

    return _anova(*args, **kwargs)


def anova_posthoc(*args, **kwargs):
    from pyprideap.stats.differential import anova_posthoc as _anova_posthoc

    return _anova_posthoc(*args, **kwargs)


__all__ = [
    "__version__",
    # Core
    "AffinityDataset",
    "DatasetStats",
    "Level",
    "LodStats",
    "LodMethod",
    "Platform",
    "PrideClient",
    "ValidationResult",
    # IO
    "read",
    "validate",
    # LOD
    "compute_lod_from_controls",
    "compute_nclod",
    "compute_lod_stats",
    "get_bundled_fixed_lod_path",
    "get_reported_lod",
    "get_proteins_above_lod",
    "get_valid_proteins",
    "load_fixed_lod",
    # QC
    "compute_qc",
    "compute_stats",
    "compute_volcano",
    "qc_report",
    "qc_report_split",
    # Filtering
    "filter_controls",
    "filter_qc",
    "get_unique_samples",
    # Normalization
    "assess_bridgeability",
    "bridge_normalize",
    "reference_median_normalize",
    "select_bridge_samples",
    "subset_normalize",
    # Plots
    "boxplot",
    "PRIDE_COLORS",
    "pride_color_discrete",
    "pride_color_gradient",
    "pride_fill_discrete",
    "set_plot_theme",
    # Statistical testing
    "ttest",
    "wilcoxon",
    "anova",
    "anova_posthoc",
    # Experimental design
    "randomize_plates",
]
