"""pyprideap — Python library for PRIDE Affinity Proteomics (PAD) archive data."""

from importlib.metadata import PackageNotFoundError, version

from pyprideap.core import AffinityDataset, Level, Platform, ValidationResult
from pyprideap.filtering import filter_controls, filter_qc
from pyprideap.lod import LodMethod, LodStats, compute_lod_from_controls, compute_nclod, compute_lod_stats, get_bundled_fixed_lod_path, get_reported_lod, get_valid_proteins, load_fixed_lod
from pyprideap.pride import PrideClient
from pyprideap.qc.compute import compute_all as compute_qc
from pyprideap.qc.report import qc_report
from pyprideap.readers.registry import read
from pyprideap.stats import DatasetStats, compute_stats
from pyprideap.validators import validate

try:
    __version__ = version("pyprideap")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__all__ = [
    "__version__",
    "AffinityDataset",
    "DatasetStats",
    "Level",
    "LodStats",
    "Platform",
    "PrideClient",
    "ValidationResult",
    "LodMethod",
    "compute_lod_from_controls",
    "compute_nclod",
    "compute_lod_stats",
    "compute_qc",
    "get_bundled_fixed_lod_path",
    "compute_stats",
    "get_reported_lod",
    "load_fixed_lod",
    "qc_report",
    "filter_controls",
    "filter_qc",
    "get_valid_proteins",
    "read",
    "validate",
]
