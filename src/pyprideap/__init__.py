"""pyprideap — Python library for PRIDE Affinity Proteomics (PAD) archive data."""

from importlib.metadata import version

from pyprideap.core import AffinityDataset, Level, Platform, ValidationResult
from pyprideap.pride import PrideClient
from pyprideap.readers.registry import read
from pyprideap.stats import DatasetStats, compute_stats
from pyprideap.validators import validate

__version__ = version("pyprideap")

__all__ = [
    "__version__",
    "AffinityDataset",
    "DatasetStats",
    "Level",
    "Platform",
    "PrideClient",
    "ValidationResult",
    "compute_stats",
    "read",
    "validate",
]
