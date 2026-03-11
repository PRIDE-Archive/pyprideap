from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import pandas as pd


class Platform(Enum):
    OLINK_EXPLORE = "olink_explore"
    OLINK_EXPLORE_HT = "olink_explore_ht"
    OLINK_TARGET = "olink_target"
    SOMASCAN = "somascan"


class Level(Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationResult:
    level: Level
    rule: str
    message: str
    details: dict[str, object] | None = None


@dataclass
class AffinityDataset:
    platform: Platform
    samples: pd.DataFrame
    features: pd.DataFrame
    expression: pd.DataFrame
    metadata: dict[str, object] = field(default_factory=dict)
