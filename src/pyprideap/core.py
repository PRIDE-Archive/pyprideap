from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import pandas as pd


class Platform(Enum):
    OLINK_EXPLORE = "olink_explore"
    OLINK_EXPLORE_HT = "olink_explore_ht"
    OLINK_REVEAL = "olink_reveal"
    OLINK_TARGET = "olink_target"
    SOMASCAN = "somascan"
    NULISA = "nulisa"


def value_unit(platform: Platform) -> str:
    """Unit of the deposited values: SomaScan RFU, NULISA NPQ (log2), Olink NPX (log2)."""
    if platform == Platform.SOMASCAN:
        return "RFU"
    if platform == Platform.NULISA:
        return "NPQ"
    return "NPX"


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
