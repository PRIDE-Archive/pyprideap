from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TypeVar, cast

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


_FrameOrSeries = TypeVar("_FrameOrSeries", pd.DataFrame, pd.Series)


def linearize(values: _FrameOrSeries, platform: Platform) -> _FrameOrSeries:
    """Deposited values on a linear scale, for SD / mean and ratio-to-median metrics.

    Olink NPX is log2, so 2^NPX. NULISA NPQ is log2 of the rescaled count plus 1
    (Alamar adds 1 before the log transform), so 2^NPQ - 1. SomaScan RFU is
    already linear and is returned unchanged.
    """
    if platform == Platform.SOMASCAN:
        return values
    linear = cast(_FrameOrSeries, values.rpow(2.0))
    if platform == Platform.NULISA:
        return cast(_FrameOrSeries, linear - 1.0)
    return linear


def linear_scale_label(platform: Platform) -> str:
    """HTML label of the linear scale used by :func:`linearize`."""
    if platform == Platform.SOMASCAN:
        return "SomaScan RFU as deposited"
    if platform == Platform.NULISA:
        return "NULISA 2<sup>NPQ</sup> &minus; 1"
    return "Olink 2<sup>NPX</sup>"


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
