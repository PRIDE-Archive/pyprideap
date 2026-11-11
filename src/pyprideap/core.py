from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, TypeVar, cast

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
_F = TypeVar("_F", bound=Callable[..., Any])

# Per-report cache of results derived from one dataset (control filtering, LOD,
# linear values). Active only inside ``dataset_cache()``; elsewhere every call
# computes afresh, so results never outlive the report that produced them.
_DATASET_CACHE: ContextVar[dict[tuple[str, int], tuple[object, object]] | None] = ContextVar(
    "pyprideap_dataset_cache", default=None
)


@contextmanager
def dataset_cache() -> Iterator[None]:
    """Reuse ``cached_per_dataset`` results while computing one report."""
    if _DATASET_CACHE.get() is not None:  # nested: keep the outer cache
        yield
        return
    token = _DATASET_CACHE.set({})
    try:
        yield
    finally:
        _DATASET_CACHE.reset(token)


def with_dataset_cache(fn: _F) -> _F:
    """Run ``fn`` inside ``dataset_cache()``."""

    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with dataset_cache():
            return fn(*args, **kwargs)

    return cast(_F, wrapper)


def cached_per_dataset(fn: _F) -> _F:
    """Cache ``fn(dataset)`` inside ``dataset_cache()``, keyed by the dataset object.

    Only calls with the dataset as the sole argument are cached. Results are
    shared between callers, so they must be treated as read-only.
    """

    @wraps(fn)
    def wrapper(dataset: object, *args: Any, **kwargs: Any) -> Any:
        cache = _DATASET_CACHE.get()
        if cache is None or args or kwargs:
            return fn(dataset, *args, **kwargs)
        key = (fn.__qualname__, id(dataset))
        hit = cache.get(key)
        if hit is not None and hit[0] is dataset:
            return hit[1]
        result = fn(dataset)
        cache[key] = (dataset, result)  # holding the dataset keeps its id unique
        return result

    return cast(_F, wrapper)


def numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    """``df.apply(pd.to_numeric, errors="coerce")`` without the per-column cost.

    Expression matrices are normally numeric already; only columns that are not
    are converted. Always returns a new frame, like ``apply`` did.
    """
    out = df.copy()
    for col in [c for c, dt in zip(df.columns, df.dtypes) if not pd.api.types.is_numeric_dtype(dt)]:
        out[col] = pd.to_numeric(df[col], errors="coerce")
    return cast(pd.DataFrame, out)


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
