from __future__ import annotations

from typing import Protocol

from pyprideap.core import AffinityDataset, ValidationResult


class BaseValidator(Protocol):
    def validate(self, dataset: AffinityDataset) -> list[ValidationResult]: ...
