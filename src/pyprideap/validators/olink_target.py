from __future__ import annotations

from pyprideap.core import AffinityDataset, ValidationResult
from pyprideap.validators.olink_explore import OlinkExploreValidator


class OlinkTargetValidator(OlinkExploreValidator):
    _rule_prefix = "olink_target"

    def validate(self, dataset: AffinityDataset) -> list[ValidationResult]:
        results = super().validate(dataset)
        return [
            ValidationResult(
                level=r.level,
                rule=r.rule.replace("olink.", f"{self._rule_prefix}.", 1),
                message=r.message,
                details=r.details,
            )
            for r in results
        ]
