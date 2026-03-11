from pyprideap.core import AffinityDataset, Platform, ValidationResult
from pyprideap.validators.base import BaseValidator
from pyprideap.validators.olink_explore import OlinkExploreValidator
from pyprideap.validators.olink_target import OlinkTargetValidator
from pyprideap.validators.somascan import SomaScanValidator

_VALIDATORS: dict[Platform, type[BaseValidator]] = {
    Platform.OLINK_EXPLORE: OlinkExploreValidator,
    Platform.OLINK_EXPLORE_HT: OlinkExploreValidator,
    Platform.OLINK_TARGET: OlinkTargetValidator,
    Platform.SOMASCAN: SomaScanValidator,
}


def validate(dataset: AffinityDataset) -> list[ValidationResult]:
    validator_cls = _VALIDATORS.get(dataset.platform)
    if validator_cls is None:
        raise ValueError(f"No validator for platform: {dataset.platform}")
    return validator_cls().validate(dataset)


__all__ = [
    "validate",
    "OlinkExploreValidator",
    "OlinkTargetValidator",
    "SomaScanValidator",
]
