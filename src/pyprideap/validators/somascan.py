from __future__ import annotations

import numpy as np
import pandas as pd

from pyprideap.core import AffinityDataset, Level, ValidationResult

_REQUIRED_FEATURE_COLS = {"SeqId", "UniProt", "Target", "Dilution"}
_REQUIRED_SAMPLE_COLS = {"SampleId", "SampleType"}
_EXPECTED_HEADER_KEYS = {"AssayVersion", "AssayType"}


class SomaScanValidator:
    def validate(self, dataset: AffinityDataset) -> list[ValidationResult]:
        results: list[ValidationResult] = []
        results.extend(self._check_schema(dataset))
        results.extend(self._check_rfu_positive(dataset))
        results.extend(self._check_not_empty(dataset))
        results.extend(self._check_header_metadata(dataset))
        return results

    def _check_schema(self, ds: AffinityDataset) -> list[ValidationResult]:
        results = []
        for col in _REQUIRED_SAMPLE_COLS:
            if col not in ds.samples.columns:
                results.append(
                    ValidationResult(
                        level=Level.ERROR,
                        rule="somascan.schema.missing_sample_column",
                        message=f"Missing required sample column: {col}",
                        details={"column": col},
                    )
                )
        for col in _REQUIRED_FEATURE_COLS:
            if col not in ds.features.columns:
                results.append(
                    ValidationResult(
                        level=Level.ERROR,
                        rule="somascan.schema.missing_feature_column",
                        message=f"Missing required feature column: {col}",
                        details={"column": col},
                    )
                )
        return results

    def _check_rfu_positive(self, ds: AffinityDataset) -> list[ValidationResult]:
        if ds.expression.empty:
            return []
        values = np.asarray(pd.to_numeric(pd.Series(ds.expression.values.flatten()), errors="coerce").dropna())
        if len(values) == 0:
            return []
        negative = values < 0
        if negative.any():
            return [
                ValidationResult(
                    level=Level.ERROR,
                    rule="somascan.rfu.positive",
                    message=f"RFU values must be positive: {int(negative.sum())} negative values found",
                    details={"count": int(negative.sum()), "min": float(values.min())},
                )
            ]
        return []

    def _check_not_empty(self, ds: AffinityDataset) -> list[ValidationResult]:
        if ds.expression.empty:
            return [
                ValidationResult(
                    level=Level.ERROR,
                    rule="somascan.expression.empty",
                    message="Expression matrix is empty — at least one data record required",
                )
            ]
        return []

    def _check_header_metadata(self, ds: AffinityDataset) -> list[ValidationResult]:
        results = []
        for key in _EXPECTED_HEADER_KEYS:
            if key not in ds.metadata:
                results.append(
                    ValidationResult(
                        level=Level.WARNING,
                        rule="somascan.metadata.missing_header",
                        message=f"Missing expected ADAT header key: {key}",
                        details={"key": key},
                    )
                )
        return results
