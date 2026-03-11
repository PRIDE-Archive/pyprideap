from __future__ import annotations

import numpy as np
import pandas as pd

from pyprideap.core import AffinityDataset, Level, ValidationResult

_REQUIRED_SAMPLE_COLS = {"SampleID", "SampleType", "SampleQC"}
_REQUIRED_FEATURE_COLS = {"OlinkID", "UniProt", "Panel"}
_VALID_QC_VALUES = {"NA", "PASS", "WARN", "FAIL"}
_NPX_MIN = -10.0
_NPX_MAX = 40.0


class OlinkExploreValidator:
    def validate(self, dataset: AffinityDataset) -> list[ValidationResult]:
        results: list[ValidationResult] = []
        results.extend(self._check_schema(dataset))
        results.extend(self._check_qc_values(dataset))
        results.extend(self._check_qc_consistency(dataset))
        results.extend(self._check_npx_range(dataset))
        results.extend(self._check_not_empty(dataset))
        return results

    def _check_schema(self, ds: AffinityDataset) -> list[ValidationResult]:
        results = []
        for col in _REQUIRED_SAMPLE_COLS:
            if col not in ds.samples.columns:
                results.append(
                    ValidationResult(
                        level=Level.ERROR,
                        rule="olink.schema.missing_sample_column",
                        message=f"Missing required sample column: {col}",
                        details={"column": col},
                    )
                )
        for col in _REQUIRED_FEATURE_COLS:
            if col not in ds.features.columns:
                results.append(
                    ValidationResult(
                        level=Level.ERROR,
                        rule="olink.schema.missing_feature_column",
                        message=f"Missing required feature column: {col}",
                        details={"column": col},
                    )
                )
        return results

    def _check_qc_values(self, ds: AffinityDataset) -> list[ValidationResult]:
        if "SampleQC" not in ds.samples.columns:
            return []
        invalid = set(ds.samples["SampleQC"].dropna().unique()) - _VALID_QC_VALUES
        if invalid:
            return [
                ValidationResult(
                    level=Level.ERROR,
                    rule="olink.SampleQC.invalid_values",
                    message=f"Invalid SampleQC values: {invalid}. Must be one of {_VALID_QC_VALUES}",
                    details={"invalid_values": list(invalid)},
                )
            ]
        return []

    def _check_qc_consistency(self, ds: AffinityDataset) -> list[ValidationResult]:
        if "SampleQC" not in ds.samples.columns:
            return []
        results = []
        for idx in range(len(ds.samples)):
            row = ds.samples.iloc[idx]
            qc = row.get("SampleQC")
            if idx >= len(ds.expression):
                continue
            expr_row = ds.expression.iloc[idx]
            if qc in ("FAIL", "NA"):
                if not expr_row.isna().all():
                    results.append(
                        ValidationResult(
                            level=Level.ERROR,
                            rule="olink.npx.qc_consistency",
                            message=f"Sample {row.get('SampleID', idx)}: SampleQC={qc} but NPX contains non-NaN values",
                            details={"sample_index": idx, "qc": qc},
                        )
                    )
            elif qc in ("PASS", "WARN"):
                if expr_row.isna().all():
                    results.append(
                        ValidationResult(
                            level=Level.ERROR,
                            rule="olink.npx.qc_consistency",
                            message=f"Sample {row.get('SampleID', idx)}: SampleQC={qc} but all NPX values are NaN",
                            details={"sample_index": idx, "qc": qc},
                        )
                    )
        return results

    def _check_npx_range(self, ds: AffinityDataset) -> list[ValidationResult]:
        if ds.expression.empty:
            return []
        values = np.asarray(pd.to_numeric(pd.Series(ds.expression.values.flatten()), errors="coerce").dropna())
        if len(values) == 0:
            return []
        out_of_range = (values < _NPX_MIN) | (values > _NPX_MAX)
        if out_of_range.any():
            return [
                ValidationResult(
                    level=Level.WARNING,
                    rule="olink.npx.range",
                    message=f"NPX values outside expected log2 range [{_NPX_MIN}, {_NPX_MAX}]: "
                    f"{int(out_of_range.sum())} values out of range",
                    details={"count": int(out_of_range.sum()), "min": float(values.min()), "max": float(values.max())},
                )
            ]
        return []

    def _check_not_empty(self, ds: AffinityDataset) -> list[ValidationResult]:
        if ds.expression.empty:
            return [
                ValidationResult(
                    level=Level.ERROR,
                    rule="olink.expression.empty",
                    message="Expression matrix is empty — at least one data record required",
                )
            ]
        return []
