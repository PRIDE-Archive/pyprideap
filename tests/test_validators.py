import numpy as np
import pandas as pd

from pyprideap.core import AffinityDataset, Level, Platform
from pyprideap.validators.olink_explore import OlinkExploreValidator
from pyprideap.validators.olink_target import OlinkTargetValidator
from pyprideap.validators.somascan import SomaScanValidator


def _make_olink_dataset(**overrides) -> AffinityDataset:
    defaults = {
        "platform": Platform.OLINK_EXPLORE,
        "samples": pd.DataFrame(
            {
                "SampleID": ["S001", "S002"],
                "SampleType": ["SAMPLE", "SAMPLE"],
                "SampleQC": ["PASS", "PASS"],
            }
        ),
        "features": pd.DataFrame(
            {
                "OlinkID": ["OID1", "OID2"],
                "UniProt": ["P12345", "Q67890"],
                "Panel": ["Inflammation", "Inflammation"],
            }
        ),
        "expression": pd.DataFrame(
            {
                "OID1": [3.5, 4.1],
                "OID2": [2.0, -0.5],
            }
        ),
        "metadata": {},
    }
    defaults.update(overrides)
    return AffinityDataset(**defaults)


class TestOlinkExploreValidator:
    def test_valid_dataset_no_errors(self):
        ds = _make_olink_dataset()
        results = OlinkExploreValidator().validate(ds)
        errors = [r for r in results if r.level == Level.ERROR]
        assert len(errors) == 0

    def test_missing_sample_column(self):
        ds = _make_olink_dataset(
            samples=pd.DataFrame({"SampleID": ["S001"]}),
        )
        results = OlinkExploreValidator().validate(ds)
        errors = [r for r in results if r.level == Level.ERROR]
        assert any("SampleType" in r.message for r in errors)

    def test_missing_feature_column(self):
        ds = _make_olink_dataset(
            features=pd.DataFrame({"OlinkID": ["OID1"]}),
        )
        results = OlinkExploreValidator().validate(ds)
        errors = [r for r in results if r.level == Level.ERROR]
        assert any("UniProt" in r.message for r in errors)

    def test_invalid_sample_qc_value(self):
        ds = _make_olink_dataset(
            samples=pd.DataFrame(
                {
                    "SampleID": ["S001"],
                    "SampleType": ["SAMPLE"],
                    "SampleQC": ["INVALID"],
                }
            ),
        )
        results = OlinkExploreValidator().validate(ds)
        errors = [r for r in results if r.level == Level.ERROR]
        assert any("SampleQC" in r.rule for r in errors)

    def test_fail_qc_with_non_nan_npx_error(self):
        ds = _make_olink_dataset(
            samples=pd.DataFrame(
                {
                    "SampleID": ["S001"],
                    "SampleType": ["SAMPLE"],
                    "SampleQC": ["FAIL"],
                }
            ),
            expression=pd.DataFrame({"OID1": [3.5]}),
        )
        results = OlinkExploreValidator().validate(ds)
        errors = [r for r in results if r.level == Level.ERROR]
        assert any("qc_consistency" in r.rule for r in errors)

    def test_pass_qc_with_nan_npx_error(self):
        ds = _make_olink_dataset(
            samples=pd.DataFrame(
                {
                    "SampleID": ["S001"],
                    "SampleType": ["SAMPLE"],
                    "SampleQC": ["PASS"],
                }
            ),
            expression=pd.DataFrame({"OID1": [np.nan]}),
        )
        results = OlinkExploreValidator().validate(ds)
        errors = [r for r in results if r.level == Level.ERROR]
        assert any("qc_consistency" in r.rule for r in errors)

    def test_npx_out_of_range_warning(self):
        ds = _make_olink_dataset(
            expression=pd.DataFrame({"OID1": [50.0, 3.5]}),
        )
        results = OlinkExploreValidator().validate(ds)
        warnings = [r for r in results if r.level == Level.WARNING]
        assert any("range" in r.rule for r in warnings)

    def test_empty_expression_error(self):
        ds = _make_olink_dataset(
            expression=pd.DataFrame(),
        )
        results = OlinkExploreValidator().validate(ds)
        errors = [r for r in results if r.level == Level.ERROR]
        assert any("empty" in r.rule for r in errors)


def _make_olink_target_dataset(**overrides) -> AffinityDataset:
    defaults = {
        "platform": Platform.OLINK_TARGET,
        "samples": pd.DataFrame(
            {
                "SampleID": ["S001", "S002"],
                "SampleType": ["SAMPLE", "SAMPLE"],
                "SampleQC": ["PASS", "PASS"],
            }
        ),
        "features": pd.DataFrame(
            {
                "OlinkID": ["OID1"],
                "UniProt": ["P12345"],
                "Panel": ["Immuno-Oncology"],
            }
        ),
        "expression": pd.DataFrame({"OID1": [3.5, 4.1]}),
        "metadata": {},
    }
    defaults.update(overrides)
    return AffinityDataset(**defaults)


class TestOlinkTargetValidator:
    def test_valid_dataset_no_errors(self):
        ds = _make_olink_target_dataset()
        results = OlinkTargetValidator().validate(ds)
        errors = [r for r in results if r.level == Level.ERROR]
        assert len(errors) == 0

    def test_missing_required_columns(self):
        ds = _make_olink_target_dataset(
            features=pd.DataFrame({"OlinkID": ["OID1"]}),
        )
        results = OlinkTargetValidator().validate(ds)
        errors = [r for r in results if r.level == Level.ERROR]
        assert len(errors) > 0

    def test_qc_consistency_checked(self):
        ds = _make_olink_target_dataset(
            samples=pd.DataFrame(
                {
                    "SampleID": ["S001"],
                    "SampleType": ["SAMPLE"],
                    "SampleQC": ["FAIL"],
                }
            ),
            expression=pd.DataFrame({"OID1": [3.5]}),
        )
        results = OlinkTargetValidator().validate(ds)
        errors = [r for r in results if r.level == Level.ERROR]
        assert any("qc_consistency" in r.rule for r in errors)

    def test_rule_prefix_rewritten(self):
        ds = _make_olink_target_dataset(
            features=pd.DataFrame({"OlinkID": ["OID1"]}),
        )
        results = OlinkTargetValidator().validate(ds)
        for r in results:
            assert not r.rule.startswith("olink."), f"Rule should use olink_target prefix: {r.rule}"
            assert r.rule.startswith("olink_target."), f"Unexpected rule prefix: {r.rule}"


def _make_somascan_dataset(**overrides) -> AffinityDataset:
    defaults = {
        "platform": Platform.SOMASCAN,
        "samples": pd.DataFrame(
            {
                "SampleId": ["S001", "S002"],
                "SampleType": ["Sample", "Sample"],
            }
        ),
        "features": pd.DataFrame(
            {
                "SeqId": ["10000-1", "10001-2"],
                "UniProt": ["P05231", "P01375"],
                "Target": ["IL-6", "TNF"],
                "Dilution": ["20", "0.5"],
            }
        ),
        "expression": pd.DataFrame(
            {
                "SL000001": [1234.5, 1100.2],
                "SL000002": [5678.9, 4567.8],
            }
        ),
        "metadata": {"AssayVersion": "v4.1", "AssayType": "SomaScan"},
    }
    defaults.update(overrides)
    return AffinityDataset(**defaults)


class TestSomaScanValidator:
    def test_valid_dataset_no_errors(self):
        ds = _make_somascan_dataset()
        results = SomaScanValidator().validate(ds)
        errors = [r for r in results if r.level == Level.ERROR]
        assert len(errors) == 0

    def test_missing_feature_column(self):
        ds = _make_somascan_dataset(
            features=pd.DataFrame({"SeqId": ["10000-1"]}),
        )
        results = SomaScanValidator().validate(ds)
        errors = [r for r in results if r.level == Level.ERROR]
        assert any("UniProt" in r.message for r in errors)

    def test_negative_rfu_error(self):
        ds = _make_somascan_dataset(
            expression=pd.DataFrame({"SL000001": [-100.0, 1234.5]}),
        )
        results = SomaScanValidator().validate(ds)
        errors = [r for r in results if r.level == Level.ERROR]
        assert any("positive" in r.message.lower() for r in errors)

    def test_empty_expression_error(self):
        ds = _make_somascan_dataset(expression=pd.DataFrame())
        results = SomaScanValidator().validate(ds)
        errors = [r for r in results if r.level == Level.ERROR]
        assert any("empty" in r.rule for r in errors)

    def test_missing_header_metadata_warning(self):
        ds = _make_somascan_dataset(metadata={})
        results = SomaScanValidator().validate(ds)
        warnings = [r for r in results if r.level == Level.WARNING]
        assert any("AssayVersion" in r.message for r in warnings)
