"""Functional tests for reading all supported data formats, auto-detection, and validation."""

import pandas as pd
import pytest

import pyprideap
from pyprideap.core import AffinityDataset, Level, Platform
from pyprideap.io.readers.registry import detect_format, read

# ---------------------------------------------------------------------------
# Format auto-detection
# ---------------------------------------------------------------------------


class TestFormatDetection:
    def test_detects_all_formats(
        self, olink_csv_path, olink_parquet_path, somascan_adat_path, olink_xlsx_path, somascan_csv_path
    ):
        assert detect_format(olink_csv_path) == "olink_csv"
        assert detect_format(olink_parquet_path) == "olink_parquet"
        assert detect_format(somascan_adat_path) == "somascan_adat"
        assert detect_format(olink_xlsx_path) == "olink_xlsx"
        assert detect_format(somascan_csv_path) == "somascan_csv"

    def test_unknown_format_raises(self, tmp_path):
        f = tmp_path / "unknown.xyz"
        f.write_text("garbage")
        with pytest.raises(ValueError, match="Cannot detect format"):
            detect_format(f)


# ---------------------------------------------------------------------------
# Reading all formats via public API
# ---------------------------------------------------------------------------


class TestReadOlink:
    def test_read_olink_csv(self, olink_csv_path):
        ds = read(olink_csv_path)
        assert isinstance(ds, AffinityDataset)
        assert ds.platform == Platform.OLINK_TARGET
        assert len(ds.samples) == 4
        assert len(ds.features) == 3
        assert ds.expression.shape == (4, 3)
        assert "SampleID" in ds.samples.columns
        assert "OlinkID" in ds.features.columns
        assert "UniProt" in ds.features.columns
        # NPX values match expected
        sample_idx = ds.samples[ds.samples["SampleID"] == "S001"].index[0]
        assert ds.expression.loc[sample_idx, "OID00001"] == pytest.approx(3.45)
        # FAIL QC samples have NaN expression
        fail_idx = ds.samples[ds.samples["SampleID"] == "S003"].index[0]
        assert ds.expression.loc[fail_idx].isna().all()

    def test_read_olink_parquet(self, olink_parquet_path):
        ds = read(olink_parquet_path)
        assert isinstance(ds, AffinityDataset)
        assert ds.platform == Platform.OLINK_TARGET
        assert len(ds.samples) == 2
        assert len(ds.features) == 2
        assert ds.expression.shape == (2, 2)

    def test_read_olink_xlsx(self, olink_xlsx_path):
        ds = read(olink_xlsx_path)
        assert isinstance(ds, AffinityDataset)
        assert len(ds.samples) == 2
        assert len(ds.features) == 2


class TestReadSomascan:
    def test_read_somascan_adat(self, somascan_adat_path):
        ds = read(somascan_adat_path)
        assert isinstance(ds, AffinityDataset)
        assert ds.platform == Platform.SOMASCAN
        assert len(ds.samples) == 3
        assert len(ds.features) == 3
        assert ds.expression.shape == (3, 3)
        assert "SampleId" in ds.samples.columns
        assert "SeqId" in ds.features.columns
        # RFU values are positive
        assert (ds.expression > 0).all().all()
        # Header metadata extracted
        assert ds.metadata["AssayVersion"] == "v4.1"

    def test_read_somascan_csv(self, somascan_csv_path):
        ds = read(somascan_csv_path)
        assert isinstance(ds, AffinityDataset)
        assert ds.platform == Platform.SOMASCAN
        assert len(ds.samples) == 3
        assert (ds.expression > 0).all().all()


class TestReadWithPlatformOverride:
    def test_forced_platform(self, olink_csv_path):
        ds = read(olink_csv_path, platform="olink")
        assert isinstance(ds, AffinityDataset)
        assert len(ds.samples) > 0

    def test_invalid_platform_raises(self, olink_csv_path):
        with pytest.raises(ValueError, match="platform must be"):
            read(olink_csv_path, platform="invalid")


class TestFileNotFound:
    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            read("nonexistent.npx.csv")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_olink_valid_dataset(self, olink_csv_path):
        ds = read(olink_csv_path)
        results = pyprideap.validate(ds)
        errors = [r for r in results if r.level == Level.ERROR]
        assert len(errors) == 0

    def test_somascan_valid_dataset(self, somascan_adat_path):
        ds = read(somascan_adat_path)
        results = pyprideap.validate(ds)
        errors = [r for r in results if r.level == Level.ERROR]
        assert len(errors) == 0

    def test_olink_missing_column_detected(self):
        ds = AffinityDataset(
            platform=Platform.OLINK_EXPLORE,
            samples=pd.DataFrame({"SampleID": ["S001"]}),  # missing SampleType, SampleQC
            features=pd.DataFrame({"OlinkID": ["OID1"], "UniProt": ["P12345"], "Panel": ["Inf"]}),
            expression=pd.DataFrame({"OID1": [3.5]}),
        )
        results = pyprideap.validate(ds)
        errors = [r for r in results if r.level == Level.ERROR]
        assert any("SampleType" in r.message for r in errors)

    def test_somascan_negative_rfu_detected(self):
        ds = AffinityDataset(
            platform=Platform.SOMASCAN,
            samples=pd.DataFrame({"SampleId": ["S1"], "SampleType": ["Sample"]}),
            features=pd.DataFrame({"SeqId": ["10000-1"], "UniProt": ["P1"], "Target": ["T1"], "Dilution": ["20"]}),
            expression=pd.DataFrame({"SL1": [-100.0]}),
            metadata={"AssayVersion": "v4.1"},
        )
        results = pyprideap.validate(ds)
        errors = [r for r in results if r.level == Level.ERROR]
        assert any("positive" in r.message.lower() for r in errors)

    def test_empty_expression_detected(self):
        ds = AffinityDataset(
            platform=Platform.OLINK_EXPLORE,
            samples=pd.DataFrame({"SampleID": ["S1"], "SampleType": ["SAMPLE"], "SampleQC": ["PASS"]}),
            features=pd.DataFrame({"OlinkID": ["OID1"], "UniProt": ["P1"], "Panel": ["Inf"]}),
            expression=pd.DataFrame(),
        )
        results = pyprideap.validate(ds)
        errors = [r for r in results if r.level == Level.ERROR]
        assert any("empty" in r.rule for r in errors)


# ---------------------------------------------------------------------------
# End-to-end: read → validate → stats
# ---------------------------------------------------------------------------


class TestEndToEnd:
    @pytest.mark.parametrize(
        "fixture",
        ["olink_csv_path", "olink_parquet_path", "somascan_adat_path", "somascan_csv_path"],
    )
    def test_read_validate_stats(self, fixture, request):
        path = request.getfixturevalue(fixture)
        ds = pyprideap.read(path)
        results = pyprideap.validate(ds)
        stats = pyprideap.compute_stats(ds)
        assert stats.n_samples > 0
        assert stats.n_features > 0
        errors = [r for r in results if r.level == Level.ERROR]
        assert len(errors) == 0
