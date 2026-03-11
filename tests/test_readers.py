from pathlib import Path

import pytest

from pyprideap.core import AffinityDataset, Platform
from pyprideap.readers.olink_csv import read_olink_csv
from pyprideap.readers.olink_parquet import read_olink_parquet
from pyprideap.readers.olink_xlsx import read_olink_xlsx
from pyprideap.readers.somascan_adat import read_somascan_adat
from pyprideap.readers.somascan_csv import read_somascan_csv


class TestOlinkCsvReader:
    def test_returns_affinity_dataset(self, olink_csv_path):
        ds = read_olink_csv(olink_csv_path)
        assert isinstance(ds, AffinityDataset)
        assert ds.platform == Platform.OLINK_EXPLORE

    def test_samples_extracted(self, olink_csv_path):
        ds = read_olink_csv(olink_csv_path)
        assert len(ds.samples) == 4
        assert "SampleID" in ds.samples.columns
        assert "SampleType" in ds.samples.columns

    def test_features_extracted(self, olink_csv_path):
        ds = read_olink_csv(olink_csv_path)
        assert len(ds.features) == 3
        assert "OlinkID" in ds.features.columns
        assert "UniProt" in ds.features.columns
        assert "Panel" in ds.features.columns

    def test_expression_matrix_shape(self, olink_csv_path):
        ds = read_olink_csv(olink_csv_path)
        assert ds.expression.shape == (4, 3)

    def test_expression_values_are_npx(self, olink_csv_path):
        ds = read_olink_csv(olink_csv_path)
        sample_idx = ds.samples[ds.samples["SampleID"] == "S001"].index[0]
        feature_col = "OID00001"
        assert ds.expression.loc[sample_idx, feature_col] == pytest.approx(3.45)

    def test_failed_qc_samples_have_nan(self, olink_csv_path):
        ds = read_olink_csv(olink_csv_path)
        sample_idx = ds.samples[ds.samples["SampleID"] == "S003"].index[0]
        assert ds.expression.loc[sample_idx].isna().all()

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            read_olink_csv("nonexistent.csv")


class TestOlinkParquetReader:
    def test_returns_affinity_dataset(self, olink_parquet_path):
        ds = read_olink_parquet(olink_parquet_path)
        assert isinstance(ds, AffinityDataset)
        assert ds.platform == Platform.OLINK_EXPLORE_HT

    def test_samples_extracted(self, olink_parquet_path):
        ds = read_olink_parquet(olink_parquet_path)
        assert len(ds.samples) == 2
        assert "SampleID" in ds.samples.columns

    def test_features_extracted(self, olink_parquet_path):
        ds = read_olink_parquet(olink_parquet_path)
        assert len(ds.features) == 2
        assert "OlinkID" in ds.features.columns
        assert "UniProt" in ds.features.columns

    def test_expression_matrix_shape(self, olink_parquet_path):
        ds = read_olink_parquet(olink_parquet_path)
        assert ds.expression.shape == (2, 2)

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            read_olink_parquet("nonexistent.parquet")


class TestSomascanAdatReader:
    def test_returns_affinity_dataset(self, somascan_adat_path):
        ds = read_somascan_adat(somascan_adat_path)
        assert isinstance(ds, AffinityDataset)
        assert ds.platform == Platform.SOMASCAN

    def test_header_metadata_extracted(self, somascan_adat_path):
        ds = read_somascan_adat(somascan_adat_path)
        assert "AssayVersion" in ds.metadata
        assert ds.metadata["AssayVersion"] == "v4.1"

    def test_samples_extracted(self, somascan_adat_path):
        ds = read_somascan_adat(somascan_adat_path)
        assert len(ds.samples) == 3
        assert "SampleId" in ds.samples.columns

    def test_features_extracted(self, somascan_adat_path):
        ds = read_somascan_adat(somascan_adat_path)
        assert len(ds.features) == 3
        assert "SeqId" in ds.features.columns
        assert "UniProt" in ds.features.columns
        assert "Target" in ds.features.columns

    def test_expression_matrix_shape(self, somascan_adat_path):
        ds = read_somascan_adat(somascan_adat_path)
        assert ds.expression.shape == (3, 3)

    def test_expression_values_are_rfu(self, somascan_adat_path):
        ds = read_somascan_adat(somascan_adat_path)
        assert (ds.expression > 0).all().all()

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            read_somascan_adat("nonexistent.adat")


class TestSomascanCsvReader:
    def test_returns_affinity_dataset(self, somascan_csv_path):
        ds = read_somascan_csv(somascan_csv_path)
        assert isinstance(ds, AffinityDataset)
        assert ds.platform == Platform.SOMASCAN

    def test_samples_extracted(self, somascan_csv_path):
        ds = read_somascan_csv(somascan_csv_path)
        assert len(ds.samples) == 3
        assert "SampleId" in ds.samples.columns

    def test_features_extracted(self, somascan_csv_path):
        ds = read_somascan_csv(somascan_csv_path)
        assert len(ds.features) == 3
        assert "SeqId" in ds.features.columns

    def test_expression_values_positive(self, somascan_csv_path):
        ds = read_somascan_csv(somascan_csv_path)
        assert (ds.expression > 0).all().all()

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            read_somascan_csv("nonexistent.csv")


class TestOlinkXlsxReader:
    @pytest.fixture
    def olink_xlsx_path(self):
        return Path(__file__).parent / "data" / "olink_sample.xlsx"

    def test_returns_affinity_dataset(self, olink_xlsx_path):
        ds = read_olink_xlsx(olink_xlsx_path)
        assert isinstance(ds, AffinityDataset)

    def test_samples_and_features(self, olink_xlsx_path):
        ds = read_olink_xlsx(olink_xlsx_path)
        assert len(ds.samples) == 2
        assert len(ds.features) == 2
