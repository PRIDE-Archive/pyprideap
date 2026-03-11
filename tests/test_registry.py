import pytest

from pyprideap.core import AffinityDataset, Platform
from pyprideap.readers.registry import detect_format, read


class TestDetectFormat:
    def test_adat_detected(self, somascan_adat_path):
        fmt = detect_format(somascan_adat_path)
        assert fmt == "somascan_adat"

    def test_parquet_detected(self, olink_parquet_path):
        fmt = detect_format(olink_parquet_path)
        assert fmt == "olink_parquet"

    def test_npx_csv_detected(self, olink_csv_path):
        fmt = detect_format(olink_csv_path)
        assert fmt == "olink_csv"

    def test_somascan_csv_detected(self, somascan_csv_path):
        fmt = detect_format(somascan_csv_path)
        assert fmt == "somascan_csv"

    def test_xlsx_detected(self, olink_xlsx_path):
        fmt = detect_format(olink_xlsx_path)
        assert fmt == "olink_xlsx"

    def test_unknown_raises(self, tmp_path):
        f = tmp_path / "unknown.xyz"
        f.write_text("garbage")
        with pytest.raises(ValueError, match="Cannot detect format"):
            detect_format(f)


class TestAutoRead:
    def test_read_olink_csv(self, olink_csv_path):
        ds = read(olink_csv_path)
        assert isinstance(ds, AffinityDataset)
        assert ds.platform == Platform.OLINK_EXPLORE

    def test_read_somascan_adat(self, somascan_adat_path):
        ds = read(somascan_adat_path)
        assert ds.platform == Platform.SOMASCAN

    def test_read_parquet(self, olink_parquet_path):
        ds = read(olink_parquet_path)
        assert ds.platform == Platform.OLINK_EXPLORE_HT

    def test_read_xlsx(self, olink_xlsx_path):
        ds = read(olink_xlsx_path)
        assert isinstance(ds, AffinityDataset)
        assert ds.platform == Platform.OLINK_EXPLORE

    def test_read_somascan_csv(self, somascan_csv_path):
        ds = read(somascan_csv_path)
        assert isinstance(ds, AffinityDataset)
        assert ds.platform == Platform.SOMASCAN
