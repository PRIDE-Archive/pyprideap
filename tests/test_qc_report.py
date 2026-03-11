import pytest

pytest.importorskip("plotly")

import pandas as pd

from pyprideap.core import AffinityDataset, Platform
from pyprideap.qc.report import qc_report


def _make_dataset():
    return AffinityDataset(
        platform=Platform.OLINK_EXPLORE,
        samples=pd.DataFrame(
            {
                "SampleID": [f"S{i}" for i in range(5)],
                "SampleType": ["SAMPLE"] * 5,
                "SampleQC": ["PASS", "PASS", "WARN", "PASS", "FAIL"],
            }
        ),
        features=pd.DataFrame(
            {
                "OlinkID": ["O1", "O2", "O3"],
                "UniProt": ["P1", "P2", "P3"],
                "Panel": ["Inf", "Inf", "Neuro"],
            }
        ),
        expression=pd.DataFrame(
            {
                "O1": [3.5, 4.1, 2.3, 5.0, 1.2],
                "O2": [2.0, -0.5, 3.1, 2.8, 0.9],
                "O3": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        ),
        metadata={},
    )


class TestQcReport:
    def test_generates_html_file(self, tmp_path):
        ds = _make_dataset()
        output = tmp_path / "report.html"
        result = qc_report(ds, output)
        assert result.exists()
        content = result.read_text()
        assert "<html" in content
        assert "Distribution" in content

    def test_contains_plotly_js(self, tmp_path):
        ds = _make_dataset()
        output = tmp_path / "report.html"
        qc_report(ds, output)
        content = (tmp_path / "report.html").read_text()
        assert "plotly" in content.lower()

    def test_contains_platform_info(self, tmp_path):
        ds = _make_dataset()
        output = tmp_path / "report.html"
        qc_report(ds, output)
        content = (tmp_path / "report.html").read_text()
        assert "Olink Explore" in content
        assert "Samples:" in content

    def test_somascan_report(self, tmp_path):
        ds = AffinityDataset(
            platform=Platform.SOMASCAN,
            samples=pd.DataFrame({"SampleId": ["S1", "S2"], "SampleType": ["Sample", "Sample"]}),
            features=pd.DataFrame(
                {
                    "SeqId": ["10000-1", "10001-2"],
                    "UniProt": ["P1", "P2"],
                    "Target": ["T1", "T2"],
                    "Dilution": ["20", "0.5"],
                }
            ),
            expression=pd.DataFrame({"SL1": [1234.5, 1100.2], "SL2": [5678.9, 4567.8]}),
            metadata={},
        )
        output = tmp_path / "somascan_report.html"
        result = qc_report(ds, output)
        assert result.exists()
        content = result.read_text()
        assert "Somascan" in content
