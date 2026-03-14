"""Functional tests for QC pipeline: compute metrics, render plots, and generate HTML reports."""

import json
from dataclasses import asdict

import pandas as pd
import pytest

from pyprideap.core import AffinityDataset, Platform

plotly = pytest.importorskip("plotly")

from pyprideap.processing.lod import get_proteins_above_lod  # noqa: E402
from pyprideap.viz.qc.compute import (  # noqa: E402
    CorrelationData,
    CvDistributionData,
    DataCompletenessData,
    DistributionData,
    HeatmapData,
    LodAnalysisData,
    PcaData,
    QcLodSummaryData,
    UmapData,
    VolcanoData,
    compute_all,
    compute_correlation,
    compute_cv_distribution,
    compute_data_completeness,
    compute_distribution,
    compute_lod_analysis,
    compute_pca,
    compute_qc_summary,
)
from pyprideap.viz.qc.render import (  # noqa: E402
    render_correlation,
    render_cv_distribution,
    render_data_completeness,
    render_distribution,
    render_heatmap,
    render_lod_analysis,
    render_pca,
    render_qc_summary,
    render_umap,
    render_volcano,
)
from pyprideap.viz.qc.report import qc_report, qc_report_split  # noqa: E402

# ---------------------------------------------------------------------------
# Dataset builders
# ---------------------------------------------------------------------------


def _make_olink_dataset():
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


def _make_somascan_dataset():
    return AffinityDataset(
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


# ---------------------------------------------------------------------------
# Compute: metrics from datasets
# ---------------------------------------------------------------------------


class TestComputeMetrics:
    def test_distribution_olink(self):
        result = compute_distribution(_make_olink_dataset())
        assert result.xlabel == "NPX Value"
        assert len(result.sample_ids) == 5
        assert len(result.sample_values) == 5

    def test_distribution_somascan(self):
        result = compute_distribution(_make_somascan_dataset())
        assert "log10" in result.xlabel.lower()
        assert len(result.sample_values) == 2

    def test_qc_summary_olink(self):
        result = compute_qc_summary(_make_olink_dataset())
        assert result is not None
        assert any("PASS" in c for c in result.categories)

    def test_qc_summary_somascan_returns_none(self):
        assert compute_qc_summary(_make_somascan_dataset()) is None

    def test_lod_analysis_with_lod(self):
        ds = _make_olink_dataset()
        ds.features["LOD"] = [1.0, 0.5, 2.0]
        result = compute_lod_analysis(ds)
        assert result is not None
        assert len(result.assay_ids) == 3
        assert all(0 <= p <= 100 for p in result.above_lod_pct)

    def test_lod_analysis_without_lod(self):
        assert compute_lod_analysis(_make_somascan_dataset()) is None

    def test_proteins_above_lod(self):
        ds = _make_olink_dataset()
        ds.features["LOD"] = [1.0, 0.5, 2.0]
        result = get_proteins_above_lod(ds)
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(p, str) for p in result)
        # All returned accessions should be from the features UniProt column
        assert all(p in ds.features["UniProt"].values for p in result)
        # Result should be sorted
        assert result == sorted(result)

    def test_proteins_above_lod_no_lod(self):
        ds = _make_somascan_dataset()
        result = get_proteins_above_lod(ds)
        assert result == []

    def test_pca(self):
        ds = _make_olink_dataset()
        result = compute_pca(ds)
        if result is None:
            pytest.skip("scikit-learn not installed")
        assert len(result.pc1) == 5
        assert result.labels == [f"S{i}" for i in range(5)]

    def test_correlation_square_matrix(self):
        ds = _make_olink_dataset()
        result = compute_correlation(ds)
        n = len(ds.samples)
        assert len(result.matrix) == n
        assert len(result.matrix[0]) == n
        for i in range(n):
            assert abs(result.matrix[i][i] - 1.0) < 1e-6

    def test_data_completeness_with_lod(self):
        ds = _make_olink_dataset()
        ds.features["LOD"] = [3.0, 3.0, 3.0]
        result = compute_data_completeness(ds)
        assert result is not None
        assert result.above_lod_rate[0] + result.below_lod_rate[0] == pytest.approx(1.0)

    def test_data_completeness_without_lod(self):
        assert compute_data_completeness(_make_somascan_dataset()) is None

    def test_cv_distribution(self):
        for ds in [_make_olink_dataset(), _make_somascan_dataset()]:
            result = compute_cv_distribution(ds)
            assert result is not None
            assert len(result.cv_values) > 0

    def test_compute_all_olink(self):
        result = compute_all(_make_olink_dataset())
        assert "distribution" in result
        assert "correlation" in result
        assert all(v is not None for v in result.values())

    def test_compute_all_somascan(self):
        result = compute_all(_make_somascan_dataset())
        assert "cv_distribution" in result
        assert result["cv_distribution"] is not None


# ---------------------------------------------------------------------------
# Dataclass serialization
# ---------------------------------------------------------------------------


class TestDataclassSerialization:
    def test_all_dataclasses_json_serializable(self):
        instances = [
            DistributionData(sample_ids=["S1"], sample_values=[[1.0]], xlabel="x"),
            QcLodSummaryData(categories=["PASS"], counts=[10]),
            LodAnalysisData(assay_ids=["A1"], above_lod_pct=[90.0], panel=["P1"]),
            PcaData(pc1=[1.0], pc2=[2.0], variance_explained=[0.5, 0.3], labels=["S1"], groups=["G1"]),
            CorrelationData(matrix=[[1.0]], labels=["S1"]),
            DataCompletenessData(
                sample_ids=["S1"],
                above_lod_rate=[0.8],
                below_lod_rate=[0.2],
                protein_ids=["P1"],
                missing_freq=[0.1],
            ),
            CvDistributionData(feature_ids=["F1"], cv_values=[0.15]),
        ]
        for inst in instances:
            serialized = json.dumps(asdict(inst))
            assert isinstance(serialized, str)


# ---------------------------------------------------------------------------
# Render: dataclass → Plotly figure
# ---------------------------------------------------------------------------


class TestRenderPlots:
    def test_render_distribution(self):
        data = DistributionData(sample_ids=["S1", "S2"], sample_values=[[1.0, 2.0], [1.5, 2.5]], xlabel="NPX")
        assert render_distribution(data) is not None

    def test_render_qc_summary(self):
        data = QcLodSummaryData(categories=["PASS", "FAIL"], counts=[10, 2])
        assert render_qc_summary(data) is not None

    def test_render_lod_analysis(self):
        data = LodAnalysisData(assay_ids=["A1", "A2"], above_lod_pct=[95.0, 80.0], panel=["Inf", "Inf"])
        assert render_lod_analysis(data) is not None

    def test_render_pca(self):
        data = PcaData(
            pc1=[1.0, 2.0],
            pc2=[3.0, 4.0],
            variance_explained=[0.6, 0.3],
            labels=["S1", "S2"],
            groups=["A", "B"],
        )
        assert render_pca(data) is not None

    def test_render_correlation(self):
        data = CorrelationData(matrix=[[1.0, 0.5], [0.5, 1.0]], labels=["S1", "S2"])
        assert render_correlation(data) is not None

    def test_render_data_completeness(self):
        data = DataCompletenessData(
            sample_ids=["S1", "S2"],
            above_lod_rate=[0.7, 0.8],
            below_lod_rate=[0.3, 0.2],
            protein_ids=["P1", "P2"],
            missing_freq=[0.1, 0.4],
        )
        assert render_data_completeness(data) is not None

    def test_render_cv_distribution(self):
        data = CvDistributionData(feature_ids=["F1", "F2"], cv_values=[0.1, 0.2])
        assert render_cv_distribution(data) is not None

    def test_render_umap(self):
        data = UmapData(x=[1.0, 2.0], y=[3.0, 4.0], labels=["S1", "S2"], groups=["A", "B"])
        assert render_umap(data) is not None

    def test_render_heatmap(self):
        data = HeatmapData(
            values=[[0.5, -0.3], [-0.5, 0.3]],
            sample_labels=["S1", "S2"],
            protein_labels=["P1", "P2"],
            sample_order=[0, 1],
            protein_order=[1, 0],
        )
        assert render_heatmap(data) is not None

    def test_render_volcano(self):
        data = VolcanoData(
            protein_ids=["P1", "P2", "P3"],
            assay_names=["A1", "A2", "A3"],
            fold_change=[2.0, -1.5, 0.1],
            neg_log10_pval=[3.0, 2.5, 0.5],
            significant=[True, True, False],
            direction=["up", "down", "ns"],
        )
        assert render_volcano(data) is not None


# ---------------------------------------------------------------------------
# Report: end-to-end HTML generation
# ---------------------------------------------------------------------------


class TestQcReport:
    def test_generates_html_with_plots(self, tmp_path):
        ds = _make_olink_dataset()
        output = tmp_path / "report.html"
        result = qc_report(ds, output)
        assert result.exists()
        content = result.read_text()
        assert "<html" in content
        assert "Distribution" in content
        assert "plotly" in content.lower()

    def test_contains_platform_and_pride_styling(self, tmp_path):
        ds = _make_olink_dataset()
        output = tmp_path / "report.html"
        qc_report(ds, output)
        content = output.read_text()
        assert "Olink Explore" in content
        assert "pride-embedded" in content
        assert "#5bc0be" in content

    def test_contains_postmessage_and_resize(self, tmp_path):
        ds = _make_olink_dataset()
        output = tmp_path / "report.html"
        qc_report(ds, output)
        content = output.read_text()
        assert "pride-qc-resize" in content
        assert "ResizeObserver" in content
        assert "window.parent.postMessage" in content

    def test_contains_empty_fallback(self, tmp_path):
        ds = _make_olink_dataset()
        output = tmp_path / "report.html"
        qc_report(ds, output)
        content = output.read_text()
        assert "pride-embedded-empty" in content

    def test_contains_summary_table(self, tmp_path):
        ds = _make_olink_dataset()
        output = tmp_path / "report.html"
        qc_report(ds, output)
        content = output.read_text()
        assert "Dataset Summary" in content
        assert "Detection rate" in content
        assert "Median CV" in content
        # At least one traffic-light status dot should be present
        assert any(dot in content for dot in ["dot-green", "dot-amber", "dot-red"])
        # QC Status should appear for Olink
        assert "PASS / WARN / FAIL" in content

    def test_split_report_creates_individual_files(self, tmp_path):
        ds = _make_olink_dataset()
        output_dir = tmp_path / "plots"
        result = qc_report_split(ds, output_dir)
        assert result.is_dir()
        # Core files should always exist
        assert (output_dir / "summary.html").exists()
        assert (output_dir / "distribution.html").exists()
        assert (output_dir / "correlation.html").exists()
        assert (output_dir / "lod_sources.html").exists()
        # Each file should be valid standalone HTML
        for html_file in output_dir.glob("*.html"):
            content = html_file.read_text()
            assert "<html" in content
            assert "</html>" in content
        # Summary should contain the table
        summary = (output_dir / "summary.html").read_text()
        assert "Dataset Summary" in summary
        assert "Detection rate" in summary

    def test_somascan_report(self, tmp_path):
        ds = _make_somascan_dataset()
        output = tmp_path / "somascan_report.html"
        result = qc_report(ds, output)
        assert result.exists()
        assert "Somascan" in result.read_text()
