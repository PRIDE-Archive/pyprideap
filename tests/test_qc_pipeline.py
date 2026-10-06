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
        assert "Features (assays)" in content
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
        # Each file should be valid standalone HTML
        for html_file in output_dir.glob("*.html"):
            content = html_file.read_text()
            assert "<html" in content
            assert "</html>" in content
        # Summary should contain the table
        summary = (output_dir / "summary.html").read_text()
        assert "Dataset Summary" in summary
        assert "Features (assays)" in summary

    def test_split_report_embedding_layout(self, tmp_path):
        from plotly.offline import get_plotlyjs_version

        embedded = qc_report_split(_make_olink_dataset(), tmp_path / "embedded", no_border=True)
        page = (embedded / "distribution.html").read_text()
        # plotly.js matches the installed plotly package, not a pinned old version
        assert f"plotly-{get_plotlyjs_version()}.min.js" in page
        # Compact, content-height layout for iframes; no fixed 1100px card wrapper
        assert "height: auto; min-height: 0;" in page
        assert "padding:28px 36px" not in page
        assert '"t":40' in page.replace(" ", "")

        bordered = qc_report_split(_make_olink_dataset(), tmp_path / "bordered", no_border=False)
        assert "padding:28px 36px" in (bordered / "distribution.html").read_text()

    def test_somascan_report(self, tmp_path):
        ds = _make_somascan_dataset()
        output = tmp_path / "somascan_report.html"
        result = qc_report(ds, output)
        assert result.exists()
        assert "Somascan" in result.read_text()


# ---------------------------------------------------------------------------
# LOD source resolution
# ---------------------------------------------------------------------------


def _make_somascan_with_buffers(n_bio=5, n_buffers=12):
    import numpy as np

    rng = np.random.default_rng(0)
    n = n_bio + n_buffers
    expr = np.abs(np.vstack([rng.normal(5000, 1000, (n_bio, 3)), rng.normal(100, 20, (n_buffers, 3))]))
    return AffinityDataset(
        platform=Platform.SOMASCAN,
        samples=pd.DataFrame(
            {"SampleId": [f"S{i}" for i in range(n)], "SampleType": ["Sample"] * n_bio + ["Buffer"] * n_buffers}
        ),
        features=pd.DataFrame({"SeqId": ["1-1", "2-2", "3-3"], "UniProt": ["P1", "P2", "P3"]}),
        expression=pd.DataFrame(expr, columns=["SL1", "SL2", "SL3"]),
        metadata={},
    )


def _make_olink_with_negative_controls(n_bio=5, n_nc=10):
    import numpy as np

    rng = np.random.default_rng(0)
    n = n_bio + n_nc
    expr = np.vstack([rng.normal(5, 1, (n_bio, 3)), rng.normal(0.5, 0.2, (n_nc, 3))])
    return AffinityDataset(
        platform=Platform.OLINK_EXPLORE,
        samples=pd.DataFrame(
            {
                "SampleID": [f"S{i}" for i in range(n)],
                "SampleType": ["SAMPLE"] * n_bio + ["NEGATIVE_CONTROL"] * n_nc,
                "SampleQC": ["PASS"] * n,
            }
        ),
        features=pd.DataFrame({"OlinkID": ["O1", "O2", "O3"], "UniProt": ["P1", "P2", "P3"], "Panel": ["Inf"] * 3}),
        expression=pd.DataFrame(expr, columns=["O1", "O2", "O3"]),
        metadata={},
    )


class TestLodResolution:
    def test_somascan_uses_elod_not_nclod_with_many_buffers(self):
        from pyprideap.processing.lod import compute_soma_elod
        from pyprideap.viz.qc.compute import resolve_lod_with_source

        ds = _make_somascan_with_buffers()
        lod, source = resolve_lod_with_source(ds)
        assert source == "eLOD"
        pd.testing.assert_series_equal(lod, compute_soma_elod(ds))

    def test_report_active_source_matches_resolver(self):
        from pyprideap.viz.qc.compute import resolve_lod_with_source
        from pyprideap.viz.qc.report import _lod_source_info

        for ds in (_make_somascan_with_buffers(), _make_olink_with_negative_controls()):
            assert _lod_source_info(ds)["active"] == resolve_lod_with_source(ds)[1]
        assert _lod_source_info(_make_olink_with_negative_controls())["active"] == "NCLOD"

    def test_qc_summary_uses_resolved_lod(self):
        # No LOD column in the file: the summary must still split by NCLOD
        result = compute_qc_summary(_make_olink_with_negative_controls())
        assert result is not None
        assert any("LOD" in c for c in result.categories)


# ---------------------------------------------------------------------------
# CV definitions (distribution and plate-level share one definition)
# ---------------------------------------------------------------------------


def _make_olink_two_plates():
    import numpy as np

    rng = np.random.default_rng(1)
    n_per_plate = 6
    types = (["SAMPLE"] * 5 + ["NEGATIVE"]) * 2
    expr = rng.normal(5, 1, (2 * n_per_plate, 2))
    expr[[5, 11], :] = -3.0  # controls: far from study samples
    return AffinityDataset(
        platform=Platform.OLINK_EXPLORE,
        samples=pd.DataFrame(
            {
                "SampleID": [f"S{i}" for i in range(2 * n_per_plate)],
                "SampleType": types,
                "PlateID": ["P1"] * n_per_plate + ["P2"] * n_per_plate,
            }
        ),
        features=pd.DataFrame({"OlinkID": ["O1", "O2"], "UniProt": ["P1", "P2"], "Panel": ["Inf"] * 2}),
        expression=pd.DataFrame(expr, columns=["O1", "O2"]),
        metadata={},
    )


class TestCvDefinitions:
    def test_cv_distribution_linear_scale_study_samples_only(self):
        import numpy as np

        ds = _make_olink_two_plates()
        study = ds.expression[ds.samples["SampleType"] == "SAMPLE"]
        expected = (2**study).std() / (2**study).mean()
        result = compute_cv_distribution(ds)
        assert result.cv_values == pytest.approx(expected.tolist())
        assert np.all(np.array(result.cv_values) > 0)

    def test_plate_cv_olink_plateid_linear_scale(self):
        from pyprideap.viz.qc.compute import compute_plate_cv

        ds = _make_olink_two_plates()
        result = compute_plate_cv(ds)
        assert result is not None
        assert result.plate_ids == ["P1", "P2"]
        p1 = ds.expression.iloc[:5]
        expected_p1 = ((2**p1).std() / (2**p1).mean()).tolist()
        assert result.intra_cv[:2] == pytest.approx(expected_p1)


# ---------------------------------------------------------------------------
# Split-report layout manifest and summary columns
# ---------------------------------------------------------------------------


class TestSplitLayout:
    def test_manifest_lists_written_files_in_complete_rows(self, tmp_path):
        out = qc_report_split(_make_olink_dataset(), tmp_path / "split")
        manifest = json.loads((out / "manifest.json").read_text())
        assert manifest["schema"] == 1
        assert [t["id"] for t in manifest["tabs"]][0] == "overview"
        for tab in manifest["tabs"]:
            half_run = 0
            for item in tab["items"]:
                assert (out / item["file"]).exists()
                if item["width"] == "half":
                    half_run += 1
                else:
                    assert half_run % 2 == 0, f"orphan half-width plot before {item['key']}"
                    half_run = 0
            assert half_run % 2 == 0
        placed = {i["key"] for t in manifest["tabs"] for i in t["items"]}
        assert "summary" in placed
        assert not placed & set(manifest["folded_into_summary"])

    def test_manifest_widens_orphan_half_plot(self):
        from pyprideap.viz.qc.report import _build_split_manifest

        manifest = _build_split_manifest(["summary", "lod_analysis", "distribution"], "olink_explore")
        widths = {i["key"]: i["width"] for t in manifest["tabs"] for i in t["items"]}
        assert widths == {"summary": "full", "lod_analysis": "full", "distribution": "full"}

    def test_summary_two_columns_split_at_group(self, tmp_path):
        out = qc_report_split(_make_olink_dataset(), tmp_path / "split")
        summary = (out / "summary.html").read_text()
        assert '<div class="summary-columns">' in summary
        assert summary.count('<table class="summary-table">') == 2

    def test_split_summary_columns_balances_groups(self):
        from pyprideap.viz.qc.report import _split_summary_columns, _summary_group, _summary_row

        rows = [_summary_group("A")] + [_summary_row("", "a", "1")] * 5
        rows += [_summary_group("B")] + [_summary_row("", "b", "1")] * 2
        rows += [_summary_group("C")] + [_summary_row("", "c", "1")] * 2
        left, right = _split_summary_columns(rows, 2)
        assert left[0] == _summary_group("A") and right[0] == _summary_group("B")
        assert left + right == rows


class TestSplitPlotTweaks:
    def test_completeness_hides_ticks_for_many_samples(self):
        from pyprideap.viz.qc.compute import DataCompletenessData
        from pyprideap.viz.qc.render import render_sample_completeness

        n = 60
        data = DataCompletenessData(
            sample_ids=[f"S{i}" for i in range(n)], above_lod_rate=[0.5] * n, below_lod_rate=[0.5] * n
        )
        fig = render_sample_completeness(data)
        assert fig.layout.xaxis.showticklabels is False

    def test_distribution_summary_legend_only_bands(self):
        from pyprideap.viz.qc.compute import DistributionData
        from pyprideap.viz.qc.render import render_distribution

        n = 30
        data = DistributionData(
            sample_ids=[f"S{i}" for i in range(n)],
            sample_values=[[float(j % 7) for j in range(50)] for _ in range(n)],
            xlabel="NPX Value",
        )
        fig = render_distribution(data)
        in_legend = [t.name for t in fig.data if t.showlegend is not False]
        assert in_legend == ["5th–95th percentile", "IQR (25th–75th)", "Median"]


# ---------------------------------------------------------------------------
# Technical QC: replicate CV, batch effect, vendor QC flags
# ---------------------------------------------------------------------------


def _make_olink_technical(n_per_plate=8, n_ctrl=4, n_assays=6, plate_shift=0.0):
    import numpy as np

    rng = np.random.default_rng(3)
    plates, types = [], []
    for p in ("P1", "P2"):
        plates += [p] * (n_per_plate + n_ctrl)
        types += ["SAMPLE"] * n_per_plate + ["SAMPLE_CONTROL"] * n_ctrl
    n = len(plates)
    expr = rng.normal(5, 1.0, (n, n_assays))
    ctrl = np.array([t == "SAMPLE_CONTROL" for t in types])
    expr[ctrl] = 5 + rng.normal(0, 0.05, (ctrl.sum(), n_assays))  # tight technical replicates
    expr[np.array(plates) == "P2"] += plate_shift
    assays = [f"O{j}" for j in range(n_assays)]
    assay_qc = pd.DataFrame("PASS", index=range(n), columns=assays)
    assay_qc.iloc[0, 0] = "WARN"
    sample_qc = pd.DataFrame("PASS", index=range(n), columns=assays)
    sample_qc.iloc[1, :3] = "FAIL"  # sample S1 fails in one block only
    return AffinityDataset(
        platform=Platform.OLINK_EXPLORE,
        samples=pd.DataFrame(
            {"SampleID": [f"S{i}" for i in range(n)], "SampleType": types, "PlateID": plates, "SampleQC": "PASS"}
        ),
        features=pd.DataFrame({"OlinkID": assays, "UniProt": assays, "Panel": ["Inf"] * 3 + ["Onc"] * 3}),
        expression=pd.DataFrame(expr, columns=assays),
        metadata={"assay_qc_matrix": assay_qc, "sample_qc_matrix": sample_qc},
    )


class TestTechnicalQc:
    def test_replicate_cv_uses_sample_controls_only(self):
        from pyprideap.viz.qc.compute import compute_replicate_cv

        r = compute_replicate_cv(_make_olink_technical())
        assert r.control_label == "Sample controls" and r.n_replicates == 8
        import numpy as np

        assert np.median(r.technical_cv) < 0.1 < np.median(r.study_cv)

    def test_replicate_cv_needs_three_replicates(self):
        from pyprideap.viz.qc.compute import compute_replicate_cv

        ds = _make_olink_technical(n_ctrl=1)
        assert compute_replicate_cv(ds) is None

    def test_batch_effect_detects_plate_shift(self):
        from pyprideap.viz.qc.compute import compute_batch_effect

        shifted = compute_batch_effect(_make_olink_technical(plate_shift=3.0))
        assert shifted.plate_r2[0] > 0.8
        assert shifted.plate_ids == ["P1", "P2"]
        assert [len(v) for v in shifted.plate_sample_medians] == [8, 8]  # controls excluded
        plain = compute_batch_effect(_make_olink_technical(plate_shift=0.0))
        assert max(plain.plate_r2) < shifted.plate_r2[0]

    def test_qc_flags_per_panel_and_worst_block(self):
        from pyprideap.viz.qc.compute import compute_qc_flags

        q = compute_qc_flags(_make_olink_technical())
        assert q.rows == ["Inf · Assay QC", "Inf · Sample QC", "Onc · Assay QC", "Onc · Sample QC"]
        assert q.sample_worst == {"PASS": 15, "FAIL": 1}
        assert q.flagged_assay_pct == pytest.approx(100 / (16 * 6), abs=0.01)

    def test_summary_uses_worst_block_sample_qc(self, tmp_path):
        out = qc_report_split(_make_olink_technical(), tmp_path / "split")
        summary = (out / "summary.html").read_text()
        assert "PASS / WARN / FAIL (study samples, worst block)" in summary
        assert "15 / 0 / 1" in summary
        assert "Technical CV, median (sample controls, n=8)" in summary
        manifest = json.loads((out / "manifest.json").read_text())
        technical = next(t for t in manifest["tabs"] if t["id"] == "technical")
        assert [i["key"] for i in technical["items"]][:3] == ["batch_effect", "plate_signal", "qc_flags"]

    def test_olink_reader_keeps_qc_flag_matrices(self, tmp_path):
        rows = []
        for sid in ("A", "B"):
            for oid, block_qc in (("OID1", "PASS"), ("OID2", "FAIL" if sid == "B" else "PASS")):
                rows.append(
                    {
                        "SampleID": sid,
                        "OlinkID": oid,
                        "NPX": 1.0,
                        "UniProt": "P1",
                        "Assay": oid,
                        "Panel": "X",
                        "SampleQC": block_qc,
                        "AssayQC": "WARN" if oid == "OID1" else "PASS",
                    }
                )
        path = tmp_path / "flags.npx.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        from pyprideap.io.readers.olink_csv import read_olink_csv

        ds = read_olink_csv(path)
        assert ds.metadata["assay_qc_matrix"].values.tolist() == [["WARN", "PASS"], ["WARN", "PASS"]]
        assert ds.metadata["sample_qc_matrix"].values.tolist() == [["PASS", "PASS"], ["PASS", "FAIL"]]
