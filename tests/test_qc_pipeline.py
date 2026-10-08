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

    def test_proteins_above_lod_no_lod_somascan_returns_measured(self):
        # SomaScan without buffer samples has no LOD: all measured proteins are returned
        ds = _make_somascan_dataset()
        assert get_proteins_above_lod(ds) == ["P1", "P2"]

    def test_proteins_above_lod_no_lod_olink_is_empty(self):
        ds = _make_olink_dataset()  # no LOD column, no negative controls, no FixedLOD match
        assert get_proteins_above_lod(ds) == []

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


# ---------------------------------------------------------------------------
# Pre-analytical indicators, dilution QC, sex consistency
# ---------------------------------------------------------------------------


def _make_marker_dataset(n=40, sexes=None, swap=None, hemolysed=None):
    """Olink-like dataset with hemolysis, platelet and sex marker assays plus filler assays."""
    import numpy as np

    rng = np.random.default_rng(7)
    genes = ["HBB", "CA1", "PRDX2", "PF4", "PPBP", "KDM5D", "KLK3", "PZP"] + [f"G{i}" for i in range(6)]
    expr = rng.normal(0, 0.3, (n, len(genes)))
    sexes = sexes or (["male"] * (n // 2) + ["female"] * (n - n // 2))
    is_male = np.array([s == "male" for s in sexes])
    if swap is not None:
        is_male[swap] = ~is_male[swap]  # protein profile of the other sex
    expr[:, genes.index("KDM5D")] += np.where(is_male, 4, 0)
    expr[:, genes.index("KLK3")] += np.where(is_male, 4, 0)
    expr[:, genes.index("PZP")] += np.where(is_male, 0, 3)
    for i in hemolysed or []:
        expr[i, :3] += 5
    return AffinityDataset(
        platform=Platform.OLINK_EXPLORE,
        samples=pd.DataFrame({"SampleID": [f"S{i}" for i in range(n)], "SampleType": "SAMPLE", "sex": sexes}),
        features=pd.DataFrame({"OlinkID": [f"O{j}" for j in range(len(genes))], "Assay": genes, "UniProt": genes}),
        expression=pd.DataFrame(expr, columns=[f"O{j}" for j in range(len(genes))]),
        metadata={},
    )


class TestPreanalyticalDilutionSex:
    def test_hemolysis_outlier_flagged(self):
        from pyprideap.viz.qc.compute import compute_preanalytical

        pa = compute_preanalytical(_make_marker_dataset(hemolysed=[3]))
        assert pa.markers["hemolysis"] == ["CA1", "HBB", "PRDX2"]
        assert pa.outliers["hemolysis"] == ["S3"]
        assert pa.outliers["platelet"] == []

    def test_sex_mismatch_against_annotation(self):
        from pyprideap.viz.qc.compute import compute_sex_check

        sc = compute_sex_check(_make_marker_dataset(swap=[5]))
        assert sc.mismatches == ["S5"]
        assert sc.markers == {"male": ["KDM5D", "KLK3"], "female": ["PZP"]}

    def test_sex_prediction_without_annotation_needs_bimodality(self):
        from pyprideap.viz.qc.compute import compute_sex_check

        bimodal = _make_marker_dataset()
        bimodal.samples["sex"] = "not available"
        sc = compute_sex_check(bimodal)
        assert sc is not None and sc.predicted.count("male") == 20 and not sc.mismatches
        single = _make_marker_dataset(sexes=["female"] * 40)
        single.samples["sex"] = "not available"
        assert compute_sex_check(single) is None

    def test_two_group_split_rejects_unimodal(self):
        import numpy as np

        from pyprideap.viz.qc.compute import _two_group_split

        rng = np.random.default_rng(0)
        assert _two_group_split(rng.normal(0, 1, 300)) is None
        assert _two_group_split(rng.lognormal(0, 0.8, 300)) is None
        assert _two_group_split(np.r_[rng.normal(-2, 0.4, 60), rng.normal(2, 0.4, 40)]) == pytest.approx(0, abs=0.5)

    def test_dilution_qc_groups_somascan_bins(self):
        import numpy as np

        from pyprideap.viz.qc.compute import compute_dilution_qc

        rng = np.random.default_rng(1)
        n_s, n_qc = 12, 4
        dil = ["20", "20", "0.5", "0.5", "0.005", "0"]
        expr = np.abs(rng.normal(1000, 200, (n_s + n_qc, len(dil))))
        ds = AffinityDataset(
            platform=Platform.SOMASCAN,
            samples=pd.DataFrame(
                {
                    "SampleId": [f"S{i}" for i in range(n_s + n_qc)],
                    "SampleType": ["Sample"] * n_s + ["QC"] * n_qc,
                    "NormScale_20": 1.0,
                    "NormScale_0_5": 1.1,
                    "NormScale_0_005": 0.9,
                }
            ),
            features=pd.DataFrame({"SeqId": [f"{i}-1" for i in range(len(dil))], "Dilution": dil}),
            expression=pd.DataFrame(expr, columns=[f"SL{i}" for i in range(len(dil))]),
            metadata={},
        )
        dq = compute_dilution_qc(ds)
        assert dq.dilutions == ["20%", "0.5%", "0.005%"]  # "0" (non-human / controls) excluded
        assert dq.n_assays == [2, 2, 1]
        assert [len(v) for v in dq.technical_cv] == [2, 2, 1]
        assert [len(v) for v in dq.norm_scale] == [16, 16, 16]

    def test_report_places_new_plots_in_technical_tab(self, tmp_path):
        out = qc_report_split(_make_marker_dataset(swap=[5], hemolysed=[3]), tmp_path / "split")
        manifest = json.loads((out / "manifest.json").read_text())
        technical = next(t for t in manifest["tabs"] if t["id"] == "technical")
        keys = [i["key"] for i in technical["items"]]
        assert "preanalytical" in keys and "sex_check" in keys
        summary = (out / "summary.html").read_text()
        assert (
            "Sex mismatches vs annotation" in summary
            and "Samples with possible hemolysis (indicator outliers)" in summary
        )

    def test_sex_check_drops_uninformative_markers_and_ignores_wide_male_range(self):
        """Mirrors PAD000003 (SomaScan): KLK3 informative, EIF1AY reagent not; PSA varies widely in men."""
        import numpy as np

        from pyprideap.viz.qc.compute import compute_sex_check

        rng = np.random.default_rng(11)
        n_m, n_f = 60, 40
        sexes = ["male"] * n_m + ["female"] * n_f
        klk3 = np.r_[rng.normal(1.5, 1.2, n_m), rng.normal(-1.0, 0.15, n_f)]  # wide in men, tight in women
        klk3[n_m + 3] = 1.5  # one annotated female with male-range KLK3
        eif1ay = rng.normal(0, 1, n_m + n_f)  # reagent that does not separate the sexes
        filler = rng.normal(0, 0.3, (n_m + n_f, 4))
        genes = ["KLK3", "EIF1AY", "G1", "G2", "G3", "G4"]
        ds = AffinityDataset(
            platform=Platform.OLINK_EXPLORE,
            samples=pd.DataFrame(
                {"SampleID": [f"S{i}" for i in range(n_m + n_f)], "SampleType": "SAMPLE", "sex": sexes}
            ),
            features=pd.DataFrame({"OlinkID": [f"O{j}" for j in range(6)], "Assay": genes, "UniProt": genes}),
            expression=pd.DataFrame(np.c_[klk3, eif1ay, filler], columns=[f"O{j}" for j in range(6)]),
            metadata={},
        )
        sc = compute_sex_check(ds)
        assert sc.markers == {"male": ["KLK3"], "female": []}
        assert sc.marker_auc["EIF1AY"] < 0.8 <= sc.marker_auc["KLK3"]
        # Low-KLK3 men are normal variation; only the female in the male range is flagged
        assert sc.mismatches == [f"S{n_m + 3}"]


class TestReviewFixes:
    def test_status_counts_without_future_stack(self):
        from pyprideap.viz.qc.compute import _status_counts

        m = pd.DataFrame({"a": ["PASS", "WARN", None], "b": ["FAIL", "odd", "PASS"]})
        assert _status_counts(m) == {"PASS": 2, "WARN": 1, "FAIL": 1, "NA": 2}

    def test_single_multitarget_reagent_is_not_two_markers(self):
        import numpy as np

        from pyprideap.viz.qc.compute import compute_preanalytical

        rng = np.random.default_rng(2)
        genes = ["HBA1|HBA2", "PF4", "PPBP", "G1", "G2"]
        ds = AffinityDataset(
            platform=Platform.SOMASCAN,
            samples=pd.DataFrame({"SampleId": [f"S{i}" for i in range(20)], "SampleType": "Sample"}),
            features=pd.DataFrame({"SeqId": [f"{i}-1" for i in range(5)], "EntrezGeneSymbol": genes}),
            expression=pd.DataFrame(np.abs(rng.normal(1000, 100, (20, 5))), columns=[f"SL{i}" for i in range(5)]),
            metadata={},
        )
        pa = compute_preanalytical(ds)
        assert "hemolysis" not in pa.scores  # one reagent, even if it lists two genes
        assert pa.markers["platelet"] == ["PF4", "PPBP"]

    def test_dilution_above_lod_counts_study_samples_only(self):
        import numpy as np

        from pyprideap.viz.qc.compute import compute_dilution_qc

        rng = np.random.default_rng(4)
        n_s, n_qc, n_buf = 12, 4, 20  # buffers outnumber study samples
        n = n_s + n_qc + n_buf
        expr = np.abs(rng.normal(5000, 300, (n, 4)))
        expr[n_s + n_qc :] = np.abs(rng.normal(50, 5, (n_buf, 4)))  # blanks, below eLOD
        ds = AffinityDataset(
            platform=Platform.SOMASCAN,
            samples=pd.DataFrame(
                {
                    "SampleId": [f"S{i}" for i in range(n)],
                    "SampleType": ["Sample"] * n_s + ["QC"] * n_qc + ["Buffer"] * n_buf,
                }
            ),
            features=pd.DataFrame({"SeqId": [f"{i}-1" for i in range(4)], "Dilution": ["20", "20", "0.5", "0.5"]}),
            expression=pd.DataFrame(expr, columns=[f"SL{i}" for i in range(4)]),
            metadata={},
        )
        assert compute_dilution_qc(ds).above_lod_pct == [100.0, 100.0]

    def test_reader_maps_warning_to_warn(self, tmp_path):
        rows = [
            {
                "SampleID": s,
                "OlinkID": "OID1",
                "NPX": 1.0,
                "UniProt": "P1",
                "Assay": "A",
                "Panel": "X",
                "QC_Warning": "Warning" if s == "B" else "Pass",
            }
            for s in ("A", "B")
        ]
        path = tmp_path / "warning.npx.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        from pyprideap.io.readers.olink_csv import read_olink_csv

        assert read_olink_csv(path).metadata["sample_qc_matrix"].values.ravel().tolist() == ["PASS", "WARN"]


# ---------------------------------------------------------------------------
# Protein counts used by PRIDE (proteins-above-lod)
# ---------------------------------------------------------------------------


class TestProteinCounts:
    def test_somascan_measured_proteins_exclude_control_reagents(self):
        ds = _make_somascan_dataset()
        ds.features["Type"] = ["Protein", "Hybridization Control Elution"]
        assert get_proteins_above_lod(ds) == ["P1"]

    def test_controls_do_not_dilute_the_above_lod_share(self):
        import numpy as np

        n_study, n_ctrl = 6, 8
        ds = AffinityDataset(
            platform=Platform.OLINK_EXPLORE,
            samples=pd.DataFrame(
                {
                    "SampleID": [f"S{i}" for i in range(n_study + n_ctrl)],
                    "SampleType": ["SAMPLE"] * n_study + ["NEGATIVE_CONTROL"] * n_ctrl,
                }
            ),
            features=pd.DataFrame({"OlinkID": ["O1"], "UniProt": ["P1"], "LOD": [1.0]}),
            expression=pd.DataFrame({"O1": np.r_[np.full(n_study, 3.0), np.full(n_ctrl, 0.0)]}),
            metadata={},
        )
        # 6/14 samples above LOD overall (43%), but 100% of study samples
        assert get_proteins_above_lod(ds) == ["P1"]

    def test_olink_reader_prefers_lodnpx_and_coerces_text(self, tmp_path):
        from pyprideap.io.readers.olink_csv import read_olink_csv

        rows = [
            {
                "SampleID": s,
                "OlinkID": "OID1",
                "UniProt": "P1",
                "Assay": "A",
                "Panel": "X",
                "NPX": npx,
                "LODNPX": 0.5,
                "LOD": 1000,
            }
            for s, npx in (("S1", "2.0"), ("S2", "1.5"), ("S3", "PlateID"))
        ]
        path = tmp_path / "lodnpx.npx.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        ds = read_olink_csv(path)
        assert ds.metadata["lod_matrix"]["OID1"].tolist() == [0.5, 0.5, 0.5]
        assert ds.expression["OID1"].isna().tolist() == [False, False, True]
        assert get_proteins_above_lod(ds) == ["P1"]

    def test_table_only_adat_is_read(self, tmp_path):
        from pyprideap.io.readers.somascan_adat import read_somascan_adat

        n_meta = 3  # sample metadata columns: PlateId, SampleId, SampleType
        pad = "\t" * n_meta
        lines = [
            f"{pad}SeqId\t10000-28\t10001-7",
            f"{pad}UniProt\tP43320\tP04049",
            f"{pad}Type\tProtein\tProtein",
            f"{pad}Dilution\t20\t0.5",
            # header: sample metadata names, one empty field above the feature-name
            # column, then empty fields over the analyte columns (as in real exports)
            "PlateId\tSampleId\tSampleType\t\t\t",
            "P1\tS1\tSample\t\t1000.5\t200.1",
            "P1\tB1\tBuffer\t\t50.2\t20.3",
        ]
        path = tmp_path / "table_only.adat"
        path.write_text("\n".join(lines) + "\n")
        ds = read_somascan_adat(path)
        assert ds.features["UniProt"].tolist() == ["P43320", "P04049"]
        assert ds.samples["SampleType"].tolist() == ["Sample", "Buffer"]
        assert ds.expression.iloc[0].tolist() == [1000.5, 200.1]


# ---------------------------------------------------------------------------
# Bridging samples, SDRF design groups and reanalysis readiness
# ---------------------------------------------------------------------------


def _write_bridged_olink(path, shift=0.5, lod_col="LOD"):
    """Two plates; samples B1-B3 measured on both, plate 2 shifted by *shift* NPX."""
    import numpy as np

    rng = np.random.default_rng(5)
    base = {s: rng.normal(5, 1, 4) for s in ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3"]}
    rows = []
    for plate, ids, offset in (
        ("P1", ["A1", "A2", "A3", "B1", "B2", "B3"], 0.0),
        ("P2", ["B1", "B2", "B3", "C1", "C2", "C3"], shift),
    ):
        for sid in ids:
            for j in range(4):
                rows.append(
                    {
                        "SampleID": sid,
                        "PlateID": plate,
                        "OlinkID": f"OID{j}",
                        "UniProt": f"P{j}",
                        "Assay": f"A{j}",
                        "Panel": "X",
                        "NPX": base[sid][j] + offset,
                        lod_col: 1.0,
                    }
                )
    pd.DataFrame(rows).to_csv(path, index=False)


class TestBridgingAndReadiness:
    def test_reader_keeps_each_plate_run(self, tmp_path):
        from pyprideap.io.readers.olink_csv import read_olink_csv

        path = tmp_path / "bridged.npx.csv"
        _write_bridged_olink(path)
        ds = read_olink_csv(path)
        assert len(ds.samples) == 12  # 9 samples, 3 of them on both plates
        assert ds.samples["SampleID"].nunique() == 9
        assert ds.samples["SampleRun"].is_unique

    def test_plate_lod_column_is_read(self, tmp_path):
        from pyprideap.io.readers.olink_csv import read_olink_csv

        path = tmp_path / "platelod.npx.csv"
        _write_bridged_olink(path, lod_col="PlateLOD")
        assert "lod_matrix" in read_olink_csv(path).metadata

    def test_bridge_agreement_reports_plate_offset(self, tmp_path):
        from pyprideap.io.readers.olink_csv import read_olink_csv
        from pyprideap.viz.qc.compute import compute_bridge_agreement

        path = tmp_path / "bridged.npx.csv"
        _write_bridged_olink(path, shift=0.5)
        b = compute_bridge_agreement(read_olink_csv(path))
        assert b.n_samples == 3 and b.plates == ["P1", "P2"]
        assert b.median_offset == pytest.approx(0.5, abs=1e-6)
        assert b.median_correlation == pytest.approx(1.0)

    def test_sdrf_factor_value_defines_groups(self, tmp_path):
        from pyprideap.io.readers.sdrf import read_sdrf, select_biological_group_column

        sdrf = pd.DataFrame(
            {
                "source name": [f"S{i}" for i in range(6)],
                "characteristics[disease]": ["obesity"] * 6,
                "characteristics[disease].1": ["a", "a", "a", "b", "b", "b"],
                "factor value[disease]": ["case"] * 3 + ["control"] * 3,
            }
        )
        path = tmp_path / "x.sdrf.tsv"
        sdrf.to_csv(path, sep="\t", index=False)
        parsed = read_sdrf(path)
        assert parsed.attrs["factor_value_columns"] == ["disease 3"]
        # the declared factor value wins over the other disease columns
        assert select_biological_group_column(parsed, parsed.attrs["factor_value_columns"]) == "disease 3"
        # without factor values, a repeated column still matches "disease" by its base name
        assert select_biological_group_column(parsed.drop(columns=["disease 3"])) == "disease 2"

    def test_sdrf_merge_records_linkage_and_factor_columns(self, tmp_path):
        from pyprideap.io.readers.sdrf import merge_sdrf, read_sdrf

        ds = _make_olink_dataset()
        sdrf = pd.DataFrame({"source name": ds.samples["SampleID"], "factor value[disease]": ["a", "a", "b", "b", "b"]})
        path = tmp_path / "y.sdrf.tsv"
        sdrf.to_csv(path, sep="\t", index=False)
        merged = merge_sdrf(ds, read_sdrf(path))
        assert merged.metadata["sdrf_merge"] == {"matched": 5, "total": 5}
        assert merged.metadata["sdrf_factor_columns"] == ["disease"]

    def test_readiness_reports_missing_items_with_impact(self):
        from pyprideap.viz.qc.compute import compute_readiness

        ds = _make_olink_dataset()
        ds.samples = ds.samples.drop(columns=["SampleType"])
        r = compute_readiness(ds, {})
        by = {i.category: i for i in r.items}
        assert by["Sample types (controls)"].status == "missing"
        assert "technical CV" in by["Sample types (controls)"].impact
        assert by["SDRF linked to samples"].status == "missing"
        assert by["Bridging / replicate samples"].status in ("n/a", "missing")
        assert r.n_available < r.n_applicable

    def test_readiness_flags_unbridged_plates(self, tmp_path):
        from pyprideap.io.readers.olink_csv import read_olink_csv
        from pyprideap.viz.qc.compute import compute_all

        path = tmp_path / "bridged.npx.csv"
        _write_bridged_olink(path, shift=0.5)
        results = compute_all(read_olink_csv(path))
        item = next(i for i in results["readiness"].items if i.category == "Bridging / replicate samples")
        assert item.status == "available" and "not bridge-normalised" in item.impact

    def test_readiness_card_in_split_report_overview(self, tmp_path):
        out = qc_report_split(_make_olink_dataset(), tmp_path / "split")
        assert (out / "readiness.html").exists()
        manifest = json.loads((out / "manifest.json").read_text())
        assert [i["key"] for i in manifest["tabs"][0]["items"]][:2] == ["summary", "readiness"]


# ---------------------------------------------------------------------------
# Fixes from the sweep over all public PAD datasets
# ---------------------------------------------------------------------------


class TestSweepFixes:
    @staticmethod
    def _olink(ids, extra=None):
        import numpy as np

        samples = pd.DataFrame({"SampleID": ids, **(extra or {})})
        return AffinityDataset(
            platform=Platform.OLINK_EXPLORE,
            samples=samples,
            features=pd.DataFrame({"OlinkID": ["O1"], "UniProt": ["P1"]}),
            expression=pd.DataFrame({"O1": np.arange(len(ids), dtype=float)}),
            metadata={},
        )

    def test_merge_with_integer_sample_ids(self):
        from pyprideap.io.readers.sdrf import merge_sdrf

        ds = self._olink([1, 2, 3])  # PAD000035: numeric SampleID crashed the merge
        sdrf = pd.DataFrame({"source name": ["PAD000035-1", "PAD000035-2", "PAD000035-3"], "sex": ["f", "m", "f"]})
        merged = merge_sdrf(ds, sdrf)
        assert merged.samples["sex"].tolist() == ["f", "m", "f"]

    def test_merge_ignores_separators_case_and_leading_zeros(self):
        from pyprideap.io.readers.sdrf import merge_sdrf

        ds = self._olink(["L14_NIST N4", "CE002303 201", "0670628553"])
        sdrf = pd.DataFrame(
            {
                "source name": ["PAD000009-L14_NIST_N4", "PAD000050-CE002303-201", "670628553"],
                "disease": ["a", "b", "c"],
            }
        )
        assert merge_sdrf(ds, sdrf).metadata["sdrf_merge"] == {"matched": 3, "total": 3}

    def test_merge_uses_rows_for_this_data_file(self):
        from pyprideap.io.readers.sdrf import merge_sdrf

        ds = self._olink(["S1", "S2"])
        ds.metadata["source_file"] = "/data/plate1.npx.csv"
        sdrf = pd.DataFrame(
            {
                "source name": ["S1", "S2", "S1", "S2"],
                "comment[data file]": ["plate2.npx.csv", "plate2.npx.csv", "plate1.npx.csv", "plate1.npx.csv"],
                "group": ["wrong", "wrong", "right", "right"],
            }
        )
        assert merge_sdrf(ds, sdrf).samples["group"].tolist() == ["right", "right"]

    def test_merge_prefers_the_most_specific_sample_column(self):
        from pyprideap.io.readers.sdrf import merge_sdrf

        # PAD000030: SampleID is constant, the names are in SampleName
        ds = self._olink(["X", "X"], {"SampleName": ["A", "B"]})
        sdrf = pd.DataFrame({"source name": ["A", "B"], "sex": ["f", "m"]})
        assert merge_sdrf(ds, sdrf).samples["sex"].tolist() == ["f", "m"]

    def test_repeated_individual_column_is_not_a_group(self):
        from pyprideap.io.readers.sdrf import get_grouping_columns

        sdrf = pd.DataFrame({"individual": ["a"] * 9, "individual 2": ["p1"] * 3 + ["p2"] * 3 + ["p3"] * 3})
        assert get_grouping_columns(sdrf) == []

    def test_deidentified_olink_parquet_columns(self, tmp_path):
        from pyprideap.io.readers.registry import read

        rows = [
            {
                "DeidentifiedSampleID": s,
                "Sample_Type": "SAMPLE",
                "DeidentifiedPlateID": "P1",
                "OlinkID": "O1",
                "UniProt": "P1",
                "Assay": "A",
                "Panel": "X",
                "NPX": 1.0,
            }
            for s in ("S1", "S2")
        ]
        path = tmp_path / "deid.parquet"
        pd.DataFrame(rows).to_parquet(path)
        ds = read(path)
        assert ds.samples["SampleID"].tolist() == ["S1", "S2"] and "SampleType" in ds.samples

    def test_olink_ct_export_gives_clear_error(self, tmp_path):
        from pyprideap.io.readers.registry import read

        path = tmp_path / "run_Ct.raw.csv"
        path.write_text("Run,Olink NPX Signature 1.17.0\nCt data\nPanel,X,X\nAssay,IL8,TNF\n")
        with pytest.raises(ValueError, match="Ct values"):
            read(path)

    def test_somascan_table_exported_as_csv(self, tmp_path):
        from pyprideap.io.readers.registry import read

        lines = [
            "SeqId,,,10000-28,10001-7",
            "UniProt,,,P43320,P04049",
            "Type,,,Protein,Protein",
            "Dilution,,,20,0.5",
            "PlateId,SampleId,SampleType,,",
            "P1,S1,Sample,1000.5,200.1",
            "P1,B1,Buffer,50.2,20.3",
        ]
        path = tmp_path / "SomaLogic_rawdata.csv"
        path.write_text("\n".join(lines) + "\n")
        ds = read(path)
        assert ds.platform == Platform.SOMASCAN
        assert ds.features["UniProt"].tolist() == ["P43320", "P04049"]
        assert ds.samples["SampleType"].tolist() == ["Sample", "Buffer"]
        assert ds.expression.iloc[1].tolist() == [50.2, 20.3]

    def test_readiness_names_sdrf_without_factor_values(self, tmp_path):
        from pyprideap.viz.qc.compute import compute_all
        from pyprideap.viz.qc.report import _prepare_qc_dataset

        ds = _make_olink_dataset()
        path = tmp_path / "x.sdrf.tsv"
        pd.DataFrame({"source name": ds.samples["SampleID"], "characteristics[disease]": ["not available"] * 5}).to_csv(
            path, sep="\t", index=False
        )
        items = {i.category: i for i in compute_all(_prepare_qc_dataset(ds, path))["readiness"].items}
        assert "no factor value" in items["Study groups"].detail
        assert items["SDRF linked to samples"].detail.startswith("SDRF provided; 5 of 5")

    def test_merge_picks_the_sample_column_that_matches(self):
        from pyprideap.io.readers.sdrf import merge_sdrf

        # PAD000014: SampleName is more varied, but the SDRF uses SampleId
        ds = self._olink(["X", "X", "X"], {"SampleId": ["5530 P", "5531 P", "5532 P"], "SampleName": ["a", "b", "c"]})
        ds.samples = ds.samples.drop(columns=["SampleID"])
        sdrf = pd.DataFrame(
            {"source name": ["PAD000014-5530_P", "PAD000014-5531_P", "PAD000014-5532_P"], "sex": list("fmf")}
        )
        assert merge_sdrf(ds, sdrf).samples["sex"].tolist() == ["f", "m", "f"]

    def test_csv_delimiter_detection(self, tmp_path):
        from pyprideap.io.readers.olink_csv import _read_delimited, _sniff_delimiter

        for sep in (",", ";", "\t"):
            path = tmp_path / "x.npx.csv"
            path.write_text(sep.join(["SampleID", "OlinkID", "NPX"]) + "\n" + sep.join(["S1", "O1", "1.5"]) + "\n")
            assert _sniff_delimiter(path) == sep
            assert _read_delimited(path).columns.tolist() == ["SampleID", "OlinkID", "NPX"]

    def test_qc_flags_first_value_wins_and_missing_stay_empty(self, tmp_path):
        from pyprideap.io.readers.olink_csv import read_olink_csv

        rows = [
            ("S1", "O1", "Pass"),
            ("S1", "O1", "FAIL"),  # duplicate measurement: first wins
            ("S1", "O2", "Warning"),
            ("S2", "O1", None),  # long form normalised; missing flag stays empty
        ]
        df = pd.DataFrame(
            [
                {"SampleID": s, "OlinkID": o, "UniProt": "P", "Assay": "A", "Panel": "X", "NPX": 1.0, "QC_Warning": f}
                for s, o, f in rows
            ]
        )
        path = tmp_path / "flags.npx.csv"
        df.to_csv(path, index=False)
        m = read_olink_csv(path).metadata["sample_qc_matrix"]
        assert m.loc[0, "O1"] == "PASS" and m.loc[0, "O2"] == "WARN"
        assert pd.isna(m.loc[1, "O1"]) and pd.isna(m.loc[1, "O2"])

    @staticmethod
    def _write_long(path, sep=",", npx=("1.25", "2.5", "-0.75", "3.0")):
        rows = [
            {
                "SampleID": sid,
                "OlinkID": oid,
                "UniProt": "P" + oid[-1],
                "Assay": "A" + oid[-1],
                "Panel": "X",
                "PlateID": "P1",
                "MissingFreq": "30%" if oid == "OID1" else "0.1",
                "NPX": v,
                "LOD": "0.5",
                "QC_Warning": "PASS",
                "Unused": "x" * 20,
            }
            for (sid, oid), v in zip([("007", "OID1"), ("007", "OID2"), ("010", "OID1"), ("010", "OID2")], npx)
        ]
        pd.DataFrame(rows).to_csv(path, sep=sep, index=False)

    def test_streamed_and_single_read_agree(self, tmp_path, monkeypatch):
        import pyprideap.io.readers.olink_csv as oc

        path = tmp_path / "long.npx.csv"
        self._write_long(path)
        small = oc.read_olink_csv(path)
        monkeypatch.setattr(oc, "_STREAM_MIN_BYTES", 0)
        streamed = oc.read_olink_csv(path)
        assert small.expression.equals(streamed.expression)
        assert small.samples.equals(streamed.samples) and small.features.equals(streamed.features)

    def test_ids_keep_leading_zeros_and_unused_columns_are_skipped(self, tmp_path):
        from pyprideap.io.readers.olink_csv import read_olink_csv

        path = tmp_path / "ids.npx.csv"
        self._write_long(path)
        ds = read_olink_csv(path)
        assert ds.samples["SampleID"].tolist() == ["007", "010"]
        assert ds.expression.to_numpy().tolist() == [[1.25, 2.5], [-0.75, 3.0]]
        assert "Unused" not in ds.samples.columns and "Unused" not in ds.features.columns
        # MissingFreq written as "30%" stays text, as before
        assert ds.features["MissingFreq"].tolist() == ["30%", "0.1"]

    def test_text_in_npx_falls_back_and_becomes_missing(self, tmp_path):
        from pyprideap.io.readers.olink_csv import read_olink_csv

        path = tmp_path / "shifted.npx.csv"
        self._write_long(path, sep=";", npx=("1.25", "PlateID", "-0.75", "3.0"))
        ds = read_olink_csv(path)
        assert ds.expression.isna().to_numpy().tolist() == [[False, True], [False, False]]


class TestQcLodSummary:
    """Issue #50: every category listed, per-measurement flags, readable labels."""

    @staticmethod
    def _dataset(flags, npx):
        import numpy as np

        return AffinityDataset(
            platform=Platform.OLINK_EXPLORE,
            samples=pd.DataFrame({"SampleID": ["S1", "S2"], "SampleType": "SAMPLE", "SampleQC": ["PASS", "PASS"]}),
            features=pd.DataFrame({"OlinkID": ["O1", "O2"], "UniProt": ["P1", "P2"], "LOD": [1.0, 1.0]}),
            expression=pd.DataFrame(np.array(npx, dtype=float), columns=["O1", "O2"]),
            metadata={"sample_qc_matrix": pd.DataFrame(flags, columns=["O1", "O2"]).astype("string")},
        )

    def test_empty_categories_are_reported(self):
        q = compute_qc_summary(self._dataset([["PASS", "PASS"], ["PASS", "PASS"]], [[2, 0.5], [3, 4]]))
        counts = dict(zip(q.categories, q.counts))
        assert counts["PASS & NPX > LOD"] == 3 and counts["PASS & NPX ≤ LOD"] == 1
        assert counts["WARN & NPX > LOD"] == 0 and counts["FAIL & NPX ≤ LOD"] == 0

    def test_flags_per_measurement_and_blanked_failures(self):
        import numpy as np

        # S2 has a WARN block on O2 only, and a FAIL on O1 whose value Olink blanked
        q = compute_qc_summary(self._dataset([["PASS", "PASS"], ["fail", "Warning"]], [[2, 0.5], [np.nan, 4]]))
        counts = dict(zip(q.categories, q.counts))
        assert counts["WARN & NPX > LOD"] == 1  # only the flagged block, not the whole sample
        assert counts["FAIL & no value"] == 1
        assert counts["PASS & NPX > LOD"] == 1 and counts["PASS & NPX ≤ LOD"] == 1

    def test_render_has_legend_and_one_decimal(self):
        fig = render_qc_summary(
            QcLodSummaryData(categories=["PASS & NPX > LOD", "WARN & NPX > LOD"], counts=[56870 - 11043, 0])
        )
        assert fig.layout.showlegend is True
        names = [t.name for t in fig.data]
        assert names[0] == "PASS & NPX > LOD: 45,827 (100.0%)" and names[1] == "WARN & NPX > LOD: 0 (0.0%)"
        assert "%{y:.1f}%" in fig.data[0].hovertemplate


class TestPlateSignalCorrection:
    """Issue #49: toggle between deposited and plate-centred signal."""

    @staticmethod
    def _two_plates(offset=1.0, seed=0):
        import numpy as np

        rng = np.random.default_rng(seed)
        n, p = 20, 30
        expr = rng.normal(0, 1, (n, p))
        expr[n // 2 :] += offset  # plate 2 shifted by a constant
        return AffinityDataset(
            platform=Platform.OLINK_EXPLORE,
            samples=pd.DataFrame(
                {
                    "SampleID": [f"S{i}" for i in range(n)],
                    "SampleType": "SAMPLE",
                    "PlateID": ["P1"] * (n // 2) + ["P2"] * (n // 2),
                }
            ),
            features=pd.DataFrame({"OlinkID": [f"O{j}" for j in range(p)], "UniProt": [f"P{j}" for j in range(p)]}),
            expression=pd.DataFrame(expr, columns=[f"O{j}" for j in range(p)]),
            metadata={},
        )

    def test_centring_removes_a_plate_offset(self):
        import numpy as np

        from pyprideap.viz.qc.compute import compute_batch_effect

        b = compute_batch_effect(self._two_plates(offset=2.0))
        raw = [np.median(v) for v in b.plate_sample_medians]
        centred = [np.median(v) for v in b.corrected_sample_medians]
        assert raw[1] - raw[0] > 1.5 and abs(centred[1] - centred[0]) < 0.3
        assert b.plate_r2[0] > 0.5 and b.corrected_plate_r2[0] < 0.2

    def test_render_has_toggle_and_pc1_caption(self):
        from pyprideap.viz.qc.compute import compute_batch_effect
        from pyprideap.viz.qc.render import render_plate_signal

        fig = render_plate_signal(compute_batch_effect(self._two_plates(offset=2.0)))
        buttons = fig.layout.updatemenus[0].buttons
        assert [b.label for b in buttons] == ["As deposited", "Plate-centred"]
        assert [t.visible for t in fig.data] == [True, True, False, False]
        assert buttons[1].args[0]["visible"] == [False, False, True, True]
        assert "of PC1" in fig.layout.xaxis.title.text and "of PC1" in buttons[1].args[1]["xaxis.title.text"]


class TestLodAnalysisControls:
    """Issue #51: study samples only, negative controls shown separately."""

    @staticmethod
    def _dataset():
        import numpy as np

        types = ["SAMPLE"] * 6 + ["PLATE_CONTROL"] * 2 + ["NEGATIVE_CONTROL"] * 2
        expr = np.array([[5.0, 5.0]] * 8 + [[0.0, 5.0], [0.0, 0.0]])  # one negative control above LOD on O2
        return AffinityDataset(
            platform=Platform.OLINK_EXPLORE,
            samples=pd.DataFrame({"SampleID": [f"S{i}" for i in range(10)], "SampleType": types}),
            features=pd.DataFrame({"OlinkID": ["O1", "O2"], "UniProt": ["P1", "P2"], "LOD": [1.0, 1.0]}),
            expression=pd.DataFrame(expr, columns=["O1", "O2"]),
            metadata={},
        )

    def test_study_samples_only_and_negatives_separate(self):
        from pyprideap.viz.qc.compute import compute_lod_analysis

        r = compute_lod_analysis(self._dataset())
        assert r.above_lod_pct == [100.0, 100.0]  # was 80% with the 2 negative controls included
        assert r.negative_above_lod_pct == [0.0, 50.0]
        assert (r.n_study, r.n_negative) == (6, 2)

    def test_render_shows_negative_controls(self):
        from pyprideap.viz.qc.compute import compute_lod_analysis
        from pyprideap.viz.qc.render import render_lod_analysis

        fig = render_lod_analysis(compute_lod_analysis(self._dataset()))
        names = [t.name for t in fig.data]
        assert any("study samples (n = 6)" in n.lower() for n in names)
        assert "Negative controls (n = 2)" in names
        assert "study samples" in fig.layout.yaxis.title.text

    def test_help_texts_have_no_double_percent(self):
        from pyprideap.viz.qc.report import _HELP_TEXT

        assert not [k for k, v in _HELP_TEXT.items() if "%%" in v]


def test_help_texts_state_meaning_before_method_details():
    """Issue #48: meaning first, scale / group details after."""
    from pyprideap.viz.qc.report import _HELP_TEXT

    mpi = _HELP_TEXT["mpi"]
    assert mpi.index("Higher is more consistent") < mpi.index("linear scale") < mpi.index("biological groups")
    rank = _HELP_TEXT["rank_concordance"]
    assert rank.rstrip().endswith("comparable between NPX and RFU within a dataset.")


class TestColourByOptions:
    """Issue #48 item 5: colour PCA / t-SNE by sample type, plate or study group."""

    @staticmethod
    def _dataset():
        import numpy as np

        rng = np.random.default_rng(3)
        n = 12
        samples = pd.DataFrame(
            {
                "SampleID": [f"S{i}" for i in range(n)],
                "SampleType": ["SAMPLE"] * 10 + ["NEGATIVE_CONTROL"] * 2,
                "PlateID": ["P1"] * 6 + ["P2"] * 6,
                "disease": ["case"] * 5 + ["control"] * 5 + [None, None],
            }
        )
        return AffinityDataset(
            platform=Platform.OLINK_EXPLORE,
            samples=samples,
            features=pd.DataFrame({"OlinkID": [f"O{j}" for j in range(5)], "UniProt": [f"P{j}" for j in range(5)]}),
            expression=pd.DataFrame(rng.normal(0, 1, (n, 5)), columns=[f"O{j}" for j in range(5)]),
            metadata={},
        )

    def test_options_include_plate_and_study_group(self):
        from pyprideap.viz.qc.compute import compute_pca

        opts = compute_pca(self._dataset()).color_options
        assert list(opts)[:2] == ["Sample type", "Plate"]
        group = opts["Study group: disease"]
        assert group[:5] == ["case"] * 5 and group[-2:] == ["Control samples"] * 2

    def test_single_valued_sample_type_is_not_offered(self):
        from pyprideap.viz.qc.compute import compute_pca

        ds = self._dataset()
        ds.samples["SampleType"] = "SAMPLE"
        assert "Sample type" not in compute_pca(ds).color_options

    def test_render_dropdown_switches_colouring(self):
        from pyprideap.viz.qc.compute import compute_pca
        from pyprideap.viz.qc.render import render_pca

        fig = render_pca(compute_pca(self._dataset()))
        buttons = fig.layout.updatemenus[0].buttons
        assert [b.label for b in buttons] == [
            "Colour by: Sample type",
            "Colour by: Plate",
            "Colour by: Study group: disease",
        ]
        visible_first = [t.name for t in fig.data if t.visible]
        assert sorted(visible_first) == ["NEGATIVE_CONTROL", "SAMPLE"]
        plate_vis = buttons[1].args[0]["visible"]
        assert sorted(t.name for t, v in zip(fig.data, plate_vis) if v) == ["P1", "P2"]
