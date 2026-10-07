"""Tests for technology-agnostic QC metrics and SDRF case/control detection."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from pyprideap.core import AffinityDataset, Platform
from pyprideap.io.readers.sdrf import (
    infer_groups_from_sample_ids,
    merge_sdrf,
    normalize_sample_id,
    resolve_biological_groups,
    select_biological_group_column,
)
from pyprideap.viz.qc.agnostic import (
    DynamicRangeData,
    MpiData,
    RankConcordanceData,
    compute_agnostic_qc,
    compute_dynamic_range,
    compute_mpi,
    compute_rank_concordance,
)
from pyprideap.viz.qc.compute import compute_all

plotly = pytest.importorskip("plotly")

from pyprideap.viz.qc.render import render_dynamic_range, render_mpi, render_rank_concordance  # noqa: E402
from pyprideap.viz.qc.report import qc_report, qc_report_split  # noqa: E402


def _dataset(
    expression: pd.DataFrame,
    sample_ids: list[str] | None = None,
    extra_sample_cols: dict | None = None,
    platform: Platform = Platform.OLINK_EXPLORE,
) -> AffinityDataset:
    n, p = expression.shape
    if sample_ids is None:
        sample_ids = [f"S{i:03d}" for i in range(n)]
    samples = pd.DataFrame({"SampleID": sample_ids})
    if extra_sample_cols:
        for col, values in extra_sample_cols.items():
            samples[col] = values
    features = pd.DataFrame(
        {
            "OlinkID": list(expression.columns),
            "UniProt": [f"P{i}" for i in range(p)],
            "Assay": [f"A{i}" for i in range(p)],
        }
    )
    return AffinityDataset(
        platform=platform,
        samples=samples,
        features=features,
        expression=expression.reset_index(drop=True),
        metadata={},
    )


def _grouped_expression(n_per_group: int = 12, n_proteins: int = 30, seed: int = 0) -> tuple[pd.DataFrame, list[str]]:
    """Two groups with a mean shift on the first proteins (disease signal)."""
    rng = np.random.default_rng(seed)
    n = n_per_group * 2
    expr = rng.normal(loc=4.0, scale=0.15, size=(n, n_proteins))
    expr[n_per_group:, :8] += 3.0  # case/control difference, not technical noise
    columns = [f"OID{i:03d}" for i in range(n_proteins)]
    groups = ["control"] * n_per_group + ["case"] * n_per_group
    return pd.DataFrame(expr, columns=columns), groups


class TestNormalizeSampleId:
    def test_strips_accession_prefix(self):
        assert normalize_sample_id("PAD000001-XB6") == "XB6"
        assert normalize_sample_id("PAD000003-C126_positive_7") == "C126_positive_7"
        assert normalize_sample_id("PAD000003-r00001-C126_positive_7") == "C126_positive_7"
        assert normalize_sample_id("XB6") == "XB6"


class TestMergeSdrfPrefix:
    def test_matches_pad_prefixed_source_names(self):
        expr = pd.DataFrame(np.ones((4, 3)), columns=["P1", "P2", "P3"])
        ds = _dataset(expr, sample_ids=["XB6", "XB65", "XB68", "XB115"])
        sdrf = pd.DataFrame(
            {
                "source name": [
                    "PAD000001-XB6",
                    "PAD000001-XB65",
                    "PAD000001-XB68",
                    "PAD000001-XB115",
                ],
                "disease": ["healthy", "COVID-19", "healthy", "COVID-19"],
            }
        )
        merged = merge_sdrf(ds, sdrf)
        assert merged.samples["disease"].tolist() == ["healthy", "COVID-19", "healthy", "COVID-19"]


class TestCaseControlFromSdrf:
    def test_prefers_disease_over_sex(self):
        frame = pd.DataFrame(
            {
                "source name": [f"S{i}" for i in range(10)],
                "sex": ["male", "female"] * 5,
                "disease": ["healthy"] * 5 + ["COVID-19"] * 5,
            }
        )
        assert select_biological_group_column(frame) == "disease"

    def test_uses_treatment_when_no_disease(self):
        frame = pd.DataFrame(
            {
                "source name": [f"S{i}" for i in range(10)],
                "treatment": ["control"] * 5 + ["treated"] * 5,
            }
        )
        assert select_biological_group_column(frame) == "treatment"

    def test_ignores_sex_as_case_control(self):
        frame = pd.DataFrame(
            {
                "source name": [f"S{i}" for i in range(10)],
                "sex": ["male", "female"] * 5,
            }
        )
        assert select_biological_group_column(frame) is None

    def test_infers_positive_negative_from_ids(self):
        ids = [f"C{i}_positive_{i}" for i in range(5)] + [f"C{i}_negative" for i in range(5)]
        labels = infer_groups_from_sample_ids(ids)
        assert labels is not None
        assert set(labels.dropna().unique()) == {"positive", "negative"}

    def test_resolve_from_sample_ids_when_sdrf_disease_missing(self):
        expr = pd.DataFrame(np.ones((8, 4)), columns=list("ABCD"))
        ids = [f"C{i}_positive" for i in range(4)] + [f"C{i}_negative" for i in range(4)]
        ds = _dataset(expr, sample_ids=ids)
        resolved = resolve_biological_groups(ds)
        assert resolved is not None
        labels, column = resolved
        assert column == "sample identifier"
        assert labels.notna().sum() == 8


class TestMpi:
    def test_higher_within_group_than_pooled(self):
        expr, groups = _grouped_expression()
        ds = _dataset(expr, extra_sample_cols={"disease": groups})
        pooled = compute_mpi(ds, groups=None)
        grouped = compute_mpi(ds, groups=pd.Series(groups))
        assert pooled is not None and grouped is not None
        assert grouped.has_case_control is True
        # Disease shift inflates pooled MAD; within-group MPI must be higher.
        assert grouped.median > pooled.median

    def test_no_groups_tag_is_false(self):
        expr, _ = _grouped_expression()
        ds = _dataset(expr)
        result = compute_mpi(ds)
        assert result is not None
        assert result.has_case_control is False


class TestDynamicRangeAndConcordance:
    def test_dynamic_range_positive(self):
        expr, groups = _grouped_expression()
        ds = _dataset(expr, extra_sample_cols={"disease": groups})
        result = compute_dynamic_range(ds, has_case_control=True)
        assert result is not None
        assert result.median > 0
        assert len(result.values) == expr.shape[1]

    def test_rank_concordance_high_for_similar_samples(self):
        rng = np.random.default_rng(1)
        # Shared protein profile plus small noise → high rank concordance
        base = rng.normal(size=40)
        rows = [base + rng.normal(scale=0.05, size=40) for _ in range(8)]
        expr = pd.DataFrame(rows, columns=[f"P{i}" for i in range(40)])
        ds = _dataset(expr)
        result = compute_rank_concordance(ds)
        assert result is not None
        assert result.median > 0.8


class TestComputeAllAndReport:
    def test_compute_all_includes_agnostic_metrics(self):
        expr, groups = _grouped_expression()
        ds = _dataset(expr, extra_sample_cols={"disease": groups})
        result = compute_all(ds)
        assert isinstance(result["mpi"], MpiData)
        assert result["mpi"].has_case_control is True
        assert isinstance(result["dynamic_range"], DynamicRangeData)
        assert isinstance(result["rank_concordance"], RankConcordanceData)

    def test_compute_agnostic_qc_from_sdrf_merge(self, tmp_path):
        expr, groups = _grouped_expression()
        ids = [f"S{i:03d}" for i in range(len(groups))]
        ds = _dataset(expr, sample_ids=ids)
        sdrf = pd.DataFrame(
            {
                "source name": [f"PAD000099-{sid}" for sid in ids],
                "disease": groups,
            }
        )
        merged = merge_sdrf(ds, sdrf)
        metrics = compute_agnostic_qc(merged)
        assert metrics["mpi"].has_case_control is True
        assert metrics["mpi"].group_counts["control"] == 12
        assert metrics["mpi"].group_counts["case"] == 12

    def test_renderers(self):
        expr, groups = _grouped_expression()
        ds = _dataset(expr, extra_sample_cols={"disease": groups})
        metrics = compute_agnostic_qc(ds)
        assert render_mpi(metrics["mpi"]) is not None
        assert render_dynamic_range(metrics["dynamic_range"]) is not None
        assert render_rank_concordance(metrics["rank_concordance"]) is not None

    def test_qc_report_contains_plots_and_captions(self, tmp_path):
        expr, groups = _grouped_expression()
        ids = [f"S{i:03d}" for i in range(len(groups))]
        ds = _dataset(expr, sample_ids=ids)
        sdrf_path = tmp_path / "study.sdrf.tsv"
        pd.DataFrame({"source name": ids, "disease": groups}).to_csv(sdrf_path, sep="\t", index=False)

        output = tmp_path / "report.html"
        qc_report(ds, output, sdrf_path=sdrf_path)
        html = output.read_text()
        assert "Measurement Precision Index" in html
        assert "Relative Spread" in html
        assert "Rank Concordance" in html
        assert "Case/control: yes" in html or r"Case\u002fcontrol: yes" in html
        assert "computed within each group" in html  # MPI help text
        assert "Precision &amp; Concordance" in html  # summary group
        assert "Median MPI (within groups)" in html

    def test_qc_report_without_groups_tags_no(self, tmp_path):
        expr, _ = _grouped_expression()
        ds = _dataset(expr)
        output = tmp_path / "report.html"
        qc_report(ds, output)
        html = output.read_text()
        assert "Case/control: no" in html or r"Case\u002fcontrol: no" in html
        assert "Measurement Precision Index" in html

    def test_split_report_writes_agnostic_files(self, tmp_path):
        expr, groups = _grouped_expression()
        ds = _dataset(expr, extra_sample_cols={"disease": groups})
        out_dir = tmp_path / "plots"
        qc_report_split(ds, out_dir)
        assert (out_dir / "mpi.html").exists()
        assert (out_dir / "dynamic_range.html").exists()
        assert (out_dir / "rank_concordance.html").exists()
        mpi_html = (out_dir / "mpi.html").read_text()
        assert "Case/control: yes" in mpi_html or r"Case\u002fcontrol: yes" in mpi_html
        assert "linear scale" in mpi_html
        manifest = json.loads((out_dir / "manifest.json").read_text())
        signal = next(t for t in manifest["tabs"] if t["id"] == "signal")
        structure = next(t for t in manifest["tabs"] if t["id"] == "structure")
        assert {"mpi", "dynamic_range"} <= {i["key"] for i in signal["items"]}
        assert "rank_concordance" in {i["key"] for i in structure["items"]}
        assert not manifest["unplaced"]


class TestReviewFixes:
    def test_mpi_and_spread_do_not_depend_on_npx_reference(self):
        """NPX is relative: shifting all values by a constant must not change precision or spread."""
        expr, _ = _grouped_expression()
        base = compute_mpi(_dataset(expr))
        shifted = compute_mpi(_dataset(expr + 5.0))
        assert shifted.median == pytest.approx(base.median, rel=1e-6)
        assert compute_dynamic_range(_dataset(expr + 5.0)).median == pytest.approx(
            compute_dynamic_range(_dataset(expr)).median, rel=1e-6
        )

    def test_control_samples_are_excluded(self):
        expr, _ = _grouped_expression()
        study = _dataset(expr, extra_sample_cols={"SampleType": ["SAMPLE"] * len(expr)})
        controls = pd.DataFrame(np.full((6, expr.shape[1]), -3.0), columns=expr.columns)
        with_controls = _dataset(
            pd.concat([expr, controls], ignore_index=True),
            extra_sample_cols={"SampleType": ["SAMPLE"] * len(expr) + ["NEGATIVE_CONTROL"] * 6},
        )
        assert compute_mpi(with_controls).median == pytest.approx(compute_mpi(study).median)
        assert compute_rank_concordance(with_controls).median == pytest.approx(compute_rank_concordance(study).median)

    def test_control_sample_names_do_not_create_groups(self):
        expr, _ = _grouped_expression()
        ids = [f"S{i:03d}" for i in range(len(expr))]
        controls = pd.DataFrame(np.full((6, expr.shape[1]), 2.0), columns=expr.columns)
        ds = _dataset(
            pd.concat([expr, controls], ignore_index=True),
            sample_ids=ids + [f"case_{i}" for i in range(3)] + [f"control_{i}" for i in range(3)],
            extra_sample_cols={"SampleType": ["SAMPLE"] * len(expr) + ["PLATE_CONTROL"] * 6},
        )
        assert compute_agnostic_qc(ds)["mpi"].has_case_control is False

    def test_sdrf_discovery_requires_same_accession(self, tmp_path):
        from pyprideap.cli import _discover_sdrf

        data = tmp_path / "PAD000002_npx.parquet"
        data.write_text("")
        (tmp_path / "PAD000001.sdrf.tsv").write_text("source name\n")
        assert _discover_sdrf(data) is None
        own = tmp_path / "PAD000002.sdrf.tsv"
        own.write_text("source name\n")
        assert _discover_sdrf(data) == own
        assert _discover_sdrf(tmp_path / "no_accession.npx.csv") is None

    def test_duplicate_sdrf_rows_do_not_add_samples(self):
        expr, groups = _grouped_expression()
        ids = [f"S{i:03d}" for i in range(len(groups))]
        ds = _dataset(expr, sample_ids=ids)
        sdrf = pd.DataFrame({"source name": ids + ids[:5], "disease": groups + groups[:5]})
        merged = merge_sdrf(ds, sdrf)
        assert len(merged.samples) == len(ds.samples)
        assert merged.samples["disease"].tolist() == groups

    def test_histograms_have_no_quality_bands(self):
        expr, _ = _grouped_expression()
        fig = render_mpi(compute_mpi(_dataset(expr)))
        assert all(shape.type == "line" for shape in fig.layout.shapes)  # only the median line
