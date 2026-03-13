import json
from dataclasses import asdict

import pandas as pd
import pytest

from pyprideap.core import AffinityDataset, Platform
from pyprideap.viz.qc.compute import (
    CorrelationData,
    CvDistributionData,
    DataCompletenessData,
    DistributionData,
    LodAnalysisData,
    PcaData,
    QcLodSummaryData,
    compute_all,
    compute_correlation,
    compute_cv_distribution,
    compute_data_completeness,
    compute_distribution,
    compute_lod_analysis,
    compute_pca,
    compute_qc_summary,
)


def _make_olink_dataset():
    return AffinityDataset(
        platform=Platform.OLINK_EXPLORE,
        samples=pd.DataFrame(
            {
                "SampleID": ["S1", "S2"],
                "SampleType": ["SAMPLE", "SAMPLE"],
                "SampleQC": ["PASS", "PASS"],
            }
        ),
        features=pd.DataFrame({"OlinkID": ["O1", "O2"], "UniProt": ["P1", "P2"], "Panel": ["Inf", "Inf"]}),
        expression=pd.DataFrame({"O1": [3.5, 4.1], "O2": [2.0, -0.5]}),
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


class TestDataclassSerialization:
    def test_distribution_data_serializable(self):
        d = DistributionData(sample_ids=["S1"], sample_values=[[1.0, 2.0]], xlabel="NPX")
        result = json.dumps(asdict(d))
        assert '"sample_values"' in result

    def test_all_dataclasses_serializable(self):
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


class TestComputeDistribution:
    def test_olink_returns_per_sample_values(self):
        ds = _make_olink_dataset()
        result = compute_distribution(ds)
        assert result.xlabel == "NPX Value"
        assert len(result.sample_ids) == 2
        assert len(result.sample_values) == 2
        assert len(result.sample_values[0]) == 2  # 2 features per sample

    def test_somascan_returns_log10_rfu(self):
        ds = _make_somascan_dataset()
        result = compute_distribution(ds)
        assert "log10" in result.xlabel.lower()
        assert len(result.sample_values) == 2


class TestComputeQcSummary:
    def test_olink_returns_qc_counts(self):
        ds = _make_olink_dataset()
        ds.samples["SampleQC"] = ["PASS", "FAIL"]
        result = compute_qc_summary(ds)
        assert result is not None
        assert any("PASS" in c for c in result.categories)

    def test_somascan_returns_none(self):
        ds = _make_somascan_dataset()
        result = compute_qc_summary(ds)
        assert result is None


class TestComputeLodAnalysis:
    def test_olink_with_lod_returns_data(self):
        ds = _make_olink_dataset()
        ds.features["LOD"] = [1.0, 0.5]
        result = compute_lod_analysis(ds)
        assert result is not None
        assert len(result.assay_ids) == 2
        assert all(0 <= p <= 100 for p in result.above_lod_pct)

    def test_no_lod_returns_none(self):
        ds = _make_somascan_dataset()
        result = compute_lod_analysis(ds)
        assert result is None


class TestComputePca:
    def test_returns_pca_data(self):
        ds = _make_olink_dataset()
        result = compute_pca(ds)
        if result is None:
            pytest.skip("scikit-learn not installed")
        assert len(result.pc1) == 2
        assert len(result.variance_explained) == 2

    def test_labels_from_sample_id(self):
        ds = _make_olink_dataset()
        result = compute_pca(ds)
        if result is None:
            pytest.skip("scikit-learn not installed")
        assert result.labels == ["S1", "S2"]

    def test_uses_qc_for_groups_when_single_type(self):
        ds = _make_olink_dataset()
        ds.samples["SampleQC"] = ["PASS", "WARN"]
        result = compute_pca(ds)
        if result is None:
            pytest.skip("scikit-learn not installed")
        assert result.groups == ["PASS", "WARN"]


class TestComputeCorrelation:
    def test_returns_square_matrix(self):
        ds = _make_olink_dataset()
        result = compute_correlation(ds)
        assert len(result.matrix) == 2
        assert len(result.matrix[0]) == 2

    def test_diagonal_is_one(self):
        ds = _make_olink_dataset()
        result = compute_correlation(ds)
        for i in range(len(result.matrix)):
            assert abs(result.matrix[i][i] - 1.0) < 1e-6


class TestComputeDataCompleteness:
    def test_no_lod_returns_none(self):
        """SomaScan has no LOD sources, so should return None."""
        ds = _make_somascan_dataset()
        result = compute_data_completeness(ds)
        assert result is None

    def test_with_reported_lod(self):
        ds = _make_olink_dataset()
        ds.features["LOD"] = [3.0, 3.0]  # O1: 3.5>3, 4.1>3; O2: 2.0<3, -0.5<3
        result = compute_data_completeness(ds)
        assert result is not None
        assert result.below_lod_rate[0] > 0  # sample 0 has O2 below LOD
        assert result.above_lod_rate[0] + result.below_lod_rate[0] == pytest.approx(1.0)

    def test_with_missing_freq(self):
        ds = _make_olink_dataset()
        ds.features["MissingFreq"] = [0.1, 0.5]  # 10% and 50% below LOD
        result = compute_data_completeness(ds)
        assert result is not None
        assert result.missing_freq[0] == pytest.approx(0.1)
        assert result.missing_freq[1] == pytest.approx(0.5)


class TestComputeCvDistribution:
    def test_somascan_returns_cv(self):
        ds = _make_somascan_dataset()
        result = compute_cv_distribution(ds)
        assert result is not None
        assert len(result.cv_values) == 2

    def test_olink_returns_cv(self):
        ds = _make_olink_dataset()
        result = compute_cv_distribution(ds)
        assert result is not None
        assert len(result.cv_values) == 2


class TestComputeAll:
    def test_olink_returns_expected_keys(self):
        ds = _make_olink_dataset()
        result = compute_all(ds)
        assert "distribution" in result
        assert "correlation" in result

    def test_somascan_includes_cv(self):
        ds = _make_somascan_dataset()
        result = compute_all(ds)
        assert "cv_distribution" in result
        assert result["cv_distribution"] is not None

    def test_none_values_excluded(self):
        ds = _make_somascan_dataset()
        result = compute_all(ds)
        assert all(v is not None for v in result.values())
