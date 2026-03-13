"""Functional tests for statistical analysis: descriptive stats, differential testing, and plate design."""

import numpy as np
import pandas as pd
import pytest

from pyprideap.core import AffinityDataset, Platform
from pyprideap.stats.descriptive import compute_stats
from pyprideap.stats.design import randomize_plates

scipy_stats = pytest.importorskip("scipy.stats")
statsmodels = pytest.importorskip("statsmodels")

from pyprideap.stats.differential import anova, anova_posthoc, ttest, wilcoxon  # noqa: E402

# ---------------------------------------------------------------------------
# Dataset builders
# ---------------------------------------------------------------------------


def _make_stats_dataset():
    return AffinityDataset(
        platform=Platform.OLINK_EXPLORE,
        samples=pd.DataFrame(
            {
                "SampleID": ["S1", "S2", "S3"],
                "SampleType": ["SAMPLE", "SAMPLE", "CONTROL"],
                "SampleQC": ["PASS", "WARN", "PASS"],
            }
        ),
        features=pd.DataFrame(
            {
                "OlinkID": ["OID1", "OID2", "OID3"],
                "Panel": ["Inflammation", "Inflammation", "Oncology"],
            }
        ),
        expression=pd.DataFrame({"OID1": [3.5, 4.1, 0.1], "OID2": [2.0, np.nan, 0.05], "OID3": [5.6, 6.0, 0.2]}),
        metadata={},
    )


def _make_two_group_dataset():
    return AffinityDataset(
        platform=Platform.OLINK_EXPLORE,
        samples=pd.DataFrame(
            {
                "SampleID": [f"S{i}" for i in range(8)],
                "SampleType": ["SAMPLE"] * 8,
                "SampleQC": ["PASS"] * 8,
                "Group": ["A", "A", "A", "A", "B", "B", "B", "B"],
            }
        ),
        features=pd.DataFrame({"OlinkID": ["O1", "O2"], "UniProt": ["P1", "P2"], "Panel": ["Inf", "Inf"]}),
        expression=pd.DataFrame(
            {"O1": [1.0, 1.5, 2.0, 1.2, 5.0, 5.5, 6.0, 5.2], "O2": [3.0, 3.1, 3.2, 3.0, 3.3, 3.4, 3.1, 3.2]}
        ),
        metadata={},
    )


def _make_three_group_dataset():
    return AffinityDataset(
        platform=Platform.OLINK_EXPLORE,
        samples=pd.DataFrame(
            {
                "SampleID": [f"S{i}" for i in range(9)],
                "SampleType": ["SAMPLE"] * 9,
                "SampleQC": ["PASS"] * 9,
                "Group": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
            }
        ),
        features=pd.DataFrame({"OlinkID": ["O1", "O2"], "UniProt": ["P1", "P2"], "Panel": ["Inf", "Inf"]}),
        expression=pd.DataFrame(
            {
                "O1": [1.0, 1.5, 2.0, 5.0, 5.5, 6.0, 10.0, 10.5, 11.0],
                "O2": [3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8],
            }
        ),
        metadata={},
    )


# ---------------------------------------------------------------------------
# Descriptive statistics
# ---------------------------------------------------------------------------


class TestDescriptiveStats:
    def test_compute_stats(self):
        stats = compute_stats(_make_stats_dataset())
        assert stats.n_samples == 3
        assert stats.n_features == 3
        assert stats.detection_rate == pytest.approx(8 / 9, abs=0.01)
        assert stats.sample_types == {"SAMPLE": 2, "CONTROL": 1}
        assert stats.panels == {"Inflammation": 2, "Oncology": 1}
        assert stats.qc_summary == {"PASS": 2, "WARN": 1}
        assert "mean" in stats.value_distribution
        text = stats.summary()
        assert "3 samples" in text
        assert "3 features" in text


# ---------------------------------------------------------------------------
# Differential expression
# ---------------------------------------------------------------------------


class TestDifferentialExpression:
    def test_ttest(self):
        ds = _make_two_group_dataset()
        result = ttest(ds, group_var="Group")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        expected_cols = {"protein_id", "estimate", "statistic", "p_value", "adj_p_value", "significant"}
        assert expected_cols.issubset(set(result.columns))
        # O1 has large difference => significant
        row_o1 = result[result["protein_id"] == "O1"].iloc[0]
        assert row_o1["p_value"] < 0.05

    def test_ttest_wrong_groups_raises(self):
        ds = _make_three_group_dataset()
        with pytest.raises(ValueError, match="exactly 2"):
            ttest(ds, group_var="Group")

    def test_wilcoxon(self):
        ds = _make_two_group_dataset()
        result = wilcoxon(ds, group_var="Group")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert all(result["p_value"].notna())

    def test_anova(self):
        ds = _make_three_group_dataset()
        result = anova(ds, group_var="Group")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        expected_cols = {"protein_id", "statistic", "df_between", "df_within", "p_value", "adj_p_value"}
        assert expected_cols.issubset(set(result.columns))
        row_o1 = result[result["protein_id"] == "O1"].iloc[0]
        assert row_o1["p_value"] < 0.05
        assert row_o1["df_between"] == 2

    def test_anova_posthoc(self):
        ds = _make_three_group_dataset()
        result = anova_posthoc(ds, group_var="Group")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 6  # 3 pairwise × 2 proteins
        assert {"protein_id", "contrast", "estimate", "p_value", "adj_p_value"}.issubset(set(result.columns))


# ---------------------------------------------------------------------------
# Experimental design
# ---------------------------------------------------------------------------


class TestPlateRandomization:
    def test_assigns_all_samples(self):
        samples = pd.DataFrame({"SampleID": [f"S{i}" for i in range(20)], "SampleType": ["SAMPLE"] * 20})
        result = randomize_plates(samples, n_plates=1, plate_size=88, seed=42)
        assert "PlateNumber" in result.columns
        assert "WellPosition" in result.columns
        assert result["PlateNumber"].notna().all()
        assert len(result) == 20

    def test_respects_capacity(self):
        samples = pd.DataFrame({"SampleID": [f"S{i}" for i in range(50)], "SampleType": ["SAMPLE"] * 50})
        result = randomize_plates(samples, n_plates=3, plate_size=20, seed=42)
        assert all(count <= 20 for count in result["PlateNumber"].value_counts().values)

    def test_reproducible_with_seed(self):
        samples = pd.DataFrame({"SampleID": [f"S{i}" for i in range(30)], "SampleType": ["SAMPLE"] * 30})
        r1 = randomize_plates(samples, n_plates=2, plate_size=20, seed=123)
        r2 = randomize_plates(samples, n_plates=2, plate_size=20, seed=123)
        pd.testing.assert_frame_equal(r1, r2)

    def test_keep_paired(self):
        paired_values = ["subj1"] * 3 + ["subj2"] * 3 + ["subj3"] * 3 + ["subj4"] * 3
        samples = pd.DataFrame(
            {"SampleID": [f"S{i}" for i in range(12)], "SampleType": ["SAMPLE"] * 12, "SubjectID": paired_values}
        )
        result = randomize_plates(samples, n_plates=2, plate_size=10, keep_paired="SubjectID", seed=42)
        for subj in ["subj1", "subj2", "subj3", "subj4"]:
            plates = result.loc[result["SubjectID"] == subj, "PlateNumber"].unique()
            assert len(plates) == 1, f"Subject {subj} split across plates"

    def test_too_many_samples_raises(self):
        samples = pd.DataFrame({"SampleID": [f"S{i}" for i in range(100)], "SampleType": ["SAMPLE"] * 100})
        with pytest.raises(ValueError, match="Too many samples"):
            randomize_plates(samples, n_plates=1, plate_size=10, seed=42)
