import pandas as pd
import pytest

scipy_stats = pytest.importorskip("scipy.stats")
statsmodels = pytest.importorskip("statsmodels")

from pyprideap.core import AffinityDataset, Platform
from pyprideap.stats.differential import anova, anova_posthoc, ttest, wilcoxon


def _make_two_group_dataset():
    """Dataset with two groups, 4 samples each."""
    return AffinityDataset(
        platform=Platform.OLINK_EXPLORE,
        samples=pd.DataFrame({
            "SampleID": [f"S{i}" for i in range(8)],
            "SampleType": ["SAMPLE"] * 8,
            "SampleQC": ["PASS"] * 8,
            "Group": ["A", "A", "A", "A", "B", "B", "B", "B"],
        }),
        features=pd.DataFrame({
            "OlinkID": ["O1", "O2"],
            "UniProt": ["P1", "P2"],
            "Panel": ["Inf", "Inf"],
        }),
        expression=pd.DataFrame({
            "O1": [1.0, 1.5, 2.0, 1.2, 5.0, 5.5, 6.0, 5.2],
            "O2": [3.0, 3.1, 3.2, 3.0, 3.3, 3.4, 3.1, 3.2],
        }),
        metadata={},
    )


def _make_three_group_dataset():
    """Dataset with three groups, 3 samples each."""
    return AffinityDataset(
        platform=Platform.OLINK_EXPLORE,
        samples=pd.DataFrame({
            "SampleID": [f"S{i}" for i in range(9)],
            "SampleType": ["SAMPLE"] * 9,
            "SampleQC": ["PASS"] * 9,
            "Group": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
        }),
        features=pd.DataFrame({
            "OlinkID": ["O1", "O2"],
            "UniProt": ["P1", "P2"],
            "Panel": ["Inf", "Inf"],
        }),
        expression=pd.DataFrame({
            "O1": [1.0, 1.5, 2.0, 5.0, 5.5, 6.0, 10.0, 10.5, 11.0],
            "O2": [3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8],
        }),
        metadata={},
    )


class TestTtest:
    def test_ttest_two_groups(self):
        """t-test on a two-group dataset should return expected columns."""
        ds = _make_two_group_dataset()
        result = ttest(ds, group_var="Group")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2  # one row per protein
        expected_cols = {
            "protein_id", "estimate", "statistic",
            "p_value", "adj_p_value", "significant",
        }
        assert expected_cols.issubset(set(result.columns))
        # O1 has a large difference between groups => should be significant
        row_o1 = result[result["protein_id"] == "O1"].iloc[0]
        assert row_o1["p_value"] < 0.05

    def test_ttest_wrong_groups_raises(self):
        """ValueError when group_var does not have exactly 2 levels."""
        ds = _make_three_group_dataset()
        with pytest.raises(ValueError, match="exactly 2"):
            ttest(ds, group_var="Group")


class TestWilcoxon:
    def test_wilcoxon_two_groups(self):
        """Mann-Whitney U test on two-group dataset returns valid output."""
        ds = _make_two_group_dataset()
        result = wilcoxon(ds, group_var="Group")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        expected_cols = {
            "protein_id", "estimate", "statistic",
            "p_value", "adj_p_value", "significant",
        }
        assert expected_cols.issubset(set(result.columns))
        # p_values should be valid numbers
        assert all(result["p_value"].notna())


class TestAnova:
    def test_anova_multi_group(self):
        """ANOVA on a 3-group dataset should return expected columns."""
        ds = _make_three_group_dataset()
        result = anova(ds, group_var="Group")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        expected_cols = {
            "protein_id", "statistic", "df_between", "df_within",
            "p_value", "adj_p_value", "significant",
        }
        assert expected_cols.issubset(set(result.columns))
        # O1 has strongly separated groups => significant
        row_o1 = result[result["protein_id"] == "O1"].iloc[0]
        assert row_o1["p_value"] < 0.05
        assert row_o1["df_between"] == 2  # 3 groups - 1

    def test_anova_posthoc_returns_contrasts(self):
        """Post-hoc Tukey HSD should return pairwise contrasts."""
        ds = _make_three_group_dataset()
        result = anova_posthoc(ds, group_var="Group")

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        expected_cols = {
            "protein_id", "contrast", "estimate",
            "p_value", "adj_p_value",
        }
        assert expected_cols.issubset(set(result.columns))
        # 3 groups => 3 pairwise contrasts per protein, 2 proteins => 6 rows
        assert len(result) == 6
        # Check contrast format
        contrasts = result["contrast"].unique()
        assert len(contrasts) == 3
