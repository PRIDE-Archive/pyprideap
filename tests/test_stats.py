import numpy as np
import pandas as pd
import pytest

from pyprideap.core import AffinityDataset, Platform
from pyprideap.stats import compute_stats


def _make_dataset():
    return AffinityDataset(
        platform=Platform.OLINK_EXPLORE,
        samples=pd.DataFrame(
            {
                "SampleID": ["S001", "S002", "S003"],
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
        expression=pd.DataFrame(
            {
                "OID1": [3.5, 4.1, 0.1],
                "OID2": [2.0, np.nan, 0.05],
                "OID3": [5.6, 6.0, 0.2],
            }
        ),
        metadata={},
    )


class TestComputeStats:
    def test_n_samples(self):
        stats = compute_stats(_make_dataset())
        assert stats.n_samples == 3

    def test_n_features(self):
        stats = compute_stats(_make_dataset())
        assert stats.n_features == 3

    def test_features_per_sample(self):
        stats = compute_stats(_make_dataset())
        assert stats.features_per_sample.iloc[0] == 3
        assert stats.features_per_sample.iloc[1] == 2

    def test_samples_per_feature(self):
        stats = compute_stats(_make_dataset())
        assert stats.samples_per_feature["OID2"] == 2

    def test_detection_rate(self):
        stats = compute_stats(_make_dataset())
        assert stats.detection_rate == pytest.approx(8 / 9, abs=0.01)

    def test_sample_types(self):
        stats = compute_stats(_make_dataset())
        assert stats.sample_types == {"SAMPLE": 2, "CONTROL": 1}

    def test_panels(self):
        stats = compute_stats(_make_dataset())
        assert stats.panels == {"Inflammation": 2, "Oncology": 1}

    def test_qc_summary(self):
        stats = compute_stats(_make_dataset())
        assert stats.qc_summary == {"PASS": 2, "WARN": 1}

    def test_value_distribution(self):
        stats = compute_stats(_make_dataset())
        assert "mean" in stats.value_distribution
        assert "median" in stats.value_distribution
        assert "std" in stats.value_distribution
        assert "min" in stats.value_distribution
        assert "max" in stats.value_distribution

    def test_summary_returns_string(self):
        stats = compute_stats(_make_dataset())
        text = stats.summary()
        assert isinstance(text, str)
        assert "3 samples" in text
        assert "3 features" in text
