"""Functional tests for the processing pipeline: LOD computation, filtering, and normalization."""

import numpy as np
import pandas as pd
import pytest

from pyprideap.core import AffinityDataset, Platform
from pyprideap.processing.filtering import filter_controls, filter_qc
from pyprideap.processing.lod import (
    LodMethod,
    compute_lod_from_controls,
    compute_lod_stats,
    compute_soma_elod,
    get_lod_values,
    get_proteins_above_lod,
    get_reported_lod,
    get_valid_proteins,
    load_fixed_lod,
)
from pyprideap.processing.normalization import (
    assess_bridgeability,
    bridge_normalize,
    lift_somascan,
    reference_median_normalize,
    scale_analytes,
    select_bridge_samples,
    subset_normalize,
)

_N_CONTROLS = 12


# ---------------------------------------------------------------------------
# Dataset builders
# ---------------------------------------------------------------------------


def _make_olink_dataset(
    n_samples=None,
    n_features=3,
    sample_types=None,
    qc_values=None,
    lod_values=None,
    expression_values=None,
):
    if n_samples is None:
        n_samples = 5 + _N_CONTROLS
    if sample_types is None:
        n_bio = n_samples - _N_CONTROLS
        sample_types = ["Sample"] * n_bio + ["Negative"] * _N_CONTROLS
    if qc_values is None:
        qc_values = ["PASS"] * n_samples

    samples = pd.DataFrame(
        {"SampleID": [f"S{i}" for i in range(n_samples)], "SampleType": sample_types, "SampleQC": qc_values}
    )
    feature_data = {
        "OlinkID": [f"OID{i}" for i in range(n_features)],
        "UniProt": [f"P{i}" for i in range(n_features)],
        "Panel": ["PanelA"] * n_features,
    }
    if lod_values is not None:
        feature_data["LOD"] = lod_values
    features = pd.DataFrame(feature_data)

    if expression_values is not None:
        expression = pd.DataFrame(expression_values, columns=[f"OID{j}" for j in range(n_features)])
    else:
        rng = np.random.default_rng(42)
        n_ctrl = sum(1 for s in sample_types if s.lower() in {"negative", "negative control"})
        n_bio = n_samples - n_ctrl
        expr_data = np.empty((n_samples, n_features))
        expr_data[:n_bio, :] = rng.normal(5.0, 2.0, size=(n_bio, n_features))
        if n_ctrl > 0:
            expr_data[n_bio:, :] = rng.normal(0.5, 0.3, size=(n_ctrl, n_features))
        expression = pd.DataFrame(expr_data, columns=[f"OID{j}" for j in range(n_features)])

    return AffinityDataset(platform=Platform.OLINK_EXPLORE, samples=samples, features=features, expression=expression)


def _make_somascan_dataset_with_buffers(n_bio=5, n_buffers=10, n_features=3):
    n_samples = n_bio + n_buffers
    samples = pd.DataFrame(
        {
            "SampleId": [f"S{i}" for i in range(n_samples)],
            "SampleType": ["Sample"] * n_bio + ["Buffer"] * n_buffers,
        }
    )
    features = pd.DataFrame(
        {
            "SeqId": [f"10000-{i}" for i in range(n_features)],
            "UniProt": [f"P{i}" for i in range(n_features)],
            "Target": [f"T{i}" for i in range(n_features)],
            "Dilution": ["20"] * n_features,
        }
    )
    rng = np.random.default_rng(42)
    expr_data = np.empty((n_samples, n_features))
    expr_data[:n_bio, :] = rng.normal(5000.0, 1000.0, size=(n_bio, n_features))
    expr_data[n_bio:, :] = rng.normal(100.0, 20.0, size=(n_buffers, n_features))
    expr_data = np.abs(expr_data)
    expression = pd.DataFrame(expr_data, columns=[f"SL{i}" for i in range(n_features)])
    return AffinityDataset(
        platform=Platform.SOMASCAN,
        samples=samples,
        features=features,
        expression=expression,
        metadata={"AssayVersion": "v4.1"},
    )


def _make_norm_dataset(sample_ids, expression_data, qc=None):
    samples_dict = {
        "SampleID": sample_ids,
        "SampleType": ["SAMPLE"] * len(sample_ids),
        "SampleQC": qc or ["PASS"] * len(sample_ids),
    }
    samples = pd.DataFrame(samples_dict)
    expr = pd.DataFrame(expression_data, index=range(len(sample_ids)))
    features = pd.DataFrame(
        {
            "OlinkID": list(expr.columns),
            "UniProt": [f"P{i}" for i in range(len(expr.columns))],
            "Panel": ["Inf"] * len(expr.columns),
        }
    )
    return AffinityDataset(
        platform=Platform.OLINK_EXPLORE, samples=samples, features=features, expression=expr, metadata={}
    )


# ---------------------------------------------------------------------------
# LOD computation
# ---------------------------------------------------------------------------


class TestLodOlink:
    def test_nclod_from_negative_controls(self):
        ds = _make_olink_dataset()
        lod = compute_lod_from_controls(ds)
        assert len(lod) == 3
        assert all(np.isfinite(lod))

    def test_nclod_requires_min_controls(self):
        ds = _make_olink_dataset(n_samples=12, sample_types=["Sample"] * 3 + ["Negative"] * 9)
        with pytest.raises(ValueError, match="at least"):
            compute_lod_from_controls(ds)

    def test_nclod_requires_sample_type_column(self):
        ds = _make_olink_dataset()
        ds.samples = ds.samples.drop(columns=["SampleType"])
        with pytest.raises(ValueError, match="SampleType"):
            compute_lod_from_controls(ds)

    def test_reported_lod_from_features(self):
        ds = _make_olink_dataset(lod_values=[1.0, 2.0, 3.0])
        lod = get_lod_values(ds)
        assert lod is not None
        assert lod["OID0"] == 1.0

    def test_reported_lod_none_without_column(self):
        ds = _make_olink_dataset()
        assert get_reported_lod(ds) is None

    def test_lod_stats_pipeline(self):
        ds = _make_olink_dataset(lod_values=[2.0, 2.0, 2.0])
        stats = compute_lod_stats(ds)
        assert stats.lod_source == "reported"
        assert stats.n_assays_with_lod == 3
        assert 0.0 < stats.above_lod_rate <= 1.0
        assert "LOD source" in stats.summary()

    def test_get_valid_proteins(self):
        ds = _make_olink_dataset(
            lod_values=[10.0, 2.0, 10.0],
            expression_values=([[1.0, 5.0, 1.0]] * 5 + [[0.5, 0.3, 0.4]] * _N_CONTROLS),
        )
        valid = get_valid_proteins(ds)
        assert "OID1" in valid
        assert "OID0" not in valid

    def test_load_fixed_lod(self, tmp_path):
        csv_path = tmp_path / "fixedLOD.csv"
        csv_path.write_text(
            "OlinkID;DataAnalysisRefID;LODNPX;LODCount;LODMethod\n"
            "OID0;REF001;1.1;100;lod_npx\n"
            "OID1;REF001;2.2;200;lod_npx\n"
            "OID2;REF001;3.3;300;lod_npx\n"
        )
        ds = _make_olink_dataset()
        lod = load_fixed_lod(ds, csv_path)
        assert isinstance(lod, pd.Series)
        assert abs(lod["OID0"] - 1.1) < 1e-10

    def test_method_dispatch(self):
        ds = _make_olink_dataset(lod_values=[1.0, 2.0, 3.0])
        assert get_lod_values(ds, method=LodMethod.REPORTED) is not None
        assert get_lod_values(ds, method="NCLOD") is not None


class TestLodSomascan:
    def test_elod_from_buffers(self):
        ds = _make_somascan_dataset_with_buffers()
        lod = compute_soma_elod(ds)
        assert len(lod) == 3
        assert all(np.isfinite(lod))
        # eLOD should be above buffer median
        buffer_medians = ds.expression.iloc[5:].median()
        for col in ds.expression.columns:
            assert lod[col] > buffer_medians[col]

    def test_elod_requires_buffers(self):
        ds = _make_somascan_dataset_with_buffers()
        ds.samples["SampleType"] = "Sample"
        with pytest.raises(ValueError, match="buffer"):
            compute_soma_elod(ds)

    def test_method_dispatch(self):
        ds = _make_somascan_dataset_with_buffers()
        lod = get_lod_values(ds, method="SOMA_ELOD")
        assert lod is not None


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


class TestFiltering:
    def test_filter_controls_removes_non_sample(self):
        ds = _make_olink_dataset(
            n_samples=4,
            sample_types=["Sample", "CONTROL", "Sample", "Negative"],
            qc_values=["PASS"] * 4,
        )
        result = filter_controls(ds)
        assert len(result.samples) == 2
        assert len(result.expression) == 2

    def test_filter_qc_keeps_pass_and_warn(self):
        ds = _make_olink_dataset(
            n_samples=4,
            sample_types=["Sample"] * 4,
            qc_values=["PASS", "FAIL", "WARN", "NA"],
        )
        result = filter_qc(ds)
        assert len(result.samples) == 2
        assert set(result.samples["SampleQC"]) == {"PASS", "WARN"}
        assert len(result.expression) == len(result.samples)


# ---------------------------------------------------------------------------
# Normalization — Olink (additive)
# ---------------------------------------------------------------------------


class TestNormalizationOlink:
    def test_bridge_normalize(self):
        ds1 = _make_norm_dataset(["S1", "S2", "S3"], {"O1": [10.0, 11.0, 12.0], "O2": [5.0, 6.0, 7.0]})
        ds2 = _make_norm_dataset(["S1", "S2", "S4"], {"O1": [8.0, 9.0, 10.0], "O2": [3.0, 4.0, 5.0]})
        result = bridge_normalize(ds1, ds2, bridge_samples=[0, 1])
        assert result.expression["O1"].iloc[0] == pytest.approx(10.0)
        assert result.expression["O1"].iloc[2] == pytest.approx(12.0)

    def test_bridge_normalize_no_overlap_raises(self):
        ds1 = _make_norm_dataset(["S1", "S2"], {"O1": [10.0, 11.0]})
        ds2 = _make_norm_dataset(["S3", "S4"], {"O1": [8.0, 9.0]})
        with pytest.raises(ValueError, match="No bridge samples"):
            bridge_normalize(ds1, ds2, bridge_samples=[10, 11])

    def test_subset_normalize(self):
        ds1 = _make_norm_dataset(["S1", "S2", "S3"], {"O1": [10.0, 12.0, 14.0], "O2": [4.0, 6.0, 8.0]})
        ds2 = _make_norm_dataset(["S4", "S5", "S6"], {"O1": [7.0, 9.0, 11.0], "O2": [1.0, 3.0, 5.0]})
        result = subset_normalize(ds1, ds2, reference_proteins=["O1"])
        assert result.expression["O1"].iloc[0] == pytest.approx(10.0)
        # O2 not a reference protein, not adjusted
        assert result.expression["O2"].iloc[0] == pytest.approx(1.0)

    def test_reference_median_normalize(self):
        ds = _make_norm_dataset(["S1", "S2", "S3"], {"O1": [2.0, 4.0, 6.0], "O2": [1.0, 3.0, 5.0]})
        result = reference_median_normalize(ds, {"O1": 10.0, "O2": 5.0})
        assert result.expression["O1"].median() == pytest.approx(10.0)
        assert result.expression["O2"].median() == pytest.approx(5.0)

    def test_select_bridge_samples(self):
        ds = _make_norm_dataset(
            ["S1", "S2", "S3", "S4", "S5"],
            {"O1": [1.0, 2.0, 3.0, 4.0, 5.0], "O2": [5.0, 4.0, 3.0, 2.0, 1.0]},
        )
        result = select_bridge_samples(ds, n=3)
        assert len(result) == 3

    def test_assess_bridgeability(self):
        ds1 = _make_norm_dataset(["S1", "S2", "S3"], {"O1": [3.0, 4.0, 5.0], "O2": [1.0, 2.0, 3.0]})
        ds2 = _make_norm_dataset(["S1", "S2", "S3"], {"O1": [3.5, 4.5, 5.5], "O2": [1.5, 2.5, 3.5]})
        result = assess_bridgeability(ds1, ds2)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert result["correlation"].iloc[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Normalization — SomaScan (multiplicative)
# ---------------------------------------------------------------------------


class TestNormalizationSomascan:
    def _make_soma(self, sample_ids, expression_data):
        samples = pd.DataFrame({"SampleId": sample_ids, "SampleType": ["Sample"] * len(sample_ids)})
        expr = pd.DataFrame(expression_data, index=range(len(sample_ids)))
        features = pd.DataFrame(
            {
                "SeqId": list(expr.columns),
                "UniProt": [f"P{i}" for i in range(len(expr.columns))],
                "Target": [f"T{i}" for i in range(len(expr.columns))],
                "Dilution": ["20"] * len(expr.columns),
            }
        )
        return AffinityDataset(
            platform=Platform.SOMASCAN,
            samples=samples,
            features=features,
            expression=expr,
            metadata={"AssayVersion": "v4.1"},
        )

    def test_scale_analytes(self):
        ds = self._make_soma(["S1", "S2"], {"SL0": [1000.0, 2000.0], "SL1": [500.0, 1500.0]})
        result = scale_analytes(ds, {"SL0": 2.0, "SL1": 0.5})
        assert result.expression["SL0"].iloc[0] == pytest.approx(2000.0)
        assert result.expression["SL1"].iloc[0] == pytest.approx(250.0)
        # Does not mutate original
        assert ds.expression["SL0"].iloc[0] == pytest.approx(1000.0)

    def test_lift_somascan(self):
        ds = self._make_soma(["S1", "S2"], {"SL0": [1000.0, 2000.0], "SL1": [500.0, 1500.0]})
        result = lift_somascan(ds, {"SL0": 1.5}, target_version="7k")
        assert result.expression["SL0"].iloc[0] == pytest.approx(1500.0)
        assert result.expression["SL1"].iloc[0] == pytest.approx(500.0)  # default 1.0
        assert result.metadata["SignalSpace"] == "7k"
        # Does not mutate original
        assert ds.expression["SL0"].iloc[0] == pytest.approx(1000.0)


# ---------------------------------------------------------------------------
# Regression tests for audit fixes
# ---------------------------------------------------------------------------


class TestAuditFixes:
    def test_col_check_filter_uses_seqid_not_position(self):
        """filter_by_col_check must map by SeqId, not by positional index."""
        from pyprideap.processing.somascan.qc_flags import filter_by_col_check

        features = pd.DataFrame(
            {
                "SeqId": ["10000-1", "10001-2", "10002-3"],
                "ColCheck": ["PASS", "FLAG", "PASS"],
            }
        )
        expression = pd.DataFrame(
            {
                "10000-1": [100.0, 200.0],
                "10001-2": [300.0, 400.0],
                "10002-3": [500.0, 600.0],
            }
        )
        samples = pd.DataFrame({"SampleId": ["S1", "S2"], "SampleType": ["Sample", "Sample"]})
        ds = AffinityDataset(
            platform=Platform.SOMASCAN,
            samples=samples,
            features=features,
            expression=expression,
        )
        result = filter_by_col_check(ds)
        assert list(result.expression.columns) == ["10000-1", "10002-3"]
        assert len(result.features) == 2

    def test_get_valid_proteins_with_filtered_lod(self):
        """get_valid_proteins must align LOD rows after filtering controls."""
        ds = _make_olink_dataset(
            lod_values=[2.0, 2.0, 2.0],
            expression_values=([[5.0, 5.0, 5.0]] * 5 + [[0.1, 0.1, 0.1]] * _N_CONTROLS),
        )
        valid = get_valid_proteins(ds)
        # All 3 proteins should be valid (bio samples are above LOD)
        assert len(valid) == 3

    def test_assay_map_uses_column_names(self):
        """_resolve_assay_map must key by OlinkID, not integer index."""
        from pyprideap.stats.differential import _resolve_assay_map

        ds = _make_olink_dataset()
        ds.features["Assay"] = ["AssayA", "AssayB", "AssayC"]
        assay_map = _resolve_assay_map(ds)
        assert assay_map.get("OID0") == "AssayA"
        assert assay_map.get("OID1") == "AssayB"

    def test_select_bridge_samples_non_standard_sample_type(self):
        """select_bridge_samples must accept non-'SAMPLE' biological types."""
        ds = _make_norm_dataset(
            ["S1", "S2", "S3", "S4", "S5"],
            {"O1": [1.0, 2.0, 3.0, 4.0, 5.0], "O2": [5.0, 4.0, 3.0, 2.0, 1.0]},
        )
        ds.samples["SampleType"] = "Subject"
        result = select_bridge_samples(ds, n=3)
        assert len(result) == 3

    def test_proteins_above_lod_returns_accessions(self):
        """get_proteins_above_lod returns UniProt strings, not assay IDs."""
        # Use a dataset with only biological samples (no controls)
        n = 5
        ds = AffinityDataset(
            platform=Platform.OLINK_EXPLORE,
            samples=pd.DataFrame(
                {
                    "SampleID": [f"S{i}" for i in range(n)],
                    "SampleType": ["Sample"] * n,
                    "SampleQC": ["PASS"] * n,
                }
            ),
            features=pd.DataFrame(
                {
                    "OlinkID": ["OID0", "OID1", "OID2"],
                    "UniProt": ["P0", "P1", "P2"],
                    "Panel": ["A"] * 3,
                    "LOD": [2.0, 2.0, 2.0],
                }
            ),
            expression=pd.DataFrame(
                {
                    "OID0": [5.0] * n,
                    "OID1": [5.0] * n,
                    "OID2": [0.1] * n,
                }
            ),
        )
        result = get_proteins_above_lod(ds, threshold=50.0)
        assert "P0" in result
        assert "P1" in result
        # P2 is below LOD for all samples
        assert "P2" not in result
