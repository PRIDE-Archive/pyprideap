"""Tests for LOD computation and analysis."""

import numpy as np
import pandas as pd
import pytest

from pyprideap.core import AffinityDataset, Platform
from pyprideap.processing.lod import (
    _MAD_TO_SD,
    _MIN_STD_FLOOR,
    _SOMA_ELOD_K,
    LodMethod,
    compute_lod_from_controls,
    compute_lod_stats,
    compute_soma_elod,
    get_lod_values,
    get_reported_lod,
    get_valid_proteins,
    load_fixed_lod,
)

# Number of negative controls that satisfies the minimum (>=10)
_N_CONTROLS = 12


def _make_olink_dataset(
    n_samples=None,
    n_features=3,
    sample_types=None,
    qc_values=None,
    lod_values=None,
    expression_values=None,
):
    if n_samples is None:
        n_samples = 5 + _N_CONTROLS  # 5 biological + 12 controls
    if sample_types is None:
        n_bio = n_samples - _N_CONTROLS
        sample_types = ["Sample"] * n_bio + ["Negative"] * _N_CONTROLS
    if qc_values is None:
        qc_values = ["PASS"] * n_samples

    samples = pd.DataFrame(
        {
            "SampleID": [f"S{i}" for i in range(n_samples)],
            "SampleType": sample_types,
            "SampleQC": qc_values,
        }
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
        expression = pd.DataFrame(
            expression_values,
            columns=[f"OID{j}" for j in range(n_features)],
        )
    else:
        rng = np.random.default_rng(42)
        # Count actual negative controls from sample_types
        n_ctrl = sum(1 for s in sample_types if s.lower() in {"negative", "negative control"})
        n_bio = n_samples - n_ctrl
        expr_data = np.empty((n_samples, n_features))
        expr_data[:n_bio, :] = rng.normal(5.0, 2.0, size=(n_bio, n_features))
        if n_ctrl > 0:
            expr_data[n_bio:, :] = rng.normal(0.5, 0.3, size=(n_ctrl, n_features))
        expression = pd.DataFrame(
            expr_data,
            columns=[f"OID{j}" for j in range(n_features)],
        )

    return AffinityDataset(
        platform=Platform.OLINK_EXPLORE,
        samples=samples,
        features=features,
        expression=expression,
    )


class TestComputeLodFromControls:
    def test_computes_lod_from_negative_controls(self):
        ds = _make_olink_dataset()
        lod = compute_lod_from_controls(ds)
        assert len(lod) == 3
        assert all(np.isfinite(lod))
        # LOD = median + max(0.2, 3*std) per Olink official method
        for col in ds.expression.columns:
            control_vals = ds.expression.iloc[-_N_CONTROLS:][col]
            med = control_vals.median()
            std = control_vals.std()
            expected = med + max(_MIN_STD_FLOOR, 3 * std)
            assert abs(lod[col] - expected) < 1e-10

    def test_std_floor_applied(self):
        """When std is very small, the 0.2 NPX floor kicks in."""
        n_controls = 12
        n_samples = 3 + n_controls
        sample_types = ["Sample"] * 3 + ["Negative"] * n_controls
        # All controls have identical values → std = 0
        expr = [[5.0, 5.0, 5.0]] * 3 + [[1.0, 1.0, 1.0]] * n_controls
        ds = _make_olink_dataset(
            n_samples=n_samples,
            sample_types=sample_types,
            expression_values=expr,
        )
        lod = compute_lod_from_controls(ds)
        # std=0 so floor of 0.2 applies: LOD = 1.0 + 0.2 = 1.2
        for col in ds.expression.columns:
            assert abs(lod[col] - 1.2) < 1e-10

    def test_raises_without_sample_type(self):
        ds = _make_olink_dataset()
        ds.samples = ds.samples.drop(columns=["SampleType"])
        with pytest.raises(ValueError, match="SampleType"):
            compute_lod_from_controls(ds)

    def test_raises_with_too_few_controls(self):
        """Need at least 10 controls per Olink specification."""
        ds = _make_olink_dataset(
            n_samples=12,
            sample_types=["Sample"] * 3 + ["Negative"] * 9,
        )
        with pytest.raises(ValueError, match="at least"):
            compute_lod_from_controls(ds)


class TestGetLodValues:
    def test_extracts_lod_from_features(self):
        ds = _make_olink_dataset(lod_values=[1.0, 2.0, 3.0])
        lod = get_lod_values(ds)
        assert lod is not None
        assert len(lod) == 3
        assert lod["OID0"] == 1.0
        assert lod["OID2"] == 3.0

    def test_returns_none_when_no_lod_column(self):
        ds = _make_olink_dataset()
        lod = get_lod_values(ds)
        assert lod is None

    def test_returns_none_when_all_lod_nan(self):
        ds = _make_olink_dataset(lod_values=[np.nan, np.nan, np.nan])
        lod = get_lod_values(ds)
        assert lod is None


class TestComputeLodStats:
    def test_stats_with_lod_from_features(self):
        n = 5 + _N_CONTROLS
        ds = _make_olink_dataset(
            lod_values=[2.0, 2.0, 2.0],
            expression_values=([[5.0, 5.0, 5.0]] * 3 + [[1.0, 5.0, 1.0]] * 2 + [[0.5, 0.3, 0.4]] * _N_CONTROLS),
        )
        stats = compute_lod_stats(ds)
        assert stats.lod_source == "reported"
        assert stats.n_assays_with_lod == 3
        assert 0.0 < stats.above_lod_rate <= 1.0
        assert len(stats.above_lod_per_sample) == n
        assert "PanelA" in stats.above_lod_per_panel

    def test_stats_computed_from_controls(self):
        ds = _make_olink_dataset()
        stats = compute_lod_stats(ds)
        assert stats.lod_source == "nclod"
        assert stats.n_assays_with_lod > 0

    def test_stats_not_available(self):
        ds = _make_olink_dataset(
            n_samples=3,
            sample_types=["Sample", "Sample", "Sample"],
        )
        stats = compute_lod_stats(ds)
        assert stats.lod_source == "not_available"
        assert stats.above_lod_rate == 0.0

    def test_summary_string(self):
        ds = _make_olink_dataset(lod_values=[2.0, 2.0, 2.0])
        stats = compute_lod_stats(ds)
        summary = stats.summary()
        assert "LOD source" in summary
        assert "above-LOD rate" in summary


class TestGetValidProteins:
    def test_returns_proteins_above_lod(self):
        ds = _make_olink_dataset(
            lod_values=[10.0, 2.0, 10.0],
            expression_values=([[1.0, 5.0, 1.0]] * 5 + [[0.5, 0.3, 0.4]] * _N_CONTROLS),
        )
        valid = get_valid_proteins(ds)
        # Only OID1 has NPX (5.0) > LOD (2.0)
        assert "OID1" in valid
        assert "OID0" not in valid
        assert "OID2" not in valid

    def test_excludes_fail_qc_samples(self):
        n_bio = 5
        ds = _make_olink_dataset(
            lod_values=[2.0, 2.0, 2.0],
            qc_values=["FAIL"] * n_bio + ["PASS"] * _N_CONTROLS,
            expression_values=([[5.0, 5.0, 5.0]] * n_bio + [[0.5, 0.3, 0.4]] * _N_CONTROLS),
        )
        # All biological samples FAIL QC; only PASS samples are controls
        # After filtering controls, no biological samples with PASS remain
        valid = get_valid_proteins(ds)
        assert valid == []

    def test_returns_all_when_no_lod(self):
        ds = _make_olink_dataset(
            n_samples=3,
            sample_types=["Sample", "Sample", "Sample"],
        )
        valid = get_valid_proteins(ds)
        # No LOD available, returns all proteins with non-NaN values
        assert len(valid) == 3


class TestGetReportedLod:
    def test_extracts_from_lod_matrix_metadata(self):
        ds = _make_olink_dataset()
        # Simulate lod_matrix in metadata (sample × assay DataFrame)
        lod_matrix = pd.DataFrame(
            [[1.0, 2.0, 3.0]] * len(ds.samples),
            columns=ds.expression.columns,
        )
        ds.metadata["lod_matrix"] = lod_matrix
        result = get_reported_lod(ds)
        assert isinstance(result, pd.DataFrame)
        assert result.shape == ds.expression.shape

    def test_falls_back_to_features_lod(self):
        ds = _make_olink_dataset(lod_values=[1.5, 2.5, 3.5])
        result = get_reported_lod(ds)
        assert isinstance(result, pd.Series)
        assert result["OID1"] == 2.5

    def test_returns_none_without_lod(self):
        ds = _make_olink_dataset()
        result = get_reported_lod(ds)
        assert result is None


class TestLoadFixedLod:
    def test_loads_from_semicolon_csv(self, tmp_path):
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
        assert abs(lod["OID1"] - 2.2) < 1e-10
        assert abs(lod["OID2"] - 3.3) < 1e-10

    def test_loads_from_comma_csv(self, tmp_path):
        csv_path = tmp_path / "fixedLOD.csv"
        csv_path.write_text(
            "OlinkID,DataAnalysisRefID,LODNPX,LODCount,LODMethod\n"
            "OID0,REF001,1.1,100,lod_npx\n"
            "OID1,REF001,2.2,200,lod_npx\n"
            "OID2,REF001,3.3,300,lod_npx\n"
        )
        ds = _make_olink_dataset()
        lod = load_fixed_lod(ds, csv_path)
        assert isinstance(lod, pd.Series)
        assert abs(lod["OID0"] - 1.1) < 1e-10

    def test_joins_on_data_analysis_ref_id(self, tmp_path):
        csv_path = tmp_path / "fixedLOD.csv"
        csv_path.write_text(
            "OlinkID;DataAnalysisRefID;LODNPX;LODCount;LODMethod\n"
            "OID0;REF001;1.0;100;lod_npx\n"
            "OID0;REF002;9.9;100;lod_npx\n"
            "OID1;REF001;2.0;200;lod_npx\n"
            "OID2;REF001;3.0;300;lod_npx\n"
        )
        ds = _make_olink_dataset()
        ds.features["DataAnalysisRefID"] = "REF001"
        lod = load_fixed_lod(ds, csv_path)
        # Should pick REF001 (1.0), not REF002 (9.9)
        assert abs(lod["OID0"] - 1.0) < 1e-10

    def test_raises_on_missing_file(self):
        ds = _make_olink_dataset()
        with pytest.raises(FileNotFoundError):
            load_fixed_lod(ds, "/nonexistent/file.csv")

    def test_raises_on_missing_columns(self, tmp_path):
        csv_path = tmp_path / "bad.csv"
        csv_path.write_text("OlinkID;SomeOtherCol\nOID0;1.0\n")
        ds = _make_olink_dataset()
        with pytest.raises(ValueError, match="LODNPX"):
            load_fixed_lod(ds, csv_path)

    def test_raises_without_bundled_or_path(self):
        ds = _make_olink_dataset()
        # olink_target has no bundled file
        ds.platform = Platform.OLINK_TARGET
        with pytest.raises(ValueError, match="No bundled"):
            load_fixed_lod(ds)


class TestLodMethodDispatch:
    def test_reported_method(self):
        ds = _make_olink_dataset(lod_values=[1.0, 2.0, 3.0])
        lod = get_lod_values(ds, method=LodMethod.REPORTED)
        assert lod is not None
        assert lod["OID0"] == 1.0

    def test_reported_is_default(self):
        ds = _make_olink_dataset(lod_values=[1.0, 2.0, 3.0])
        lod = get_lod_values(ds)
        assert lod is not None
        assert lod["OID0"] == 1.0

    def test_nclod_method(self):
        ds = _make_olink_dataset()
        lod = get_lod_values(ds, method="NCLOD")
        assert lod is not None

    def test_fixed_without_bundled_raises(self):
        ds = _make_olink_dataset()
        ds.platform = Platform.OLINK_TARGET
        with pytest.raises(ValueError, match="No bundled"):
            get_lod_values(ds, method=LodMethod.FIXED)

    def test_fixed_with_file(self, tmp_path):
        csv_path = tmp_path / "fixedLOD.csv"
        csv_path.write_text(
            "OlinkID;DataAnalysisRefID;LODNPX;LODCount;LODMethod\n"
            "OID0;REF001;1.1;100;lod_npx\n"
            "OID1;REF001;2.2;200;lod_npx\n"
            "OID2;REF001;3.3;300;lod_npx\n"
        )
        ds = _make_olink_dataset()
        lod = get_lod_values(ds, method="FIXED", lod_file_path=csv_path)
        assert abs(lod["OID0"] - 1.1) < 1e-10

    def test_string_method_names(self):
        ds = _make_olink_dataset(lod_values=[1.0, 2.0, 3.0])
        assert get_lod_values(ds, method="REPORTED") is not None
        assert get_lod_values(ds, method="NCLOD") is not None

    def test_soma_elod_method(self):
        ds = _make_somascan_dataset_with_buffers()
        lod = get_lod_values(ds, method="SOMA_ELOD")
        assert lod is not None
        assert len(lod) == 3

    def test_soma_elod_returns_none_without_buffers(self):
        ds = _make_somascan_dataset_with_buffers()
        # Remove buffer samples
        ds.samples["SampleType"] = "Sample"
        lod = get_lod_values(ds, method=LodMethod.SOMA_ELOD)
        assert lod is None


# ---------------------------------------------------------------------------
# SomaScan eLOD
# ---------------------------------------------------------------------------

_N_BUFFERS = 10


def _make_somascan_dataset_with_buffers(
    n_bio=5,
    n_buffers=_N_BUFFERS,
    n_features=3,
):
    """Build a SomaScan dataset with buffer samples for eLOD testing."""
    n_samples = n_bio + n_buffers
    samples = pd.DataFrame({
        "SampleId": [f"S{i}" for i in range(n_samples)],
        "SampleType": ["Sample"] * n_bio + ["Buffer"] * n_buffers,
    })

    features = pd.DataFrame({
        "SeqId": [f"10000-{i}" for i in range(n_features)],
        "UniProt": [f"P{i}" for i in range(n_features)],
        "Target": [f"T{i}" for i in range(n_features)],
        "Dilution": ["20"] * n_features,
    })

    rng = np.random.default_rng(42)
    expr_data = np.empty((n_samples, n_features))
    # Biological: high RFU
    expr_data[:n_bio, :] = rng.normal(5000.0, 1000.0, size=(n_bio, n_features))
    # Buffer: low RFU (background signal)
    expr_data[n_bio:, :] = rng.normal(100.0, 20.0, size=(n_buffers, n_features))
    # Ensure all positive (RFU)
    expr_data = np.abs(expr_data)

    expression = pd.DataFrame(
        expr_data,
        columns=[f"SL{i}" for i in range(n_features)],
    )

    return AffinityDataset(
        platform=Platform.SOMASCAN,
        samples=samples,
        features=features,
        expression=expression,
        metadata={"AssayVersion": "v4.1", "AssayType": "SomaScan"},
    )


class TestComputeSomaElod:
    def test_computes_elod_from_buffers(self):
        ds = _make_somascan_dataset_with_buffers()
        lod = compute_soma_elod(ds)
        assert len(lod) == 3
        assert all(np.isfinite(lod))
        # eLOD should be higher than buffer median (added positive offset)
        buffer_medians = ds.expression.iloc[5:].median()
        for col in ds.expression.columns:
            assert lod[col] > buffer_medians[col]

    def test_elod_formula_matches_somadataio(self):
        """Verify: eLOD = median + 3.3 * 1.4826 * MAD."""
        ds = _make_somascan_dataset_with_buffers()
        lod = compute_soma_elod(ds)

        buffer_expr = ds.expression.iloc[5:].apply(pd.to_numeric, errors="coerce")
        for col in ds.expression.columns:
            vals = buffer_expr[col]
            med = vals.median()
            mad = (vals - med).abs().median()
            expected = med + _SOMA_ELOD_K * _MAD_TO_SD * mad
            assert abs(lod[col] - expected) < 1e-10

    def test_elod_with_constant_buffer(self):
        """When all buffer values are identical, MAD=0 so eLOD = median."""
        ds = _make_somascan_dataset_with_buffers()
        # Set all buffer RFU to constant
        ds.expression.iloc[5:] = 100.0
        lod = compute_soma_elod(ds)
        for col in ds.expression.columns:
            assert abs(lod[col] - 100.0) < 1e-10

    def test_raises_without_buffer_samples(self):
        ds = _make_somascan_dataset_with_buffers()
        ds.samples["SampleType"] = "Sample"
        with pytest.raises(ValueError, match="buffer"):
            compute_soma_elod(ds)

    def test_raises_without_sample_type(self):
        ds = _make_somascan_dataset_with_buffers()
        ds.samples = ds.samples.drop(columns=["SampleType"])
        with pytest.raises(ValueError, match="SampleType"):
            compute_soma_elod(ds)
