"""End-to-end tests using the public API."""

import pyprideap


def test_read_validate_stats_olink(olink_csv_path):
    dataset = pyprideap.read(olink_csv_path)
    results = pyprideap.validate(dataset)
    stats = pyprideap.compute_stats(dataset)
    assert stats.n_samples > 0
    assert stats.n_features > 0
    assert isinstance(results, list)
    errors = [r for r in results if r.level == pyprideap.Level.ERROR]
    assert len(errors) == 0, f"Unexpected validation errors: {[r.message for r in errors]}"


def test_read_validate_stats_somascan(somascan_adat_path):
    dataset = pyprideap.read(somascan_adat_path)
    results = pyprideap.validate(dataset)
    stats = pyprideap.compute_stats(dataset)
    assert stats.n_samples > 0
    assert stats.n_features > 0
    assert isinstance(results, list)
    errors = [r for r in results if r.level == pyprideap.Level.ERROR]
    assert len(errors) == 0, f"Unexpected validation errors: {[r.message for r in errors]}"


def test_read_validate_stats_parquet(olink_parquet_path):
    dataset = pyprideap.read(olink_parquet_path)
    results = pyprideap.validate(dataset)
    stats = pyprideap.compute_stats(dataset)
    assert stats.n_samples > 0
    assert stats.n_features > 0
    assert isinstance(results, list)
    errors = [r for r in results if r.level == pyprideap.Level.ERROR]
    assert len(errors) == 0, f"Unexpected validation errors: {[r.message for r in errors]}"


def test_read_validate_stats_somascan_csv(somascan_csv_path):
    dataset = pyprideap.read(somascan_csv_path)
    results = pyprideap.validate(dataset)
    stats = pyprideap.compute_stats(dataset)
    assert stats.n_samples > 0
    assert stats.n_features > 0
    assert isinstance(results, list)
