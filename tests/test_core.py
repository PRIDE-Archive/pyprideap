import pandas as pd

from pyprideap.core import AffinityDataset, Level, Platform, ValidationResult


def test_platform_enum_values():
    assert Platform.OLINK_EXPLORE.value == "olink_explore"
    assert Platform.OLINK_EXPLORE_HT.value == "olink_explore_ht"
    assert Platform.OLINK_TARGET.value == "olink_target"
    assert Platform.SOMASCAN.value == "somascan"


def test_affinity_dataset_creation():
    ds = AffinityDataset(
        platform=Platform.OLINK_EXPLORE,
        samples=pd.DataFrame({"SampleID": ["S1", "S2"]}),
        features=pd.DataFrame({"OlinkID": ["OID1"], "UniProt": ["P12345"]}),
        expression=pd.DataFrame({"OID1": [1.5, 2.3]}),
        metadata={"version": "1.0"},
    )
    assert ds.platform == Platform.OLINK_EXPLORE
    assert len(ds.samples) == 2
    assert len(ds.features) == 1
    assert ds.expression.shape == (2, 1)
    assert ds.metadata["version"] == "1.0"


def test_validation_result_creation():
    vr = ValidationResult(
        level=Level.ERROR,
        rule="olink.schema.missing_column",
        message="Missing required column: SampleID",
        details={"column": "SampleID"},
    )
    assert vr.level == Level.ERROR
    assert "SampleID" in vr.message


def test_level_enum_ordering():
    assert Level.ERROR.value == "error"
    assert Level.WARNING.value == "warning"
    assert Level.INFO.value == "info"
