from pathlib import Path

import pytest

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def olink_csv_path():
    return DATA_DIR / "olink_sample.npx.csv"


@pytest.fixture
def olink_parquet_path():
    return DATA_DIR / "olink_sample.parquet"


@pytest.fixture
def somascan_adat_path():
    return DATA_DIR / "somascan_sample.adat"


@pytest.fixture
def olink_xlsx_path():
    return DATA_DIR / "olink_sample.xlsx"


@pytest.fixture
def somascan_csv_path():
    return DATA_DIR / "somascan_sample.csv"
