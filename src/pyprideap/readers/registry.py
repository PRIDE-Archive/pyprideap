from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from pyprideap.core import AffinityDataset
from pyprideap.readers.olink_csv import read_olink_csv
from pyprideap.readers.olink_parquet import read_olink_parquet
from pyprideap.readers.olink_xlsx import read_olink_xlsx
from pyprideap.readers.somascan_adat import read_somascan_adat
from pyprideap.readers.somascan_csv import read_somascan_csv

_OLINK_MARKER_COLS = {"OlinkID", "NPX", "SampleID"}
_SOMASCAN_MARKER_COLS = {"SeqId", "SomaId"}


def detect_format(path: str | Path) -> str:
    path = Path(path)
    suffix = path.suffix.lower()
    name = path.name.lower()

    if suffix == ".adat":
        return "somascan_adat"

    if suffix == ".parquet":
        schema = pq.read_schema(path)
        cols = set(schema.names)
        if _OLINK_MARKER_COLS.issubset(cols):
            return "olink_parquet"
        raise ValueError(f"Cannot detect format: parquet file lacks Olink marker columns at {path}")

    if name.endswith(".npx.csv") or name.endswith(".ct.csv"):
        return "olink_csv"

    if suffix == ".csv":
        df_head = pd.read_csv(path, nrows=1)
        cols = set(df_head.columns)
        has_seqid_cols = any(c.startswith("SeqId.") for c in cols)
        if has_seqid_cols or _SOMASCAN_MARKER_COLS.intersection(cols):
            return "somascan_csv"
        if _OLINK_MARKER_COLS.intersection(cols):
            return "olink_csv"

    if suffix == ".xlsx":
        df_head = pd.read_excel(path, nrows=1)
        cols = set(df_head.columns)
        if _OLINK_MARKER_COLS.intersection(cols):
            return "olink_xlsx"

    raise ValueError(f"Cannot detect format for file: {path}")


def read(path: str | Path) -> AffinityDataset:
    fmt = detect_format(path)
    readers = {
        "somascan_adat": read_somascan_adat,
        "olink_parquet": read_olink_parquet,
        "olink_csv": read_olink_csv,
        "olink_xlsx": read_olink_xlsx,
        "somascan_csv": read_somascan_csv,
    }
    return readers[fmt](path)
