from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from pyprideap.core import AffinityDataset

logger = logging.getLogger(__name__)
from pyprideap.io.readers.olink_csv import read_olink_csv
from pyprideap.io.readers.olink_parquet import read_olink_parquet
from pyprideap.io.readers.olink_xlsx import read_olink_xlsx
from pyprideap.io.readers.somascan_adat import read_somascan_adat
from pyprideap.io.readers.somascan_csv import read_somascan_csv

_OLINK_MARKER_COLS = {"OlinkID", "NPX", "SampleID"}
_SOMASCAN_MARKER_COLS = {"SeqId", "SomaId"}


def detect_format(path: str | Path) -> str:
    path = Path(path)
    suffix = path.suffix.lower()
    name = path.name.lower()

    if suffix == ".adat":
        logger.debug("Format detected: somascan_adat (suffix=%s)", suffix)
        return "somascan_adat"

    if suffix == ".parquet":
        schema = pq.read_schema(path)
        cols = set(schema.names)
        if _OLINK_MARKER_COLS.issubset(cols):
            logger.debug("Format detected: olink_parquet (suffix=%s)", suffix)
            return "olink_parquet"
        raise ValueError(f"Cannot detect format: parquet file lacks Olink marker columns at {path}")

    if name.endswith(".npx.csv") or name.endswith(".ct.csv"):
        logger.debug("Format detected: olink_csv (name pattern=%s)", name)
        return "olink_csv"

    if suffix == ".csv":
        df_head = pd.read_csv(path, nrows=1)
        cols = set(df_head.columns)
        has_seqid_cols = any(c.startswith("SeqId.") for c in cols)
        if has_seqid_cols or _SOMASCAN_MARKER_COLS.issubset(cols):
            logger.debug("Format detected: somascan_csv (column inspection)")
            return "somascan_csv"
        if _OLINK_MARKER_COLS.issubset(cols):
            logger.debug("Format detected: olink_csv (column inspection)")
            return "olink_csv"

    if suffix == ".xlsx":
        df_head = pd.read_excel(path, nrows=1)
        cols = set(df_head.columns)
        if _OLINK_MARKER_COLS.issubset(cols):
            logger.debug("Format detected: olink_xlsx (column inspection)")
            return "olink_xlsx"

    raise ValueError(f"Cannot detect format for file: {path}")


def read(path: str | Path, *, platform: str | None = None) -> AffinityDataset:
    """Read an affinity proteomics data file.

    Parameters
    ----------
    path : str or Path
        Path to the data file.
    platform : str or None
        Force platform type: ``"olink"`` or ``"somascan"``.
        If *None* (default), the format is auto-detected from the file.
    """
    if platform is not None:
        platform = platform.lower()
        if platform not in ("olink", "somascan"):
            raise ValueError(f"platform must be 'olink' or 'somascan', got '{platform}'")

    if platform is not None:
        path = Path(path)
        suffix = path.suffix.lower()
        logger.debug("Platform forced: %s, suffix=%s", platform, suffix)
        if platform == "somascan":
            if suffix == ".adat":
                return read_somascan_adat(path)
            return read_somascan_csv(path)
        else:  # olink
            if suffix == ".parquet":
                return read_olink_parquet(path)
            if suffix == ".xlsx":
                return read_olink_xlsx(path)
            return read_olink_csv(path)

    fmt = detect_format(path)
    logger.debug("Auto-detected format: %s, dispatching reader", fmt)
    readers = {
        "somascan_adat": read_somascan_adat,
        "olink_parquet": read_olink_parquet,
        "olink_csv": read_olink_csv,
        "olink_xlsx": read_olink_xlsx,
        "somascan_csv": read_somascan_csv,
    }
    return readers[fmt](path)
