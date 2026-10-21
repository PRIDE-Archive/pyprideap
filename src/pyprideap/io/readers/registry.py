from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from pyprideap.core import AffinityDataset
from pyprideap.io.readers.nulisa import (
    NULISA_LONG_MARKER_COLS,
    is_nulisa_workbook,
    is_nulisa_xml,
    read_nulisa_csv,
    read_nulisa_xlsx,
)
from pyprideap.io.readers.olink_csv import _OLINK_COLUMN_ALIASES, _sniff_delimiter, read_olink_csv
from pyprideap.io.readers.olink_parquet import read_olink_parquet
from pyprideap.io.readers.olink_xlsx import read_olink_xlsx
from pyprideap.io.readers.somascan_adat import is_somascan_table_csv, read_somascan_adat
from pyprideap.io.readers.somascan_csv import read_somascan_csv

logger = logging.getLogger(__name__)

_OLINK_MARKER_COLS = {"OlinkID", "NPX", "SampleID"}
_SOMASCAN_MARKER_COLS = {"SeqId", "SomaId"}


def _nulisa_xml_message(path: Path) -> str:
    return (
        f"{path.name} is a NULISAseq instrument XML (raw read counts); QC uses the NPQ export of "
        "the same run (NPQ long format CSV or NPQ workbook). The XML is read alongside it for "
        "UniProt IDs and plate QC."
    )


def detect_format(path: str | Path) -> str:
    path = Path(path)
    suffix = path.suffix.lower()
    name = path.name.lower()

    if suffix == ".adat":
        logger.debug("Format detected: somascan_adat (suffix=%s)", suffix)
        return "somascan_adat"

    if suffix == ".xml" and is_nulisa_xml(path):
        raise ValueError(_nulisa_xml_message(path))

    if suffix == ".parquet":
        schema = pq.read_schema(path)
        cols = {_OLINK_COLUMN_ALIASES.get(c, c) for c in schema.names}
        if _OLINK_MARKER_COLS.issubset(cols):
            logger.debug("Format detected: olink_parquet (suffix=%s)", suffix)
            return "olink_parquet"
        raise ValueError(f"Cannot detect format: parquet file lacks Olink marker columns at {path}")

    if suffix == ".csv":
        with open(path, encoding="utf-8-sig", errors="replace") as f:
            head = [f.readline() for _ in range(2)]
        if "Olink NPX Signature" in head[0] and head[1].strip().lower().startswith("ct data"):
            raise ValueError(
                f"{path.name} contains Olink Ct values (raw qPCR cycle thresholds before NPX "
                "normalisation); QC needs the NPX export of the same run"
            )
        if is_somascan_table_csv(path):
            logger.debug("Format detected: somascan table CSV (ADAT table exported as CSV)")
            return "somascan_adat"
        try:
            header_cols = set(pd.read_csv(path, nrows=0, sep=_sniff_delimiter(path) or ",").columns)
        except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError):
            header_cols = set()
        if NULISA_LONG_MARKER_COLS <= header_cols:
            logger.debug("Format detected: nulisa_csv (NPQ long format)")
            return "nulisa_csv"

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

    if suffix == ".xlsx" and is_nulisa_workbook(path):
        logger.debug("Format detected: nulisa_xlsx (NPQ workbook)")
        return "nulisa_xlsx"

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
        Force platform type: ``"olink"``, ``"somascan"`` or ``"nulisa"``.
        If *None* (default), the format is auto-detected from the file.
    """
    if platform is not None:
        platform = platform.lower()
        if platform not in ("olink", "somascan", "nulisa"):
            raise ValueError(f"platform must be 'olink', 'somascan' or 'nulisa', got '{platform}'")

    if platform is not None:
        path = Path(path)
        suffix = path.suffix.lower()
        logger.debug("Platform forced: %s, suffix=%s", platform, suffix)
        if platform == "nulisa":
            if suffix == ".xml" and is_nulisa_xml(path):
                raise ValueError(_nulisa_xml_message(path))
            return read_nulisa_xlsx(path) if suffix == ".xlsx" else read_nulisa_csv(path)
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
        "nulisa_csv": read_nulisa_csv,
        "nulisa_xlsx": read_nulisa_xlsx,
    }
    return readers[fmt](path)
