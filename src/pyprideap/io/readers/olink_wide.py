"""Olink wide NPX exports (NPX Manager / NPX Signature "NPX data" layout).

One row per sample and one column per assay, with the assay annotation in header
rows above the samples and per-assay summaries in footer rows below::

    Olink LC0175,,Olink NPX Signature 1.7.1.0      <- optional title rows
    NPX data
    Panel,,Olink Target 96 Immuno-Oncology(v.3113),...
    Assay,,IL8,TNFRSF9,...,Plate ID,QC Warning
    Uniprot ID,,P10145,Q07011,...
    OlinkID,,OID00752,OID00753,...
    SC1,SC1,6.46497,5.67991,...,LC0175,Pass       <- samples
    LOD,,0.37940,-0.02032,...                      <- footers
    Missing Data freq.,,0%,0%,...
    Normalization,,IPC Normalized,...

The grid is turned into the long table (SampleID, OlinkID, UniProt, Assay, Panel,
NPX, LOD, MissingFreq, Normalization, PlateID, QC_Warning) that the long-format
reader already handles, so LOD, QC flags and platform detection are shared.
"""

from __future__ import annotations

import csv
import logging
import re
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_OID = re.compile(r"^OID\d{5}$")
_HEADER_SCAN_ROWS = 30
_MIN_ASSAYS = 3

# Header-row labels (first column, lower-case) -> long-table column
_HEADER_LABELS = {"panel": "Panel", "assay": "Assay", "uniprot id": "UniProt", "uniprot": "UniProt"}
# Footer-row labels -> long-table column
_FOOTER_LABELS = {
    "lod": "LOD",
    "missing data freq.": "MissingFreq",
    "missing data freq": "MissingFreq",
    "normalization": "Normalization",
}
# Trailing per-sample columns, named in the Assay row -> long-table column
_SAMPLE_COLUMNS = {"plate id": "PlateID", "qc warning": "QC_Warning"}


def _cell(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip()


def _header_row(grid: pd.DataFrame) -> int | None:
    """Index of the row that holds the OlinkIDs, or None when the grid is not wide."""
    for i in range(min(len(grid), _HEADER_SCAN_ROWS)):
        if sum(bool(_OID.match(_cell(v))) for v in grid.iloc[i]) >= _MIN_ASSAYS:
            return i
    return None


def is_olink_wide_grid(grid: pd.DataFrame) -> bool:
    return _header_row(grid) is not None


def _read_csv_grid(path: Path, nrows: int | None = None) -> pd.DataFrame:
    with open(path, encoding="utf-8-sig", errors="replace", newline="") as fh:
        sample = "".join(fh.readline() for _ in range(_HEADER_SCAN_ROWS))
    # The OlinkID line has the most fields; pick the delimiter that splits it most
    sep = max((",", ";", "\t"), key=lambda d: max((line.count(d) for line in sample.splitlines()), default=0))
    with open(path, encoding="utf-8-sig", errors="replace", newline="") as fh:
        rows = []
        for i, row in enumerate(csv.reader(fh, delimiter=sep)):
            if nrows is not None and i >= nrows:
                break
            rows.append(row)
    width = max((len(r) for r in rows), default=0)
    grid = pd.DataFrame([r + [""] * (width - len(r)) for r in rows], dtype=object)
    grid.attrs["decimal_comma"] = sep == ";"
    return cast(pd.DataFrame, grid)


def is_olink_wide_csv(path: str | Path) -> bool:
    try:
        return is_olink_wide_grid(_read_csv_grid(Path(path), nrows=_HEADER_SCAN_ROWS))
    except (OSError, UnicodeDecodeError, csv.Error):
        return False


def olink_wide_sheets(path: str | Path) -> list[str]:
    """Sheets of an Excel workbook that hold a wide NPX grid."""
    try:
        book = pd.ExcelFile(path)
    except Exception:  # not a readable workbook
        return []
    sheets = []
    for name in book.sheet_names:
        head = pd.read_excel(book, sheet_name=name, header=None, dtype=object, nrows=_HEADER_SCAN_ROWS)
        if is_olink_wide_grid(head):
            sheets.append(str(name))
    return sheets


def _numbers(values: pd.Series, decimal_comma: bool) -> pd.Series:
    text = values.map(_cell)
    if decimal_comma:
        text = text.str.replace(",", ".", regex=False)
    return pd.to_numeric(text.str.rstrip("%"), errors="coerce")


def wide_grid_to_long(grid: pd.DataFrame, *, source: str = "", sheet: str | None = None) -> pd.DataFrame:
    """Convert one wide NPX grid to the long Olink table."""
    header = _header_row(grid)
    if header is None:
        raise ValueError(f"{source}: no OlinkID header row found")
    decimal_comma = bool(grid.attrs.get("decimal_comma", False))

    oid_row = grid.iloc[header].map(_cell)
    assay_cols = [j for j, v in enumerate(oid_row) if _OID.match(v)]

    # Annotation rows above the OlinkID row, by their first-column label
    annotations: dict[str, pd.Series] = {}
    names_row: pd.Series | None = None
    for i in range(header):
        label = _cell(grid.iat[i, 0]).lower()
        if label in _HEADER_LABELS:
            annotations[_HEADER_LABELS[label]] = grid.iloc[i].map(_cell)
            if label == "assay":
                names_row = annotations["Assay"]

    # Trailing per-sample columns (Plate ID, QC Warning) are named in the Assay row
    extra_cols: dict[str, int] = {}
    if names_row is not None:
        for j, name in enumerate(names_row):
            key = name.lower()
            if j not in assay_cols and key in _SAMPLE_COLUMNS and _SAMPLE_COLUMNS[key] not in extra_cols:
                extra_cols[_SAMPLE_COLUMNS[key]] = j

    # Sample rows follow the header until the footer; footers are labelled rows
    sample_rows: list[int] = []
    footers: dict[str, pd.Series] = {}
    for i in range(header + 1, len(grid)):
        label = _cell(grid.iat[i, 0])
        if not label:
            continue
        if label.lower() in _FOOTER_LABELS:
            footers.setdefault(_FOOTER_LABELS[label.lower()], grid.iloc[i])
            continue
        values = _numbers(grid.iloc[i, assay_cols], decimal_comma)
        if values.notna().any():
            sample_rows.append(i)

    if not sample_rows:
        raise ValueError(f"{source}: wide NPX grid has no sample rows")

    n_assays = len(assay_cols)
    n_samples = len(sample_rows)
    block = grid.iloc[sample_rows, assay_cols]
    npx = np.column_stack([_numbers(block.iloc[:, k], decimal_comma).to_numpy() for k in range(n_assays)])

    sample_ids = [_cell(grid.iat[i, 0]) for i in sample_rows]
    long = pd.DataFrame(
        {
            "SampleID": np.repeat(sample_ids, n_assays),
            "OlinkID": np.tile([oid_row.iat[j] for j in assay_cols], n_samples),
            "NPX": npx.reshape(-1),
        }
    )
    for col, row in annotations.items():
        long[col] = np.tile([row.iat[j] or None for j in assay_cols], n_samples)
    for col, row in footers.items():
        cells = [row.iat[j] for j in assay_cols]
        if col in ("LOD", "MissingFreq"):
            numeric = _numbers(pd.Series(cells, dtype=object), decimal_comma)
            if col == "MissingFreq" and any("%" in _cell(v) for v in cells):
                numeric = numeric / 100.0
            long[col] = np.tile(numeric.to_numpy(), n_samples)
        else:
            long[col] = np.tile([_cell(v) for v in cells], n_samples)
    for col, j in extra_cols.items():
        long[col] = np.repeat([_cell(grid.iat[i, j]) or None for i in sample_rows], n_assays)
    if sheet is not None:
        # Several sheets in one workbook: keep their samples apart even when IDs repeat
        if "PlateID" in long.columns:
            long["PlateID"] = [sheet if p is None else f"{sheet}: {p}" for p in long["PlateID"]]
        else:
            long["PlateID"] = sheet
    logger.debug("%s: wide NPX grid, %d samples x %d assays", source, n_samples, n_assays)
    return cast(pd.DataFrame, long)


def read_olink_wide(path: str | Path) -> pd.DataFrame:
    """Long Olink table from a wide NPX CSV, or from every wide sheet of a workbook."""
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        sheets = olink_wide_sheets(path)
        if not sheets:
            raise ValueError(f"{path.name}: no wide Olink NPX sheet found")
        frames = []
        for name in sheets:
            grid = pd.read_excel(path, sheet_name=name, header=None, dtype=object)
            frames.append(
                wide_grid_to_long(grid, source=f"{path.name}[{name}]", sheet=name if len(sheets) > 1 else None)
            )
        return pd.concat(frames, ignore_index=True)
    return wide_grid_to_long(_read_csv_grid(path), source=path.name)
