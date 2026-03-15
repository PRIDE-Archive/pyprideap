from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from pyprideap.core import AffinityDataset, Platform
from pyprideap.io.readers.olink_csv import _warn_data_quality

logger = logging.getLogger(__name__)


def read_somascan_adat(path: str | Path) -> AffinityDataset:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    header, col_data, row_data = _parse_adat_sections(path)
    logger.debug("ADAT parsed: %d header keys", len(header))

    features = col_data.reset_index(drop=True)

    soma_ids = features["Name"].tolist()
    sample_cols = [c for c in row_data.columns if c not in soma_ids]

    samples = row_data[sample_cols].reset_index(drop=True)
    expression = row_data[soma_ids].astype(float).reset_index(drop=True)
    logger.debug("ADAT shape: %d samples x %d features", len(samples), len(features))

    dataset = AffinityDataset(
        platform=Platform.SOMASCAN,
        samples=samples,
        features=features,
        expression=expression,
        metadata=header,
    )
    _warn_data_quality(dataset, source=path.name)
    return dataset


def _parse_adat_sections(path: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Parse an ADAT file into header, column metadata, and row data.

    Supports two ADAT layouts:
    - **Legacy**: ^HEADER, ^COL_DATA (feature meta rows), ^ROW_DATA (header + data rows)
    - **TABLE_BEGIN**: ^HEADER, ^COL_DATA, ^ROW_DATA (row meta definitions only),
      ^TABLE_BEGIN (combined feature meta + data in a single block)
    """
    header: dict[str, str] = {}
    col_lines: list[str] = []
    row_lines: list[str] = []
    table_lines: list[str] = []
    current_section = None

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("^HEADER"):
                current_section = "HEADER"
                continue
            elif line.startswith("^COL_DATA"):
                current_section = "COL_DATA"
                continue
            elif line.startswith("^ROW_DATA"):
                current_section = "ROW_DATA"
                continue
            elif line.startswith("^TABLE_BEGIN"):
                current_section = "TABLE_BEGIN"
                continue

            if current_section == "HEADER":
                stripped = line.lstrip("\\").lstrip("!")
                if line.startswith("\\!") or line.startswith("!"):
                    key, _, value = stripped.partition("\t")
                    header[key] = value
                elif "\t" in line:
                    key, _, value = line.partition("\t")
                    header[key] = value
            elif current_section == "COL_DATA":
                col_lines.append(line)
            elif current_section == "ROW_DATA":
                row_lines.append(line)
            elif current_section == "TABLE_BEGIN":
                table_lines.append(line)

    def strip_meta(s: str) -> str:
        return s.lstrip("\\").lstrip("!")

    # TABLE_BEGIN format: combined col metadata + data in one block
    if table_lines:
        logger.debug("ADAT layout: TABLE_BEGIN format (%d table lines)", len(table_lines))
        return _parse_table_begin(header, col_lines, row_lines, table_lines)

    # Legacy format: COL_DATA has feature metadata, ROW_DATA has all data
    logger.debug("ADAT layout: legacy format (col_lines=%d, row_lines=%d)", len(col_lines), len(row_lines))
    if not col_lines:
        raise ValueError(f"ADAT file has no COL_DATA section content: {path}")
    if not row_lines:
        raise ValueError(f"ADAT file has no ROW_DATA section content: {path}")

    col_header = strip_meta(col_lines[0]).split("\t")
    col_rows = [line.split("\t") for line in col_lines[1:]]
    col_data = pd.DataFrame(col_rows, columns=col_header)

    row_header = strip_meta(row_lines[0]).split("\t")
    row_rows = [line.split("\t") for line in row_lines[1:]]
    row_data = pd.DataFrame(row_rows, columns=row_header)

    logger.debug("Legacy sections: %d features, %d data rows", len(col_data), len(row_data))
    return header, col_data, row_data


def _parse_table_begin(
    header: dict,
    col_lines: list[str],
    row_lines: list[str],
    table_lines: list[str],
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Parse the ^TABLE_BEGIN combined format.

    In this format:
    - ^COL_DATA has column metadata field definitions (Name, Type rows)
    - ^ROW_DATA has row metadata field definitions (Name, Type rows)
    - ^TABLE_BEGIN contains:
      1. Feature metadata rows (prefixed with leading tabs, one row per
         COL_DATA field like SeqId, Target, UniProt, etc.)
      2. A header row (sample metadata column names + analyte IDs)
      3. Data rows (sample metadata values + RFU values)

    The number of leading tabs in the feature metadata rows equals the
    number of sample metadata columns.
    """

    def strip_meta(s: str) -> str:
        return s.lstrip("\\").lstrip("!")

    # Separate feature metadata rows (start with tabs) from data rows
    feature_meta_rows: list[tuple[str, list[str]]] = []
    data_header_line: str | None = None
    data_lines: list[str] = []

    for line in table_lines:
        if line.startswith("\t"):
            # Feature metadata row: leading tabs (one per sample col) + field name + values
            parts = line.split("\t")
            non_empty_start = 0
            for i, p in enumerate(parts):
                if p.strip():
                    non_empty_start = i
                    break
            field_name = parts[non_empty_start].strip()
            values = parts[non_empty_start + 1 :]
            feature_meta_rows.append((field_name, values))
        elif data_header_line is None:
            # First non-tab line is the sample column header
            data_header_line = line
        else:
            data_lines.append(line)

    if not feature_meta_rows:
        raise ValueError("TABLE_BEGIN section has no feature metadata rows")
    if not data_lines:
        raise ValueError("TABLE_BEGIN section has no data rows")

    # Build feature metadata (col_data)
    first_field, first_values = feature_meta_rows[0]
    n_analytes = len(first_values)

    col_dict: dict[str, list[str]] = {}
    for field_name, values in feature_meta_rows:
        col_dict[field_name] = values[:n_analytes]
    col_data = pd.DataFrame(col_dict)

    # Ensure Name column exists (needed by read_somascan_adat)
    if "Name" not in col_data.columns:
        if "SeqId" in col_data.columns:
            col_data.insert(
                0, "Name", col_data["SeqId"].apply(lambda s: f"seq.{s.replace('-', '.')}" if isinstance(s, str) else s)
            )
        else:
            col_data.insert(0, "Name", [f"Analyte_{i}" for i in range(n_analytes)])

    # The TABLE_BEGIN header line contains all columns: sample meta + analyte IDs.
    # We replace the analyte portion with our generated Name column for consistency.
    analyte_names = col_data["Name"].tolist()

    if data_header_line:
        header_parts = data_header_line.split("\t")
        # Sample columns are the first (total - n_analytes) columns
        n_sample_cols = len(header_parts) - n_analytes
        if n_sample_cols < 0:
            n_sample_cols = 0
        sample_col_names = header_parts[:n_sample_cols]
    else:
        sample_col_names = []
        n_sample_cols = 0

    full_header = sample_col_names + analyte_names
    expected_len = len(full_header)

    parsed_rows = []
    for line in data_lines:
        parts = line.split("\t")
        if len(parts) >= expected_len:
            parsed_rows.append(parts[:expected_len])

    row_data = pd.DataFrame(parsed_rows, columns=full_header)

    return header, col_data, row_data
