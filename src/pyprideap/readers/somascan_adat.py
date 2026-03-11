from __future__ import annotations

from pathlib import Path

import pandas as pd

from pyprideap.core import AffinityDataset, Platform


def read_somascan_adat(path: str | Path) -> AffinityDataset:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    header, col_data, row_data = _parse_adat_sections(path)

    features = col_data.reset_index(drop=True)

    soma_ids = features["Name"].tolist()
    sample_cols = [c for c in row_data.columns if c not in soma_ids]

    samples = row_data[sample_cols].reset_index(drop=True)
    expression = row_data[soma_ids].astype(float).reset_index(drop=True)

    return AffinityDataset(
        platform=Platform.SOMASCAN,
        samples=samples,
        features=features,
        expression=expression,
        metadata=header,
    )


def _parse_adat_sections(path: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    header: dict[str, str] = {}
    col_lines: list[str] = []
    row_lines: list[str] = []
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

            if current_section == "HEADER":
                # Lines may start with \! or !
                stripped = line.lstrip("\\").lstrip("!")
                if line.startswith("\\!") or line.startswith("!"):
                    key, _, value = stripped.partition("\t")
                    header[key] = value
            elif current_section == "COL_DATA":
                col_lines.append(line)
            elif current_section == "ROW_DATA":
                row_lines.append(line)

    # Strip leading \! or ! from header lines
    def strip_meta(s: str) -> str:
        return s.lstrip("\\").lstrip("!")

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

    return header, col_data, row_data
