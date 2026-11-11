"""Olink wide NPX exports (NPX Manager / NPX Signature "NPX data" layout)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pyprideap.core import Platform
from pyprideap.io.readers.registry import detect_format, read
from pyprideap.processing.lod import get_proteins_above_lod

ASSAYS = [("IL8", "P10145", "OID00752"), ("TNFRSF9", "Q07011", "OID00753"), ("TIE2", "Q02763", "OID00754")]
PANEL = "Olink Target 96 Immuno-Oncology(v.3113)"


def _grid(sep: str = ",", decimal: str = ".") -> str:
    n = len(ASSAYS)
    pad = [""] * 3

    def num(x: float) -> str:
        return f"{x:.5f}".replace(".", decimal)

    rows = [
        ["Olink LC0175", "", "Olink NPX Signature 1.7.1.0", *[""] * (n - 1), *pad],
        ["NPX data", "", *[""] * n, *pad],
        ["Panel", "", *[PANEL] * n, *pad],
        ["Assay", "", *[a for a, _, _ in ASSAYS], "Plate ID", "QC Warning", "QC Deviation from median"],
        ["Uniprot ID", "", *[u for _, u, _ in ASSAYS], "", "", "Inc Ctrl 2"],
        ["OlinkID", "sample name", *[o for _, _, o in ASSAYS], "", "", ""],
        ["", "", *[""] * n, *pad],
        ["S1", "a", num(6.5), num(5.7), num(-1.0), "P1", "Pass", num(0.1)],
        ["S2", "b", num(7.0), num(5.1), num(-2.0), "P1", "Warning", num(0.3)],
        ["S3", "c", num(8.0), "", num(-1.5), "P1", "Pass", num(0.2)],
        ["", "", *[""] * n, *pad],
        ["LOD", "", num(0.4), num(0.0), num(1.0), *pad],
        ["Missing Data freq.", "", "0%", "33%", "100%", *pad],
        ["Normalization", "", *["IPC Normalized"] * n, *pad],
    ]
    return "\n".join(sep.join(r) for r in rows) + "\n"


@pytest.fixture
def wide_csv(tmp_path: Path) -> Path:
    path = tmp_path / "NSCLC_Olink_Target96.npx.csv"
    path.write_text(_grid())
    return path


def test_detected_and_read(wide_csv: Path):
    assert detect_format(wide_csv) == "olink_wide"
    ds = read(wide_csv)
    assert ds.platform == Platform.OLINK_TARGET
    assert ds.expression.shape == (3, 3)
    assert list(ds.samples["SampleID"]) == ["S1", "S2", "S3"]
    assert list(ds.samples["PlateID"]) == ["P1"] * 3
    assert list(ds.features["UniProt"]) == ["P10145", "Q07011", "Q02763"]
    assert list(ds.features["Panel"]) == [PANEL] * 3
    assert ds.expression.loc[0, "OID00752"] == pytest.approx(6.5)
    assert pd.isna(ds.expression.loc[2, "OID00753"])
    assert ds.features.set_index("OlinkID").loc["OID00753", "MissingFreq"] == pytest.approx(0.33)


def test_lod_and_qc_from_footer(wide_csv: Path):
    ds = read(wide_csv)
    lod = ds.metadata["lod_matrix"]
    assert lod.loc[0, "OID00754"] == pytest.approx(1.0)
    assert ds.metadata["sample_qc_matrix"].loc[1, "OID00752"] == "WARN"
    # IL8 and TNFRSF9 are above LOD in the samples; TIE2 never is
    assert sorted(get_proteins_above_lod(ds)) == ["P10145", "Q07011"]


def test_semicolon_decimal_comma(tmp_path: Path):
    path = tmp_path / "wide.csv"
    path.write_text(_grid(sep=";", decimal=","))
    ds = read(path)
    assert ds.expression.loc[1, "OID00752"] == pytest.approx(7.0)
    assert ds.metadata["lod_matrix"].loc[0, "OID00752"] == pytest.approx(0.4)


def test_workbook_all_wide_sheets(tmp_path: Path):
    rows = [r.split(",") for r in _grid().strip().split("\n")[2:]]  # no title rows
    width = max(len(r) for r in rows)
    grid = pd.DataFrame([r + [""] * (width - len(r)) for r in rows])
    path = tmp_path / "wide.xlsx"
    with pd.ExcelWriter(path) as xl:
        grid.to_excel(xl, sheet_name="Npx cellculture", header=False, index=False)
        grid.to_excel(xl, sheet_name="Npx Sera", header=False, index=False)
        pd.DataFrame({"note": ["not data"]}).to_excel(xl, sheet_name="Readme", index=False)
    assert detect_format(path) == "olink_wide"
    ds = read(path)
    assert ds.expression.shape == (6, 3)
    assert set(ds.samples["PlateID"]) == {"Npx cellculture: P1", "Npx Sera: P1"}


def test_long_csv_unchanged(tmp_path: Path):
    path = tmp_path / "long.npx.csv"
    pd.DataFrame(
        {
            "SampleID": ["S1", "S1"],
            "OlinkID": ["OID00752", "OID00753"],
            "UniProt": ["P10145", "Q07011"],
            "Assay": ["IL8", "TNFRSF9"],
            "NPX": [1.0, 2.0],
        }
    ).to_csv(path, index=False)
    assert detect_format(path) == "olink_csv"
    assert read(path).expression.shape == (1, 2)
