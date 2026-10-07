"""CLI tests for commands that download PAD data (download is mocked)."""

from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner

from pyprideap import cli


def _fake_download(accession: str, dest_dir: Path) -> tuple[list[Path], Path | None]:
    rows = [
        {"SampleID": sid, "SampleType": stype, "OlinkID": oid, "UniProt": "P1", "Assay": "A1", "Panel": "X", "NPX": 1.0}
        for sid, stype in (("S1", "SAMPLE"), ("S2", "SAMPLE"), ("NC1", "NEGATIVE_CONTROL"))
        for oid in ("OID1", "OID2")
    ]
    path = dest_dir / f"{accession}_npx.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return [path], None


@pytest.mark.parametrize("exclude_controls, expected", [(False, ["NC1", "S1", "S2"]), (True, ["S1", "S2"])])
def test_unique_samples_from_accession(monkeypatch, exclude_controls, expected):
    monkeypatch.setattr(cli, "_download_pad_files", _fake_download)
    args = ["unique-samples", "-a", "PAD000001"] + (["--exclude-controls"] if exclude_controls else [])
    result = CliRunner().invoke(cli.main, args)
    assert result.exit_code == 0, result.output
    assert result.stdout.split() == expected
