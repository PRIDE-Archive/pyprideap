"""PRIDE file download: mirror selection, fallback and parallel fetch."""

from __future__ import annotations

from pathlib import Path

import pytest

from pyprideap import cli

FTP = "ftp://ftp.pride.ebi.ac.uk/pride/data/archive/2026/10/PAD000068/NPQ_Long_Format.csv"
HTTPS = "https://ftp.pride.ebi.ac.uk/pride/data/archive/2026/10/PAD000068/NPQ_Long_Format.csv"
FIRE = "https://hh.fire.sdo.ebi.ac.uk/pride-public/2026/10/PAD000068/NPQ_Long_Format.csv"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in ("PYPRIDEAP_DOWNLOAD_PROTOCOL", "PYPRIDEAP_FIRE_URL", "PYPRIDEAP_DOWNLOAD_WORKERS"):
        monkeypatch.delenv(var, raising=False)


def test_fire_first_then_https_by_default():
    assert cli._candidate_urls(FTP) == [FIRE, HTTPS]
    assert cli._candidate_urls(HTTPS) == [FIRE, HTTPS]


def test_fire_url_override(monkeypatch):
    monkeypatch.setenv("PYPRIDEAP_FIRE_URL", "http://fire.internal:8080/")
    assert cli._candidate_urls(FTP)[0] == "http://fire.internal:8080/pride-public/2026/10/PAD000068/NPQ_Long_Format.csv"


def test_protocol_choices(monkeypatch):
    monkeypatch.setenv("PYPRIDEAP_DOWNLOAD_PROTOCOL", "https")
    assert cli._candidate_urls(FTP) == [HTTPS]
    monkeypatch.setenv("PYPRIDEAP_DOWNLOAD_PROTOCOL", "ftp")
    assert cli._candidate_urls(HTTPS) == [FTP, HTTPS]


def test_non_pride_url_untouched():
    assert cli._candidate_urls("https://example.org/x.csv") == ["https://example.org/x.csv"]


class _Resp:
    def __init__(self, data: bytes):
        self._data = data

    def read(self, n: int = -1) -> bytes:
        out, self._data = self._data, b""
        return out

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_fetch_falls_back_to_https(monkeypatch, tmp_path: Path):
    tried: list[str] = []

    def fake_urlopen(url, timeout=None):
        tried.append(url)
        if "fire" in url:
            raise OSError("404")
        return _Resp(b"data")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    dest = tmp_path / "f.csv"
    cli._fetch(FTP, dest)
    assert tried == [FIRE, HTTPS]
    assert dest.read_bytes() == b"data"


def test_fetch_raises_when_all_mirrors_fail(monkeypatch, tmp_path: Path):
    def fake_urlopen(url, timeout=None):
        raise OSError("down")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with pytest.raises(OSError):
        cli._fetch(FTP, tmp_path / "f.csv")


def test_download_pad_files_parallel(monkeypatch, tmp_path: Path):
    base = "ftp://ftp.pride.ebi.ac.uk/pride/data/archive/2026/10/PAD000068/"
    urls = {n: base + n for n in ("NPQ_Long_Format.csv", "run.xml", "study.sdrf.tsv", "notes.pdf")}

    class FakeClient:
        def get_download_files(self, accession):
            return {n: (u, None) for n, u in urls.items()}

    monkeypatch.setattr("pyprideap.api.pride.PrideClient", FakeClient)

    def fake_fetch(url, dest):
        if dest.name == "run.xml":
            raise OSError("aux missing")
        dest.write_text("x")
        return dest

    monkeypatch.setattr(cli, "_fetch", fake_fetch)
    files, sdrf = cli._download_pad_files("PAD000068", tmp_path)
    assert [f.name for f in files] == ["NPQ_Long_Format.csv"]
    assert sdrf is not None and sdrf.name == "study.sdrf.tsv"
    assert not (tmp_path / "notes.pdf").exists()


def test_download_pad_files_data_failure_raises(monkeypatch, tmp_path: Path):
    class FakeClient:
        def get_download_files(self, accession):
            return {"a.csv": ("ftp://ftp.pride.ebi.ac.uk/pride/data/archive/a.csv", "RESULT")}

    monkeypatch.setattr("pyprideap.api.pride.PrideClient", FakeClient)
    monkeypatch.setattr(cli, "_fetch", lambda url, dest: (_ for _ in ()).throw(OSError("boom")))
    with pytest.raises(OSError):
        cli._download_pad_files("PAD000068", tmp_path)


def test_select_data_files_keeps_result_and_raw():
    files = {
        "NPQ_Long_Format.csv": ("u1", "RESULT"),
        "NPQ_Values.xlsx": ("u2", "OTHER"),
        "run.adat": ("u3", "RAW"),
        "dep_analysis.csv": ("u4", "SEARCH"),
    }
    assert cli._select_data_files(files) == {"NPQ_Long_Format.csv": "u1", "run.adat": "u3"}


def test_select_data_files_untyped_keeps_all():
    files = {"a.npx.csv": ("u1", "OTHER"), "b.xlsx": ("u2", None)}
    assert cli._select_data_files(files) == {"a.npx.csv": "u1", "b.xlsx": "u2"}


def test_download_pad_files_skips_other_copies(monkeypatch, tmp_path: Path):
    base = "ftp://ftp.pride.ebi.ac.uk/pride/data/archive/2026/10/PAD000068/"

    class FakeClient:
        def get_download_files(self, accession):
            return {
                "NPQ_Long_Format.csv": (base + "NPQ_Long_Format.csv", "RESULT"),
                "NPQ_Values.xlsx": (base + "NPQ_Values.xlsx", "OTHER"),
                "study.sdrf.tsv": (base + "study.sdrf.tsv", "EXPERIMENTAL DESIGN"),
            }

    monkeypatch.setattr("pyprideap.api.pride.PrideClient", FakeClient)

    def fake_fetch(url, dest):
        dest.write_text("x")
        return dest

    monkeypatch.setattr(cli, "_fetch", fake_fetch)
    files, sdrf = cli._download_pad_files("PAD000068", tmp_path)
    assert [f.name for f in files] == ["NPQ_Long_Format.csv"]
    assert sdrf is not None and sdrf.name == "study.sdrf.tsv"
