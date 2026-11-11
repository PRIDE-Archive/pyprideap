from __future__ import annotations

import logging

import requests  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

_BASE_URL = "https://www.ebi.ac.uk/pride/ws/archive/v3"


_DEFAULT_TIMEOUT = 30


class PrideClient:
    def __init__(self, base_url: str = _BASE_URL, timeout: int = _DEFAULT_TIMEOUT):
        self.base_url = base_url
        self.timeout = timeout
        self._session = requests.Session()

    def __enter__(self) -> PrideClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        self._session.close()

    def get_project(self, accession: str) -> dict:  # type: ignore[type-arg]
        url = f"{self.base_url}/projects/{accession}"
        logger.debug("GET %s", url)
        resp = self._session.get(url, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()  # type: ignore[no-any-return]

    def list_files(self, accession: str) -> list[dict]:  # type: ignore[type-arg]
        url = f"{self.base_url}/projects/{accession}/files"
        logger.debug("GET %s", url)
        resp = self._session.get(url, timeout=self.timeout)
        resp.raise_for_status()
        files = resp.json()
        logger.debug("Files found: %d for %s", len(files), accession)
        return files  # type: ignore[no-any-return]

    def get_download_files(self, accession: str) -> dict[str, tuple[str, str | None]]:
        """File name -> (FTP URL, file category such as RESULT, RAW or OTHER)."""
        files = self.list_files(accession)
        out: dict[str, tuple[str, str | None]] = {}
        for f in files:
            name = f.get("fileName", "")
            category = (f.get("fileCategory") or {}).get("value")
            for loc in f.get("publicFileLocations", []):
                if loc.get("name") == "FTP Protocol":
                    out[name] = (loc["value"], category.upper() if isinstance(category, str) else None)
                    break
        logger.debug("Download URLs extracted: %d FTP links", len(out))
        return out

    def get_download_urls(self, accession: str) -> dict[str, str]:
        return {name: url for name, (url, _category) in self.get_download_files(accession).items()}
