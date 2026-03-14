from __future__ import annotations

import requests  # type: ignore[import-untyped]

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
        resp = self._session.get(f"{self.base_url}/projects/{accession}", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()  # type: ignore[no-any-return]

    def list_files(self, accession: str) -> list[dict]:  # type: ignore[type-arg]
        resp = self._session.get(
            f"{self.base_url}/projects/{accession}/files",
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()  # type: ignore[no-any-return]

    def get_download_urls(self, accession: str) -> dict[str, str]:
        files = self.list_files(accession)
        urls = {}
        for f in files:
            name = f.get("fileName", "")
            locations = f.get("publicFileLocations", [])
            for loc in locations:
                if loc.get("name") == "FTP Protocol":
                    urls[name] = loc["value"]
                    break
        return urls
