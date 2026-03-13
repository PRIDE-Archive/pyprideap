"""Functional tests for PRIDE Archive API client."""

from unittest.mock import MagicMock, patch

import pytest

from pyprideap.api.pride import PrideClient


class TestPrideClient:
    def test_get_project(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "accession": "PAD000001",
            "title": "Test Project",
            "projectDescription": "A test",
        }
        with patch("pyprideap.api.pride.requests.Session.get", return_value=mock_response):
            client = PrideClient()
            project = client.get_project("PAD000001")
            assert project["accession"] == "PAD000001"
            assert project["title"] == "Test Project"

    def test_list_files(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"fileName": "olink_npx.csv", "fileSizeBytes": 4535478},
            {"fileName": "checksum.txt", "fileSizeBytes": 252},
        ]
        with patch("pyprideap.api.pride.requests.Session.get", return_value=mock_response):
            client = PrideClient()
            files = client.list_files("PAD000001")
            assert len(files) == 2
            assert files[0]["fileName"] == "olink_npx.csv"

    def test_get_project_not_found(self):
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = Exception("404 Not Found")
        with patch("pyprideap.api.pride.requests.Session.get", return_value=mock_response):
            client = PrideClient()
            with pytest.raises(Exception, match="404"):
                client.get_project("PAD999999")

    def test_get_download_urls(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {
                "fileName": "olink_npx.csv",
                "publicFileLocations": [
                    {
                        "name": "FTP Protocol",
                        "value": "ftp://ftp.pride.ebi.ac.uk/pride/data/archive/2025/03/PAD000001/olink_npx.csv",
                    },
                ],
            },
        ]
        with patch("pyprideap.api.pride.requests.Session.get", return_value=mock_response):
            client = PrideClient()
            urls = client.get_download_urls("PAD000001")
            assert "olink_npx.csv" in urls
            assert "ftp://" in urls["olink_npx.csv"]
