from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from open_licenseplate.app import create_app
from open_licenseplate.config import load_settings
from open_licenseplate.database import Database, upgrade_database
from open_licenseplate.models import catalog as catalog_module
from open_licenseplate.models.catalog import (
    CATALOG_ID,
    CATALOG_RELEASE_TAG,
    CATALOG_REPOSITORY,
    CATALOG_ROOT,
    CatalogEntry,
    CatalogError,
    FixedCatalogDownloader,
    ModelCatalog,
    load_model_catalog,
)
from open_licenseplate.models.manifest import parse_manifest
from open_licenseplate.models.repository import ModelRepository
from open_licenseplate.models.service import ModelImportError


@dataclass
class FakeDownloader:
    """Network-free downloader that writes a selected test archive."""

    content: bytes = b""
    error: BaseException | None = None
    calls: list[dict[str, object]] | None = None

    def __post_init__(self) -> None:
        if self.calls is None:
            self.calls = []

    def download(
        self,
        *,
        url: str,
        archive_asset: str,
        expected_size: int,
        destination: Path,
    ) -> None:
        assert self.calls is not None
        self.calls.append(
            {
                "url": url,
                "archive_asset": archive_asset,
                "expected_size": expected_size,
            }
        )
        if self.error is not None:
            raise self.error
        destination.write_bytes(self.content)


class FakeResponse:
    def __init__(
        self,
        *,
        status: int = 200,
        headers: dict[str, str] | None = None,
        chunks: list[bytes] | None = None,
        read_error: BaseException | None = None,
        location: str | None = None,
    ) -> None:
        self.status = status
        self._headers = headers or {}
        self._chunks = list(chunks or [])
        self._read_error = read_error
        self._location = location

    def getheader(self, name: str) -> str | None:
        if name.casefold() == "location" and self._location is not None:
            return self._location
        return self._headers.get(name)

    def read(self, _size: int) -> bytes:
        if self._read_error is not None:
            raise self._read_error
        if not self._chunks:
            return b""
        return self._chunks.pop(0)

    def close(self) -> None:
        return None


class FakeConnection:
    def __init__(self, response: FakeResponse) -> None:
        self.response = response

    def request(self, _method: str, _target: str, *, headers: dict[str, str]) -> None:
        assert headers["Accept"] == "application/octet-stream"

    def getresponse(self) -> FakeResponse:
        return self.response

    def close(self) -> None:
        return None


def _settings(tmp_path: Path) -> Any:
    return load_settings(
        cli_overrides={
            "storage.data_dir": tmp_path / "data",
            "storage.log_dir": tmp_path / "logs",
        }
    )


def _prepare_database(tmp_path: Path) -> Any:
    settings = _settings(tmp_path)
    upgrade_database(settings.storage.data_dir / "open-licenseplate.sqlite3")
    return settings


def _test_catalog(tmp_path: Path) -> tuple[ModelCatalog, bytes]:
    package = tmp_path / "asset" / "license-plate-yolov11n.mlpackage"
    (package / "Data").mkdir(parents=True)
    (package / "Manifest.json").write_text("{}", encoding="utf-8")
    (package / "Data" / "weights.bin").write_bytes(b"catalog test bytes")

    from open_licenseplate.models.archive import compute_artifact_sha256

    package_sha256 = compute_artifact_sha256(package)
    archive_path = tmp_path / "asset.zip"
    with zipfile.ZipFile(archive_path, "w") as output:
        for path in package.rglob("*"):
            if path.is_file():
                output.write(path, path.relative_to(package.parent).as_posix())
    archive_bytes = archive_path.read_bytes()
    archive_sha256 = hashlib.sha256(archive_bytes).hexdigest()
    source_repository = "https://huggingface.co/morsetechlab/yolov11-license-plate-detection"
    source_revision = "251a30d7daedca065f56e04b0af04052c907c68f"
    manifest_value: dict[str, Any] = {
        "schema_version": 1,
        "id": "license-plate-yolov11n",
        "display_name": "Catalog test model",
        "task": "object_detection",
        "backend": "coreml",
        "adapter": "ultralytics_yolo_nms",
        "artifact": "license-plate-yolov11n.mlpackage",
        "artifact_sha256": package_sha256,
        "input": {
            "name": "image",
            "kind": "image",
            "width": 640,
            "height": 640,
            "color_space": "rgb",
        },
        "preprocessing": {"resize": "letterbox"},
        "outputs": {
            "boxes": "coordinates",
            "scores": "confidence",
            "box_format": "xyxy",
            "coordinate_space": "model_pixels",
        },
        "labels": ["license_plate"],
        "defaults": {"confidence_threshold": 0.35, "iou_threshold": 0.45},
        "source": {
            "url": f"{source_repository}/resolve/{source_revision}/weights.pt",
            "repository": source_repository,
            "revision": source_revision,
            "license": "AGPL-3.0",
        },
        "conversion": {
            "source_weight": "weights.pt",
            "tool_versions": {},
            "arguments": {},
        },
        "distribution": {
            "archive": "open-licenseplate-model-catalog-license-plate-yolov11n.zip",
            "archive_sha256": archive_sha256,
            "archive_size": len(archive_bytes),
            "recommendation": "fast_default",
            "release_tag": CATALOG_RELEASE_TAG,
        },
    }
    manifest_bytes = json.dumps(manifest_value, separators=(",", ":")).encode("utf-8")
    manifest = parse_manifest(manifest_bytes)
    entry = CatalogEntry(
        catalog_id=manifest.model_id,
        display_name=manifest.display_name,
        recommendation="fast_default",
        archive_asset="open-licenseplate-model-catalog-license-plate-yolov11n.zip",
        archive_url=(
            "https://github.com/linuxlewis/open-licenseplate/releases/download/"
            "model-catalog-v1/open-licenseplate-model-catalog-license-plate-yolov11n.zip"
        ),
        archive_size=len(archive_bytes),
        archive_sha256=archive_sha256,
        package_sha256=package_sha256,
        license="AGPL-3.0",
        source_repository=source_repository,
        source_revision=source_revision,
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        manifest_bytes=manifest_bytes,
        manifest=manifest,
    )
    return (
        ModelCatalog(
            catalog_id=CATALOG_ID,
            repository=CATALOG_REPOSITORY,
            release_tag=CATALOG_RELEASE_TAG,
            entries=(entry,),
        ),
        archive_bytes,
    )


def _assert_staging_empty(settings: Any) -> None:
    staging = settings.storage.data_dir / "staging"
    assert staging.is_dir()
    assert list(staging.iterdir()) == []


def test_catalog_api_lists_only_fixed_entries_without_archive_urls(tmp_path: Path) -> None:
    settings = _prepare_database(tmp_path)

    with TestClient(create_app(settings)) as client:
        response = client.get("/api/v1/models/catalog")

    assert response.status_code == 200
    payload = response.json()
    assert payload["catalog_id"] == CATALOG_ID
    assert [entry["catalog_id"] for entry in payload["models"]] == [
        "license-plate-yolov11n",
        "license-plate-yolov11s",
        "license-plate-yolov11m",
    ]
    for entry in payload["models"]:
        assert {
            "catalog_id",
            "display_name",
            "recommendation",
            "archive_size",
            "license",
            "source",
            "installed",
            "install_available",
            "catalog",
        } == set(entry)
        assert "archive_url" not in json.dumps(entry)
        assert entry["installed"] is False
        assert entry["install_available"] is True
        assert entry["source"]["revision"]


def test_catalog_loader_requires_the_committed_manifest_checksum(tmp_path: Path) -> None:
    root = tmp_path / "committed-catalog"
    manifest_directory = root / "manifests"
    manifest_directory.mkdir(parents=True)
    lock = json.loads((CATALOG_ROOT / "model-catalog-lock.json").read_text(encoding="utf-8"))
    assert isinstance(lock, dict)
    raw_models = lock["models"]
    assert isinstance(raw_models, list)
    for raw_model in raw_models:
        assert isinstance(raw_model, dict)
        manifest_name = str(raw_model["manifest_asset"])
        manifest_bytes = (CATALOG_ROOT / "manifests" / manifest_name).read_bytes()
        (manifest_directory / manifest_name).write_bytes(manifest_bytes)
        raw_model["manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
    (root / "model-catalog-lock.json").write_text(
        json.dumps(lock),
        encoding="utf-8",
    )

    loaded = load_model_catalog(root)
    assert len(loaded.entries) == 3

    manifest_path = manifest_directory / "license-plate-yolov11n.json"
    original_manifest = manifest_path.read_bytes()
    manifest_path.write_bytes(original_manifest + b"\n")
    with pytest.raises(CatalogError, match="manifest checksum"):
        load_model_catalog(root)


def test_catalog_install_is_verified_idempotent_and_not_active(tmp_path: Path) -> None:
    settings = _prepare_database(tmp_path)
    catalog, archive_bytes = _test_catalog(tmp_path)
    downloader = FakeDownloader(archive_bytes)

    with TestClient(
        create_app(
            settings,
            model_catalog=catalog,
            catalog_downloader=downloader,
        )
    ) as client:
        first = client.post("/api/v1/models/catalog/license-plate-yolov11n/install")
        second = client.post("/api/v1/models/catalog/license-plate-yolov11n/install")
        listed = client.get("/api/v1/models/catalog")

    assert first.status_code == 201, first.text
    assert second.status_code == 200, second.text
    assert first.json()["catalog"] == {
        "catalog_id": CATALOG_ID,
        "entry_id": "license-plate-yolov11n",
    }
    assert first.json()["active"] is False
    assert second.json()["id"] == first.json()["id"]
    assert len(downloader.calls or []) == 1
    installed = listed.json()["models"][0]
    assert installed["installed"] is True
    assert installed["install_available"] is False

    database = Database(settings.storage.data_dir / "open-licenseplate.sqlite3")
    try:
        models = ModelRepository(database).list()
        assert len(models) == 1
        assert models[0].active is False
    finally:
        database.dispose()
    _assert_staging_empty(settings)


def test_catalog_install_rejects_unknown_id_without_downloading(tmp_path: Path) -> None:
    settings = _prepare_database(tmp_path)
    catalog, _archive_bytes = _test_catalog(tmp_path)
    downloader = FakeDownloader()

    with TestClient(
        create_app(
            settings,
            model_catalog=catalog,
            catalog_downloader=downloader,
        )
    ) as client:
        response = client.post("/api/v1/models/catalog/not-in-catalog/install")

    assert response.status_code == 404
    assert downloader.calls == []
    _assert_staging_empty(settings)


def test_catalog_archive_checksum_mismatch_cleans_up(tmp_path: Path) -> None:
    settings = _prepare_database(tmp_path)
    catalog, archive_bytes = _test_catalog(tmp_path)
    entry = catalog.entries[0]
    bad_entry = replace(entry, archive_sha256="0" * 64)
    bad_catalog = replace(catalog, entries=(bad_entry, *catalog.entries[1:]))
    downloader = FakeDownloader(archive_bytes)

    with TestClient(
        create_app(
            settings,
            model_catalog=bad_catalog,
            catalog_downloader=downloader,
        )
    ) as client:
        response = client.post("/api/v1/models/catalog/license-plate-yolov11n/install")

    assert response.status_code == 422
    assert "SHA-256" in response.json()["detail"]
    _assert_staging_empty(settings)
    database = Database(settings.storage.data_dir / "open-licenseplate.sqlite3")
    try:
        assert ModelRepository(database).list() == []
    finally:
        database.dispose()


def test_catalog_package_checksum_mismatch_cleans_up(tmp_path: Path) -> None:
    settings = _prepare_database(tmp_path)
    catalog, archive_bytes = _test_catalog(tmp_path)
    package = tmp_path / "wrong" / "license-plate-yolov11n.mlpackage"
    (package / "Data").mkdir(parents=True)
    (package / "Manifest.json").write_text("{}", encoding="utf-8")
    (package / "Data" / "weights.bin").write_bytes(b"different package")
    wrong_archive_path = tmp_path / "wrong.zip"
    with zipfile.ZipFile(wrong_archive_path, "w") as output:
        for path in package.rglob("*"):
            if path.is_file():
                output.write(path, path.relative_to(package.parent).as_posix())
    wrong_archive = wrong_archive_path.read_bytes()
    bad_entry = replace(
        catalog.entries[0],
        archive_size=len(wrong_archive),
        archive_sha256=hashlib.sha256(wrong_archive).hexdigest(),
    )
    bad_catalog = replace(catalog, entries=(bad_entry, *catalog.entries[1:]))
    downloader = FakeDownloader(wrong_archive)

    with TestClient(
        create_app(
            settings,
            model_catalog=bad_catalog,
            catalog_downloader=downloader,
        )
    ) as client:
        response = client.post("/api/v1/models/catalog/license-plate-yolov11n/install")

    assert response.status_code == 422
    assert "artifact_sha256" in response.json()["detail"]
    assert archive_bytes != wrong_archive
    _assert_staging_empty(settings)


def test_catalog_manifest_mismatch_cleans_up(tmp_path: Path) -> None:
    settings = _prepare_database(tmp_path)
    catalog, archive_bytes = _test_catalog(tmp_path)
    raw_manifest = dict(catalog.entries[0].manifest.raw)
    raw_manifest["artifact"] = "unexpected.mlpackage"
    bad_manifest_bytes = json.dumps(raw_manifest, separators=(",", ":")).encode("utf-8")
    bad_manifest = parse_manifest(bad_manifest_bytes)
    bad_entry = replace(
        catalog.entries[0],
        display_name=bad_manifest.display_name,
        manifest_sha256=hashlib.sha256(bad_manifest_bytes).hexdigest(),
        manifest_bytes=bad_manifest_bytes,
        manifest=bad_manifest,
    )
    bad_catalog = replace(catalog, entries=(bad_entry, *catalog.entries[1:]))
    downloader = FakeDownloader(archive_bytes)

    with TestClient(
        create_app(
            settings,
            model_catalog=bad_catalog,
            catalog_downloader=downloader,
        )
    ) as client:
        response = client.post("/api/v1/models/catalog/license-plate-yolov11n/install")

    assert response.status_code == 422
    assert "one top-level .mlpackage" in response.json()["detail"]
    _assert_staging_empty(settings)


def test_catalog_import_failure_cleans_up(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _prepare_database(tmp_path)
    catalog, archive_bytes = _test_catalog(tmp_path)
    downloader = FakeDownloader(archive_bytes)

    def fail_import(**_kwargs: object) -> object:
        raise ModelImportError("catalog import failed")

    monkeypatch.setattr(catalog_module, "import_model", fail_import)
    with TestClient(
        create_app(
            settings,
            model_catalog=catalog,
            catalog_downloader=downloader,
        )
    ) as client:
        response = client.post("/api/v1/models/catalog/license-plate-yolov11n/install")

    assert response.status_code == 422
    assert response.json()["detail"] == "catalog import failed"
    _assert_staging_empty(settings)


def test_catalog_download_timeout_is_bounded_and_cleans_up(tmp_path: Path) -> None:
    settings = _prepare_database(tmp_path)
    catalog, _archive_bytes = _test_catalog(tmp_path)
    downloader = FakeDownloader(error=TimeoutError())

    with TestClient(
        create_app(
            settings,
            model_catalog=catalog,
            catalog_downloader=downloader,
        )
    ) as client:
        response = client.post("/api/v1/models/catalog/license-plate-yolov11n/install")

    assert response.status_code == 502
    assert response.json()["detail"] == "catalog asset download timed out"
    _assert_staging_empty(settings)


def test_fixed_downloader_rejects_wrong_hosts_and_unsafe_redirects(tmp_path: Path) -> None:
    archive_asset = "asset.zip"
    fixed_url = (
        "https://github.com/linuxlewis/open-licenseplate/releases/download/"
        f"model-catalog-v1/{archive_asset}"
    )
    invalid_urls = (
        fixed_url.replace("https://", "http://"),
        fixed_url.replace("github.com", "evil.example"),
        fixed_url.replace("github.com", "user:pass@github.com"),
        fixed_url.replace("github.com", "github.com:8443"),
        f"{fixed_url}#fragment",
        fixed_url.replace(
            "/linuxlewis/open-licenseplate/",
            "/other/repository/",
        ),
    )
    for invalid_url in invalid_urls:
        with pytest.raises(CatalogError):
            catalog_module.validate_catalog_url(invalid_url, archive_asset)

    downloader = FixedCatalogDownloader(
        connection_factory=lambda _host, _port, _timeout: FakeConnection(
            FakeResponse(
                status=302,
                location="https://evil.example/asset.zip",
            )
        )
    )
    with pytest.raises(CatalogError, match="redirect host"):
        downloader.download(
            url=fixed_url,
            archive_asset=archive_asset,
            expected_size=1,
            destination=tmp_path / "asset.zip",
        )


def test_fixed_downloader_rejects_too_many_redirects(tmp_path: Path) -> None:
    archive_asset = "asset.zip"
    fixed_url = (
        "https://github.com/linuxlewis/open-licenseplate/releases/download/"
        f"model-catalog-v1/{archive_asset}"
    )
    responses = [
        FakeResponse(
            status=302,
            location=(
                "https://release-assets.githubusercontent.com/"
                f"github-production-release-asset/{index}?signature=test"
            ),
        )
        for index in range(3)
    ]

    def connection_factory(_host: str, _port: int, _timeout: float) -> FakeConnection:
        return FakeConnection(responses.pop(0))

    downloader = FixedCatalogDownloader(
        max_redirects=2,
        connection_factory=connection_factory,
    )
    with pytest.raises(CatalogError, match="too many redirects"):
        downloader.download(
            url=fixed_url,
            archive_asset=archive_asset,
            expected_size=1,
            destination=tmp_path / "asset.zip",
        )


@pytest.mark.parametrize(
    ("response", "expected_message"),
    [
        (
            FakeResponse(
                headers={"Content-Length": "4"},
                chunks=[b"abc"],
            ),
            "size does not match",
        ),
        (
            FakeResponse(chunks=[b"abcde"]),
            "exceeds the locked size limit",
        ),
        (
            FakeResponse(
                headers={"Content-Length": "3"},
                chunks=[b"abc"],
            ),
            "Content-Length does not match",
        ),
        (
            FakeResponse(read_error=TimeoutError()),
            "download timed out",
        ),
    ],
)
def test_fixed_downloader_enforces_stream_limits_and_timeouts(
    tmp_path: Path,
    response: FakeResponse,
    expected_message: str,
) -> None:
    archive_asset = "asset.zip"
    fixed_url = (
        "https://github.com/linuxlewis/open-licenseplate/releases/download/"
        f"model-catalog-v1/{archive_asset}"
    )
    downloader = FixedCatalogDownloader(
        connection_factory=lambda _host, _port, _timeout: FakeConnection(response)
    )

    with pytest.raises(CatalogError, match=expected_message):
        downloader.download(
            url=fixed_url,
            archive_asset=archive_asset,
            expected_size=4,
            destination=tmp_path / "asset.zip",
        )
