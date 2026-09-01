"""Fixed, verified model catalog metadata and download support."""

from __future__ import annotations

import hashlib
import http.client
import json
import logging
import os
import re
import stat
import tempfile
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, BinaryIO, Protocol, cast
from urllib.parse import quote, unquote, urljoin, urlsplit

from ..database import Database, database_status
from ..paths import ManagedPaths
from .archive import (
    MAX_ARCHIVE_BYTES,
    ModelArchiveError,
    compute_artifact_sha256,
    validate_package_directory,
)
from .manifest import ModelManifest, ModelManifestError, parse_manifest
from .repository import Model, ModelRepository, manifest_from_record
from .service import ModelConflictError, ModelImportError, import_model

CATALOG_SCHEMA_VERSION = 1
CATALOG_REPOSITORY = "linuxlewis/open-licenseplate"
CATALOG_RELEASE_TAG = "model-catalog-v1"
CATALOG_ID = "open-licenseplate-model-catalog-v1"
CATALOG_ENTRY_IDS = (
    "license-plate-yolov11n",
    "license-plate-yolov11s",
    "license-plate-yolov11m",
)
CATALOG_ROOT = Path(__file__).resolve().parents[3] / "model-catalog"
CATALOG_LOCK_PATH = CATALOG_ROOT / "model-catalog-lock.json"
CATALOG_MAX_REDIRECTS = 3
CATALOG_CONNECT_TIMEOUT = 10.0
CATALOG_READ_TIMEOUT = 30.0
CATALOG_TOTAL_TIMEOUT = 120.0
DOWNLOAD_CHUNK_BYTES = 1024 * 1024
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
MODEL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_GITHUB_HOST = "github.com"
_RELEASE_REDIRECT_HOSTS = frozenset(
    {
        "objects.githubusercontent.com",
        "release-assets.githubusercontent.com",
    }
)
_RELEASE_CDN_PATH_PREFIX = "/github-production-release-asset/"
_RELEASE_PATH_PREFIX = f"/{CATALOG_REPOSITORY}/releases/download/{CATALOG_RELEASE_TAG}/"
_REQUEST_HEADERS = {
    "Accept": "application/octet-stream",
    "User-Agent": "open-licenseplate-model-catalog",
}
_CATALOG_RESOURCE_DIRECTORY = "catalog_data"
_CATALOG_LOCK_FILENAME = "model-catalog-lock.json"
logger = logging.getLogger("open_licenseplate.models.catalog")


class CatalogError(ValueError):
    """Raised when fixed catalog metadata or an install is invalid."""


class CatalogDownloadError(CatalogError):
    """Raised when a catalog asset cannot be downloaded."""


class CatalogIntegrityError(CatalogError):
    """Raised when a downloaded catalog asset does not match its lock entry."""


@dataclass(frozen=True)
class CatalogEntry:
    """One fixed catalog entry loaded from committed metadata."""

    catalog_id: str
    display_name: str
    recommendation: str
    archive_asset: str
    archive_url: str
    archive_size: int
    archive_sha256: str
    package_sha256: str
    license: str
    source_repository: str
    source_revision: str
    manifest_sha256: str
    manifest_bytes: bytes
    manifest: ModelManifest


@dataclass(frozen=True)
class ModelCatalog:
    """The committed catalog lock and its matching manifests."""

    catalog_id: str
    repository: str
    release_tag: str
    entries: tuple[CatalogEntry, ...]

    def get(self, catalog_id: str) -> CatalogEntry | None:
        """Return one entry by its fixed ID."""
        for entry in self.entries:
            if entry.catalog_id == catalog_id:
                return entry
        return None


class CatalogDownloader(Protocol):
    """Downloader contract used by the catalog installer and its tests."""

    def download(
        self,
        *,
        url: str,
        archive_asset: str,
        expected_size: int,
        destination: BinaryIO,
        cancel_event: threading.Event | None = None,
    ) -> None:
        """Stream one fixed catalog asset into a private destination."""


class CatalogInstallLocks:
    """Serialize installs for one catalog model inside one application process."""

    def __init__(self) -> None:
        self._guard = threading.Lock()
        self._locks: dict[str, threading.Lock] = {}

    @contextmanager
    def acquire(self, catalog_id: str) -> Iterator[None]:
        """Hold the lock for one catalog entry until its install finishes."""
        with self._guard:
            lock = self._locks.setdefault(catalog_id, threading.Lock())
        with lock:
            yield


_DEFAULT_INSTALL_LOCKS = CatalogInstallLocks()


class FixedCatalogDownloader:
    """Download only verified GitHub release assets with bounded redirects."""

    def __init__(
        self,
        *,
        connect_timeout: float = CATALOG_CONNECT_TIMEOUT,
        read_timeout: float = CATALOG_READ_TIMEOUT,
        total_timeout: float = CATALOG_TOTAL_TIMEOUT,
        max_redirects: int = CATALOG_MAX_REDIRECTS,
        connection_factory: Callable[[str, int, float], Any] | None = None,
    ) -> None:
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout
        self.total_timeout = total_timeout
        self.max_redirects = max_redirects
        self._connection_factory = connection_factory or _https_connection

    def download(
        self,
        *,
        url: str,
        archive_asset: str,
        expected_size: int,
        destination: BinaryIO,
        cancel_event: threading.Event | None = None,
    ) -> None:
        """Download one exact-size asset and reject unsafe redirect chains."""
        current_url = validate_catalog_url(url, archive_asset)
        deadline = time.monotonic() + self.total_timeout
        for redirect_count in range(self.max_redirects + 1):
            connection: Any | None = None
            response: Any | None = None
            try:
                _check_download_state(deadline, cancel_event)
                parsed = _parse_url(current_url)
                host = parsed.hostname
                if host is None:
                    raise CatalogDownloadError("catalog asset URL has no host")
                port = parsed.port or 443
                connection = self._connection_factory(
                    host,
                    port,
                    min(self.connect_timeout, _remaining(deadline)),
                )
                connection.request("GET", _request_target(parsed), headers=_REQUEST_HEADERS)
                response = connection.getresponse()
                status = int(response.status)
                if 300 <= status < 400:
                    if redirect_count >= self.max_redirects:
                        raise CatalogDownloadError("catalog asset returned too many redirects")
                    location = response.getheader("Location")
                    if not isinstance(location, str) or not location.strip():
                        raise CatalogDownloadError("catalog asset returned an invalid redirect")
                    current_url = validate_catalog_redirect(
                        urljoin(current_url, location),
                        archive_asset,
                    )
                    continue
                if status < 200 or status >= 300:
                    raise CatalogDownloadError("catalog asset returned an unexpected status")
                _set_read_timeout(response, min(self.read_timeout, _remaining(deadline)))
                _stream_response(
                    response,
                    destination=destination,
                    expected_size=expected_size,
                    deadline=deadline,
                    read_timeout=self.read_timeout,
                    cancel_event=cancel_event,
                )
                return
            except CatalogError:
                raise
            except (http.client.HTTPException, OSError, TimeoutError) as error:
                if isinstance(error, TimeoutError):
                    raise CatalogDownloadError("catalog asset download timed out") from error
                raise CatalogDownloadError("catalog asset download failed") from error
            finally:
                if response is not None:
                    _close_response(response)
                if connection is not None:
                    _close_connection(connection)

        raise CatalogDownloadError("catalog asset returned too many redirects")


def load_model_catalog(root: Path | None = None) -> ModelCatalog:
    """Load and validate the committed lock and its matching manifests."""
    if root is None:
        resource_root = resources.files("open_licenseplate").joinpath(_CATALOG_RESOURCE_DIRECTORY)
        if resource_root.is_dir():
            return _load_resource_catalog(resource_root)
        root = CATALOG_ROOT
    return _load_filesystem_catalog(root)


def _load_filesystem_catalog(root: Path) -> ModelCatalog:
    try:
        lock_path = _contained_file(root, Path(_CATALOG_LOCK_FILENAME))
        lock = _read_json_object(lock_path)
        return _parse_catalog_lock(
            lock,
            lambda reference: _contained_file(root, Path(reference)).read_bytes(),
        )
    except CatalogError:
        raise
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        raise CatalogError("model catalog metadata is invalid") from error


def _load_resource_catalog(resource_root: Any) -> ModelCatalog:
    try:
        lock_resource = resource_root.joinpath(_CATALOG_LOCK_FILENAME)
        lock = _read_json_object_bytes(cast(bytes, lock_resource.read_bytes()))

        def read_manifest(reference: str) -> bytes:
            parts = _manifest_reference_parts(reference)
            manifest_resource = resource_root.joinpath(*parts)
            if not manifest_resource.is_file():
                raise CatalogError("model catalog metadata is invalid")
            return cast(bytes, manifest_resource.read_bytes())

        return _parse_catalog_lock(lock, read_manifest)
    except CatalogError:
        raise
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        raise CatalogError("model catalog metadata is invalid") from error


def catalog_entry_payload(
    catalog: ModelCatalog,
    entry: CatalogEntry,
    *,
    installed: bool,
    install_available: bool,
) -> dict[str, Any]:
    """Return the public catalog fields without any download URL."""
    return {
        "catalog_id": entry.catalog_id,
        "display_name": entry.display_name,
        "recommendation": entry.recommendation,
        "archive_size": entry.archive_size,
        "license": entry.license,
        "source": {
            "repository": entry.source_repository,
            "revision": entry.source_revision,
        },
        "installed": installed,
        "install_available": install_available,
        "catalog": {
            "catalog_id": catalog.catalog_id,
            "release_tag": catalog.release_tag,
        },
    }


def install_catalog_model(
    *,
    catalog: ModelCatalog,
    entry: CatalogEntry,
    paths: ManagedPaths,
    repository: ModelRepository,
    downloader: CatalogDownloader,
    install_locks: CatalogInstallLocks | None = None,
    cancel_event: threading.Event | None = None,
) -> tuple[Model, bool]:
    """Download, verify, and import one fixed entry.

    The existing model importer remains the authority for package extraction,
    manifest checks, package hashing, and the database commit.
    """
    effective_locks = install_locks or _DEFAULT_INSTALL_LOCKS
    effective_cancel_event = cancel_event or threading.Event()
    with effective_locks.acquire(entry.catalog_id):
        return _install_catalog_model_locked(
            catalog=catalog,
            entry=entry,
            paths=paths,
            repository=repository,
            downloader=downloader,
            cancel_event=effective_cancel_event,
        )


def _install_catalog_model_locked(
    *,
    catalog: ModelCatalog,
    entry: CatalogEntry,
    paths: ManagedPaths,
    repository: ModelRepository,
    downloader: CatalogDownloader,
    cancel_event: threading.Event,
) -> tuple[Model, bool]:
    """Run one serialized catalog install with atomic staging cleanup."""
    _check_cancelled(cancel_event)
    _validate_catalog_entry(catalog, entry)
    existing = repository.get(entry.manifest.model_id)
    if existing is not None:
        if _is_identical_installed_model(existing, entry):
            if catalog_model_is_installed(existing, entry, paths):
                return existing, False
            raise ModelConflictError(
                "the catalog model record exists but its artifact is missing or damaged"
            )
        raise ModelConflictError(
            "a managed model with this catalog id already exists with different provenance"
        )

    try:
        paths.ensure_directories()
        _ensure_private_staging(paths.staging)
        staging_path, staging_file = _new_staging_file(paths.staging)
    except OSError as error:
        raise CatalogError("catalog staging could not be prepared") from error

    committed = False
    try:
        try:
            downloader.download(
                url=entry.archive_url,
                archive_asset=entry.archive_asset,
                expected_size=entry.archive_size,
                destination=staging_file,
                cancel_event=cancel_event,
            )
        except CatalogError:
            raise
        except TimeoutError as error:
            raise CatalogDownloadError("catalog asset download timed out") from error
        except OSError as error:
            raise CatalogDownloadError("catalog asset download failed") from error
        except Exception as error:
            raise CatalogDownloadError("catalog asset download failed") from error

        try:
            staging_file.flush()
            os.fsync(staging_file.fileno())
        except (OSError, ValueError) as error:
            raise CatalogDownloadError("catalog staging write failed") from error
        _check_cancelled(cancel_event)
        _verify_downloaded_archive(
            staging_file,
            expected_size=entry.archive_size,
            expected_sha256=entry.archive_sha256,
        )
        staging_file.close()
        try:
            model = import_model(
                manifest_value=entry.manifest_bytes,
                source_path=staging_path,
                paths=paths,
                repository=repository,
            )
        except (CatalogError, ModelImportError):
            raise
        except ValueError as error:
            raise ModelConflictError(
                "a managed model with this catalog id already exists"
            ) from error
        except Exception as error:
            raise ModelImportError("catalog model import failed safely") from error
        committed = True
        return model, True
    finally:
        if not staging_file.closed:
            with suppress(OSError, ValueError):
                staging_file.close()
        try:
            _remove_staging_file(staging_path)
        except OSError:
            logger.error(
                "catalog staging cleanup failed",
                extra={"archive_asset": entry.archive_asset},
            )
            if not committed:
                raise CatalogError("catalog staging cleanup failed") from None


def catalog_model_is_installed(
    model: Model,
    entry: CatalogEntry,
    paths: ManagedPaths,
) -> bool:
    """Verify model metadata and the managed package before reporting installed."""
    if not _is_identical_installed_model(model, entry):
        return False
    try:
        artifact_path = ManagedPaths.validate_contained_path(
            paths.models / model.artifact_path,
            paths.models,
        )
        model_directory = ManagedPaths.validate_contained_path(
            paths.models / model.id,
            paths.models,
        )
        if (
            not artifact_path.is_dir()
            or artifact_path.is_symlink()
            or artifact_path.name != entry.manifest.artifact
            or artifact_path.parent != model_directory
        ):
            return False
        validate_package_directory(
            artifact_path,
            expected_artifact=entry.manifest.artifact,
        )
        return compute_artifact_sha256(artifact_path) == entry.package_sha256
    except (ModelArchiveError, ModelManifestError, OSError, ValueError, TypeError):
        return False


def reconcile_orphaned_model_directories(paths: ManagedPaths) -> int:
    """Remove model directories left by an interrupted import with no DB row."""
    if paths.models.is_symlink() or not paths.models.is_dir():
        raise CatalogError("managed model directory is not safe")
    if database_status(paths.database)["status"] != "ok":
        return 0
    database = Database(paths.database)
    try:
        known_ids = {model.id for model in ModelRepository(database).list()}
    finally:
        database.dispose()

    removed = 0
    for child in tuple(paths.models.iterdir()):
        if child.name in known_ids:
            continue
        ManagedPaths.validate_contained_path(child, paths.models)
        _remove_orphan_model_entry(child)
        removed += 1
    return removed


def validate_catalog_url(url: str, archive_asset: str) -> str:
    """Validate the initial fixed GitHub release URL."""
    _validate_archive_asset(archive_asset)
    parsed = _parse_url(url)
    _validate_common_url_parts(parsed)
    if parsed.hostname != _GITHUB_HOST:
        raise CatalogDownloadError("catalog asset host is not allowed")
    if parsed.port not in (None, 443):
        raise CatalogDownloadError("catalog asset port is not allowed")
    if parsed.query:
        raise CatalogDownloadError("catalog asset URL contains an unexpected query")
    expected_path = _RELEASE_PATH_PREFIX + quote(archive_asset, safe="")
    if parsed.path != expected_path:
        raise CatalogDownloadError("catalog asset path is not allowed")
    return url


def validate_catalog_redirect(url: str, archive_asset: str) -> str:
    """Validate one redirect without accepting arbitrary network targets."""
    _validate_archive_asset(archive_asset)
    parsed = _parse_url(url)
    _validate_common_url_parts(parsed)
    host = parsed.hostname
    if host == _GITHUB_HOST:
        if parsed.port not in (None, 443) or parsed.query:
            raise CatalogDownloadError("catalog redirect is not allowed")
        expected_path = _RELEASE_PATH_PREFIX + quote(archive_asset, safe="")
        if parsed.path != expected_path:
            raise CatalogDownloadError("catalog redirect path is not allowed")
        return url
    if host not in _RELEASE_REDIRECT_HOSTS:
        raise CatalogDownloadError("catalog redirect host is not allowed")
    if parsed.port not in (None, 443):
        raise CatalogDownloadError("catalog redirect port is not allowed")
    decoded_path = unquote(parsed.path)
    if (
        not decoded_path.startswith(_RELEASE_CDN_PATH_PREFIX)
        or "\x00" in decoded_path
        or not parsed.query
    ):
        raise CatalogDownloadError("catalog redirect path is not allowed")
    path_parts = decoded_path.split("/")
    if any(part in {".", ".."} for part in path_parts[1:]):
        raise CatalogDownloadError("catalog redirect path is not allowed")
    return url


def _parse_catalog_lock(
    lock: dict[str, Any],
    read_manifest: Callable[[str], bytes],
) -> ModelCatalog:
    if set(lock) != {"catalog_id", "models", "release", "schema_version"}:
        raise CatalogError("model catalog metadata is invalid")
    if lock["schema_version"] != CATALOG_SCHEMA_VERSION:
        raise CatalogError("model catalog metadata is invalid")
    catalog_id = _required_text(lock, "catalog_id", 128)
    if catalog_id != CATALOG_ID:
        raise CatalogError("model catalog is not approved")
    release = _required_mapping(lock, "release")
    if set(release) != {"repository", "tag", "prerelease"}:
        raise CatalogError("model catalog metadata is invalid")
    if release != {
        "repository": CATALOG_REPOSITORY,
        "tag": CATALOG_RELEASE_TAG,
        "prerelease": True,
    }:
        raise CatalogError("model catalog release is not approved")

    raw_models = lock["models"]
    if not isinstance(raw_models, list) or not raw_models:
        raise CatalogError("model catalog metadata is invalid")
    entries: list[CatalogEntry] = []
    seen_ids: set[str] = set()
    for raw_model in raw_models:
        if not isinstance(raw_model, dict):
            raise CatalogError("model catalog metadata is invalid")
        entry = _parse_catalog_entry(raw_model, read_manifest)
        if entry.catalog_id in seen_ids:
            raise CatalogError("model catalog contains duplicate IDs")
        seen_ids.add(entry.catalog_id)
        entries.append(entry)
    if tuple(entry.catalog_id for entry in entries) != CATALOG_ENTRY_IDS:
        raise CatalogError("model catalog entries are not approved")
    return ModelCatalog(
        catalog_id=catalog_id,
        repository=CATALOG_REPOSITORY,
        release_tag=CATALOG_RELEASE_TAG,
        entries=tuple(entries),
    )


def _parse_catalog_entry(
    raw: dict[str, Any],
    read_manifest: Callable[[str], bytes],
) -> CatalogEntry:
    expected_keys = {
        "id",
        "manifest",
        "manifest_asset",
        "manifest_sha256",
        "archive_asset",
        "archive_url",
        "package_sha256",
        "archive_sha256",
        "archive_size",
        "source",
        "license",
        "release_tag",
    }
    if set(raw) != expected_keys:
        raise CatalogError("model catalog entry metadata is invalid")
    catalog_id = _required_text(raw, "id", 128)
    if MODEL_ID_RE.fullmatch(catalog_id) is None:
        raise CatalogError("model catalog entry ID is invalid")
    if raw["release_tag"] != CATALOG_RELEASE_TAG:
        raise CatalogError("model catalog entry release is not approved")
    archive_asset = _required_text(raw, "archive_asset", 255)
    _validate_archive_asset(archive_asset)
    archive_url = _required_text(raw, "archive_url", 2048)
    validate_catalog_url(archive_url, archive_asset)
    archive_size = raw["archive_size"]
    if (
        isinstance(archive_size, bool)
        or not isinstance(archive_size, int)
        or archive_size <= 0
        or archive_size > MAX_ARCHIVE_BYTES
    ):
        raise CatalogError("model catalog archive size is invalid")
    archive_sha256 = _required_sha256(raw, "archive_sha256")
    package_sha256 = _required_sha256(raw, "package_sha256")
    license_name = _required_text(raw, "license", 255)
    source = _required_mapping(raw, "source")
    source_repository = _required_text(source, "repository", 2048)
    source_revision = _required_text(source, "revision", 128)

    manifest_reference = _required_text(raw, "manifest", 255)
    manifest_asset = _required_text(raw, "manifest_asset", 255)
    manifest_sha256 = _required_sha256(raw, "manifest_sha256")
    if _manifest_reference_parts(manifest_reference)[-1] != manifest_asset:
        raise CatalogError("model catalog manifest reference is invalid")
    try:
        manifest_bytes = read_manifest(manifest_reference)
        if hashlib.sha256(manifest_bytes).hexdigest() != manifest_sha256:
            raise CatalogError("model catalog manifest checksum does not match the lock")
        manifest = parse_manifest(manifest_bytes)
    except CatalogError:
        raise
    except (ModelManifestError, OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        raise CatalogError("model catalog manifest is invalid") from error

    distribution = _required_mapping(manifest.raw, "distribution")
    recommendation = _required_text(distribution, "recommendation", 64)
    if (
        manifest.model_id != catalog_id
        or manifest.display_name != _required_text(manifest.raw, "display_name", 255)
        or manifest.source_license != license_name
        or manifest.raw.get("source", {}).get("repository") != source_repository
        or manifest.raw.get("source", {}).get("revision") != source_revision
        or manifest.artifact_sha256 != package_sha256
        or distribution.get("archive") != archive_asset
        or distribution.get("archive_sha256") != archive_sha256
        or distribution.get("archive_size") != archive_size
        or distribution.get("release_tag") != CATALOG_RELEASE_TAG
    ):
        raise CatalogError("model catalog manifest does not match its lock entry")
    return CatalogEntry(
        catalog_id=catalog_id,
        display_name=manifest.display_name,
        recommendation=recommendation,
        archive_asset=archive_asset,
        archive_url=archive_url,
        archive_size=archive_size,
        archive_sha256=archive_sha256,
        package_sha256=package_sha256,
        license=license_name,
        source_repository=source_repository,
        source_revision=source_revision,
        manifest_sha256=manifest_sha256,
        manifest_bytes=manifest_bytes,
        manifest=manifest,
    )


def _is_identical_installed_model(model: Model, entry: CatalogEntry) -> bool:
    if model.id != entry.manifest.model_id or model.artifact_sha256 != entry.package_sha256:
        return False
    try:
        manifest = manifest_from_record(model)
    except (ModelManifestError, ValueError):
        return False
    return manifest.snapshot_json == entry.manifest.snapshot_json


def _validate_catalog_entry(catalog: ModelCatalog, entry: CatalogEntry) -> None:
    """Keep injected test catalogs subject to the same fixed-entry contract."""
    if (
        catalog.catalog_id != CATALOG_ID
        or catalog.repository != CATALOG_REPOSITORY
        or catalog.release_tag != CATALOG_RELEASE_TAG
        or catalog.get(entry.catalog_id) != entry
    ):
        raise CatalogError("model catalog metadata is invalid")
    try:
        parsed_manifest = parse_manifest(entry.manifest_bytes)
    except (ModelManifestError, ValueError, TypeError) as error:
        raise CatalogError("model catalog entry metadata is invalid") from error
    if (
        parsed_manifest.snapshot_json != entry.manifest.snapshot_json
        or entry.manifest.model_id != entry.catalog_id
        or entry.manifest.display_name != entry.display_name
        or entry.manifest.artifact_sha256 != entry.package_sha256
        or entry.manifest.source_license != entry.license
        or not SHA256_RE.fullmatch(entry.manifest_sha256)
        or hashlib.sha256(entry.manifest_bytes).hexdigest() != entry.manifest_sha256
        or entry.archive_size <= 0
        or entry.archive_size > MAX_ARCHIVE_BYTES
        or not SHA256_RE.fullmatch(entry.archive_sha256)
        or not SHA256_RE.fullmatch(entry.package_sha256)
    ):
        raise CatalogError("model catalog entry metadata is invalid")
    validate_catalog_url(entry.archive_url, entry.archive_asset)


def _verify_downloaded_archive(
    handle: BinaryIO,
    *,
    expected_size: int,
    expected_sha256: str,
) -> None:
    try:
        file_stat = os.fstat(handle.fileno())
        if not stat.S_ISREG(file_stat.st_mode):
            raise CatalogIntegrityError("catalog archive is not a regular file")
        actual_size = file_stat.st_size
        if actual_size != expected_size:
            raise CatalogIntegrityError("catalog archive size does not match the lock")
        digest = hashlib.sha256()
        handle.seek(0)
        for chunk in iter(lambda: handle.read(DOWNLOAD_CHUNK_BYTES), b""):
            digest.update(chunk)
        handle.seek(0)
    except CatalogError:
        raise
    except (OSError, ValueError) as error:
        raise CatalogIntegrityError("catalog archive could not be verified") from error
    if digest.hexdigest() != expected_sha256:
        raise CatalogIntegrityError("catalog archive SHA-256 does not match the lock")


def _stream_response(
    response: Any,
    *,
    destination: BinaryIO,
    expected_size: int,
    deadline: float,
    read_timeout: float,
    cancel_event: threading.Event | None,
) -> None:
    content_length = response.getheader("Content-Length")
    if content_length is not None:
        if not isinstance(content_length, str) or not content_length.strip().isdigit():
            raise CatalogIntegrityError("catalog archive Content-Length is invalid")
        if int(content_length.strip()) != expected_size:
            raise CatalogIntegrityError("catalog archive Content-Length does not match the lock")

    received = 0
    try:
        while True:
            _check_download_state(deadline, cancel_event)
            _set_read_timeout(response, min(read_timeout, _remaining(deadline)))
            chunk = response.read(DOWNLOAD_CHUNK_BYTES)
            _check_download_state(deadline, cancel_event)
            if not chunk:
                break
            if not isinstance(chunk, bytes):
                raise CatalogDownloadError("catalog asset returned an invalid body")
            received += len(chunk)
            if received > expected_size:
                raise CatalogIntegrityError("catalog archive exceeds the locked size limit")
            destination.write(chunk)
    except CatalogError:
        raise
    except (OSError, TimeoutError, ValueError) as error:
        if isinstance(error, TimeoutError):
            raise CatalogDownloadError("catalog asset download timed out") from error
        raise CatalogDownloadError("catalog asset download failed") from error
    if received != expected_size:
        raise CatalogIntegrityError("catalog archive size does not match the lock")


def _new_staging_file(staging_directory: Path) -> tuple[Path, BinaryIO]:
    file_descriptor, path_value = tempfile.mkstemp(
        prefix="catalog-",
        suffix=".download",
        dir=staging_directory,
    )
    path = Path(path_value)
    try:
        os.chmod(path, 0o600, follow_symlinks=False)
        return path, os.fdopen(file_descriptor, "w+b")
    except OSError:
        with suppress(OSError):
            os.close(file_descriptor)
        with suppress(OSError):
            path.unlink(missing_ok=True)
        raise


def _ensure_private_staging(staging_directory: Path) -> None:
    """Repair safe directory mode and reject a replaceable staging root."""
    if staging_directory.is_symlink() or not staging_directory.is_dir():
        raise CatalogError("catalog staging directory is not safe")
    try:
        mode = stat.S_IMODE(staging_directory.stat().st_mode)
        if mode & 0o077:
            staging_directory.chmod(0o700)
            mode = stat.S_IMODE(staging_directory.stat().st_mode)
    except OSError as error:
        raise CatalogError("catalog staging directory is not safe") from error
    if mode & 0o077:
        raise CatalogError("catalog staging directory is not private")


def _remove_staging_file(path: Path) -> None:
    path.unlink(missing_ok=True)


def _remove_orphan_model_entry(path: Path) -> None:
    """Remove one validated orphan without targeting the managed root."""
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
        return
    if not path.is_dir():
        raise CatalogError("orphaned model storage is not safe")
    for child in tuple(path.iterdir()):
        _remove_orphan_model_entry(child)
    path.rmdir()


def _parse_url(value: str) -> Any:
    try:
        parsed = urlsplit(value)
        _ = parsed.port
    except (TypeError, ValueError) as error:
        raise CatalogDownloadError("catalog asset URL is invalid") from error
    return parsed


def _validate_common_url_parts(parsed: Any) -> None:
    if parsed.scheme.lower() != "https":
        raise CatalogDownloadError("catalog asset URL must use HTTPS")
    if parsed.username is not None or parsed.password is not None:
        raise CatalogDownloadError("catalog asset URL may not contain credentials")
    if parsed.fragment:
        raise CatalogDownloadError("catalog asset URL may not contain a fragment")
    if parsed.hostname is None:
        raise CatalogDownloadError("catalog asset URL has no host")


def _validate_archive_asset(value: str) -> None:
    if (
        not value
        or "/" in value
        or "\\" in value
        or value in {".", ".."}
        or "\x00" in value
        or len(value) > 255
    ):
        raise CatalogDownloadError("catalog archive asset name is invalid")


def _contained_file(root: Path, relative: Path) -> Path:
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise CatalogError("model catalog metadata is invalid")
    resolved_root = root.expanduser().resolve()
    candidate = resolved_root / relative
    if any(part.is_symlink() for part in (resolved_root, *candidate.parents)):
        raise CatalogError("model catalog metadata is invalid")
    if candidate.is_symlink():
        raise CatalogError("model catalog metadata is invalid")
    path = candidate.resolve()
    try:
        path.relative_to(resolved_root)
    except ValueError as error:
        raise CatalogError("model catalog metadata is invalid") from error
    if path.is_symlink() or not path.is_file():
        raise CatalogError("model catalog metadata is invalid")
    return path


def _read_json_object(path: Path) -> dict[str, Any]:
    return _read_json_object_bytes(path.read_bytes())


def _read_json_object_bytes(data: bytes) -> dict[str, Any]:
    value = json.loads(data.decode("utf-8"))
    if not isinstance(value, dict):
        raise CatalogError("model catalog metadata is invalid")
    return value


def _manifest_reference_parts(value: str) -> tuple[str, ...]:
    if not value or "\\" in value:
        raise CatalogError("model catalog manifest reference is invalid")
    parts = tuple(value.split("/"))
    if (
        len(parts) != 2
        or parts[0] != "manifests"
        or any(not part or part in {".", ".."} for part in parts)
    ):
        raise CatalogError("model catalog manifest reference is invalid")
    return parts


def _required_mapping(values: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = values.get(key)
    if not isinstance(value, dict):
        raise CatalogError("model catalog metadata is invalid")
    return value


def _required_text(values: Mapping[str, Any], key: str, maximum: int) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > maximum:
        raise CatalogError("model catalog metadata is invalid")
    return value.strip()


def _required_sha256(values: Mapping[str, Any], key: str) -> str:
    value = _required_text(values, key, 64).lower()
    if SHA256_RE.fullmatch(value) is None:
        raise CatalogError("model catalog metadata is invalid")
    return value


def _https_connection(host: str, port: int, timeout: float) -> Any:
    return http.client.HTTPSConnection(host, port=port, timeout=timeout)


def _request_target(parsed: Any) -> str:
    target = parsed.path or "/"
    if parsed.query:
        target += f"?{parsed.query}"
    return target


def _remaining(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise CatalogDownloadError("catalog asset download exceeded its deadline")
    return remaining


def _check_download_state(
    deadline: float,
    cancel_event: threading.Event | None,
) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise CatalogDownloadError("catalog asset download was cancelled")
    _remaining(deadline)


def _check_cancelled(cancel_event: threading.Event) -> None:
    if cancel_event.is_set():
        raise CatalogDownloadError("catalog asset download was cancelled")


def _set_read_timeout(response: Any, timeout: float) -> None:
    raw = getattr(getattr(response, "fp", None), "raw", None)
    sock = getattr(raw, "_sock", None)
    if sock is not None:
        sock.settimeout(timeout)


def _close_response(response: Any) -> None:
    with suppress(OSError):
        response.close()


def _close_connection(connection: Any) -> None:
    with suppress(OSError):
        connection.close()
