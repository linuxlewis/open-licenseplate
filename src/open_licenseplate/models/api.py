"""Managed model registry JSON API."""

from __future__ import annotations

import asyncio
import base64
import binascii
import math
import tempfile
from contextlib import suppress
from pathlib import Path
from time import perf_counter
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from starlette.datastructures import UploadFile
from starlette.formparsers import MultiPartException

from ..database import Database, database_status
from ..inference import (
    MAX_STILL_IMAGE_BYTES,
    BackendContractError,
    BackendOptions,
    BackendUnavailableError,
    DetectionValidationError,
    DetectorRegistry,
    InferenceError,
    ModelDescriptor,
    StillImageDecodeError,
    decode_still_image,
)
from ..inference.contract import ComputeUnit
from ..paths import ManagedPaths
from ..redaction import redact_text
from .archive import MAX_ARCHIVE_BYTES, compute_artifact_sha256
from .catalog import (
    CatalogDownloader,
    CatalogDownloadError,
    CatalogEntry,
    CatalogError,
    ModelCatalog,
    catalog_entry_payload,
    install_catalog_model,
)
from .manifest import ModelManifest
from .repository import Model, ModelRepository, manifest_from_record, model_payload
from .service import (
    RUNTIME_VALID,
    ModelConflictError,
    ModelDeletionError,
    ModelImportError,
    delete_model,
    import_model,
    validate_model,
)

router = APIRouter(prefix="/api/v1/models", tags=["models"])
MAX_MANIFEST_BYTES = 256 * 1024
MAX_UPLOAD_CHUNK = 1024 * 1024
MAX_STILL_IMAGE_REQUEST_BYTES = MAX_STILL_IMAGE_BYTES + 512 * 1024


@router.get("")
async def list_models(request: Request) -> JSONResponse:
    database, error = _open_database(request)
    if error is not None:
        return error
    assert database is not None
    try:
        repository = ModelRepository(database)
        models = [
            model_payload(model, artifact_exists=_artifact_exists(request.app.state.paths, model))
            for model in repository.list()
        ]
        return _json({"models": models})
    finally:
        database.dispose()


@router.get("/catalog")
async def list_model_catalog(request: Request) -> JSONResponse:
    database, error = _open_database(request)
    if error is not None:
        return error
    assert database is not None
    try:
        catalog: ModelCatalog = request.app.state.model_catalog
        repository = ModelRepository(database)
        entries = []
        for entry in catalog.entries:
            installed_model = repository.get(entry.manifest.model_id)
            installed = installed_model is not None and _catalog_model_matches(
                installed_model,
                entry,
            )
            entries.append(
                catalog_entry_payload(
                    catalog,
                    entry,
                    installed=installed,
                    install_available=installed_model is None,
                )
            )
        return _json({"catalog_id": catalog.catalog_id, "models": entries})
    finally:
        database.dispose()


@router.post("/catalog/{catalog_id}/install", status_code=201)
async def install_model_catalog_entry(
    catalog_id: str,
    request: Request,
) -> JSONResponse:
    catalog: ModelCatalog = request.app.state.model_catalog
    entry = catalog.get(catalog_id)
    if entry is None:
        return _error("catalog entry was not found", status_code=404)

    database, error = _open_database(request)
    if error is not None:
        return error
    assert database is not None
    try:
        worker = asyncio.create_task(
            asyncio.to_thread(
                _install_model_catalog_sync,
                catalog,
                entry,
                request.app.state.paths,
                ModelRepository(database),
                request.app.state.catalog_downloader,
            )
        )
        try:
            model, installed = await asyncio.shield(worker)
        except asyncio.CancelledError:
            with suppress(BaseException):
                await asyncio.shield(worker)
            raise
        payload = model_payload(
            model,
            artifact_exists=_artifact_exists(request.app.state.paths, model),
        )
        payload["catalog"] = {
            "catalog_id": catalog.catalog_id,
            "entry_id": entry.catalog_id,
        }
        return _json(payload, status_code=201 if installed else 200)
    except ModelConflictError as exception:
        return _error(str(exception), status_code=409)
    except CatalogDownloadError as exception:
        return _error(str(exception), status_code=502)
    except CatalogError as exception:
        status_code = 422
        return _error(str(exception), status_code=status_code)
    except (ModelImportError, ValueError) as exception:
        return _error(str(exception), status_code=422)
    finally:
        database.dispose()


@router.post("/import", status_code=201)
async def import_model_api(request: Request) -> JSONResponse:
    database, error = _open_database(request)
    if error is not None:
        return error
    assert database is not None
    archive_path: Path | None = None
    try:
        manifest_value, archive_path = await _read_import_request(request)
        model = import_model(
            manifest_value=manifest_value,
            source_path=archive_path,
            paths=request.app.state.paths,
            repository=ModelRepository(database),
        )
        return _json(
            model_payload(model, artifact_exists=True),
            status_code=201,
        )
    except ModelConflictError as exception:
        return _error(str(exception), status_code=409)
    except (ModelImportError, ValueError) as exception:
        return _error(str(exception), status_code=422)
    finally:
        if archive_path is not None:
            archive_path.unlink(missing_ok=True)
        database.dispose()


@router.get("/{model_id}")
async def get_model(model_id: str, request: Request) -> JSONResponse:
    database, error = _open_database(request)
    if error is not None:
        return error
    assert database is not None
    try:
        model = ModelRepository(database).get(model_id)
        if model is None:
            return _error("model was not found", status_code=404)
        return _json(
            model_payload(
                model,
                artifact_exists=_artifact_exists(request.app.state.paths, model),
            )
        )
    finally:
        database.dispose()


@router.post("/{model_id}/validate")
async def validate_model_api(model_id: str, request: Request) -> JSONResponse:
    database, error = _open_database(request)
    if error is not None:
        return error
    assert database is not None
    try:
        repository = ModelRepository(database)
        model = repository.get(model_id)
        if model is None:
            return _error("model was not found", status_code=404)
        result = await asyncio.to_thread(
            validate_model,
            model=model,
            paths=request.app.state.paths,
            repository=repository,
            backend=request.app.state.inference_backend_factory(),
        )
        refreshed = repository.get(model_id)
        if refreshed is None:
            return _error("model was removed during validation", status_code=409)
        return _json(
            {
                "model": model_payload(
                    refreshed,
                    artifact_exists=_artifact_exists(request.app.state.paths, refreshed),
                ),
                "structural_valid": result.structural_valid,
                "runtime_valid": result.runtime_valid,
                "validation_state": result.state,
                "validation_details": result.details,
            }
        )
    except ModelImportError as exception:
        return _error(str(exception), status_code=422)
    finally:
        database.dispose()


@router.post("/{model_id}/detect-image")
async def detect_image_api(model_id: str, request: Request) -> JSONResponse:
    """Decode and detect one bounded image without blocking the ASGI loop."""
    content_length = _content_length(request)
    if content_length is not None and content_length > MAX_STILL_IMAGE_REQUEST_BYTES:
        return _error("image upload exceeds the size limit", status_code=413)

    try:
        image_bytes, threshold, options = await _read_detection_request(request)
        payload = await asyncio.to_thread(
            _detect_image_sync,
            model_id,
            request.app.state.paths,
            request.app.state.detector_registry,
            image_bytes,
            threshold,
            options,
        )
        return _json(payload)
    except _ModelNotFoundError:
        return _error("model was not found", status_code=404)
    except StillImageDecodeError as exception:
        return _error(str(exception), status_code=422)
    except BackendUnavailableError:
        return _error(
            "the selected inference backend is not available on this system",
            status_code=503,
        )
    except BackendContractError:
        return _error(
            "the model manifest is incompatible with the selected inference backend",
            status_code=409,
        )
    except (DetectionValidationError, InferenceError) as exception:
        return _error(str(exception), status_code=422)
    except ModelImportError as exception:
        message = str(exception)
        status_code = 413 if "size limit" in message else 409
        return _error(message, status_code=status_code)
    except (MultiPartException, ValueError) as exception:
        return _error(str(exception), status_code=422)
    except (OSError, RuntimeError, TypeError):
        return _error(
            "still-image detection failed safely; check the model and image format",
            status_code=422,
        )


@router.post("/{model_id}/activate")
async def activate_model(model_id: str, request: Request) -> JSONResponse:
    try:
        active = await _requested_active_state(request, default=True)
    except ModelImportError as exception:
        return _error(str(exception), status_code=422)
    return await _set_model_active(model_id, request, active=active)


@router.post("/{model_id}/deactivate")
async def deactivate_model(model_id: str, request: Request) -> JSONResponse:
    return await _set_model_active(model_id, request, active=False)


@router.delete("/{model_id}")
async def remove_model(model_id: str, request: Request) -> JSONResponse:
    database, error = _open_database(request)
    if error is not None:
        return error
    assert database is not None
    try:
        repository = ModelRepository(database)
        model = repository.get(model_id)
        if model is None:
            return _error("model was not found", status_code=404)
        delete_model(model=model, paths=request.app.state.paths, repository=repository)
        return _json({"deleted": True, "model_id": model_id})
    except ModelDeletionError as exception:
        return _error(str(exception), status_code=500)
    except ModelImportError as exception:
        status_code = 409 if "active" in str(exception) else 422
        return _error(str(exception), status_code=status_code)
    finally:
        database.dispose()


async def _set_model_active(
    model_id: str,
    request: Request,
    *,
    active: bool,
) -> JSONResponse:
    database, error = _open_database(request)
    if error is not None:
        return error
    assert database is not None
    try:
        repository = ModelRepository(database)
        model = repository.get(model_id)
        if model is None:
            return _error("model was not found", status_code=404)
        live_pipeline = getattr(request.app.state, "live_pipeline", None)
        if live_pipeline is not None and live_pipeline.is_running:
            active_model_id = live_pipeline.model_id
            if active and active_model_id != model_id:
                return _error(
                    "stop the live pipeline before switching the model",
                    status_code=409,
                )
            if not active and active_model_id == model_id:
                return _error(
                    "stop the live pipeline before switching the model",
                    status_code=409,
                )
        if active:
            if model.validation_state != RUNTIME_VALID:
                return _error(
                    "runtime model validation is required before activation",
                    status_code=409,
                )
            if not _artifact_exists(request.app.state.paths, model):
                return _error("model artifact is missing", status_code=409)
        updated = repository.set_active(model_id, active)
        if updated is None:
            return _error("model was removed during activation", status_code=409)
        return _json(
            model_payload(
                updated,
                artifact_exists=_artifact_exists(request.app.state.paths, updated),
            )
        )
    finally:
        database.dispose()


async def _requested_active_state(request: Request, *, default: bool) -> bool:
    """Accept an optional explicit active flag while keeping empty POSTs simple."""
    if "application/json" not in request.headers.get("content-type", "").lower():
        return default
    try:
        body = await request.json()
    except ValueError as error:
        raise ModelImportError("activation request must be valid JSON") from error
    if not isinstance(body, dict) or "active" not in body:
        return default
    active = body["active"]
    if not isinstance(active, bool):
        raise ModelImportError("active must be a boolean")
    return active


async def _read_detection_request(
    request: Request,
) -> tuple[bytes, float | None, BackendOptions]:
    content_type = request.headers.get("content-type", "").lower()
    if not content_type.startswith("multipart/form-data"):
        raise ModelImportError("image detection requires multipart form data")
    try:
        received_bytes = 0

        async def limited_receive() -> Any:
            nonlocal received_bytes
            message = await request.receive()
            if message.get("type") == "http.request":
                received_bytes += len(message.get("body", b""))
                if received_bytes > MAX_STILL_IMAGE_REQUEST_BYTES:
                    raise ModelImportError("image upload exceeds the size limit")
            return message

        limited_request = Request(request.scope, limited_receive)
        form_context = limited_request.form(
            max_files=1,
            max_fields=4,
            max_part_size=MAX_STILL_IMAGE_BYTES,
        )
        async with form_context as form:
            upload = form.get("image") or form.get("file")
            if not isinstance(upload, UploadFile):
                raise ModelImportError("image detection requires an image file")
            image_bytes = await _read_upload(
                upload,
                MAX_STILL_IMAGE_BYTES,
                "image",
            )
            threshold_value = form.get("confidence_threshold")
            compute_units_value = form.get("compute_units")
    except MultiPartException:
        raise
    except ModelImportError:
        raise
    except (OSError, ValueError):
        raise ModelImportError("image upload could not be read") from None

    threshold = _parse_confidence_threshold(threshold_value)
    compute_units = (
        ComputeUnit.ALL
        if compute_units_value in (None, "")
        else ComputeUnit.parse(str(compute_units_value))
    )
    return image_bytes, threshold, BackendOptions(compute_units=compute_units)


def _parse_confidence_threshold(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(str(value))
    except (TypeError, ValueError):
        raise ModelImportError("confidence_threshold must be a number from 0 through 1") from None
    if not math.isfinite(parsed) or not 0 <= parsed <= 1:
        raise ModelImportError("confidence_threshold must be a number from 0 through 1")
    return parsed


class _ModelNotFoundError(ModelImportError):
    """Raised inside the worker thread when a model record is absent."""


def _detect_image_sync(
    model_id: str,
    paths: ManagedPaths,
    registry: DetectorRegistry,
    image_bytes: bytes,
    threshold: float | None,
    options: BackendOptions,
) -> dict[str, Any]:
    """Run all synchronous model and image work in a worker thread."""
    total_started = perf_counter()
    database = Database(paths.database)
    try:
        repository = ModelRepository(database)
        model = repository.get(model_id)
        if model is None:
            raise _ModelNotFoundError("model was not found")
        if model.validation_state != RUNTIME_VALID:
            raise ModelImportError("runtime model validation is required before image detection")

        manifest = manifest_from_record(model)
        if manifest.artifact_sha256 != model.artifact_sha256:
            raise ModelImportError("model manifest and registry provenance are inconsistent")
        artifact_path = ManagedPaths.validate_contained_path(
            paths.models / model.artifact_path,
            paths.models,
        )
        if (
            not artifact_path.is_dir()
            or artifact_path.is_symlink()
            or artifact_path.name != manifest.artifact
        ):
            raise ModelImportError("managed model artifact is missing or invalid")
        if compute_artifact_sha256(artifact_path) != model.artifact_sha256:
            raise ModelImportError("managed model checksum does not match its provenance")

        decoded_started = perf_counter()
        decoded = decode_still_image(image_bytes)
        image_decode_ms = (perf_counter() - decoded_started) * 1000
        descriptor = _descriptor_from_model(model, manifest, artifact_path)
        run, reloaded = registry.detect(
            decoded.image,
            descriptor,
            options,
            confidence_threshold=threshold,
        )
        total_ms = (perf_counter() - total_started) * 1000
        effective_threshold = (
            threshold
            if threshold is not None
            else float(manifest.raw["defaults"]["confidence_threshold"])
        )
        return {
            "image_base64": base64.b64encode(decoded.raw_bytes).decode("ascii"),
            "image_content_type": decoded.content_type,
            "source_width": decoded.image.width,
            "source_height": decoded.image.height,
            "detections": [
                {
                    "box_xyxy": list(detection.box_xyxy),
                    "class_id": detection.class_id,
                    "label": detection.label,
                    "confidence": detection.confidence,
                    "model_id": detection.model_id,
                    "model_checksum": detection.model_sha256,
                }
                for detection in run.batch.detections
            ],
            "rejected_candidates": run.batch.rejected_count,
            "model_id": model.id,
            "model_checksum": model.artifact_sha256,
            "compute_units": options.compute_units.value,
            "compute_units_display": options.compute_units.display_name,
            "confidence_threshold": effective_threshold,
            "model_reload": {
                "reloaded": reloaded,
                "instance_id": run.model_instance_id,
            },
            "inspection": run.inspection.as_dict(),
            "transform": run.transform.as_dict(),
            "timings": {
                "image_decode_ms": _milliseconds(image_decode_ms),
                "model_load_ms": _milliseconds(run.model_load_ms),
                "preprocessing_ms": _milliseconds(image_decode_ms + run.preprocessing_ms),
                "inference_ms": _milliseconds(run.inference_ms),
                "postprocessing_ms": _milliseconds(run.postprocessing_ms),
                "total_ms": _milliseconds(total_ms),
            },
        }
    finally:
        database.dispose()


def _descriptor_from_model(
    model: Model,
    manifest: ModelManifest,
    artifact_path: Path,
) -> ModelDescriptor:
    return ModelDescriptor(
        model_id=model.id,
        artifact_path=str(artifact_path),
        artifact_sha256=model.artifact_sha256,
        manifest=manifest,
    )


def _milliseconds(value: float) -> float:
    return round(max(0.0, value), 3)


def _content_length(request: Request) -> int | None:
    value = request.headers.get("content-length")
    if value is None:
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed >= 0 else None


async def _read_import_request(request: Request) -> tuple[bytes | str | dict[str, Any], Path]:
    content_type = request.headers.get("content-type", "").lower()
    if content_type.startswith("multipart/form-data"):
        async with request.form() as form:
            manifest = form.get("manifest") or form.get("manifest_text")
            archive = form.get("archive") or form.get("package")
            if not isinstance(archive, UploadFile):
                raise ModelImportError("import requires manifest and archive files")
            if isinstance(manifest, UploadFile):
                manifest_value: bytes | str | dict[str, Any] = await _read_upload(
                    manifest,
                    MAX_MANIFEST_BYTES,
                    "manifest",
                )
            elif isinstance(manifest, str):
                manifest_value = manifest
            else:
                raise ModelImportError("import requires a manifest file or text field")
            archive_path = await _spool_upload(request.app.state.paths, archive)
            return manifest_value, archive_path

    try:
        body = await request.json()
    except ValueError as error:
        raise ModelImportError("import request must be multipart form data") from error
    if not isinstance(body, dict):
        raise ModelImportError("import request must be an object")
    json_manifest = body.get("manifest")
    if not isinstance(json_manifest, (dict, str)):
        raise ModelImportError("JSON import requires a manifest object or text value")
    manifest_value = json_manifest
    archive_encoded = body.get("archive_base64", body.get("archive"))
    if not isinstance(archive_encoded, str):
        raise ModelImportError("JSON import requires archive_base64")
    if len(archive_encoded) > MAX_ARCHIVE_BYTES * 2:
        raise ModelImportError("model archive exceeds the upload size limit")
    try:
        archive_bytes = base64.b64decode(archive_encoded, validate=True)
    except (ValueError, binascii.Error) as error:
        raise ModelImportError("archive_base64 is invalid") from error
    if len(archive_bytes) > MAX_ARCHIVE_BYTES:
        raise ModelImportError("model archive exceeds the upload size limit")
    request.app.state.paths.ensure_directories()
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix="upload-",
            suffix=".zip",
            dir=request.app.state.paths.staging,
            delete=False,
        ) as handle:
            archive_path = Path(handle.name)
            handle.write(archive_bytes)
    except OSError:
        raise ModelImportError("model archive could not be staged") from None
    return manifest_value, archive_path


async def _read_upload(upload: UploadFile, maximum: int, label: str) -> bytes:
    chunks: list[bytes] = []
    size = 0
    while True:
        chunk = await upload.read(MAX_UPLOAD_CHUNK)
        if not chunk:
            break
        size += len(chunk)
        if size > maximum:
            raise ModelImportError(f"{label} exceeds the upload size limit")
        chunks.append(chunk)
    return b"".join(chunks)


async def _spool_upload(paths: ManagedPaths, upload: UploadFile) -> Path:
    paths.ensure_directories()
    size = 0
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix="upload-",
            suffix=".zip",
            dir=paths.staging,
            delete=False,
        ) as handle:
            archive_path = Path(handle.name)
            while True:
                chunk = await upload.read(MAX_UPLOAD_CHUNK)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_ARCHIVE_BYTES:
                    raise ModelImportError("model archive exceeds the upload size limit")
                handle.write(chunk)
    except (OSError, ModelImportError):
        archive_path.unlink(missing_ok=True)
        raise
    return archive_path


def _artifact_exists(paths: ManagedPaths, model: Any) -> bool:
    try:
        path = ManagedPaths.validate_contained_path(
            paths.models / model.artifact_path,
            paths.models,
        )
    except ValueError:
        return False
    return path.is_dir() and not path.is_symlink()


def _install_model_catalog_sync(
    catalog: ModelCatalog,
    entry: CatalogEntry,
    paths: ManagedPaths,
    repository: ModelRepository,
    downloader: CatalogDownloader,
) -> tuple[Model, bool]:
    """Run catalog network, file, and import work outside the event loop."""
    return install_catalog_model(
        catalog=catalog,
        entry=entry,
        paths=paths,
        repository=repository,
        downloader=downloader,
    )


def _catalog_model_matches(model: Model, entry: CatalogEntry) -> bool:
    if model.id != entry.manifest.model_id or model.artifact_sha256 != entry.package_sha256:
        return False
    try:
        return manifest_from_record(model).snapshot_json == entry.manifest.snapshot_json
    except (ValueError, TypeError):
        return False


def _open_database(request: Request) -> tuple[Database | None, JSONResponse | None]:
    paths = request.app.state.paths
    status = database_status(paths.database)
    if status["status"] != "ok":
        return None, _error(
            "database is not ready; run `open-licenseplate db upgrade` first",
            status_code=409,
        )
    return Database(paths.database), None


def _json(payload: object, *, status_code: int = 200) -> JSONResponse:
    return JSONResponse(content=payload, status_code=status_code)


def _error(message: str, *, status_code: int) -> JSONResponse:
    return _json({"detail": redact_text(message)}, status_code=status_code)
