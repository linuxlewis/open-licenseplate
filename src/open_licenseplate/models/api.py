"""Managed model registry JSON API."""

from __future__ import annotations

import base64
import binascii
import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from starlette.datastructures import UploadFile

from ..database import Database, database_status
from ..paths import ManagedPaths
from ..redaction import redact_text
from .archive import MAX_ARCHIVE_BYTES
from .repository import ModelRepository, model_payload
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
        result = validate_model(
            model=model,
            paths=request.app.state.paths,
            repository=repository,
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
