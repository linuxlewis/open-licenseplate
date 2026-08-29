"""Camera configuration JSON API."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from ..database import Database, database_status
from ..redaction import redact_text
from .credentials import credential_status, parse_credential_ref
from .repository import Camera, CameraRepository, camera_config_from_record
from .service import (
    CameraConfigurationError,
    prepare_camera_config,
    test_camera_configuration,
)

router = APIRouter(prefix="/api/v1/cameras", tags=["cameras"])
_SECRET_INPUT_KEYS = frozenset(
    {"password", "passwd", "secret", "token", "api_key", "api-key", "authorization"}
)


@router.get("")
async def list_cameras(request: Request) -> JSONResponse:
    database, error = _open_database(request)
    if error is not None:
        return error
    assert database is not None
    try:
        cameras = CameraRepository(database).list()
        return _json({"cameras": [_camera_payload(camera) for camera in cameras]})
    finally:
        database.dispose()


@router.post("", status_code=201)
async def create_camera(request: Request) -> JSONResponse:
    body, error = await _read_body(request)
    if error is not None:
        return error
    try:
        _reject_secret_inputs(body)
        config = prepare_camera_config(**_camera_values(body))
    except (CameraConfigurationError, ValueError) as exception:
        return _error(str(exception), status_code=422)

    database, error = _open_database(request)
    if error is not None:
        return error
    assert database is not None
    try:
        camera = CameraRepository(database).create(config)
        return _json(_camera_payload(camera), status_code=201)
    finally:
        database.dispose()


@router.get("/{camera_id}")
async def get_camera(camera_id: str, request: Request) -> JSONResponse:
    database, error = _open_database(request)
    if error is not None:
        return error
    assert database is not None
    try:
        camera = CameraRepository(database).get(camera_id)
        if camera is None:
            return _error("camera was not found", status_code=404)
        return _json(_camera_payload(camera))
    finally:
        database.dispose()


@router.patch("/{camera_id}")
async def update_camera(camera_id: str, request: Request) -> JSONResponse:
    body, error = await _read_body(request)
    if error is not None:
        return error
    try:
        _reject_secret_inputs(body)
    except CameraConfigurationError as exception:
        return _error(str(exception), status_code=422)

    database, error = _open_database(request)
    if error is not None:
        return error
    assert database is not None
    try:
        repository = CameraRepository(database)
        camera = repository.get(camera_id)
        if camera is None:
            return _error("camera was not found", status_code=404)
        try:
            config = prepare_camera_config(
                **_camera_values(
                    body,
                    current=camera,
                )
            )
        except (CameraConfigurationError, ValueError) as exception:
            return _error(str(exception), status_code=422)
        updated = repository.update(camera, config)
        return _json(_camera_payload(updated))
    finally:
        database.dispose()


@router.delete("/{camera_id}")
async def delete_camera(camera_id: str, request: Request) -> JSONResponse:
    database, error = _open_database(request)
    if error is not None:
        return error
    assert database is not None
    try:
        if not CameraRepository(database).delete(camera_id):
            return _error("camera was not found", status_code=404)
        return _json({"deleted": True, "camera_id": camera_id})
    finally:
        database.dispose()


@router.post("/{camera_id}/test")
async def test_camera(camera_id: str, request: Request) -> JSONResponse:
    # The test is deliberately configuration-only in this PR. Network capture
    # belongs to the later FrameSource and preview slice.
    return _test_camera_by_id(camera_id, request)


def _test_camera_by_id(camera_id: str, request: Request) -> JSONResponse:
    database, error = _open_database(request)
    if error is not None:
        return error
    assert database is not None
    try:
        camera = CameraRepository(database).get(camera_id)
        if camera is None:
            return _error("camera was not found", status_code=404)
        result = test_camera_configuration(camera)
        return _json(result.as_dict(camera))
    finally:
        database.dispose()


def camera_payload(camera: Camera) -> dict[str, object]:
    """Build the public camera representation without resolving secrets."""
    return _camera_payload(camera)


def camera_test_payload(camera: Camera) -> dict[str, object]:
    """Build the public test representation for HTML routes."""
    return test_camera_configuration(camera).as_dict(camera)


def _camera_payload(camera: Camera) -> dict[str, object]:
    config = camera_config_from_record(camera)
    reference = parse_credential_ref(config.credential_ref)
    return {
        "id": camera.id,
        "name": redact_text(camera.name),
        "endpoint": redact_text(camera.endpoint),
        "rtsp_url": redact_text(camera.endpoint),
        "credential_ref": config.credential_ref,
        "credential": credential_status(reference),
        "connection_options": config.connection_options,
        "preferred_stream": config.preferred_stream,
        "region_of_interest": config.region_of_interest,
        "enabled": config.enabled,
        "created_at": _timestamp(camera.created_at),
        "updated_at": _timestamp(camera.updated_at),
    }


def _camera_values(
    body: Mapping[str, Any],
    *,
    current: Camera | None = None,
) -> dict[str, Any]:
    current_config = None if current is None else camera_config_from_record(current)
    endpoint = _first(body, "rtsp_url", "url", "endpoint")
    if endpoint is None and current_config is not None:
        endpoint = current_config.endpoint
    if endpoint is None:
        endpoint = ""

    credential_ref = body.get(
        "credential_ref",
        body.get(
            "credential_reference",
            None if current_config is None else current_config.credential_ref,
        ),
    )
    options = _first(body, "connection_options", "options")
    if options is None and current_config is not None:
        options = current_config.connection_options

    transport = body.get("transport")
    if transport is None and isinstance(options, dict):
        transport = options.get("transport", "tcp")
    if transport is None:
        transport = "tcp"

    return {
        "name": body.get("name", "" if current_config is None else current_config.name),
        "rtsp_url": endpoint,
        "credential_ref": credential_ref,
        "transport": transport,
        "connection_options": options,
        "preferred_stream": _first(
            body,
            "preferred_stream",
            "stream_profile",
        )
        or ("main" if current_config is None else current_config.preferred_stream),
        "region_of_interest": _first(
            body,
            "region_of_interest",
            "roi",
        )
        if any(key in body for key in ("region_of_interest", "roi"))
        else (None if current_config is None else current_config.region_of_interest),
        "enabled": body.get("enabled", True if current_config is None else current_config.enabled),
    }


def _first(values: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in values:
            return values[key]
    return None


def _reject_secret_inputs(body: Mapping[str, Any]) -> None:
    keys = {str(key).lower().replace("-", "_") for key in body}
    if keys & _SECRET_INPUT_KEYS:
        raise CameraConfigurationError(
            "camera secrets must be supplied through credential_ref, not in the request body"
        )
    for key in ("credentials", "credential"):
        if key in body:
            raise CameraConfigurationError(
                "camera secrets must be supplied through credential_ref, not in the request body"
            )


async def _read_body(request: Request) -> tuple[dict[str, Any], JSONResponse | None]:
    try:
        body = await request.json()
    except ValueError:
        return {}, _error("request body must be valid JSON", status_code=400)
    if not isinstance(body, dict):
        return {}, _error("request body must be a JSON object", status_code=422)
    return body, None


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


def _timestamp(value: Any) -> str:
    if not isinstance(value, datetime):
        raise ValueError("camera timestamp is invalid")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
