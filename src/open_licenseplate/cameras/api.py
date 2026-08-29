"""Camera configuration JSON API."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from ..capture import ActiveCameraConflict, SourceFactory, encode_jpeg, preview_chunks
from ..database import Database, database_status
from ..redaction import redact_text
from .credentials import credential_status, parse_credential_ref
from .repository import Camera, CameraRepository, camera_config_from_record
from .service import (
    CameraConfigurationError,
    prepare_camera_config,
    test_camera_configuration,
    test_camera_connection,
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
        live_pipeline = getattr(request.app.state, "live_pipeline", None)
        if (
            live_pipeline is not None
            and live_pipeline.is_running
            and live_pipeline.camera_id == camera_id
        ):
            return _error(
                "stop the live pipeline before changing its active camera settings",
                status_code=409,
            )
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
        live_pipeline = getattr(request.app.state, "live_pipeline", None)
        if (
            live_pipeline is not None
            and live_pipeline.is_running
            and live_pipeline.camera_id == camera_id
        ):
            return _error(
                "stop the live pipeline before deleting its active camera",
                status_code=409,
            )
        if not CameraRepository(database).delete(camera_id):
            return _error("camera was not found", status_code=404)
        return _json({"deleted": True, "camera_id": camera_id})
    finally:
        database.dispose()


@router.post("/{camera_id}/test")
async def test_camera(camera_id: str, request: Request) -> JSONResponse:
    database, error = _open_database(request)
    if error is not None:
        return error
    assert database is not None
    try:
        camera = CameraRepository(database).get(camera_id)
        if camera is None:
            return _error("camera was not found", status_code=404)
        source_factory = request.app.state.camera_source_factory
        result = await asyncio.to_thread(
            test_camera_connection,
            camera,
            source_factory=source_factory,
        )
        return _json(result.as_dict(camera))
    finally:
        database.dispose()


@router.post("/{camera_id}/start")
async def start_camera(camera_id: str, request: Request) -> JSONResponse:
    database, error = _open_database(request)
    if error is not None:
        return error
    assert database is not None
    try:
        camera = CameraRepository(database).get(camera_id)
        if camera is None:
            return _error("camera was not found", status_code=404)
        config = camera_config_from_record(camera)
    finally:
        database.dispose()

    try:
        status = request.app.state.camera_runtime.start(camera.id, config)
    except ActiveCameraConflict as exception:
        return _error(str(exception), status_code=409)
    except RuntimeError as exception:
        return _error(str(exception), status_code=409)
    return _json(status.as_dict())


@router.post("/{camera_id}/stop")
async def stop_camera(camera_id: str, request: Request) -> JSONResponse:
    database, error = _open_database(request)
    if error is not None:
        return error
    assert database is not None
    try:
        if CameraRepository(database).get(camera_id) is None:
            return _error("camera was not found", status_code=404)
    finally:
        database.dispose()

    try:
        live_pipeline = getattr(request.app.state, "live_pipeline", None)
        if (
            live_pipeline is not None
            and live_pipeline.is_running
            and live_pipeline.camera_id == camera_id
        ):
            return _error(
                "stop the live pipeline before stopping its active camera",
                status_code=409,
            )
        status = await asyncio.to_thread(
            request.app.state.camera_runtime.stop,
            camera_id,
        )
    except RuntimeError as exception:
        return _error(str(exception), status_code=409)
    return _json(status.as_dict())


@router.get("/{camera_id}/status")
async def camera_status(camera_id: str, request: Request) -> JSONResponse:
    database, error = _open_database(request)
    if error is not None:
        return error
    assert database is not None
    try:
        if CameraRepository(database).get(camera_id) is None:
            return _error("camera was not found", status_code=404)
    finally:
        database.dispose()
    return _json(request.app.state.camera_runtime.status(camera_id).as_dict())


@router.get("/{camera_id}/preview.mjpeg")
async def camera_preview(camera_id: str, request: Request) -> StreamingResponse:
    database, error = _open_database(request)
    if error is not None:
        raise HTTPException(
            status_code=error.status_code,
            detail="database is not ready; run `open-licenseplate db upgrade` first",
        )
    assert database is not None
    try:
        if CameraRepository(database).get(camera_id) is None:
            raise _http_error("camera was not found", 404)
    finally:
        database.dispose()

    runtime = request.app.state.camera_runtime
    if runtime.active_camera_id != camera_id:
        raise _http_error(
            "camera is not streaming; start it before opening the preview",
            409,
        )
    chunks = preview_chunks(runtime.iter_preview(camera_id))
    return StreamingResponse(
        chunks,
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


@router.get("/{camera_id}/snapshot.jpg")
async def camera_snapshot(camera_id: str, request: Request) -> Response:
    database, error = _open_database(request)
    if error is not None:
        raise HTTPException(
            status_code=error.status_code,
            detail="database is not ready; run `open-licenseplate db upgrade` first",
        )
    assert database is not None
    try:
        if CameraRepository(database).get(camera_id) is None:
            raise _http_error("camera was not found", 404)
    finally:
        database.dispose()

    runtime = request.app.state.camera_runtime
    frame = runtime.latest_frame(camera_id)
    if frame is None:
        raise _http_error(
            "no decoded frame is available; start the camera and wait for a frame",
            409,
        )
    try:
        jpeg = await asyncio.to_thread(encode_jpeg, frame)
    except Exception:
        raise _http_error("the current frame could not be encoded as JPEG", 503) from None
    return Response(
        content=jpeg,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store"},
    )


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


def camera_test_payload(
    camera: Camera,
    *,
    source_factory: SourceFactory | None = None,
    network: bool = False,
) -> dict[str, object]:
    """Build the public test representation for HTML routes."""
    result = (
        test_camera_connection(camera, source_factory=source_factory)
        if network
        else test_camera_configuration(camera)
    )
    return result.as_dict(camera)


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


def _http_error(message: str, status_code: int) -> Exception:
    return HTTPException(status_code=status_code, detail=redact_text(message))


def _timestamp(value: Any) -> str:
    if not isinstance(value, datetime):
        raise ValueError("camera timestamp is invalid")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
