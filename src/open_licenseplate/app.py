"""FastAPI application factory, page routes, and health endpoints."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from urllib.parse import parse_qs, quote

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .cameras.api import camera_test_payload
from .cameras.api import router as camera_api_router
from .cameras.repository import CameraRepository
from .cameras.service import CameraConfigurationError, prepare_camera_config
from .capture import CameraRuntime, PyAVRTSPSource, SourceFactory
from .config import AppSettings, UISettings, load_settings
from .database import Database, database_status
from .logging import configure_logging
from .models.api import _read_import_request
from .models.api import router as model_api_router
from .models.repository import ModelRepository
from .models.service import (
    ModelConflictError,
    ModelImportError,
    delete_model,
    import_model,
    validate_model,
)
from .paths import ManagedPaths
from .redaction import redact_text
from .settings_store import SettingsStore
from .web import STATIC_DIRECTORY, render_page

logger = logging.getLogger("open_licenseplate")
CONTENT_SECURITY_POLICY = (
    "default-src 'self'; "
    "script-src 'self'; "
    "style-src 'self'; "
    "img-src 'self' data:; "
    "font-src 'self'; "
    "connect-src 'self'; "
    "object-src 'none'; "
    "base-uri 'self'; "
    "frame-ancestors 'none'; "
    "form-action 'self'"
)


def _readiness_payload(settings: AppSettings, paths: ManagedPaths) -> tuple[dict[str, Any], int]:
    directories = paths.directory_checks()
    directories_ready = all(directories.values())
    database = database_status(paths.database)
    database_ready = database["status"] == "ok"
    ready = directories_ready and database_ready
    payload = {
        "status": "ready" if ready else "not_ready",
        "checks": {
            "configuration": {"status": "ok"},
            "directories": {
                "status": "ok" if directories_ready else "not_ready",
                "details": directories,
            },
            "database": database,
        },
        "settings": {
            "server.host": {
                "value": settings.server.host,
                "source": settings.sources.get("server.host", "unknown"),
            },
            "server.port": {
                "value": settings.server.port,
                "source": settings.sources.get("server.port", "unknown"),
            },
        },
    }
    return payload, 200 if ready else 503


def _default_source_factory(camera: Any, camera_id: str) -> Any:
    """Create the production RTSP source at the runtime boundary."""
    return PyAVRTSPSource(camera, camera_id=camera_id)


def create_app(
    settings: AppSettings | None = None,
    *,
    source_factory: SourceFactory | None = None,
) -> FastAPI:
    """Create the application without starting a server."""
    effective_settings = settings or load_settings()
    paths = ManagedPaths.from_settings(effective_settings)
    effective_source_factory = source_factory or _default_source_factory
    camera_runtime = CameraRuntime(effective_source_factory)

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        paths.ensure_directories()
        configure_logging(level=effective_settings.log_level, log_file=paths.app_log)
        application.state.startup_complete = True
        logger.info(
            "application started",
            extra={
                "app_name": effective_settings.app_name,
                "environment": effective_settings.environment,
            },
        )
        try:
            yield
        finally:
            await asyncio.to_thread(camera_runtime.close)
            application.state.startup_complete = False
            logger.info("application stopped")

    application = FastAPI(
        title=effective_settings.app_name,
        version=__version__,
        lifespan=lifespan,
    )
    application.state.settings = effective_settings
    application.state.paths = paths
    application.state.startup_complete = False
    application.state.camera_runtime = camera_runtime
    application.state.camera_source_factory = effective_source_factory
    application.mount("/static", StaticFiles(directory=str(STATIC_DIRECTORY)), name="static")
    application.include_router(camera_api_router)
    application.include_router(model_api_router)

    @application.exception_handler(RequestValidationError)
    async def request_validation_error(
        request: Request,
        exception: RequestValidationError,
    ) -> JSONResponse:
        del request, exception
        return JSONResponse(
            content={"detail": "request body is invalid"},
            status_code=422,
        )

    @application.exception_handler(HTTPException)
    async def http_error(request: Request, exception: HTTPException) -> JSONResponse:
        del request
        return JSONResponse(
            content={"detail": redact_text(str(exception.detail))},
            status_code=exception.status_code,
            headers=exception.headers,
        )

    @application.middleware("http")
    async def add_security_headers(request: Request, call_next: Any) -> Any:
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = CONTENT_SECURITY_POLICY
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    @application.get("/", include_in_schema=False)
    async def home() -> RedirectResponse:
        return RedirectResponse("/live", status_code=307)

    @application.get("/live", name="live_page", include_in_schema=False)
    async def live_page(request: Request) -> Any:
        return render_page(request, effective_settings, paths, "live")

    @application.get("/events", name="events_page", include_in_schema=False)
    async def events_page(request: Request) -> Any:
        return render_page(request, effective_settings, paths, "events")

    @application.get("/jobs", name="jobs_page", include_in_schema=False)
    async def jobs_page(request: Request) -> Any:
        return render_page(request, effective_settings, paths, "jobs")

    @application.get("/cameras", name="cameras_page", include_in_schema=False)
    async def cameras_page(request: Request) -> Any:
        return render_page(request, effective_settings, paths, "cameras")

    @application.post("/cameras", include_in_schema=False)
    async def create_camera_from_page(request: Request) -> RedirectResponse:
        form = _form_values(await request.body())
        try:
            config = prepare_camera_config(
                name=form.get("name", ""),
                rtsp_url=form.get("rtsp_url", ""),
                credential_ref=form.get("credential_ref"),
                transport=form.get("transport", "tcp"),
                preferred_stream=form.get("preferred_stream", "main"),
                enabled=True,
            )
            _create_camera(paths, config)
        except (CameraConfigurationError, ValueError) as error:
            return _camera_redirect("error", str(error))
        return _camera_redirect("created")

    @application.post("/cameras/{camera_id}/edit", include_in_schema=False)
    async def update_camera_from_page(camera_id: str, request: Request) -> RedirectResponse:
        form = _form_values(await request.body())
        database = _ready_database(paths)
        if database is None:
            return _camera_redirect("database")
        try:
            repository = CameraRepository(database)
            camera = repository.get(camera_id)
            if camera is None:
                return _camera_redirect("missing")
            config = prepare_camera_config(
                name=form.get("name", camera.name),
                rtsp_url=form.get("rtsp_url") or camera.endpoint,
                credential_ref=form.get("credential_ref") or camera.credential_ref,
                transport=form.get("transport", "tcp"),
                connection_options=None,
                preferred_stream=form.get("preferred_stream", camera.preferred_stream),
                enabled="enabled" in form,
            )
            repository.update(camera, config)
        except (CameraConfigurationError, ValueError) as error:
            return _camera_redirect("error", str(error))
        finally:
            database.dispose()
        return _camera_redirect("updated")

    @application.post("/cameras/{camera_id}/test", include_in_schema=False)
    async def test_camera_from_page(camera_id: str) -> RedirectResponse:
        database = _ready_database(paths)
        if database is None:
            return _camera_redirect("database")
        try:
            camera = CameraRepository(database).get(camera_id)
            if camera is None:
                return _camera_redirect("missing")
            result = await asyncio.to_thread(
                camera_test_payload,
                camera,
                source_factory=effective_source_factory,
                network=True,
            )
        finally:
            database.dispose()
        return _camera_redirect(
            "test",
            str(result.get("status", "invalid")),
            str(result.get("message", "Camera test completed.")),
        )

    @application.post("/cameras/{camera_id}/delete", include_in_schema=False)
    async def delete_camera_from_page(camera_id: str) -> RedirectResponse:
        database = _ready_database(paths)
        if database is None:
            return _camera_redirect("database")
        try:
            deleted = CameraRepository(database).delete(camera_id)
        finally:
            database.dispose()
        return _camera_redirect("deleted" if deleted else "missing")

    @application.get("/models", name="models_page", include_in_schema=False)
    async def models_page(request: Request) -> Any:
        return render_page(request, effective_settings, paths, "models")

    @application.post("/models/import", include_in_schema=False)
    async def import_model_from_page(request: Request) -> RedirectResponse:
        if database_status(paths.database)["status"] != "ok":
            return _model_redirect("database")
        archive_path = None
        database = None
        try:
            manifest_value, archive_path = await _read_import_request(request)
            database = Database(paths.database)
            import_model(
                manifest_value=manifest_value,
                source_path=archive_path,
                paths=paths,
                repository=ModelRepository(database),
            )
        except ModelConflictError as error:
            return _model_redirect("error", str(error))
        except (ModelImportError, ValueError) as error:
            return _model_redirect("error", str(error))
        finally:
            if archive_path is not None:
                archive_path.unlink(missing_ok=True)
            if database is not None:
                database.dispose()
        return _model_redirect("imported")

    @application.post("/models/{model_id}/validate", include_in_schema=False)
    async def validate_model_from_page(model_id: str) -> RedirectResponse:
        database = _ready_database(paths)
        if database is None:
            return _model_redirect("database")
        try:
            repository = ModelRepository(database)
            model = repository.get(model_id)
            if model is None:
                return _model_redirect("missing")
            result = validate_model(model=model, paths=paths, repository=repository)
        except ModelImportError as error:
            return _model_redirect("error", str(error))
        finally:
            database.dispose()
        return _model_redirect("validated" if result.valid else "invalid")

    @application.post("/models/{model_id}/activate", include_in_schema=False)
    async def activate_model_from_page(model_id: str) -> RedirectResponse:
        return _model_state_from_page(model_id, paths, active=True)

    @application.post("/models/{model_id}/deactivate", include_in_schema=False)
    async def deactivate_model_from_page(model_id: str) -> RedirectResponse:
        return _model_state_from_page(model_id, paths, active=False)

    @application.post("/models/{model_id}/delete", include_in_schema=False)
    async def delete_model_from_page(model_id: str) -> RedirectResponse:
        database = _ready_database(paths)
        if database is None:
            return _model_redirect("database")
        try:
            repository = ModelRepository(database)
            model = repository.get(model_id)
            if model is None:
                return _model_redirect("missing")
            delete_model(model=model, paths=paths, repository=repository)
        except ModelImportError as error:
            return _model_redirect("error", str(error))
        finally:
            database.dispose()
        return _model_redirect("deleted")

    @application.get("/system", name="system_page", include_in_schema=False)
    async def system_page(request: Request) -> Any:
        return render_page(request, effective_settings, paths, "system")

    @application.post("/system/preferences", include_in_schema=False)
    async def save_system_preferences(request: Request) -> RedirectResponse:
        form = parse_qs((await request.body()).decode("utf-8"))
        density = form.get("density", [""])[0]
        try:
            updated_ui = UISettings.model_validate({"density": density})
        except ValueError as error:
            raise HTTPException(
                status_code=400,
                detail="density must be comfortable or compact",
            ) from error

        status = database_status(paths.database)
        if status["status"] != "ok":
            raise HTTPException(
                status_code=409,
                detail="database is not ready; run `open-licenseplate db upgrade` first",
            )

        database = Database(paths.database)
        try:
            SettingsStore(database).set("ui.density", updated_ui.density)
        finally:
            database.dispose()

        effective_settings.ui.density = updated_ui.density
        effective_settings._sources["ui.density"] = "persisted"
        return RedirectResponse("/system", status_code=303)

    @application.get("/api/v1/health/live")
    async def live_health() -> dict[str, str]:
        return {"status": "ok"}

    @application.get("/api/v1/health/ready")
    async def ready_health() -> JSONResponse:
        payload, status_code = _readiness_payload(effective_settings, paths)
        return JSONResponse(content=payload, status_code=status_code)

    @application.get("/api/v1/live/state")
    async def live_state() -> dict[str, Any]:
        return camera_runtime.status().as_dict()

    @application.get("/api/v1/system/status")
    async def system_status() -> dict[str, Any]:
        database = database_status(paths.database)
        return {
            "application": effective_settings.app_name,
            "version": __version__,
            "database": database,
            "directories": paths.directory_checks(),
            "runtime": camera_runtime.status().as_dict(),
        }

    @application.get("/api/v1/system/metrics")
    async def system_metrics() -> dict[str, Any]:
        return {
            "camera": camera_runtime.status().as_dict(),
        }

    return application


def _form_values(body: bytes) -> dict[str, str]:
    parsed = parse_qs(body.decode("utf-8"), keep_blank_values=True)
    return {key: values[0] for key, values in parsed.items() if values}


def _ready_database(paths: ManagedPaths) -> Database | None:
    if database_status(paths.database)["status"] != "ok":
        return None
    return Database(paths.database)


def _create_camera(paths: ManagedPaths, config: Any) -> None:
    database = _ready_database(paths)
    if database is None:
        raise CameraConfigurationError(
            "database is not ready; run `open-licenseplate db upgrade` first"
        )
    try:
        CameraRepository(database).create(config)
    finally:
        database.dispose()


def _camera_redirect(kind: str, status: str = "", message: str = "") -> RedirectResponse:
    query = f"notice={quote(redact_text(kind), safe='')}"
    if status:
        query += f"&status={quote(redact_text(status), safe='')}"
    if message:
        query += f"&message={quote(redact_text(message), safe='')}"
    return RedirectResponse(f"/cameras?{query}", status_code=303)


def _model_state_from_page(
    model_id: str,
    paths: ManagedPaths,
    *,
    active: bool,
) -> RedirectResponse:
    database = _ready_database(paths)
    if database is None:
        return _model_redirect("database")
    try:
        repository = ModelRepository(database)
        model = repository.get(model_id)
        if model is None:
            return _model_redirect("missing")
        if active and model.validation_state != "valid":
            return _model_redirect("error", "only valid models can be activated")
        if active:
            artifact_path = ManagedPaths.validate_contained_path(
                paths.models / model.artifact_path,
                paths.models,
            )
            if not artifact_path.is_dir() or artifact_path.is_symlink():
                return _model_redirect("error", "model artifact is missing")
        updated = repository.set_active(model_id, active)
        if updated is None:
            return _model_redirect("missing")
    except (ModelImportError, ValueError) as error:
        return _model_redirect("error", str(error))
    finally:
        database.dispose()
    return _model_redirect("activated" if active else "deactivated")


def _model_redirect(kind: str, message: str = "") -> RedirectResponse:
    query = f"notice={quote(redact_text(kind), safe='')}"
    if message:
        query += f"&message={quote(redact_text(message), safe='')}"
    return RedirectResponse(f"/models?{query}", status_code=303)
