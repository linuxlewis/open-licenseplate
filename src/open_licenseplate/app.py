"""FastAPI application factory, page routes, and health endpoints."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .config import AppSettings, load_settings
from .database import database_status
from .logging import configure_logging
from .paths import ManagedPaths
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


def create_app(settings: AppSettings | None = None) -> FastAPI:
    """Create the application without starting a server."""
    effective_settings = settings or load_settings()
    paths = ManagedPaths.from_settings(effective_settings)

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
    application.mount("/static", StaticFiles(directory=str(STATIC_DIRECTORY)), name="static")

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

    @application.get("/models", name="models_page", include_in_schema=False)
    async def models_page(request: Request) -> Any:
        return render_page(request, effective_settings, paths, "models")

    @application.get("/system", name="system_page", include_in_schema=False)
    async def system_page(request: Request) -> Any:
        return render_page(request, effective_settings, paths, "system")

    @application.get("/api/v1/health/live")
    async def live_health() -> dict[str, str]:
        return {"status": "ok"}

    @application.get("/api/v1/health/ready")
    async def ready_health() -> JSONResponse:
        payload, status_code = _readiness_payload(effective_settings, paths)
        return JSONResponse(content=payload, status_code=status_code)

    return application
