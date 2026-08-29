"""FastAPI application factory and health endpoints."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from html import escape
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

from .config import AppSettings, load_settings
from .database import database_status
from .logging import configure_logging
from .paths import ManagedPaths

logger = logging.getLogger("open_licenseplate")


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
        version="0.1.0",
        lifespan=lifespan,
    )
    application.state.settings = effective_settings
    application.state.paths = paths
    application.state.startup_complete = False

    @application.get("/", response_class=HTMLResponse)
    async def home() -> str:
        name = escape(effective_settings.app_name)
        return (
            "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            f"<title>{name}</title></head><body>"
            f"<h1>{name}</h1><p>Application shell is running.</p>"
            "<ul><li><a href='/api/v1/health/live'>Liveness</a></li>"
            "<li><a href='/api/v1/health/ready'>Readiness</a></li></ul>"
            "</body></html>"
        )

    @application.get("/api/v1/health/live")
    async def live_health() -> dict[str, str]:
        return {"status": "ok"}

    @application.get("/api/v1/health/ready")
    async def ready_health() -> JSONResponse:
        payload, status_code = _readiness_payload(effective_settings, paths)
        return JSONResponse(content=payload, status_code=status_code)

    return application
