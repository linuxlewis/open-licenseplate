"""Jinja templates and the M0 application shell."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import Request
from fastapi.templating import Jinja2Templates
from starlette.responses import Response

from . import __version__
from .cameras.api import camera_payload
from .cameras.repository import CameraRepository
from .config import AppSettings
from .database import Database, database_status
from .paths import ManagedPaths
from .redaction import redact_text

_WEB_DIRECTORY = Path(__file__).resolve().parent
STATIC_DIRECTORY = _WEB_DIRECTORY / "static"
templates = Jinja2Templates(directory=str(_WEB_DIRECTORY / "templates"))


@dataclass(frozen=True)
class PageDefinition:
    """Static content for one primary application page."""

    key: str
    label: str
    path: str
    eyebrow: str
    title: str
    description: str
    empty_title: str = ""
    empty_description: str = ""
    planned_milestone: str = ""


PAGES = (
    PageDefinition(
        key="live",
        label="Live",
        path="/live",
        eyebrow="Operations",
        title="Live view",
        description=(
            "The live camera surface will keep the current frame and detection state in view."
        ),
        empty_title="No live source is configured",
        empty_description=(
            "Camera connection and detection controls will appear here in a later milestone. "
            "The shell is ready for those controls."
        ),
        planned_milestone="M1 and M3",
    ),
    PageDefinition(
        key="events",
        label="Events",
        path="/events",
        eyebrow="Review",
        title="Plate events",
        description="Review confirmed plate appearances, evidence, and processing results.",
        empty_title="No plate events yet",
        empty_description=(
            "Events will appear after the live pipeline confirms a plate track. "
            "No event data is created by the application shell."
        ),
        planned_milestone="M4",
    ),
    PageDefinition(
        key="jobs",
        label="Jobs",
        path="/jobs",
        eyebrow="Recovery",
        title="Processing jobs",
        description="Inspect durable work, attempts, leases, and recovery actions.",
        empty_title="No processing jobs yet",
        empty_description=(
            "Durable jobs will appear when event processing is implemented. "
            "This empty state confirms that no work is waiting."
        ),
        planned_milestone="M5",
    ),
    PageDefinition(
        key="cameras",
        label="Cameras",
        path="/cameras",
        eyebrow="Sources",
        title="Camera sources",
        description=(
            "Save RTSP camera profiles with external credential references. "
            "The configuration test does not open a network stream in this slice."
        ),
        empty_title="No cameras configured",
        empty_description=(
            "Add a camera profile below. Passwords and complete secret-bearing RTSP URLs "
            "never enter the database or the browser."
        ),
        planned_milestone="M1-A",
    ),
    PageDefinition(
        key="models",
        label="Models",
        path="/models",
        eyebrow="Inference",
        title="Detection models",
        description="Manage validated detector packages, adapters, and provenance.",
        empty_title="No models imported",
        empty_description=(
            "Model import and validation will appear when the Core ML slice is implemented. "
            "No model is loaded by this shell."
        ),
        planned_milestone="M2 and M7",
    ),
    PageDefinition(
        key="system",
        label="System",
        path="/system",
        eyebrow="Diagnostics",
        title="System status",
        description="Inspect local readiness, configuration sources, and managed storage.",
    ),
)

PAGE_BY_KEY = {page.key: page for page in PAGES}


def _nav_items(active_key: str) -> list[dict[str, Any]]:
    return [
        {
            "label": page.label,
            "path": page.path,
            "active": page.key == active_key,
        }
        for page in PAGES
    ]


def _database_summary(paths: ManagedPaths) -> dict[str, Any]:
    status = database_status(paths.database)
    status["detail"] = redact_text(str(status["detail"]))
    return status


def _global_status(database: dict[str, Any], directories: dict[str, bool]) -> dict[str, str]:
    if database["status"] == "ok" and all(directories.values()):
        return {
            "label": "Ready",
            "detail": "Database and managed directories are ready.",
            "tone": "positive",
        }
    return {
        "label": "Setup needed",
        "detail": str(database["detail"]),
        "tone": "attention",
    }


def _setting_rows(settings: AppSettings) -> list[dict[str, str]]:
    values: tuple[tuple[str, str, Any], ...] = (
        ("Application name", "app_name", settings.app_name),
        ("Environment", "environment", settings.environment),
        ("Log level", "log_level", settings.log_level),
        ("Server host", "server.host", settings.server.host),
        ("Server port", "server.port", settings.server.port),
        ("UI density", "ui.density", settings.ui.density),
        (
            "Unsafe development binding",
            "server.unsafe_development",
            "Enabled" if settings.server.unsafe_development else "Disabled",
        ),
    )
    return [
        {
            "label": label,
            "key": key,
            "value": str(value),
            "source": settings.sources.get(key, "unknown"),
        }
        for label, key, value in values
    ]


def _path_rows(paths: ManagedPaths) -> list[dict[str, str]]:
    values = (
        ("Data directory", paths.data_dir),
        ("Database", paths.database),
        ("Models", paths.models),
        ("Artifacts", paths.artifacts),
        ("Staging", paths.staging),
        ("Settings", paths.settings),
        ("Log directory", paths.log_dir),
        ("Application log", paths.app_log),
        ("Worker log", paths.worker_log),
    )
    return [{"label": label, "value": str(path)} for label, path in values]


def _runtime_rows() -> list[dict[str, str]]:
    return [
        {
            "label": "Active camera",
            "value": "Not configured",
            "detail": "Camera support is not part of M0.",
        },
        {
            "label": "Active model",
            "value": "Not loaded",
            "detail": "Core ML support is not part of M0.",
        },
        {
            "label": "Worker",
            "value": "Not started",
            "detail": "Durable processing is not part of M0.",
        },
        {
            "label": "Unresolved failures",
            "value": "None recorded",
            "detail": "Runtime work is not enabled by the application shell.",
        },
    ]


def _camera_rows(paths: ManagedPaths, database: dict[str, Any]) -> list[dict[str, Any]]:
    if database["status"] != "ok":
        return []

    owned_database = Database(paths.database)
    try:
        return [camera_payload(camera) for camera in CameraRepository(owned_database).list()]
    finally:
        owned_database.dispose()


def _camera_feedback(request: Request) -> dict[str, str] | None:
    notice = request.query_params.get("notice")
    if notice is None:
        return None

    status = request.query_params.get("status", "")
    message = request.query_params.get("message", "")
    messages = {
        "created": ("positive", "Camera profile saved."),
        "updated": ("positive", "Camera profile updated."),
        "deleted": ("positive", "Camera profile deleted."),
        "missing": ("attention", "Camera profile was not found."),
        "database": (
            "attention",
            "Database is not ready. Run `open-licenseplate db upgrade` first.",
        ),
    }
    if notice == "test":
        tone = "positive" if status == "valid" else "attention"
        return {
            "tone": tone,
            "message": redact_text(message or "Camera test completed."),
        }
    if notice == "error":
        return {
            "tone": "attention",
            "message": redact_text(message or "Camera profile was not saved."),
        }
    tone, default_message = messages.get(
        notice,
        ("attention", "Camera operation completed."),
    )
    return {"tone": tone, "message": default_message}


def _context(
    request: Request,
    settings: AppSettings,
    paths: ManagedPaths,
    active_key: str,
) -> dict[str, Any]:
    database = _database_summary(paths)
    directories = paths.directory_checks()
    page = PAGE_BY_KEY[active_key]
    return {
        "request": request,
        "app_name": settings.app_name,
        "app_version": __version__,
        "ui_density": settings.ui.density,
        "page": page,
        "nav_items": _nav_items(active_key),
        "global_status": _global_status(database, directories),
        "global_runtime": {
            "camera": "Not configured",
            "model": "Not loaded",
            "worker": "Not started",
            "failures": "None",
        },
        "database": database,
        "settings": _setting_rows(settings),
        "managed_paths": _path_rows(paths),
        "runtime": _runtime_rows(),
        "directory_checks": directories,
        "cameras": _camera_rows(paths, database),
        "camera_feedback": _camera_feedback(request),
    }


def render_page(
    request: Request,
    settings: AppSettings,
    paths: ManagedPaths,
    page_key: str,
) -> Response:
    """Render one shell page with Jinja autoescaping enabled."""
    if page_key not in PAGE_BY_KEY:
        raise ValueError(f"unknown shell page: {page_key}")
    return templates.TemplateResponse(
        request=request,
        name="page.html",
        context=_context(request, settings, paths, page_key),
    )
