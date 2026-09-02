"""Jinja templates and the M0 application shell."""

from __future__ import annotations

import hashlib
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
from .events.api import _get_event_sync, _list_events_sync
from .models.repository import ModelRepository, model_payload
from .paths import ManagedPaths
from .redaction import redact_text

_WEB_DIRECTORY = Path(__file__).resolve().parent
STATIC_DIRECTORY = _WEB_DIRECTORY / "static"


def _static_asset_version(static_directory: Path) -> str:
    """Return one stable version for the complete packaged static tree."""
    digest = hashlib.sha256()
    for path in sorted(path for path in static_directory.rglob("*") if path.is_file()):
        relative_path = path.relative_to(static_directory).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative_path).to_bytes(8, "big"))
        digest.update(relative_path)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()[:16]


STATIC_ASSET_VERSION = _static_asset_version(STATIC_DIRECTORY)
templates = Jinja2Templates(directory=str(_WEB_DIRECTORY / "templates"))


@dataclass(frozen=True)
class PageDefinition:
    """Static content for one primary application page."""

    key: str
    label: str
    path: str
    eyebrow: str
    title: str
    empty_title: str = ""
    nav: bool = True


PAGES = (
    PageDefinition(
        key="live",
        label="Live",
        path="/live",
        eyebrow="Operations",
        title="Live view",
        empty_title="No live source",
    ),
    PageDefinition(
        key="events",
        label="Events",
        path="/events",
        eyebrow="Review",
        title="Plate events",
        empty_title="No events",
    ),
    PageDefinition(
        key="event_detail",
        label="Events",
        path="/events",
        eyebrow="Review",
        title="Event review",
        nav=False,
    ),
    PageDefinition(
        key="jobs",
        label="Jobs",
        path="/jobs",
        eyebrow="Jobs",
        title="Processing jobs",
        empty_title="No jobs",
    ),
    PageDefinition(
        key="cameras",
        label="Cameras",
        path="/cameras",
        eyebrow="Sources",
        title="Camera sources",
        empty_title="No cameras",
    ),
    PageDefinition(
        key="models",
        label="Models",
        path="/models",
        eyebrow="Inference",
        title="Detection models",
        empty_title="No models",
    ),
    PageDefinition(
        key="system",
        label="System",
        path="/system",
        eyebrow="Diagnostics",
        title="System status",
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
        if page.nav
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


def _runtime_rows(runtime_status: dict[str, Any]) -> list[dict[str, str]]:
    camera_id = runtime_status.get("camera_id")
    state = str(runtime_status.get("state", "stopped"))
    if camera_id and state != "stopped":
        camera_value = str(runtime_status.get("camera_name") or camera_id)
        camera_detail = f"Lifecycle: {state}"
    else:
        camera_value = "Not configured"
        camera_detail = "No camera is running."
    return [
        {
            "label": "Active camera",
            "value": camera_value,
            "detail": camera_detail,
        },
        {
            "label": "Active model",
            "value": "Not loaded",
        },
        {
            "label": "Worker",
            "value": "Not started",
        },
        {
            "label": "Unresolved failures",
            "value": "None recorded",
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


def _model_rows(paths: ManagedPaths, database: dict[str, Any]) -> list[dict[str, Any]]:
    if database["status"] != "ok":
        return []

    owned_database = Database(paths.database)
    try:
        repository = ModelRepository(owned_database)
        rows = []
        for model in repository.list():
            try:
                artifact_path = ManagedPaths.validate_contained_path(
                    paths.models / model.artifact_path,
                    paths.models,
                )
                artifact_exists = artifact_path.is_dir() and not artifact_path.is_symlink()
            except ValueError:
                artifact_exists = False
            rows.append(model_payload(model, artifact_exists=artifact_exists))
        return rows
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


def _model_feedback(request: Request) -> dict[str, str] | None:
    notice = request.query_params.get("notice")
    if notice is None:
        return None
    messages = {
        "imported": ("positive", "Model package imported."),
        "validated": ("positive", "Model package checks passed."),
        "invalid": ("attention", "Model package checks failed."),
        "activated": ("positive", "Model activated."),
        "deactivated": ("positive", "Model deactivated."),
        "deleted": ("positive", "Model deleted."),
        "missing": ("attention", "Model was not found."),
        "database": (
            "attention",
            "Database is not ready. Run `open-licenseplate db upgrade` first.",
        ),
    }
    if notice == "error":
        return {
            "tone": "attention",
            "message": redact_text(request.query_params.get("message", "Model operation failed.")),
        }
    tone, message = messages.get(notice, ("attention", "Model operation completed."))
    return {"tone": tone, "message": message}


def _context(
    request: Request,
    settings: AppSettings,
    paths: ManagedPaths,
    active_key: str,
    event_id: str | None = None,
) -> dict[str, Any]:
    database = _database_summary(paths)
    directories = paths.directory_checks()
    page = PAGE_BY_KEY[active_key]
    runtime_status = request.app.state.camera_runtime.status().as_dict()
    live_status = request.app.state.live_pipeline.status().as_dict()
    cameras = _camera_rows(paths, database)
    selected_camera_id = (
        runtime_status.get("camera_id")
        if runtime_status.get("camera_id") in {camera["id"] for camera in cameras}
        else (cameras[0]["id"] if cameras else None)
    )
    events: list[dict[str, Any]] = []
    event_detail: dict[str, Any] | None = None
    if database["status"] == "ok" and active_key == "events":
        try:
            events = _list_events_sync(paths, 100)["events"]
        except Exception:
            events = []
    elif database["status"] == "ok" and active_key == "event_detail" and event_id:
        try:
            event_detail = _get_event_sync(paths, event_id)
        except LookupError:
            raise
        except Exception:
            event_detail = None
    return {
        "request": request,
        "app_name": settings.app_name,
        "app_version": __version__,
        "static_asset_version": STATIC_ASSET_VERSION,
        "ui_density": settings.ui.density,
        "page": page,
        "page_index": "M4"
        if active_key in {"events", "event_detail"}
        else (
            "M3"
            if active_key == "live"
            else ("M2" if active_key == "models" else ("M1" if active_key == "cameras" else "M0"))
        ),
        "nav_items": _nav_items(active_key),
        "global_status": _global_status(database, directories),
        "global_runtime": {
            "camera": runtime_status.get("camera_name") or "Stopped",
            "model": "Not loaded",
            "worker": "Not started",
            "failures": "None",
        },
        "database": database,
        "settings": _setting_rows(settings),
        "managed_paths": _path_rows(paths),
        "runtime": _runtime_rows(runtime_status),
        "directory_checks": directories,
        "cameras": cameras,
        "models": _model_rows(paths, database),
        "camera_feedback": _camera_feedback(request),
        "model_feedback": _model_feedback(request),
        "runtime_status": runtime_status,
        "live_status": live_status,
        "selected_camera_id": selected_camera_id,
        "selected_model_id": live_status.get("model_id"),
        "events": events,
        "event_detail": event_detail,
    }


def build_page_context(
    request: Request,
    settings: AppSettings,
    paths: ManagedPaths,
    page_key: str,
    *,
    event_id: str | None = None,
) -> dict[str, Any]:
    """Build page data so blocking review reads can run in a worker thread."""
    return _context(request, settings, paths, page_key, event_id=event_id)


def render_page(
    request: Request,
    settings: AppSettings,
    paths: ManagedPaths,
    page_key: str,
    *,
    context: dict[str, Any] | None = None,
    event_id: str | None = None,
) -> Response:
    """Render one shell page with Jinja autoescaping enabled."""
    if page_key not in PAGE_BY_KEY:
        raise ValueError(f"unknown shell page: {page_key}")
    page_context = context or _context(request, settings, paths, page_key, event_id=event_id)
    return templates.TemplateResponse(
        request=request,
        name="page.html",
        context=page_context,
    )
