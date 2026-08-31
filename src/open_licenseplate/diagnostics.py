"""Safe diagnostics shared by doctor, system APIs, and secret audits."""

from __future__ import annotations

from typing import Any

from . import __version__
from .config import AppSettings
from .database import database_status
from .paths import ManagedPaths
from .redaction import redact_value


def build_diagnostics(
    settings: AppSettings,
    paths: ManagedPaths,
    *,
    runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a local diagnostics payload without camera secrets or pixels."""
    payload = {
        "application": settings.app_name,
        "version": __version__,
        "configuration": {
            "status": "ok",
            "sources": settings.sources,
        },
        "paths": {
            "data_dir": str(paths.data_dir),
            "database": str(paths.database),
            "models": str(paths.models),
            "artifacts": str(paths.artifacts),
            "staging": str(paths.staging),
            "settings": str(paths.settings),
            "log_dir": str(paths.log_dir),
        },
        "directories": paths.directory_checks(),
        "database": database_status(paths.database),
        "runtime": runtime or {"state": "stopped"},
    }
    result = redact_value(payload)
    if not isinstance(result, dict):
        raise TypeError("diagnostics payload must be a mapping")
    return result


__all__ = ["build_diagnostics"]
