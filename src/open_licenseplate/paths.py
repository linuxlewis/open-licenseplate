"""Managed application data and log paths."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .config import AppSettings


@dataclass(frozen=True)
class ManagedPaths:
    """All paths owned by the application."""

    data_dir: Path
    database: Path
    models: Path
    artifacts: Path
    staging: Path
    settings: Path
    log_dir: Path
    app_log: Path
    worker_log: Path

    @classmethod
    def from_settings(cls, settings: AppSettings) -> ManagedPaths:
        data_dir = settings.storage.data_dir.expanduser()
        log_dir = settings.storage.log_dir.expanduser()
        return cls(
            data_dir=data_dir,
            database=data_dir / "open-licenseplate.sqlite3",
            models=data_dir / "models",
            artifacts=data_dir / "artifacts",
            staging=data_dir / "staging",
            settings=data_dir / "settings.json",
            log_dir=log_dir,
            app_log=log_dir / "app.log",
            worker_log=log_dir / "worker.log",
        )

    @property
    def data_directories(self) -> tuple[Path, ...]:
        return (self.data_dir, self.models, self.artifacts, self.staging)

    @property
    def all_directories(self) -> tuple[Path, ...]:
        return (*self.data_directories, self.log_dir)

    def ensure_directories(self) -> None:
        """Create managed directories with restrictive permissions."""
        for directory in self.all_directories:
            directory.mkdir(parents=True, exist_ok=True, mode=0o700)

    def directory_checks(self) -> dict[str, bool]:
        """Return existence and write checks for doctor and readiness."""
        return {
            "data_dir_exists": self.data_dir.is_dir(),
            "models_dir_exists": self.models.is_dir(),
            "artifacts_dir_exists": self.artifacts.is_dir(),
            "staging_dir_exists": self.staging.is_dir(),
            "log_dir_exists": self.log_dir.is_dir(),
            "data_dir_writable": self._is_writable_directory(self.data_dir),
            "models_dir_writable": self._is_writable_directory(self.models),
            "artifacts_dir_writable": self._is_writable_directory(self.artifacts),
            "staging_dir_writable": self._is_writable_directory(self.staging),
            "log_dir_writable": self._is_writable_directory(self.log_dir),
        }

    @staticmethod
    def _is_writable_directory(path: Path) -> bool:
        return path.is_dir() and os.access(path, os.W_OK | os.X_OK)

    @staticmethod
    def validate_contained_path(path: Path, root: Path) -> Path:
        """Resolve a path and require it to stay below its managed root."""
        resolved_root = root.expanduser().resolve()
        resolved_path = path.expanduser().resolve()
        try:
            resolved_path.relative_to(resolved_root)
        except ValueError as error:
            raise ValueError(f"path is outside managed root: {resolved_path}") from error
        return resolved_path
