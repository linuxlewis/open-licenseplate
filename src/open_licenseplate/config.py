"""Typed application settings and P00 source precedence.

P00 supports CLI > environment > built-in default. Persisted settings arrive
with the database work in P01.
"""

from __future__ import annotations

from collections.abc import Mapping
from ipaddress import ip_address
from pathlib import Path
from typing import Any, Literal

from platformdirs import user_data_dir, user_log_dir
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class SettingsError(ValueError):
    """Raised when effective settings are invalid."""


class ServerSettings(BaseModel):
    """HTTP server settings."""

    model_config = ConfigDict(extra="ignore")

    host: str = "127.0.0.1"
    port: int = Field(default=8421, ge=1, le=65535)
    unsafe_development: bool = False

    @model_validator(mode="after")
    def validate_bind_address(self) -> ServerSettings:
        """Require loopback binding unless explicitly disabled for development."""
        if self.unsafe_development:
            return self

        if self.host.lower() == "localhost":
            return self

        try:
            is_loopback = ip_address(self.host).is_loopback
        except ValueError as error:
            raise ValueError("server.host must be a loopback address") from error
        if not is_loopback:
            raise ValueError(
                "server.host must be loopback unless server.unsafe_development is true"
            )
        return self


class StorageSettings(BaseModel):
    """Application data and log roots."""

    model_config = ConfigDict(extra="ignore")

    data_dir: Path = Field(default_factory=lambda: Path(user_data_dir("open-licenseplate")))
    log_dir: Path = Field(default_factory=lambda: Path(user_log_dir("open-licenseplate")))


class LiveSettings(BaseModel):
    """Defaults for the live pipeline."""

    model_config = ConfigDict(extra="ignore")

    preview_fps: float = Field(default=10.0, gt=0)
    preview_jpeg_quality: int = Field(default=80, ge=1, le=100)
    detection_confidence: float = Field(default=0.35, ge=0, le=1)
    compute_units: Literal["all", "cpu_only", "cpu_and_gpu", "cpu_and_ne"] = "all"


class TrackingSettings(BaseModel):
    """Defaults for track aggregation."""

    model_config = ConfigDict(extra="ignore")

    confirmation_hits: int = Field(default=3, ge=1)
    close_after_seconds: float = Field(default=1.0, gt=0)
    retained_crops: int = Field(default=3, ge=1)


class WorkerSettings(BaseModel):
    """Defaults for a future durable worker."""

    model_config = ConfigDict(extra="ignore")

    poll_interval_seconds: float = Field(default=1.0, gt=0)
    lease_seconds: float = Field(default=60.0, gt=0)
    heartbeat_seconds: float = Field(default=20.0, gt=0)
    default_max_attempts: int = Field(default=5, ge=1)


class RetentionSettings(BaseModel):
    """Defaults for future managed-data retention."""

    model_config = ConfigDict(extra="ignore")

    event_metadata_days: int = Field(default=90, ge=1)
    crop_days: int = Field(default=30, ge=1)
    result_days: int = Field(default=90, ge=1)


class AppSettings(BaseSettings):
    """Effective settings after default, environment, and CLI layers."""

    model_config = SettingsConfigDict(
        env_prefix="OPEN_LICENSEPLATE_",
        env_nested_delimiter="__",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "open-licenseplate"
    environment: str = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    server: ServerSettings = Field(default_factory=ServerSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    live: LiveSettings = Field(default_factory=LiveSettings)
    tracking: TrackingSettings = Field(default_factory=TrackingSettings)
    worker: WorkerSettings = Field(default_factory=WorkerSettings)
    retention: RetentionSettings = Field(default_factory=RetentionSettings)

    _sources: dict[str, str] = PrivateAttr(default_factory=dict)

    @field_validator("log_level", mode="before")
    @classmethod
    def normalise_log_level(cls, value: Any) -> Any:
        return value.upper() if isinstance(value, str) else value

    @property
    def sources(self) -> dict[str, str]:
        """Return the source label for each effective setting."""
        return dict(self._sources)


def _merge_dicts(base: dict[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """Merge nested mappings without mutating either input."""
    result = dict(base)
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _merge_dicts(dict(result[key]), value)
        else:
            result[key] = value
    return result


def _explicit_values(model: BaseModel) -> dict[str, Any]:
    """Return only fields explicitly supplied to a Pydantic model."""
    result: dict[str, Any] = {}
    for field_name in model.model_fields_set:
        value = getattr(model, field_name)
        if isinstance(value, BaseModel):
            nested = _explicit_values(value)
            if nested:
                result[field_name] = nested
        else:
            result[field_name] = value
    return result


def _flatten_paths(
    values: Mapping[str, Any],
    source: str,
    *,
    prefix: str = "",
) -> dict[str, str]:
    """Create dotted source labels for diagnostics."""
    result: dict[str, str] = {}
    for key, value in values.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            result.update(_flatten_paths(value, source, prefix=path))
        else:
            result[path] = source
    return result


def _normalise_overrides(overrides: Mapping[str, Any] | None) -> dict[str, Any]:
    """Accept either nested mappings or dotted CLI setting names."""
    if not overrides:
        return {}

    result: dict[str, Any] = {}
    for key, value in overrides.items():
        parts = key.split(".")
        cursor = result
        for part in parts[:-1]:
            child = cursor.setdefault(part, {})
            if not isinstance(child, dict):
                raise SettingsError(f"conflicting CLI setting path: {key}")
            cursor = child
        cursor[parts[-1]] = value
    return result


def load_settings(
    *,
    cli_overrides: Mapping[str, Any] | None = None,
) -> AppSettings:
    """Load settings with CLI > environment > default precedence."""
    defaults = AppSettings.model_validate({})
    environment = AppSettings()
    cli = _normalise_overrides(cli_overrides)
    env_values = _explicit_values(environment)
    merged = _merge_dicts(defaults.model_dump(mode="python"), env_values)
    merged = _merge_dicts(merged, cli)
    settings = AppSettings.model_validate(merged)

    sources = _flatten_paths(defaults.model_dump(mode="python"), "default")
    sources.update(_flatten_paths(env_values, "environment"))
    sources.update(_flatten_paths(cli, "cli"))
    settings._sources = sources
    return settings


Settings = AppSettings
