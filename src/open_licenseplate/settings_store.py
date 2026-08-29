"""Persistence for non-secret application settings."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import Integer, String, Text, TypeDecorator, select
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from .database import Database

PERSISTABLE_SETTING_KEYS = frozenset(
    {
        "app_name",
        "environment",
        "log_level",
        "server.host",
        "server.port",
    }
)
"""Settings that are safe and stable enough for generic persistence.

Storage paths are intentionally excluded because they are needed to locate the
database before persisted settings can be read.
"""


class SettingsBase(DeclarativeBase):
    """Base metadata for settings persistence."""


class UTCDateTime(TypeDecorator[datetime]):
    """Store aware UTC timestamps as ISO-8601 text in SQLite."""

    impl = String
    cache_ok = True

    def process_bind_param(self, value: datetime | None, _dialect: Any) -> str | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("updated_at must be timezone-aware")
        return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")

    def process_result_value(self, value: str | None, _dialect: Any) -> datetime | None:
        if value is None:
            return None
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("stored updated_at is not timezone-aware")
        return parsed.astimezone(UTC)


class ApplicationSetting(SettingsBase):
    """One versioned JSON setting value."""

    __tablename__ = "application_settings"

    setting_key: Mapped[str] = mapped_column(String(255), primary_key=True)
    value_json: Mapped[str] = mapped_column(Text, nullable=False)
    schema_version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)


def validate_setting_key(setting_key: str) -> str:
    """Validate and normalize a setting key allowed in SQLite."""
    normalized = setting_key.strip()
    if normalized not in PERSISTABLE_SETTING_KEYS:
        raise ValueError(f"setting is not persistable: {normalized or '<empty>'}")
    return normalized


def _validate_value(setting_key: str, value: Any) -> str:
    if _contains_secret_key(value):
        raise ValueError("secret values cannot be persisted")
    if setting_key in {"app_name", "environment", "log_level", "server.host"} and not isinstance(
        value, str
    ):
        raise ValueError(f"{setting_key} must be a string")
    if setting_key == "server.port" and (
        isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 65535
    ):
        raise ValueError("server.port must be an integer from 1 through 65535")
    try:
        return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as error:
        raise ValueError("setting value must be JSON serializable") from error


def _contains_secret_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            any(secret_word in str(key).lower() for secret_word in ("password", "secret", "token"))
            or _contains_secret_key(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_secret_key(item) for item in value)
    return False


class SettingsStore:
    """Read and write the generic application settings table."""

    def __init__(self, database: Database) -> None:
        self.database = database

    def get_all(self) -> dict[str, Any]:
        """Return all settings decoded from their JSON representation."""
        with self.database.session() as session:
            rows = session.scalars(select(ApplicationSetting)).all()
            return {row.setting_key: _decode_value(row.value_json) for row in rows}

    def get(self, setting_key: str) -> Any | None:
        """Return one setting or ``None`` when it is not stored."""
        key = validate_setting_key(setting_key)
        with self.database.session() as session:
            row = session.get(ApplicationSetting, key)
            return None if row is None else _decode_value(row.value_json)

    def set(self, setting_key: str, value: Any, *, schema_version: int = 1) -> None:
        """Insert or replace one non-secret setting."""
        key = validate_setting_key(setting_key)
        value_json = _validate_value(key, value)
        if schema_version < 1:
            raise ValueError("schema_version must be greater than zero")

        with self.database.session() as session:
            row = session.get(ApplicationSetting, key)
            now = datetime.now(UTC)
            if row is None:
                session.add(
                    ApplicationSetting(
                        setting_key=key,
                        value_json=value_json,
                        schema_version=schema_version,
                        updated_at=now,
                    )
                )
            else:
                row.value_json = value_json
                row.schema_version = schema_version
                row.updated_at = now


def _decode_value(value_json: str) -> Any:
    try:
        return json.loads(value_json)
    except json.JSONDecodeError as error:
        raise ValueError("stored application setting contains invalid JSON") from error


def read_persisted_settings(path: Path) -> dict[str, Any]:
    """Read settings when the migrated settings table exists.

    An absent database or a pre-migration database has no persisted settings.
    Other database errors are returned to the caller so startup does not hide
    damaged local state.
    """
    if not path.expanduser().is_file():
        return {}

    database = Database(path)
    try:
        return SettingsStore(database).get_all()
    except OperationalError as error:
        if "no such table" in str(error).lower():
            return {}
        raise
    finally:
        database.dispose()
