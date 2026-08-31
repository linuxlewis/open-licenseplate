"""SQLite persistence for camera configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import Boolean, String, Text, select
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from ..database import Database
from ..redaction import redact_text, redact_url, redact_value
from ..settings_store import UTCDateTime


class CameraBase(DeclarativeBase):
    """Declarative base for camera persistence."""


class Camera(CameraBase):
    """Redacted camera configuration stored in SQLite."""

    __tablename__ = "cameras"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    endpoint: Mapped[str] = mapped_column(Text, nullable=False)
    credential_ref: Mapped[str | None] = mapped_column(String(255), nullable=True)
    connection_options_json: Mapped[str] = mapped_column(Text, nullable=False)
    preferred_stream: Mapped[str] = mapped_column(String(100), nullable=False)
    region_of_interest_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)


@dataclass(frozen=True)
class CameraConfig:
    """Validated camera values prepared for persistence."""

    name: str
    endpoint: str
    credential_ref: str | None
    connection_options: dict[str, Any]
    preferred_stream: str
    region_of_interest: dict[str, float] | None
    enabled: bool


class CameraRepository:
    """Create, read, update, and delete redacted camera records."""

    def __init__(self, database: Database) -> None:
        self.database = database

    def list(self) -> list[Camera]:
        with self.database.session() as session:
            return list(session.scalars(select(Camera).order_by(Camera.name, Camera.id)))

    def get(self, camera_id: str) -> Camera | None:
        with self.database.session() as session:
            return session.get(Camera, camera_id)

    def create(self, config: CameraConfig) -> Camera:
        now = datetime.now(UTC)
        camera = Camera(
            id=str(uuid4()),
            name=redact_text(config.name),
            endpoint=redact_url(config.endpoint),
            credential_ref=config.credential_ref,
            connection_options_json=_dump_json(redact_value(config.connection_options)),
            preferred_stream=redact_text(config.preferred_stream),
            region_of_interest_json=(
                None if config.region_of_interest is None else _dump_json(config.region_of_interest)
            ),
            enabled=config.enabled,
            created_at=now,
            updated_at=now,
        )
        with self.database.session() as session:
            session.add(camera)
            session.flush()
        return camera

    def update(self, camera: Camera, config: CameraConfig) -> Camera:
        camera.name = redact_text(config.name)
        camera.endpoint = redact_url(config.endpoint)
        camera.credential_ref = config.credential_ref
        camera.connection_options_json = _dump_json(redact_value(config.connection_options))
        camera.preferred_stream = redact_text(config.preferred_stream)
        camera.region_of_interest_json = (
            None if config.region_of_interest is None else _dump_json(config.region_of_interest)
        )
        camera.enabled = config.enabled
        camera.updated_at = datetime.now(UTC)
        with self.database.session() as session:
            stored = session.get(Camera, camera.id)
            if stored is None:
                raise KeyError(camera.id)
            stored.name = camera.name
            stored.endpoint = camera.endpoint
            stored.credential_ref = camera.credential_ref
            stored.connection_options_json = camera.connection_options_json
            stored.preferred_stream = camera.preferred_stream
            stored.region_of_interest_json = camera.region_of_interest_json
            stored.enabled = camera.enabled
            stored.updated_at = camera.updated_at
            session.flush()
            session.expunge(stored)
        return stored

    def delete(self, camera_id: str) -> bool:
        with self.database.session() as session:
            camera = session.get(Camera, camera_id)
            if camera is None:
                return False
            session.delete(camera)
        return True


def camera_config_from_record(camera: Camera) -> CameraConfig:
    """Decode one persisted record into safe configuration values."""
    options = json.loads(camera.connection_options_json)
    region = (
        None
        if camera.region_of_interest_json is None
        else json.loads(camera.region_of_interest_json)
    )
    return CameraConfig(
        name=camera.name,
        endpoint=camera.endpoint,
        credential_ref=camera.credential_ref,
        connection_options=options,
        preferred_stream=camera.preferred_stream,
        region_of_interest=region,
        enabled=camera.enabled,
    )


def _dump_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
