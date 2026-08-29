"""SQLite persistence for the managed model registry."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Boolean, String, Text, select, update
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from ..database import Database
from ..settings_store import UTCDateTime
from .manifest import ModelManifest, parse_manifest


class ModelBase(DeclarativeBase):
    """Declarative base for model persistence."""


class Model(ModelBase):
    """One immutable imported model package and its provenance."""

    __tablename__ = "models"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    display_name: Mapped[str] = mapped_column(String(255), nullable=False)
    backend: Mapped[str] = mapped_column(String(64), nullable=False)
    adapter: Mapped[str] = mapped_column(String(128), nullable=False)
    artifact_path: Mapped[str] = mapped_column(Text, nullable=False)
    artifact_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    manifest_json: Mapped[str] = mapped_column(Text, nullable=False)
    validation_state: Mapped[str] = mapped_column(String(32), nullable=False)
    validation_details_json: Mapped[str] = mapped_column(Text, nullable=False)
    source_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_license: Mapped[str | None] = mapped_column(String(255), nullable=True)
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    last_validated_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)


@dataclass(frozen=True)
class ModelCreate:
    """Validated values used to create one registry record."""

    manifest: ModelManifest
    artifact_path: str
    validation_state: str
    validation_details: dict[str, Any]


class ModelRepository:
    """Create, read, validate, activate, and delete model records."""

    def __init__(self, database: Database) -> None:
        self.database = database

    def list(self) -> list[Model]:
        with self.database.session() as session:
            return list(session.scalars(select(Model).order_by(Model.display_name, Model.id)))

    def get(self, model_id: str) -> Model | None:
        with self.database.session() as session:
            return session.get(Model, model_id)

    def create(self, values: ModelCreate) -> Model:
        now = datetime.now(UTC)
        model = Model(
            id=values.manifest.model_id,
            display_name=values.manifest.display_name,
            backend=values.manifest.backend,
            adapter=values.manifest.adapter,
            artifact_path=values.artifact_path,
            artifact_sha256=values.manifest.artifact_sha256,
            manifest_json=values.manifest.snapshot_json,
            validation_state=values.validation_state,
            validation_details_json=_dump_json(values.validation_details),
            source_url=values.manifest.source_url,
            source_license=values.manifest.source_license,
            active=False,
            created_at=now,
            last_validated_at=now,
        )
        with self.database.session() as session:
            if session.get(Model, model.id) is not None:
                raise ValueError("a model with this manifest id already exists")
            session.add(model)
            session.flush()
            session.expunge(model)
        return model

    def update_validation(
        self,
        model_id: str,
        *,
        state: str,
        details: dict[str, Any],
    ) -> Model | None:
        with self.database.session() as session:
            model = session.get(Model, model_id)
            if model is None:
                return None
            model.validation_state = state
            model.validation_details_json = _dump_json(details)
            model.last_validated_at = datetime.now(UTC)
            session.flush()
            session.expunge(model)
            return model

    def set_active(self, model_id: str, active: bool) -> Model | None:
        with self.database.session() as session:
            model = session.get(Model, model_id)
            if model is None:
                return None
            if active:
                session.execute(update(Model).where(Model.id != model_id).values(active=False))
            model.active = active
            session.flush()
            session.expunge(model)
            return model

    def delete(self, model_id: str) -> Model | None:
        with self.database.session() as session:
            model = session.get(Model, model_id)
            if model is None:
                return None
            if model.active:
                raise ValueError("active models cannot be deleted")
            session.delete(model)
            session.flush()
            session.expunge(model)
            return model


def model_payload(model: Model, *, artifact_exists: bool = True) -> dict[str, Any]:
    """Return a JSON-safe public model representation."""
    return {
        "id": model.id,
        "display_name": model.display_name,
        "backend": model.backend,
        "adapter": model.adapter,
        "artifact_path": model.artifact_path,
        "artifact_sha256": model.artifact_sha256,
        "manifest": _load_json_object(model.manifest_json, "model manifest"),
        "validation_state": model.validation_state,
        "validation_details": _load_json_object(
            model.validation_details_json,
            "model validation details",
        ),
        "source": {
            "url": model.source_url,
            "license": model.source_license,
        },
        "active": model.active,
        "artifact_exists": artifact_exists,
        "created_at": _timestamp(model.created_at),
        "last_validated_at": (
            None if model.last_validated_at is None else _timestamp(model.last_validated_at)
        ),
    }


def manifest_from_record(model: Model) -> ModelManifest:
    """Decode and validate the immutable manifest snapshot."""
    return parse_manifest(model.manifest_json)


def _load_json_object(value: str, field_name: str) -> dict[str, Any]:
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"stored {field_name} contains invalid JSON") from error
    if not isinstance(decoded, dict):
        raise ValueError(f"stored {field_name} must be an object")
    return decoded


def _dump_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _timestamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
