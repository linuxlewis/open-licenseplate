"""Minimal M4 event persistence models and repository seams.

Live tracking does not write these rows yet. The durable closure transaction is
owned by the next milestone.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Float, Integer, String, Text, select
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from ..database import Database
from ..settings_store import UTCDateTime
from ..tracking.contracts import ClosedTrackEvent


class EventBase(DeclarativeBase):
    """Declarative base for M4 event persistence."""


class CaptureSession(EventBase):
    """Persisted provenance for one camera/model stream lifecycle."""

    __tablename__ = "capture_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    camera_id: Mapped[str] = mapped_column(String(36), nullable=False)
    model_id: Mapped[str] = mapped_column(String(128), nullable=False)
    model_checksum: Mapped[str] = mapped_column(String(64), nullable=False)
    compute_configuration_json: Mapped[str] = mapped_column(Text, nullable=False)
    started_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    ended_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    end_reason: Mapped[str | None] = mapped_column(String(64), nullable=True)
    negotiated_codec: Mapped[str | None] = mapped_column(String(64), nullable=True)
    negotiated_width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    negotiated_height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    negotiated_fps: Mapped[float | None] = mapped_column(Float, nullable=True)
    application_version: Mapped[str] = mapped_column(String(64), nullable=False)


class DetectionEvent(EventBase):
    """One durable closed-track event aggregate."""

    __tablename__ = "detection_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    camera_id: Mapped[str] = mapped_column(String(36), nullable=False)
    capture_session_id: Mapped[str] = mapped_column(String(36), nullable=False)
    track_id: Mapped[int] = mapped_column(Integer, nullable=False)
    model_id: Mapped[str] = mapped_column(String(128), nullable=False)
    model_checksum: Mapped[str] = mapped_column(String(64), nullable=False)
    first_seen_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    last_seen_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    duration_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    observation_count: Mapped[int] = mapped_column(Integer, nullable=False)
    maximum_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    event_state: Mapped[str] = mapped_column(String(32), nullable=False)
    best_artifact_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    crop_ranking_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)


class EventArtifact(EventBase):
    """Managed event artifact metadata without file-system behavior."""

    __tablename__ = "event_artifacts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    event_id: Mapped[str] = mapped_column(String(36), nullable=False)
    artifact_kind: Mapped[str] = mapped_column(String(64), nullable=False)
    managed_relative_path: Mapped[str] = mapped_column(Text, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    mime_type: Mapped[str] = mapped_column(String(128), nullable=False)
    byte_size: Mapped[int] = mapped_column(Integer, nullable=False)
    width: Mapped[int] = mapped_column(Integer, nullable=False)
    height: Mapped[int] = mapped_column(Integer, nullable=False)
    source_frame_sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    source_timestamp: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    detection_confidence: Mapped[float] = mapped_column(Float, nullable=False)
    quality_score: Mapped[float] = mapped_column(Float, nullable=False)
    quality_scoring_version: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    deleted_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)


@dataclass(frozen=True, slots=True)
class CaptureSessionCreate:
    """Values needed for a future capture-session row."""

    id: str
    camera_id: str
    model_id: str
    model_checksum: str
    started_at: datetime
    compute_configuration: dict[str, Any]
    application_version: str


class EventRepository:
    """Provide only the small persistence seams required by M4-A tests."""

    def __init__(self, database: Database) -> None:
        self.database = database

    def create_capture_session(self, values: CaptureSessionCreate) -> CaptureSession:
        """Insert one capture-session provenance row."""
        row = CaptureSession(
            id=values.id,
            camera_id=values.camera_id,
            model_id=values.model_id,
            model_checksum=values.model_checksum,
            compute_configuration_json=_dump_json(values.compute_configuration),
            started_at=_aware_utc(values.started_at),
            application_version=values.application_version,
        )
        with self.database.session() as session:
            session.add(row)
            session.flush()
            session.expunge(row)
        return row

    def create_closed_event(self, event: ClosedTrackEvent) -> DetectionEvent:
        """Insert one aggregate and leave uniqueness to SQLite."""
        now = datetime.now(UTC)
        row = DetectionEvent(
            id=event.event_id,
            camera_id=event.provenance.camera_id,
            capture_session_id=event.provenance.capture_session_id,
            track_id=event.track_id,
            model_id=event.provenance.model_id,
            model_checksum=event.provenance.model_checksum,
            first_seen_at=_aware_utc(event.first_seen_at),
            last_seen_at=_aware_utc(event.last_seen_at),
            duration_seconds=event.duration_seconds,
            observation_count=event.observation_count,
            maximum_confidence=event.maximum_confidence,
            event_state=event.event_state,
            created_at=now,
            updated_at=now,
        )
        with self.database.session() as session:
            session.add(row)
            session.flush()
            session.expunge(row)
        return row

    def get(self, event_id: str) -> DetectionEvent | None:
        """Read one event by its immutable identifier."""
        with self.database.session() as session:
            return session.get(DetectionEvent, event_id)

    def list(self, *, limit: int = 100) -> list[DetectionEvent]:
        """Read a bounded newest-first event slice."""
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("event limit must be between 1 and 1000")
        with self.database.session() as session:
            return list(
                session.scalars(
                    select(DetectionEvent)
                    .order_by(DetectionEvent.first_seen_at.desc(), DetectionEvent.id)
                    .limit(limit)
                )
            )


def _aware_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("event timestamps must be timezone-aware")
    return value.astimezone(UTC)


def _dump_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


__all__ = [
    "CaptureSession",
    "CaptureSessionCreate",
    "DetectionEvent",
    "EventArtifact",
    "EventRepository",
]
