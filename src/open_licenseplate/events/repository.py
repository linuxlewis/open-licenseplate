"""M4 event persistence models and the short event closure transaction."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Float, Integer, String, Text, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

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
    crop_ranking_version: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)


class EventArtifact(EventBase):
    """Managed event artifact metadata without file-system behavior."""

    __tablename__ = "event_artifacts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    event_id: Mapped[str] = mapped_column(String(36), nullable=False)
    artifact_rank: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
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
    quality_evidence_json: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        server_default="{}",
    )
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False)
    deleted_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)


@dataclass(frozen=True, slots=True)
class CaptureSessionCreate:
    """Values needed for a capture-session provenance row."""

    id: str
    camera_id: str
    model_id: str
    model_checksum: str
    started_at: datetime
    compute_configuration: dict[str, Any]
    application_version: str
    ended_at: datetime | None = None
    end_reason: str | None = None
    negotiated_codec: str | None = None
    negotiated_width: int | None = None
    negotiated_height: int | None = None
    negotiated_fps: float | None = None


@dataclass(frozen=True, slots=True)
class CommittedArtifact:
    """Immutable artifact values inserted with one event transaction."""

    id: str
    event_id: str
    artifact_rank: int
    artifact_kind: str
    managed_relative_path: str
    sha256: str
    mime_type: str
    byte_size: int
    width: int
    height: int
    source_frame_sequence: int
    source_timestamp: datetime
    detection_confidence: float
    quality_score: float
    quality_scoring_version: str
    quality_evidence: dict[str, Any]


class EventRepository:
    """Provide bounded event reads and the M4-B closure transaction."""

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
            ended_at=None if values.ended_at is None else _aware_utc(values.ended_at),
            end_reason=values.end_reason,
            negotiated_codec=values.negotiated_codec,
            negotiated_width=values.negotiated_width,
            negotiated_height=values.negotiated_height,
            negotiated_fps=values.negotiated_fps,
            application_version=values.application_version,
        )
        with self.database.session() as session:
            session.add(row)
            session.flush()
            session.expunge(row)
        return row

    def commit_closed_event(
        self,
        event: ClosedTrackEvent,
        *,
        artifacts: Sequence[CommittedArtifact],
        crop_ranking_version: str,
        capture_session: CaptureSessionCreate | None = None,
    ) -> DetectionEvent:
        """Commit provenance, one event, and all selected artifacts together."""
        if not crop_ranking_version.strip():
            raise ValueError("crop_ranking_version is required")
        if len(artifacts) > 3:
            raise ValueError("an event may commit at most three artifacts")
        if any(artifact.event_id != event.event_id for artifact in artifacts):
            raise ValueError("artifact event IDs must match the closed event")

        session_values = capture_session or CaptureSessionCreate(
            id=event.capture_session_id,
            camera_id=event.camera_id,
            model_id=event.model_id,
            model_checksum=event.model_checksum,
            started_at=event.first_seen_at,
            compute_configuration={"capture": "live"},
            application_version="unknown",
        )
        now = datetime.now(UTC)
        with (
            self.database.engine.begin() as connection,
            Session(
                bind=connection,
                autoflush=False,
                expire_on_commit=False,
            ) as session,
        ):
            existing = session.scalar(
                select(DetectionEvent).where(
                    DetectionEvent.capture_session_id == event.capture_session_id,
                    DetectionEvent.track_id == event.track_id,
                )
            )
            if existing is not None:
                session.expunge(existing)
                return existing

            session_row = session.get(CaptureSession, session_values.id)
            if session_row is None:
                session_row = CaptureSession(
                    id=session_values.id,
                    camera_id=session_values.camera_id,
                    model_id=session_values.model_id,
                    model_checksum=session_values.model_checksum,
                    compute_configuration_json=_dump_json(session_values.compute_configuration),
                    started_at=_aware_utc(session_values.started_at),
                    ended_at=(
                        None
                        if session_values.ended_at is None
                        else _aware_utc(session_values.ended_at)
                    ),
                    end_reason=session_values.end_reason,
                    negotiated_codec=session_values.negotiated_codec,
                    negotiated_width=session_values.negotiated_width,
                    negotiated_height=session_values.negotiated_height,
                    negotiated_fps=session_values.negotiated_fps,
                    application_version=session_values.application_version,
                )
                session.add(session_row)
                session.flush()
            elif (
                session_row.camera_id != event.camera_id
                or session_row.model_id != event.model_id
                or session_row.model_checksum != event.model_checksum
            ):
                raise ValueError("capture-session provenance does not match the event")

            event_row = DetectionEvent(
                id=event.event_id,
                camera_id=event.camera_id,
                capture_session_id=event.capture_session_id,
                track_id=event.track_id,
                model_id=event.model_id,
                model_checksum=event.model_checksum,
                first_seen_at=_aware_utc(event.first_seen_at),
                last_seen_at=_aware_utc(event.last_seen_at),
                duration_seconds=event.duration_seconds,
                observation_count=event.observation_count,
                maximum_confidence=event.maximum_confidence,
                event_state=event.event_state,
                best_artifact_id=None,
                crop_ranking_version=crop_ranking_version,
                created_at=now,
                updated_at=now,
            )
            session.add(event_row)
            session.flush()

            artifact_rows = [
                EventArtifact(
                    id=artifact.id,
                    event_id=artifact.event_id,
                    artifact_rank=artifact.artifact_rank,
                    artifact_kind=artifact.artifact_kind,
                    managed_relative_path=artifact.managed_relative_path,
                    sha256=artifact.sha256,
                    mime_type=artifact.mime_type,
                    byte_size=artifact.byte_size,
                    width=artifact.width,
                    height=artifact.height,
                    source_frame_sequence=artifact.source_frame_sequence,
                    source_timestamp=_aware_utc(artifact.source_timestamp),
                    detection_confidence=artifact.detection_confidence,
                    quality_score=artifact.quality_score,
                    quality_scoring_version=artifact.quality_scoring_version,
                    quality_evidence_json=_dump_json(artifact.quality_evidence),
                    created_at=now,
                )
                for artifact in artifacts
            ]
            session.add_all(artifact_rows)
            session.flush()
            if artifact_rows:
                event_row.best_artifact_id = artifact_rows[0].id
                event_row.updated_at = now
                session.flush()

            session.expunge(event_row)
            return event_row

    def create_closed_event(
        self,
        event: ClosedTrackEvent,
        *,
        crop_ranking_version: str,
    ) -> DetectionEvent:
        """Insert one aggregate and leave uniqueness to SQLite."""
        if not crop_ranking_version.strip():
            raise ValueError("crop_ranking_version is required")
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
            crop_ranking_version=crop_ranking_version,
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

    def get_by_durable_key(self, capture_session_id: str, track_id: int) -> DetectionEvent | None:
        """Read one event by the durable capture-session and track key."""
        with self.database.session() as session:
            return session.scalar(
                select(DetectionEvent).where(
                    DetectionEvent.capture_session_id == capture_session_id,
                    DetectionEvent.track_id == track_id,
                )
            )

    def artifacts_for_event(self, event_id: str) -> list[EventArtifact]:
        """Read committed artifacts in deterministic best-first order."""
        with self.database.session() as session:
            return list(
                session.scalars(
                    select(EventArtifact)
                    .where(EventArtifact.event_id == event_id)
                    .order_by(
                        EventArtifact.artifact_rank.asc(),
                        EventArtifact.quality_score.desc(),
                        EventArtifact.detection_confidence.desc(),
                        EventArtifact.source_frame_sequence.asc(),
                        EventArtifact.id.asc(),
                    )
                )
            )

    def managed_relative_paths(self) -> set[str]:
        """Return all stored paths for startup reconciliation."""
        with self.database.session() as session:
            return set(session.scalars(select(EventArtifact.managed_relative_path)))

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
    "CommittedArtifact",
    "DetectionEvent",
    "EventArtifact",
    "EventRepository",
]
