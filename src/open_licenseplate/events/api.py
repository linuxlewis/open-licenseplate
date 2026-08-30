"""Bounded event review API and secure managed crop responses."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse, Response

from ..database import Database, database_status
from ..paths import ManagedPaths
from ..redaction import redact_text
from .artifacts import (
    ARTIFACT_MIME_TYPE,
    ArtifactUnavailable,
    ManagedArtifactService,
)
from .repository import EventArtifact, EventRepository, EventReview

router = APIRouter(prefix="/api/v1/events", tags=["events"])

MAX_EVENT_LIST_LIMIT = 100
MAX_QUALITY_EVIDENCE_BYTES = 32 * 1024
NO_OCR_PAYLOAD = {
    "state": "not_available",
    "text": None,
    "message": "OCR is not available in this workflow.",
}
EVENT_NOT_FOUND = "event was not found"
ARTIFACT_NOT_FOUND = "event artifact was not found"


class EventDatabaseUnavailable(RuntimeError):
    """Raised when the event store is not ready for a review read."""


@router.get("")
async def list_events(
    request: Request,
    limit: int = Query(default=MAX_EVENT_LIST_LIMIT, ge=1, le=MAX_EVENT_LIST_LIMIT),
) -> JSONResponse:
    """Return the newest bounded event slice."""
    try:
        payload = await asyncio.to_thread(
            _list_events_sync,
            request.app.state.paths,
            limit,
        )
    except EventDatabaseUnavailable:
        return _error(
            "database is not ready; run `open-licenseplate db upgrade` first",
            status_code=409,
        )
    except (OSError, RuntimeError, TypeError, ValueError):
        return _error("event review is temporarily unavailable", status_code=500)
    return _json(payload)


@router.get("/{event_id}")
async def get_event(event_id: str, request: Request) -> JSONResponse:
    """Return one event with provenance and bounded crop metadata."""
    try:
        payload = await asyncio.to_thread(
            _get_event_sync,
            request.app.state.paths,
            event_id,
        )
    except EventDatabaseUnavailable:
        return _error(
            "database is not ready; run `open-licenseplate db upgrade` first",
            status_code=409,
        )
    except LookupError:
        return _error(EVENT_NOT_FOUND, status_code=404)
    except (OSError, RuntimeError, TypeError, ValueError):
        return _error("event review is temporarily unavailable", status_code=500)
    return _json(payload)


@router.get("/{event_id}/artifacts/{artifact_id}")
async def get_event_artifact(
    event_id: str,
    artifact_id: str,
    request: Request,
) -> Response:
    """Return only verified JPEG bytes for an owned committed crop."""
    try:
        payload = await asyncio.to_thread(
            _read_artifact_sync,
            request.app.state.paths,
            event_id,
            artifact_id,
        )
    except EventDatabaseUnavailable:
        return _error(
            "database is not ready; run `open-licenseplate db upgrade` first",
            status_code=409,
        )
    except (ArtifactUnavailable, LookupError):
        return _error(ARTIFACT_NOT_FOUND, status_code=404)
    except (OSError, RuntimeError, TypeError, ValueError):
        return _error("event artifact is temporarily unavailable", status_code=500)
    return Response(
        content=payload,
        media_type=ARTIFACT_MIME_TYPE,
        headers={
            "Cache-Control": "private, no-store",
            "Pragma": "no-cache",
        },
    )


def _list_events_sync(paths: ManagedPaths, limit: int) -> dict[str, Any]:
    database = _open_database(paths)
    try:
        repository = EventRepository(database)
        reviews = repository.list_reviews(limit=limit)
        artifact_service = ManagedArtifactService(paths)
        return {
            "events": [_event_summary_payload(review, artifact_service) for review in reviews],
            "limit": limit,
        }
    finally:
        database.dispose()


def _get_event_sync(paths: ManagedPaths, event_id: str) -> dict[str, Any]:
    database = _open_database(paths)
    try:
        review = EventRepository(database).review_event(event_id)
        if review is None:
            raise LookupError(EVENT_NOT_FOUND)
        return _event_detail_payload(review, ManagedArtifactService(paths))
    finally:
        database.dispose()


def _read_artifact_sync(paths: ManagedPaths, event_id: str, artifact_id: str) -> bytes:
    database = _open_database(paths)
    try:
        artifact = EventRepository(database).get_artifact(event_id, artifact_id)
        if artifact is None:
            raise LookupError(ARTIFACT_NOT_FOUND)
    finally:
        database.dispose()
    return ManagedArtifactService(paths).read_committed_artifact(artifact)


def _open_database(paths: ManagedPaths) -> Database:
    try:
        status = database_status(paths.database)
    except Exception:
        raise EventDatabaseUnavailable from None
    if status["status"] != "ok":
        raise EventDatabaseUnavailable
    return Database(paths.database)


def _event_summary_payload(
    review: EventReview,
    artifact_service: ManagedArtifactService,
) -> dict[str, Any]:
    event = review.event
    best_artifact = review.artifacts[0] if review.artifacts else None
    best_crop = None
    if best_artifact is not None:
        best_crop = _artifact_payload(
            event.id,
            best_artifact,
            available=artifact_service.artifact_is_available(best_artifact),
        )
    return {
        "event_id": event.id,
        "event_time_utc": _timestamp(event.first_seen_at),
        "first_seen_utc": _timestamp(event.first_seen_at),
        "last_seen_utc": _timestamp(event.last_seen_at),
        "duration_seconds": round(event.duration_seconds, 3),
        "camera": {
            "id": event.camera_id,
            "display_name": redact_text(review.camera_display_name),
        },
        "camera_id": event.camera_id,
        "camera_display_name": redact_text(review.camera_display_name),
        "track_id": event.track_id,
        "observation_count": event.observation_count,
        "maximum_confidence": event.maximum_confidence,
        "event_state": event.event_state,
        "model_provenance": _model_provenance(event, review),
        "crop_ranking_version": event.crop_ranking_version,
        "best_crop": best_crop,
        "ocr": dict(NO_OCR_PAYLOAD),
    }


def _event_detail_payload(
    review: EventReview,
    artifact_service: ManagedArtifactService,
) -> dict[str, Any]:
    event = review.event
    artifacts = [
        _artifact_payload(
            event.id,
            artifact,
            available=artifact_service.artifact_is_available(artifact),
        )
        for artifact in review.artifacts
    ]
    return {
        **_event_summary_payload(review, artifact_service),
        "capture_session_id": event.capture_session_id,
        "model": _model_provenance(event, review),
        "artifacts": artifacts,
        "ocr_state": "not_available",
    }


def _model_provenance(event: Any, review: EventReview) -> dict[str, str]:
    return {
        "model_id": event.model_id,
        "model_checksum": event.model_checksum,
        "display_name": redact_text(review.model_display_name),
    }


def _artifact_payload(
    event_id: str,
    artifact: EventArtifact,
    *,
    available: bool,
) -> dict[str, Any]:
    return {
        "artifact_id": artifact.id,
        "event_id": event_id,
        "rank": artifact.artifact_rank,
        "artifact_kind": artifact.artifact_kind,
        "sha256": artifact.sha256,
        "mime_type": artifact.mime_type,
        "byte_size": artifact.byte_size,
        "width": artifact.width,
        "height": artifact.height,
        "source_frame_sequence": artifact.source_frame_sequence,
        "source_timestamp_utc": _timestamp(artifact.source_timestamp),
        "detection_confidence": artifact.detection_confidence,
        "quality_score": artifact.quality_score,
        "quality_scoring_version": artifact.quality_scoring_version,
        "quality_evidence": _quality_evidence(artifact),
        "availability": "available" if available else "unavailable",
        "url": (
            f"/api/v1/events/{quote(event_id, safe='')}/artifacts/{quote(artifact.id, safe='')}"
        ),
    }


def _quality_evidence(artifact: EventArtifact) -> dict[str, Any]:
    value = artifact.quality_evidence_json
    if not isinstance(value, str) or len(value.encode("utf-8")) > MAX_QUALITY_EVIDENCE_BYTES:
        return {}
    try:
        decoded = json.loads(value)
    except (TypeError, ValueError):
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _timestamp(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("event timestamp is invalid")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _json(payload: object, *, status_code: int = 200) -> JSONResponse:
    return JSONResponse(content=payload, status_code=status_code)


def _error(message: str, *, status_code: int) -> JSONResponse:
    return _json({"detail": message}, status_code=status_code)


__all__ = [
    "ARTIFACT_NOT_FOUND",
    "EVENT_NOT_FOUND",
    "MAX_EVENT_LIST_LIMIT",
    "NO_OCR_PAYLOAD",
    "router",
]
