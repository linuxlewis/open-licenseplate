"""Bounded contracts for live detection tracking and event aggregation."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal, Protocol
from uuid import uuid4

from ..inference.contract import Detection
from .crops import CropCandidate

TrackLifecycleState = Literal["candidate", "confirmed", "active", "closed"]


@dataclass(frozen=True, slots=True)
class TrackingProvenance:
    """Identity that scopes one tracker state."""

    camera_id: str
    capture_session_id: str
    generation_number: int
    stream_epoch: str
    model_id: str
    model_checksum: str

    def __post_init__(self) -> None:
        for value, field_name in (
            (self.camera_id, "camera_id"),
            (self.capture_session_id, "capture_session_id"),
            (self.stream_epoch, "stream_epoch"),
            (self.model_id, "model_id"),
            (self.model_checksum, "model_checksum"),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} is required")
        if len(self.model_checksum) != 64 or any(
            character not in "0123456789abcdefABCDEF" for character in self.model_checksum
        ):
            raise ValueError("model_checksum must be a SHA-256 hexadecimal value")
        if type(self.generation_number) is not int or self.generation_number < 0:
            raise ValueError("generation_number must be a non-negative integer")

    def as_dict(self) -> dict[str, str | int]:
        """Return the safe live metadata representation."""
        return {
            "camera_id": self.camera_id,
            "capture_session_id": self.capture_session_id,
            "generation_number": self.generation_number,
            "stream_epoch": self.stream_epoch,
            "model_id": self.model_id,
            "model_checksum": self.model_checksum,
        }


@dataclass(frozen=True, slots=True)
class LiveDetectionFrame:
    """Validated detections and all provenance for one processed frame."""

    provenance: TrackingProvenance
    frame_sequence: int
    captured_at: datetime
    detections: tuple[Detection, ...] = ()
    frame_width: int = 8192
    frame_height: int = 8192
    source_pixels: Any | None = None
    pixel_format: str = "bgr24"

    def __post_init__(self) -> None:
        if type(self.frame_sequence) is not int or self.frame_sequence < 0:
            raise ValueError("frame_sequence must be a non-negative integer")
        if type(self.frame_width) is not int or self.frame_width <= 0:
            raise ValueError("frame_width must be a positive integer")
        if type(self.frame_height) is not int or self.frame_height <= 0:
            raise ValueError("frame_height must be a positive integer")
        if self.captured_at.tzinfo is None or self.captured_at.utcoffset() is None:
            raise ValueError("captured_at must be timezone-aware")
        if not isinstance(self.pixel_format, str) or not self.pixel_format.strip():
            raise ValueError("pixel_format is required")
        captured_at = self.captured_at.astimezone(UTC)
        object.__setattr__(self, "captured_at", captured_at)
        for detection in self.detections:
            if (
                not isinstance(detection.label, str)
                or not detection.label.strip()
                or type(detection.class_id) is not int
                or detection.class_id < 0
                or detection.model_id != self.provenance.model_id
                or detection.model_sha256 != self.provenance.model_checksum
                or detection.frame_sequence != self.frame_sequence
                or detection.detected_at is None
                or detection.detected_at.astimezone(UTC) != captured_at
            ):
                raise ValueError("detection provenance does not match its live frame")
            _validate_box(detection.box_xyxy)
            if (
                detection.x2 > self.frame_width
                or detection.y2 > self.frame_height
                or not math.isfinite(float(detection.confidence))
                or not 0 <= detection.confidence <= 1
            ):
                raise ValueError("detection geometry or confidence is invalid")


@dataclass(frozen=True, slots=True)
class TrackedDetection:
    """Internal tracker output without exposing a third-party tracker type."""

    track_id: int
    detection: Detection

    def __post_init__(self) -> None:
        if type(self.track_id) is not int or self.track_id < 0:
            raise ValueError("track_id must be a non-negative integer")


class TrackerAdapter(Protocol):
    """Narrow contract required from a ByteTrack implementation."""

    def update(
        self,
        detections: Sequence[Detection],
        *,
        frame_width: int,
        frame_height: int,
    ) -> tuple[TrackedDetection, ...]:
        """Associate the current frame detections with bounded track IDs."""

    def reset(self) -> None:
        """Release all tracker state."""


@dataclass(frozen=True, slots=True)
class ActiveTrack:
    """Safe bounded metadata for a confirmed track shown in the live UI."""

    provenance: TrackingProvenance
    track_id: int
    state: Literal["confirmed", "active"]
    first_seen_at: datetime
    last_seen_at: datetime
    last_frame_sequence: int
    last_box_xyxy: tuple[float, float, float, float]
    last_confidence: float
    observation_count: int
    maximum_confidence: float

    def __post_init__(self) -> None:
        if self.state not in {"confirmed", "active"}:
            raise ValueError("active track state must be confirmed or active")
        if type(self.track_id) is not int or self.track_id < 0:
            raise ValueError("track_id must be a non-negative integer")
        if type(self.last_frame_sequence) is not int or self.last_frame_sequence < 0:
            raise ValueError("last_frame_sequence must be a non-negative integer")
        if type(self.observation_count) is not int or self.observation_count <= 0:
            raise ValueError("observation_count must be positive")
        _validate_box(self.last_box_xyxy)
        _validate_confidence(self.last_confidence)
        _validate_confidence(self.maximum_confidence)

    def as_dict(self) -> dict[str, Any]:
        """Return safe JSON metadata for the synchronized live protocol."""
        return {
            "camera_id": self.provenance.camera_id,
            "capture_session_id": self.provenance.capture_session_id,
            "generation_number": self.provenance.generation_number,
            "stream_epoch": self.provenance.stream_epoch,
            "model_id": self.provenance.model_id,
            "model_checksum": self.provenance.model_checksum,
            "track_id": self.track_id,
            "state": self.state,
            "first_seen_utc": _timestamp(self.first_seen_at),
            "last_seen_utc": _timestamp(self.last_seen_at),
            "last_frame_sequence": self.last_frame_sequence,
            "last_box_xyxy": list(self.last_box_xyxy),
            "last_confidence": self.last_confidence,
            "observation_count": self.observation_count,
            "maximum_confidence": self.maximum_confidence,
        }


@dataclass(frozen=True, slots=True)
class ClosedTrackEvent:
    """One immutable aggregate emitted exactly once for a confirmed track."""

    event_id: str
    provenance: TrackingProvenance
    track_id: int
    first_seen_at: datetime
    last_seen_at: datetime
    duration_seconds: float
    observation_count: int
    maximum_confidence: float
    event_state: Literal["closed"] = "closed"
    crop_candidates: tuple[CropCandidate, ...] = ()

    @property
    def camera_id(self) -> str:
        """Return the source camera identity."""
        return self.provenance.camera_id

    @property
    def capture_session_id(self) -> str:
        """Return the capture-session identity."""
        return self.provenance.capture_session_id

    @property
    def generation_number(self) -> int:
        """Return the live pipeline generation."""
        return self.provenance.generation_number

    @property
    def stream_epoch(self) -> str:
        """Return the synchronized display epoch."""
        return self.provenance.stream_epoch

    @property
    def model_id(self) -> str:
        """Return the detector model identity."""
        return self.provenance.model_id

    @property
    def model_checksum(self) -> str:
        """Return the detector model checksum."""
        return self.provenance.model_checksum

    @property
    def id(self) -> str:
        """Return the event identifier used by persistence."""
        return self.event_id

    def __post_init__(self) -> None:
        if not self.event_id:
            raise ValueError("event_id is required")
        if type(self.track_id) is not int or self.track_id < 0:
            raise ValueError("track_id must be a non-negative integer")
        if self.first_seen_at.tzinfo is None or self.last_seen_at.tzinfo is None:
            raise ValueError("event timestamps must be timezone-aware")
        first_seen_at = self.first_seen_at.astimezone(UTC)
        last_seen_at = self.last_seen_at.astimezone(UTC)
        object.__setattr__(self, "first_seen_at", first_seen_at)
        object.__setattr__(self, "last_seen_at", last_seen_at)
        if last_seen_at < first_seen_at:
            raise ValueError("event timestamps must be ordered")
        if self.duration_seconds < 0 or not math.isfinite(self.duration_seconds):
            raise ValueError("duration_seconds must be finite and non-negative")
        if type(self.observation_count) is not int or self.observation_count < 3:
            raise ValueError("closed events require at least three observations")
        _validate_confidence(self.maximum_confidence)
        if len(self.crop_candidates) > 3:
            raise ValueError("closed events may retain at most three crop candidates")

    @classmethod
    def create(
        cls,
        *,
        provenance: TrackingProvenance,
        track_id: int,
        first_seen_at: datetime,
        last_seen_at: datetime,
        observation_count: int,
        maximum_confidence: float,
        crop_candidates: Sequence[CropCandidate] = (),
    ) -> ClosedTrackEvent:
        """Create one immutable event aggregate with a fresh local ID."""
        first_seen = first_seen_at.astimezone(UTC)
        last_seen = last_seen_at.astimezone(UTC)
        duration_seconds = max(0.0, (last_seen - first_seen).total_seconds())
        return cls(
            event_id=str(uuid4()),
            provenance=provenance,
            track_id=track_id,
            first_seen_at=first_seen,
            last_seen_at=last_seen,
            duration_seconds=duration_seconds,
            observation_count=observation_count,
            maximum_confidence=maximum_confidence,
            crop_candidates=tuple(crop_candidates),
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe closed event payload."""
        return {
            "event_id": self.event_id,
            **self.provenance.as_dict(),
            "track_id": self.track_id,
            "first_seen_utc": _timestamp(self.first_seen_at),
            "last_seen_utc": _timestamp(self.last_seen_at),
            "duration_seconds": round(self.duration_seconds, 3),
            "observation_count": self.observation_count,
            "maximum_confidence": self.maximum_confidence,
            "event_state": self.event_state,
        }

    def without_crop_candidates(self) -> ClosedTrackEvent:
        """Return the bounded public event view without retained crop pixels."""
        return ClosedTrackEvent(
            event_id=self.event_id,
            provenance=self.provenance,
            track_id=self.track_id,
            first_seen_at=self.first_seen_at,
            last_seen_at=self.last_seen_at,
            duration_seconds=self.duration_seconds,
            observation_count=self.observation_count,
            maximum_confidence=self.maximum_confidence,
            event_state=self.event_state,
        )


@dataclass(frozen=True, slots=True)
class TrackingUpdate:
    """Result of one frame or clock tick."""

    active_tracks: tuple[ActiveTrack, ...] = ()
    closed_events: tuple[ClosedTrackEvent, ...] = ()
    accepted: bool = True
    stale: bool = False


def _validate_box(box: Sequence[float]) -> None:
    if len(box) != 4:
        raise ValueError("track box must contain four coordinates")
    values = tuple(float(value) for value in box)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("track box must contain finite coordinates")
    if values[0] < 0 or values[1] < 0 or values[0] >= values[2] or values[1] >= values[3]:
        raise ValueError("track box must have positive dimensions")


def _validate_confidence(value: float) -> None:
    if not math.isfinite(float(value)) or not 0 <= float(value) <= 1:
        raise ValueError("confidence must be between 0 and 1")


def _timestamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


__all__ = [
    "ActiveTrack",
    "ClosedTrackEvent",
    "LiveDetectionFrame",
    "TrackedDetection",
    "TrackerAdapter",
    "TrackingProvenance",
    "TrackingUpdate",
    "TrackLifecycleState",
]
