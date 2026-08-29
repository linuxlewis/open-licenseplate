"""Typed contracts and runtime values for decoded video frames."""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable
from uuid import uuid4

from ..redaction import redact_text


class SourceError(RuntimeError):
    """A safe, user-facing source failure."""

    def __init__(self, message: str) -> None:
        super().__init__(redact_text(message))


class SourceLifecycleError(SourceError):
    """Raised when a source method is called in an invalid lifecycle state."""


class SourceOpenError(SourceError):
    """Raised when a source cannot open its input."""


class SourceReadError(SourceError):
    """Raised when a source cannot decode its next frame."""


class CaptureShutdownError(SourceError):
    """Raised when a capture worker does not stop before its deadline."""


class Clock(Protocol):
    """Clock methods used by capture components."""

    def now(self) -> datetime:
        """Return the current UTC wall-clock time."""

    def monotonic(self) -> float:
        """Return a process-local monotonic time."""


class SystemClock:
    """Use the host UTC and monotonic clocks."""

    def now(self) -> datetime:
        return datetime.now(UTC)

    def monotonic(self) -> float:
        return time.monotonic()


def new_capture_session_id() -> str:
    """Return a fresh runtime capture-session identifier."""
    return str(uuid4())


@dataclass(frozen=True, slots=True)
class CaptureSession:
    """Runtime identity and timestamps for one source lifecycle."""

    id: str
    camera_id: str | None
    started_at: datetime
    started_monotonic: float
    ended_at: datetime | None = None
    end_reason: str | None = None

    @property
    def capture_session_id(self) -> str:
        """Return the stable identifier used on every frame."""
        return self.id


@dataclass(frozen=True, slots=True)
class SourceInfo:
    """Safe metadata negotiated when a source opens."""

    source_name: str
    session: CaptureSession
    codec: str | None = None
    width: int | None = None
    height: int | None = None
    nominal_fps: float | None = None
    has_camera_pts: bool = False
    transport: str | None = None
    endpoint: str | None = None

    @property
    def capture_session_id(self) -> str:
        """Return the runtime session identity."""
        return self.session.id

    @property
    def session_id(self) -> str:
        """Alias used by downstream live-processing code."""
        return self.session.id

    @property
    def started_at(self) -> datetime:
        """Return the UTC time at which the source opened."""
        return self.session.started_at


@dataclass(frozen=True, slots=True)
class VideoFrame:
    """One decoded frame and its independent host and source timestamps."""

    sequence: int
    data: Any
    pixel_format: str
    host_received_at: datetime
    host_received_monotonic: float
    capture_session_id: str
    width: int
    height: int
    camera_pts: int | None = None
    camera_pts_seconds: float | None = None

    @property
    def pixels(self) -> Any:
        """Return pixel data using the domain term used by the specification."""
        return self.data

    @property
    def received_at(self) -> datetime:
        """Return the host UTC receipt time."""
        return self.host_received_at

    @property
    def received_monotonic(self) -> float:
        """Return the host monotonic receipt time."""
        return self.host_received_monotonic

    @property
    def pts(self) -> int | None:
        """Return the camera or media presentation timestamp."""
        return self.camera_pts


@runtime_checkable
class FrameSource(Protocol):
    """Synchronous source contract owned by a capture worker thread."""

    def open(self) -> SourceInfo:
        """Open the source and start a new capture session."""

    def read(self) -> VideoFrame | None:
        """Return the next frame, or None at normal end of input."""

    def close(self) -> None:
        """Stop decode and release source resources promptly."""
