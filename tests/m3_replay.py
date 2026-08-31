"""Deterministic M3 replay fixtures and bounded source helpers."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

from open_licenseplate.capture import (
    CaptureSession,
    SourceInfo,
    VideoFrame,
)


@dataclass(frozen=True, slots=True)
class M3ReplayFixture:
    """Small synthetic road scene with explicit no-plate and plate frames."""

    fixture_id: str
    width: int
    height: int
    frame_rate: float
    background_value: int
    plate_value: int
    plate_box: tuple[int, int, int, int]
    one_hour_frames: int
    checkpoint_frames: int
    memory_tolerance_bytes: int
    sustained_growth_checkpoints: int

    @classmethod
    def load(cls) -> M3ReplayFixture:
        path = Path(__file__).parent / "fixtures" / "replay" / "m3_live_replay.json"
        values = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            fixture_id=str(values["fixture_id"]),
            width=int(values["width"]),
            height=int(values["height"]),
            frame_rate=float(values["frame_rate"]),
            background_value=int(values["background_value"]),
            plate_value=int(values["plate_value"]),
            plate_box=tuple(int(value) for value in values["plate_box"]),
            one_hour_frames=int(values["one_hour_frames"]),
            checkpoint_frames=int(values["checkpoint_frames"]),
            memory_tolerance_bytes=int(values["memory_tolerance_bytes"]),
            sustained_growth_checkpoints=int(values["sustained_growth_checkpoints"]),
        )

    @property
    def checkpoint_count(self) -> int:
        """Return the bounded number of logical one-hour checkpoints."""
        return self.one_hour_frames // self.checkpoint_frames

    def frame(self, kind: str, index: int = 0) -> np.ndarray:
        """Render one synthetic BGR frame without external image data."""
        if kind not in {"no_plate", "plate"}:
            raise ValueError("M3 replay frame kind must be no_plate or plate")
        frame = np.full(
            (self.height, self.width, 3),
            self.background_value,
            dtype=np.uint8,
        )
        if kind == "plate":
            x1, y1, x2, y2 = self.plate_box
            frame[y1:y2, x1:x2, :] = self.plate_value
            stripe_x = x1 + (index % max(1, x2 - x1))
            frame[y1:y2, stripe_x : min(x2, stripe_x + 1), :] = self.background_value
        return frame


class ReplayClock:
    """Thread-safe logical clock advanced by the replay source."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._wall = datetime(2026, 8, 29, tzinfo=UTC)
        self._ticks = 0.0

    def now(self) -> datetime:
        with self._condition:
            return self._wall + timedelta(seconds=self._ticks)

    def monotonic(self) -> float:
        with self._condition:
            return self._ticks

    def advance(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("replay clock cannot move backwards")
        with self._condition:
            self._ticks += seconds
            self._condition.notify_all()


class ReplayGate:
    """Condition-based barrier for deterministic source checkpoints."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._reached = 0
        self._released = 0

    def mark(self, checkpoint: int) -> None:
        with self._condition:
            self._reached = max(self._reached, checkpoint)
            self._condition.notify_all()

    def wait_until_reached(self, checkpoint: int, timeout: float = 5.0) -> None:
        with self._condition:
            if not self._condition.wait_for(
                lambda: self._reached >= checkpoint,
                timeout=max(0.0, timeout),
            ):
                raise AssertionError(f"replay did not reach checkpoint {checkpoint}")

    def release(self, checkpoint: int) -> None:
        with self._condition:
            self._released = max(self._released, checkpoint)
            self._condition.notify_all()

    def wait_until_released(
        self,
        checkpoint: int,
        stop_requested: threading.Event | None = None,
    ) -> bool:
        with self._condition:
            return (
                self._condition.wait_for(
                    lambda: (
                        self._released >= checkpoint
                        or (stop_requested is not None and stop_requested.is_set())
                    )
                )
                and self._released >= checkpoint
            )


FrameFactory = Callable[[int], np.ndarray]


class GeneratedReplaySource:
    """Generate a bounded replay and wait at the end until the owner stops it."""

    def __init__(
        self,
        *,
        camera_id: str,
        total_frames: int,
        frame_rate: float,
        frame_factory: FrameFactory,
        clock: ReplayClock,
        session_id: str,
        start_gate: threading.Event | None = None,
        checkpoint_gate: ReplayGate | None = None,
        checkpoint_frames: int | None = None,
        logical_seconds_per_frame: float | None = None,
        hold_after_frames: int | None = None,
        hold_gate: ReplayGate | None = None,
    ) -> None:
        if total_frames <= 0:
            raise ValueError("replay must contain at least one frame")
        if frame_rate <= 0:
            raise ValueError("replay frame rate must be positive")
        if logical_seconds_per_frame is not None and logical_seconds_per_frame <= 0:
            raise ValueError("logical frame duration must be positive")
        if hold_after_frames is not None and hold_after_frames <= 0:
            raise ValueError("hold_after_frames must be positive")
        self._camera_id = camera_id
        self._total_frames = total_frames
        self._frame_rate = frame_rate
        self._frame_factory = frame_factory
        self._clock = clock
        self._session_id = session_id
        self._start_gate = start_gate
        self._checkpoint_gate = checkpoint_gate
        self._checkpoint_frames = checkpoint_frames
        self._logical_seconds_per_frame = logical_seconds_per_frame or 1.0 / frame_rate
        self._hold_after_frames = hold_after_frames
        self._hold_gate = hold_gate
        self._stop_requested = threading.Event()
        self._opened = threading.Event()
        self._closed = threading.Event()
        self._exhausted = threading.Event()
        self._frame_index = 0
        self._pending_checkpoint: int | None = None
        self._pending_hold = False
        self._session: CaptureSession | None = None

    def open(self) -> SourceInfo:
        if self._session is not None:
            raise RuntimeError("replay source is already open")
        session = CaptureSession(
            id=self._session_id,
            camera_id=self._camera_id,
            started_at=self._clock.now(),
            started_monotonic=self._clock.monotonic(),
        )
        self._session = session
        self._opened.set()
        return SourceInfo(
            source_name="m3-replay",
            session=session,
            width=self._frame_factory(0).shape[1],
            height=self._frame_factory(0).shape[0],
            nominal_fps=self._frame_rate,
            has_camera_pts=True,
        )

    def read(self) -> VideoFrame | None:
        if self._session is None:
            raise RuntimeError("replay source is not open")
        if self._start_gate is not None:
            start_gate = self._start_gate
            self._start_gate = None
            start_gate.wait()
            if self._stop_requested.is_set():
                return None
        if self._pending_checkpoint is not None:
            checkpoint = self._pending_checkpoint
            self._pending_checkpoint = None
            if self._checkpoint_gate is not None and not self._checkpoint_gate.wait_until_released(
                checkpoint,
                self._stop_requested,
            ):
                return None
        if self._pending_hold:
            self._pending_hold = False
            if self._hold_gate is None or not self._hold_gate.wait_until_released(
                1,
                self._stop_requested,
            ):
                return None
        if self._stop_requested.is_set():
            return None
        if self._frame_index >= self._total_frames:
            self._exhausted.set()
            self._stop_requested.wait()
            return None

        frame_number = self._frame_index + 1
        self._clock.advance(self._logical_seconds_per_frame)
        data = np.ascontiguousarray(self._frame_factory(self._frame_index))
        self._frame_index += 1
        if (
            self._checkpoint_gate is not None
            and self._checkpoint_frames is not None
            and frame_number % self._checkpoint_frames == 0
        ):
            self._pending_checkpoint = frame_number // self._checkpoint_frames
            self._checkpoint_gate.mark(self._pending_checkpoint)
        if self._hold_after_frames is not None and frame_number == self._hold_after_frames:
            self._pending_hold = True
            if self._hold_gate is not None:
                self._hold_gate.mark(1)
        return VideoFrame(
            sequence=frame_number,
            data=data,
            pixel_format="bgr24",
            host_received_at=self._clock.now(),
            host_received_monotonic=self._clock.monotonic(),
            capture_session_id=self._session.id,
            width=int(data.shape[1]),
            height=int(data.shape[0]),
            camera_pts=frame_number - 1,
            camera_pts_seconds=(frame_number - 1) / self._frame_rate,
        )

    def close(self) -> None:
        self._stop_requested.set()
        if self._session is not None and self._session.ended_at is None:
            self._session = replace(
                self._session,
                ended_at=self._clock.now(),
                end_reason="stopped",
            )
        self._closed.set()

    @property
    def opened(self) -> threading.Event:
        return self._opened

    @property
    def closed(self) -> threading.Event:
        return self._closed

    @property
    def exhausted(self) -> threading.Event:
        return self._exhausted

    @property
    def frames_emitted(self) -> int:
        return self._frame_index

    @property
    def capture_session_id(self) -> str:
        return self._session_id


__all__ = [
    "GeneratedReplaySource",
    "M3ReplayFixture",
    "ReplayClock",
    "ReplayGate",
]
