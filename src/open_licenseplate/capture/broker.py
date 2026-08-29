"""Capacity-one latest-frame exchange between capture and consumers."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from .contracts import VideoFrame


@dataclass(frozen=True, slots=True)
class BrokerMetrics:
    """Point-in-time broker counters and latest-frame age."""

    captured_frames: int
    consumed_frames: int
    replaced_frames: int
    newest_frame_age_seconds: float | None
    has_frame: bool
    closed: bool

    @property
    def frame_age_seconds(self) -> float | None:
        """Alias for consumers that use the shorter metric name."""
        return self.newest_frame_age_seconds


class LatestFrameBroker:
    """Thread-safe capacity-one storage that always favors the newest frame."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._frame: VideoFrame | None = None
        self._closed = False
        self._captured_frames = 0
        self._consumed_frames = 0
        self._replaced_frames = 0

    def put(self, frame: VideoFrame) -> bool:
        """Publish one frame without waiting for a consumer.

        Return False when the broker is closed. An unread frame is replaced in
        place, so the broker never creates a backlog.
        """
        old_frame: VideoFrame | None
        with self._condition:
            if self._closed:
                return False
            old_frame = self._frame
            self._frame = frame
            self._captured_frames += 1
            if old_frame is not None:
                self._replaced_frames += 1
            self._condition.notify()
        del old_frame
        return True

    publish = put

    def get(self, timeout: float | None = None) -> VideoFrame | None:
        """Wait for and consume the current frame.

        A closed broker returns None. A timeout also returns None. The timeout
        is measured with the process-local monotonic clock.
        """
        deadline = None if timeout is None else time.monotonic() + max(timeout, 0.0)
        with self._condition:
            while self._frame is None and not self._closed:
                if deadline is None:
                    self._condition.wait()
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(remaining)

            if self._frame is None:
                return None
            frame = self._frame
            self._frame = None
            self._consumed_frames += 1
            return frame

    consume = get
    get_latest = get
    read = get
    take = get

    def peek(self) -> VideoFrame | None:
        """Return the current frame without consuming it."""
        with self._condition:
            return self._frame

    def clear(self) -> None:
        """Release an unread frame while keeping the broker open."""
        with self._condition:
            old_frame = self._frame
            self._frame = None
        del old_frame

    def close(self) -> None:
        """Close the broker and release its unread frame reference."""
        with self._condition:
            old_frame = self._frame
            self._frame = None
            self._closed = True
            self._condition.notify_all()
        del old_frame

    @property
    def closed(self) -> bool:
        """Return whether the broker accepts no more frames."""
        with self._condition:
            return self._closed

    def metrics(self, *, now_monotonic: float | None = None) -> BrokerMetrics:
        """Return counters and the age of the current unread frame."""
        now = time.monotonic() if now_monotonic is None else now_monotonic
        with self._condition:
            frame = self._frame
            age = None
            if frame is not None:
                age = max(0.0, now - frame.host_received_monotonic)
            return BrokerMetrics(
                captured_frames=self._captured_frames,
                consumed_frames=self._consumed_frames,
                replaced_frames=self._replaced_frames,
                newest_frame_age_seconds=age,
                has_frame=frame is not None,
                closed=self._closed,
            )

    stats = metrics
