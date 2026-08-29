"""Camera preview lifecycle and reconnect recovery."""

from __future__ import annotations

import random
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, Protocol

from ..cameras.repository import CameraConfig
from ..redaction import redact_text, redact_value
from .broker import LatestFrameBroker
from .contracts import Clock, FrameSource, SourceError, SourceInfo, SystemClock, VideoFrame
from .worker import FrameCaptureWorker

LifecycleState = Literal[
    "stopped",
    "connecting",
    "streaming",
    "degraded",
    "reconnecting",
    "stopping",
    "failed",
]
SourceFactory = Callable[[CameraConfig, str], FrameSource]


class WaitScheduler(Protocol):
    """Wait for a stop signal without coupling time to the runtime."""

    def wait(self, stop_requested: threading.Event, timeout: float) -> bool:
        """Wait for a timeout or a stop signal and return whether stopped."""

    def wake(self) -> None:
        """Wake a wait that is in progress."""


class EventWaitScheduler:
    """Use a real event wait for production runtime execution."""

    def wait(self, stop_requested: threading.Event, timeout: float) -> bool:
        return stop_requested.wait(max(0.0, timeout))

    def wake(self) -> None:
        """The event itself wakes when it is set."""


class ActiveCameraConflict(RuntimeError):
    """Raised when a second camera is started while one is active."""

    def __init__(self, active_camera_id: str, requested_camera_id: str) -> None:
        super().__init__(
            "another camera is active "
            f"({active_camera_id}); stop it before starting camera {requested_camera_id}"
        )
        self.active_camera_id = active_camera_id
        self.requested_camera_id = requested_camera_id


@dataclass(frozen=True, slots=True)
class ReconnectBackoff:
    """Bounded exponential backoff with injectable jitter."""

    base_delay_seconds: float = 0.5
    cap_seconds: float = 10.0
    jitter_ratio: float = 0.2
    random_value: Callable[[], float] = random.random

    def __post_init__(self) -> None:
        if self.base_delay_seconds <= 0:
            raise ValueError("base_delay_seconds must be positive")
        if self.cap_seconds < self.base_delay_seconds:
            raise ValueError("cap_seconds must be at least base_delay_seconds")
        if not 0 <= self.jitter_ratio <= 1:
            raise ValueError("jitter_ratio must be between 0 and 1")

    def delay_for(self, attempt: int) -> float:
        """Return the delay for a one-based reconnect attempt."""
        if attempt < 1:
            raise ValueError("attempt must be at least 1")
        exponential = min(
            self.cap_seconds,
            self.base_delay_seconds * (2 ** (attempt - 1)),
        )
        jitter = (float(self.random_value()) * 2 - 1) * self.jitter_ratio
        result = max(0.0, min(self.cap_seconds, exponential * (1 + jitter)))
        return float(result)


class ReconnectStateMachine:
    """Clock-driven state machine for initial open, disconnect, and stop."""

    def __init__(
        self,
        *,
        clock: Clock,
        backoff: ReconnectBackoff,
        stable_stream_seconds: float,
    ) -> None:
        self._clock = clock
        self._backoff = backoff
        self._stable_stream_seconds = stable_stream_seconds
        self.state: LifecycleState = "stopped"
        self.reconnect_attempt = 0
        self.next_retry_at: float | None = None
        self.last_error: str | None = None
        self.has_successful_session = False
        self._stable_since: float | None = None
        self._history: list[LifecycleState] = ["stopped"]

    @property
    def history(self) -> tuple[LifecycleState, ...]:
        """Return the state transitions for the current runtime generation."""
        return tuple(self._history)

    def start(self) -> None:
        """Enter the initial connection attempt."""
        self._transition("connecting")
        self.reconnect_attempt = 0
        self.next_retry_at = None
        self.last_error = None
        self.has_successful_session = False
        self._stable_since = None

    def opened(self) -> None:
        """Record a successful source open and start the stable timer."""
        self.has_successful_session = True
        self._stable_since = self._clock.monotonic()
        self.next_retry_at = None
        self.last_error = None
        self._transition("streaming")

    def initial_open_failed(self, error: str) -> None:
        """Enter terminal failed state for an initial source-open error."""
        if self.has_successful_session:
            self.disconnected(error)
            return
        self.last_error = redact_text(error)
        self.next_retry_at = None
        self._transition("failed")

    def disconnected(self, error: str) -> None:
        """Enter degraded state after a previously successful session ends."""
        self.last_error = redact_text(error)
        self.next_retry_at = None
        self._transition("degraded")

    def schedule_reconnect(self) -> float:
        """Enter reconnecting state and return its bounded wait duration."""
        if not self.has_successful_session:
            raise RuntimeError("reconnect is not available before a successful session")
        self.reconnect_attempt += 1
        delay = self._backoff.delay_for(self.reconnect_attempt)
        self.next_retry_at = self._clock.monotonic() + delay
        self._transition("reconnecting")
        return delay

    def poll_stability(self) -> bool:
        """Reset retry state after the stream remains stable long enough."""
        if (
            self.state != "streaming"
            or self._stable_since is None
            or self._clock.monotonic() - self._stable_since < self._stable_stream_seconds
        ):
            return False
        self.reconnect_attempt = 0
        return True

    def stopping(self) -> None:
        """Record a user or application stop request."""
        self.next_retry_at = None
        self._transition("stopping")

    def stopped(self) -> None:
        """Record that all owned resources are closed."""
        self.next_retry_at = None
        self.reconnect_attempt = 0
        self._transition("stopped")

    def _transition(self, state: LifecycleState) -> None:
        if self.state != state:
            self.state = state
            self._history.append(state)


@dataclass(frozen=True, slots=True)
class RuntimeCounters:
    """Cumulative counters for one active camera run."""

    captured_frames: int = 0
    consumed_frames: int = 0
    replaced_frames: int = 0
    decode_errors: int = 0
    reconnect_count: int = 0


@dataclass(frozen=True, slots=True)
class CameraRuntimeStatus:
    """Safe immutable snapshot for API, HTML, and diagnostics."""

    camera_id: str | None
    camera_name: str | None
    state: LifecycleState
    reconnect_attempt: int
    next_retry_in_seconds: float | None
    last_error: str | None
    source_info: SourceInfo | None
    metrics: dict[str, Any]
    active_camera_id: str | None = None
    detail: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe status without credentials or frame pixels."""
        source = _source_payload(self.source_info)
        payload: dict[str, Any] = {
            "camera_id": self.camera_id,
            "camera_name": self.camera_name,
            "state": self.state,
            "lifecycle_state": self.state,
            "reconnect_attempt": self.reconnect_attempt,
            "next_retry_in_seconds": self.next_retry_in_seconds,
            "last_error": redact_text(self.last_error) if self.last_error else None,
            "source": source,
            "stream_metadata": source,
            "metrics": redact_value(self.metrics),
        }
        if self.active_camera_id is not None:
            payload["active_camera_id"] = self.active_camera_id
        if self.detail is not None:
            payload["detail"] = redact_text(self.detail)
        return payload


class CameraRuntime:
    """Own one camera source, capture worker, broker, and reconnect loop."""

    def __init__(
        self,
        source_factory: SourceFactory,
        *,
        clock: Clock | None = None,
        backoff: ReconnectBackoff | None = None,
        stable_stream_seconds: float = 3.0,
        poll_interval_seconds: float = 0.05,
        worker_stop_timeout_seconds: float = 2.0,
        degraded_hold_seconds: float = 0.05,
        scheduler: WaitScheduler | None = None,
    ) -> None:
        if stable_stream_seconds < 0:
            raise ValueError("stable_stream_seconds must not be negative")
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive")
        if degraded_hold_seconds < 0:
            raise ValueError("degraded_hold_seconds must not be negative")
        self._source_factory = source_factory
        self._clock = clock or SystemClock()
        self._backoff = backoff or ReconnectBackoff()
        self._stable_stream_seconds = stable_stream_seconds
        self._poll_interval_seconds = poll_interval_seconds
        self._worker_stop_timeout_seconds = worker_stop_timeout_seconds
        self._degraded_hold_seconds = degraded_hold_seconds
        self._scheduler = scheduler or EventWaitScheduler()
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._stop_requested = threading.Event()
        self._thread: threading.Thread | None = None
        self._worker: FrameCaptureWorker | None = None
        self._broker: LatestFrameBroker | None = None
        self._camera_id: str | None = None
        self._camera_name: str | None = None
        self._camera_config: CameraConfig | None = None
        self._source_info: SourceInfo | None = None
        self._state: LifecycleState = "stopped"
        self._reconnect_attempt = 0
        self._next_retry_at: float | None = None
        self._last_error: str | None = None
        self._counters = RuntimeCounters()
        self._last_camera_id: str | None = None
        self._last_camera_name: str | None = None
        self._state_machine = ReconnectStateMachine(
            clock=self._clock,
            backoff=self._backoff,
            stable_stream_seconds=self._stable_stream_seconds,
        )

    def start(self, camera_id: str, camera: CameraConfig) -> CameraRuntimeStatus:
        """Start one camera asynchronously and return its connecting status."""
        with self._condition:
            if self._thread is not None and self._thread.is_alive():
                if self._camera_id != camera_id:
                    raise ActiveCameraConflict(self._camera_id or "unknown", camera_id)
                return self._status_locked()
            if not camera.enabled:
                raise RuntimeError("camera is disabled; enable it before starting the preview")
            self._stop_requested = threading.Event()
            self._camera_id = camera_id
            self._camera_name = camera.name
            self._camera_config = camera
            self._last_camera_id = camera_id
            self._last_camera_name = camera.name
            self._state_machine = ReconnectStateMachine(
                clock=self._clock,
                backoff=self._backoff,
                stable_stream_seconds=self._stable_stream_seconds,
            )
            self._state_machine.start()
            self._source_info = None
            self._sync_state_machine_locked()
            self._counters = RuntimeCounters()
            self._worker = None
            self._broker = None
            self._thread = threading.Thread(
                target=self._run,
                name="open-licenseplate-camera-runtime",
                daemon=True,
            )
            self._thread.start()
            return self._status_locked()

    def stop(self, camera_id: str | None = None, *, timeout: float = 5.0) -> CameraRuntimeStatus:
        """Stop the active camera and cancel any reconnect wait."""
        with self._condition:
            if self._thread is None or not self._thread.is_alive():
                if camera_id is not None and self._last_camera_id not in {None, camera_id}:
                    return self._status_locked(
                        camera_id=camera_id,
                        camera_name=None,
                        detail="camera is not active",
                        inactive=True,
                    )
                if self._state != "stopped":
                    self._state_machine.stopping()
                    self._state_machine.stopped()
                    self._sync_state_machine_locked()
                return self._status_locked()
            if camera_id is not None and self._camera_id != camera_id:
                return self._status_locked(
                    camera_id=camera_id,
                    camera_name=None,
                    detail="camera is not active",
                    inactive=True,
                )
            thread = self._thread
            worker = self._worker
            self._stop_requested.set()
            self._state_machine.stopping()
            self._sync_state_machine_locked()
            self._scheduler.wake()
            self._condition.notify_all()

        if worker is not None:
            worker.stop(timeout=max(0.0, timeout))
        if thread is not None:
            thread.join(max(0.0, timeout))
        if thread is not None and thread.is_alive():
            raise RuntimeError("camera runtime did not stop before the deadline")

        with self._condition:
            self._state_machine.stopped()
            self._sync_state_machine_locked()
            self._condition.notify_all()
            return self._status_locked()

    def close(self, *, timeout: float = 5.0) -> None:
        """Release all runtime resources during application shutdown."""
        with self._condition:
            active_camera_id = self._camera_id
        if active_camera_id is not None:
            self.stop(active_camera_id, timeout=timeout)

    def status(self, camera_id: str | None = None) -> CameraRuntimeStatus:
        """Return the current status, or a stopped status for another camera."""
        with self._condition:
            if camera_id is not None and self._camera_id not in {None, camera_id}:
                return self._status_locked(
                    camera_id=camera_id,
                    camera_name=None,
                    detail=f"camera {camera_id} is not active",
                    inactive=True,
                )
            return self._status_locked()

    @property
    def state_history(self) -> tuple[LifecycleState, ...]:
        """Return state transitions for the current camera run."""
        with self._condition:
            return self._state_machine.history

    def wait_for_state(
        self,
        state: LifecycleState,
        *,
        timeout: float = 5.0,
        after_history_index: int = 0,
    ) -> CameraRuntimeStatus:
        """Wait for one state without polling or sleeping in the caller."""
        with self._condition:
            reached = self._condition.wait_for(
                lambda: state in self._state_machine.history[after_history_index:],
                timeout=max(0.0, timeout),
            )
            if not reached:
                raise TimeoutError(f"camera runtime did not reach {state}")
            return self._status_locked()

    def latest_frame(self, camera_id: str) -> VideoFrame | None:
        """Return the newest decoded frame without removing it from the broker."""
        with self._condition:
            if self._camera_id != camera_id or self._broker is None:
                return None
            broker = self._broker
        return broker.peek()

    def wait_for_frame(self, camera_id: str, timeout: float) -> VideoFrame | None:
        """Wait for a frame or for the camera to stop."""
        deadline = self._clock.monotonic() + max(0.0, timeout)
        while True:
            frame = self.latest_frame(camera_id)
            if frame is not None:
                return frame
            with self._condition:
                if self._camera_id != camera_id or self._state == "stopped":
                    return None
                remaining = deadline - self._clock.monotonic()
                if remaining <= 0:
                    return None
                self._condition.wait(min(self._poll_interval_seconds, remaining))

    def iter_preview(self, camera_id: str, *, fps: float = 10.0) -> Iterator[VideoFrame]:
        """Yield the latest frame at a bounded preview rate."""
        interval = 1.0 / max(fps, 0.1)
        while True:
            frame = self.wait_for_frame(camera_id, interval)
            if frame is None:
                with self._condition:
                    if self._camera_id != camera_id or self._state == "stopped":
                        return
                continue
            yield frame
            if self._stop_requested.wait(interval):
                return

    @property
    def active_camera_id(self) -> str | None:
        """Return the active camera identifier, if any."""
        with self._condition:
            return self._camera_id if self._state not in {"stopped", "failed"} else None

    def _run(self) -> None:
        initial_attempt = True
        try:
            while not self._stop_requested.is_set():
                broker = LatestFrameBroker()
                worker: FrameCaptureWorker | None = None
                opened = False
                failure: str | None = None
                try:
                    with self._condition:
                        camera_id = self._camera_id
                        camera_config = self._camera_config
                    if camera_id is None or camera_config is None:
                        return
                    source = self._source_factory(camera_config, camera_id)
                    worker = FrameCaptureWorker(source, broker)
                    with self._condition:
                        self._worker = worker
                        self._broker = broker
                        self._source_info = None
                        self._condition.notify_all()
                    worker.start()

                    while not self._stop_requested.is_set():
                        worker_metrics = worker.metrics()
                        source_info = worker.source_info
                        if source_info is not None and not opened:
                            self._set_streaming(source_info)
                            opened = True
                        if opened:
                            self._poll_stability()
                        if not worker_metrics.running:
                            failure = worker_metrics.error or "camera stream ended"
                            break
                        if self._scheduler.wait(
                            self._stop_requested,
                            self._poll_interval_seconds,
                        ):
                            break

                    if self._stop_requested.is_set():
                        break
                    worker_metrics = worker.metrics()
                    failure = worker_metrics.error or "camera stream ended"
                    worker.stop(timeout=self._worker_stop_timeout_seconds)
                except SourceError as error:
                    failure = str(error)
                except Exception as error:
                    failure = redact_text(str(error)) or "camera source failed"
                finally:
                    if worker is not None:
                        self._accumulate(
                            worker.metrics(),
                            reconnect=(opened or not initial_attempt)
                            and not self._stop_requested.is_set(),
                        )
                    elif not initial_attempt and not self._stop_requested.is_set():
                        self._record_reconnect()
                    broker.close()
                    with self._condition:
                        if self._worker is worker:
                            self._worker = None
                        if self._broker is broker:
                            self._broker = None
                        self._condition.notify_all()

                if self._stop_requested.is_set():
                    break

                if not opened and initial_attempt and not self._has_successful_session():
                    self._set_failed(failure or "camera source could not be opened")
                    break
                self._set_degraded(failure or "camera stream ended")
                if self._scheduler.wait(
                    self._stop_requested,
                    self._degraded_hold_seconds,
                ):
                    break
                delay = self._schedule_reconnect()
                initial_attempt = False
                if self._scheduler.wait(self._stop_requested, delay):
                    break
        finally:
            with self._condition:
                self._worker = None
                finished_broker = self._broker
                self._broker = None
                if self._stop_requested.is_set() or self._state != "failed":
                    self._state_machine.stopped()
                    self._sync_state_machine_locked()
                self._condition.notify_all()
            if finished_broker is not None:
                finished_broker.close()

    def _set_streaming(self, source_info: SourceInfo) -> None:
        with self._condition:
            self._source_info = source_info
            self._state_machine.opened()
            self._sync_state_machine_locked()
            self._condition.notify_all()

    def _set_degraded(self, error: str) -> None:
        with self._condition:
            self._state_machine.disconnected(error)
            self._sync_state_machine_locked()
            self._condition.notify_all()

    def _set_failed(self, error: str) -> None:
        with self._condition:
            self._state_machine.initial_open_failed(error)
            self._sync_state_machine_locked()
            self._condition.notify_all()

    def _schedule_reconnect(self) -> float:
        with self._condition:
            delay = self._state_machine.schedule_reconnect()
            self._sync_state_machine_locked()
            self._condition.notify_all()
            return delay

    def _poll_stability(self) -> None:
        with self._condition:
            self._state_machine.poll_stability()
            self._sync_state_machine_locked()
            self._condition.notify_all()

    def _has_successful_session(self) -> bool:
        with self._condition:
            return self._state_machine.has_successful_session

    def _sync_state_machine_locked(self) -> None:
        self._state = self._state_machine.state
        self._reconnect_attempt = self._state_machine.reconnect_attempt
        self._next_retry_at = self._state_machine.next_retry_at
        self._last_error = self._state_machine.last_error

    def _accumulate(self, metrics: Any, *, reconnect: bool) -> None:
        with self._condition:
            self._counters = RuntimeCounters(
                captured_frames=self._counters.captured_frames + metrics.captured_frames,
                consumed_frames=self._counters.consumed_frames + metrics.consumed_frames,
                replaced_frames=self._counters.replaced_frames + metrics.replaced_frames,
                decode_errors=self._counters.decode_errors + metrics.decode_errors,
                reconnect_count=self._counters.reconnect_count + (1 if reconnect else 0),
            )

    def _record_reconnect(self) -> None:
        with self._condition:
            self._counters = RuntimeCounters(
                captured_frames=self._counters.captured_frames,
                consumed_frames=self._counters.consumed_frames,
                replaced_frames=self._counters.replaced_frames,
                decode_errors=self._counters.decode_errors,
                reconnect_count=self._counters.reconnect_count + 1,
            )

    def _status_locked(
        self,
        *,
        camera_id: str | None = None,
        camera_name: str | None = None,
        detail: str | None = None,
        inactive: bool = False,
    ) -> CameraRuntimeStatus:
        requested_id = camera_id if camera_id is not None else self._camera_id
        requested_name = camera_name if camera_name is not None else self._camera_name
        now = self._clock.monotonic()
        next_retry = None if self._next_retry_at is None else max(0.0, self._next_retry_at - now)
        worker = None if inactive else self._worker
        broker = None if inactive else self._broker
        current_metrics = worker.metrics() if worker is not None else None
        broker_metrics = broker.metrics(now_monotonic=now) if broker is not None else None
        metrics = {
            "captured_frames": self._counters.captured_frames
            + (0 if current_metrics is None else current_metrics.captured_frames),
            "consumed_frames": self._counters.consumed_frames
            + (0 if current_metrics is None else current_metrics.consumed_frames),
            "replaced_frames": self._counters.replaced_frames
            + (0 if current_metrics is None else current_metrics.replaced_frames),
            "decode_errors": self._counters.decode_errors
            + (0 if current_metrics is None else current_metrics.decode_errors),
            "reconnect_count": self._counters.reconnect_count,
            "newest_frame_age_seconds": (
                None if broker_metrics is None else broker_metrics.newest_frame_age_seconds
            ),
            "frame_age_seconds": (
                None if broker_metrics is None else broker_metrics.newest_frame_age_seconds
            ),
        }
        return CameraRuntimeStatus(
            camera_id=requested_id,
            camera_name=requested_name,
            state="stopped" if inactive else self._state,
            reconnect_attempt=0 if inactive else self._reconnect_attempt,
            next_retry_in_seconds=None if inactive else next_retry,
            last_error=None if inactive else self._last_error,
            source_info=None if inactive else self._source_info,
            metrics=metrics,
            active_camera_id=(
                self._camera_id
                if self._camera_id is not None and self._state not in {"stopped", "failed"}
                else None
            ),
            detail=detail,
        )


def _source_payload(info: SourceInfo | None) -> dict[str, Any] | None:
    if info is None:
        return None
    resolution = None
    if info.width is not None and info.height is not None:
        resolution = f"{info.width}x{info.height}"
    payload = {
        "source_name": info.source_name,
        "capture_session_id": info.capture_session_id,
        "codec": info.codec,
        "width": info.width,
        "height": info.height,
        "resolution": resolution,
        "nominal_fps": info.nominal_fps,
        "transport": info.transport,
        "camera_pts_available": info.has_camera_pts,
        "has_camera_pts": info.has_camera_pts,
        "endpoint": redact_text(info.endpoint) if info.endpoint else None,
        "started_at": _timestamp(info.started_at),
    }
    result = redact_value(payload)
    if not isinstance(result, dict):
        raise TypeError("source metadata must be a mapping")
    return result


def _timestamp(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


__all__ = [
    "ActiveCameraConflict",
    "CameraRuntime",
    "CameraRuntimeStatus",
    "EventWaitScheduler",
    "LifecycleState",
    "ReconnectBackoff",
    "ReconnectStateMachine",
    "RuntimeCounters",
    "SourceFactory",
    "WaitScheduler",
]
