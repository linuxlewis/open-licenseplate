"""Camera preview lifecycle and reconnect recovery."""

from __future__ import annotations

import random
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from ..cameras.repository import CameraConfig
from ..redaction import redact_text, redact_value
from .broker import LatestFrameBroker
from .contracts import Clock, FrameSource, SourceError, SourceInfo, SystemClock, VideoFrame
from .worker import FrameCaptureWorker

LifecycleState = Literal["stopped", "connecting", "streaming", "degraded", "reconnecting"]
SourceFactory = Callable[[CameraConfig, str], FrameSource]


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
            self._source_info = None
            self._state = "connecting"
            self._reconnect_attempt = 0
            self._next_retry_at = None
            self._last_error = None
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
                self._state = "stopped"
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
            self._condition.notify_all()

        if worker is not None:
            worker.stop(timeout=max(0.0, timeout))
        if thread is not None:
            thread.join(max(0.0, timeout))
        if thread is not None and thread.is_alive():
            raise RuntimeError("camera runtime did not stop before the deadline")

        with self._condition:
            self._state = "stopped"
            self._next_retry_at = None
            self._reconnect_attempt = 0
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
            return self._camera_id if self._state != "stopped" else None

    def _run(self) -> None:
        first_attempt = True
        retry_index = 0
        try:
            while not self._stop_requested.is_set():
                self._set_state("connecting" if first_attempt else "reconnecting")
                broker = LatestFrameBroker()
                worker: FrameCaptureWorker | None = None
                stream_started: float | None = None
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
                        if source_info is not None and stream_started is None:
                            stream_started = self._clock.monotonic()
                            self._set_streaming(source_info)
                        if (
                            stream_started is not None
                            and self._clock.monotonic() - stream_started
                            >= self._stable_stream_seconds
                        ):
                            retry_index = 0
                            self._set_reconnect_attempt(0)
                        if not worker_metrics.running:
                            failure = worker_metrics.error or "camera stream ended"
                            break
                        self._stop_requested.wait(self._poll_interval_seconds)

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
                            worker.metrics(), reconnect=not self._stop_requested.is_set()
                        )
                    elif not self._stop_requested.is_set():
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

                if stream_started is not None and (
                    self._clock.monotonic() - stream_started >= self._stable_stream_seconds
                ):
                    retry_index = 0
                self._set_degraded(failure or "camera stream ended")
                if self._stop_requested.wait(self._degraded_hold_seconds):
                    break
                retry_index += 1
                delay = self._backoff.delay_for(retry_index)
                self._set_reconnecting(retry_index, delay)
                first_attempt = False
                if self._stop_requested.wait(delay):
                    break
        finally:
            with self._condition:
                self._worker = None
                finished_broker = self._broker
                self._broker = None
                self._state = "stopped"
                self._next_retry_at = None
                self._reconnect_attempt = 0
                self._condition.notify_all()
            if finished_broker is not None:
                finished_broker.close()

    def _set_streaming(self, source_info: SourceInfo) -> None:
        with self._condition:
            self._source_info = source_info
            self._state = "streaming"
            self._next_retry_at = None
            self._last_error = None
            self._condition.notify_all()

    def _set_degraded(self, error: str) -> None:
        with self._condition:
            self._state = "degraded"
            self._last_error = redact_text(error)
            self._next_retry_at = None
            self._condition.notify_all()

    def _set_reconnecting(self, attempt: int, delay: float) -> None:
        with self._condition:
            self._state = "reconnecting"
            self._reconnect_attempt = attempt
            self._next_retry_at = self._clock.monotonic() + delay
            self._condition.notify_all()

    def _set_state(self, state: LifecycleState) -> None:
        with self._condition:
            self._state = state
            self._next_retry_at = None
            self._condition.notify_all()

    def _set_reconnect_attempt(self, attempt: int) -> None:
        with self._condition:
            self._reconnect_attempt = attempt
            self._condition.notify_all()

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
                if self._camera_id is not None and self._state != "stopped"
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
    "LifecycleState",
    "ReconnectBackoff",
    "RuntimeCounters",
    "SourceFactory",
]
