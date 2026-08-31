"""Thread ownership for blocking frame-source decode."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass

from .broker import LatestFrameBroker
from .contracts import (
    CaptureSession,
    CaptureShutdownError,
    Clock,
    FrameSource,
    SourceError,
    SourceInfo,
    SystemClock,
)


@dataclass(frozen=True, slots=True)
class CaptureMetrics:
    """Capture-worker counters and terminal state."""

    captured_frames: int
    consumed_frames: int
    replaced_frames: int
    newest_frame_age_seconds: float | None
    decode_errors: int
    reconnect_count: int
    running: bool
    failed: bool
    error: str | None


class FrameCaptureWorker:
    """Run a synchronous FrameSource away from the FastAPI event loop."""

    def __init__(
        self,
        source: FrameSource,
        broker: LatestFrameBroker,
        *,
        clock: Clock | None = None,
        on_error: Callable[[SourceError], None] | None = None,
        thread_name: str = "open-licenseplate-capture",
    ) -> None:
        self.source = source
        self.broker = broker
        self._clock = clock or SystemClock()
        self._on_error = on_error
        self._thread_name = thread_name
        self._stop_requested = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._source_info: SourceInfo | None = None
        self._session: CaptureSession | None = None
        self._captured_frames = 0
        self._decode_errors = 0
        self._reconnect_count = 0
        self._error: SourceError | None = None

    def start(self) -> None:
        """Start the capture thread and return without opening the source."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                raise RuntimeError("capture worker is already running")
            self._stop_requested.clear()
            self._source_info = None
            self._session = None
            self._captured_frames = 0
            self._decode_errors = 0
            self._reconnect_count = 0
            self._error = None
            self._thread = threading.Thread(
                target=self._run,
                name=self._thread_name,
                daemon=True,
            )
            self._thread.start()

    def stop(self, *, timeout: float = 5.0) -> None:
        """Cancel source I/O, release the broker, and join the worker."""
        self._stop_requested.set()
        self.broker.close()
        try:
            self.source.close()
        except Exception as error:
            self._record_error(SourceError(str(error)))
        thread = self._thread
        if thread is not None:
            thread.join(max(timeout, 0.0))
            if thread.is_alive():
                raise CaptureShutdownError("capture worker did not stop before the deadline")

    @property
    def source_info(self) -> SourceInfo | None:
        """Return the safe metadata negotiated by the source."""
        with self._lock:
            return self._source_info

    @property
    def capture_session(self) -> CaptureSession | None:
        """Return the runtime session with its terminal timestamp when known."""
        with self._lock:
            return self._session

    @property
    def failure(self) -> SourceError | None:
        """Return the safe source failure, if any."""
        with self._lock:
            return self._error

    def metrics(self) -> CaptureMetrics:
        """Return capture counters without touching the event loop."""
        with self._lock:
            thread = self._thread
            broker_metrics = self.broker.metrics()
            return CaptureMetrics(
                captured_frames=self._captured_frames,
                consumed_frames=broker_metrics.consumed_frames,
                replaced_frames=broker_metrics.replaced_frames,
                newest_frame_age_seconds=broker_metrics.newest_frame_age_seconds,
                decode_errors=self._decode_errors,
                reconnect_count=self._reconnect_count,
                running=thread is not None and thread.is_alive(),
                failed=self._error is not None,
                error=None if self._error is None else str(self._error),
            )

    def _run(self) -> None:
        session: CaptureSession | None = None
        reason = "end_of_input"
        try:
            info = self.source.open()
            session = info.session
            with self._lock:
                self._source_info = info
            while not self._stop_requested.is_set():
                frame = self.source.read()
                if frame is None:
                    break
                if self._stop_requested.is_set():
                    break
                if not self.broker.put(frame):
                    reason = "broker_closed"
                    break
                with self._lock:
                    self._captured_frames += 1
            if self._stop_requested.is_set():
                reason = "stopped"
        except SourceError as error:
            if self._stop_requested.is_set():
                reason = "stopped"
            else:
                reason = "failed"
                self._record_error(error)
                if self._on_error is not None:
                    self._on_error(error)
        except Exception:
            reason = "failed"
            worker_error = SourceError("capture worker failed")
            self._record_error(worker_error)
            if self._on_error is not None:
                self._on_error(worker_error)
        finally:
            try:
                self.source.close()
            except Exception as error:
                self._record_error(SourceError(str(error)))
            self.broker.close()
            if session is not None:
                with self._lock:
                    self._session = CaptureSession(
                        id=session.id,
                        camera_id=session.camera_id,
                        started_at=session.started_at,
                        started_monotonic=session.started_monotonic,
                        ended_at=self._clock.now(),
                        end_reason=reason,
                    )

    def _record_error(self, error: SourceError) -> None:
        with self._lock:
            self._error = error
            self._decode_errors += 1


CaptureWorker = FrameCaptureWorker
