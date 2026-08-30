"""Bounded live capture-to-inference coordination."""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Literal
from uuid import uuid4

import numpy as np

from ..cameras.repository import CameraConfig
from ..capture import (
    CameraRuntime,
    Clock,
    LatestFrameSubscription,
    SystemClock,
    VideoFrame,
)
from ..inference import (
    BackendContractError,
    BackendOptions,
    BackendUnavailableError,
    Detection,
    DetectionBatch,
    DetectionValidationError,
    DetectorSession,
    InferenceBackend,
    ModelDescriptor,
    PreprocessingError,
    StillImage,
)
from ..tracking import (
    ActiveTrack,
    ClosedTrackEvent,
    LiveDetectionFrame,
    TrackerFactory,
    TrackingConfig,
    TrackingEventAggregator,
    TrackingProvenance,
    default_tracker_factory,
)
from .display import (
    DisplayShutdownError,
    ProcessedDisplayService,
    ProcessedDisplaySubscription,
    build_display_candidate,
)

PipelineState = Literal["stopped", "starting", "warming", "running", "stopping", "failed"]
BackendFactory = Callable[[], InferenceBackend]
EpochFactory = Callable[[], str]
EventSink = Callable[[ClosedTrackEvent], None]


class LivePipelineError(RuntimeError):
    """Base error for live pipeline control failures."""


class LivePipelineConflict(LivePipelineError):
    """Raised when a live resource cannot be changed in place."""


class LivePipelineShutdownError(LivePipelineError):
    """Raised when a live worker does not stop before its deadline."""


@dataclass(frozen=True, slots=True)
class SourcePixelRegionOfInterest:
    """A bounded source-image pixel crop used before model preprocessing."""

    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        values = (self.x, self.y, self.width, self.height)
        if any(type(value) is not int for value in values):
            raise ValueError("region_of_interest values must be integers")
        if self.x < 0 or self.y < 0 or self.width <= 0 or self.height <= 0:
            raise ValueError("region_of_interest must have positive bounded dimensions")
        if self.x > 8192 or self.y > 8192 or self.width > 8192 or self.height > 8192:
            raise ValueError("region_of_interest is outside the safe size limit")

    @classmethod
    def from_value(cls, value: object) -> SourcePixelRegionOfInterest | None:
        """Parse one strict API or test value."""
        if value is None:
            return None
        if not isinstance(value, Mapping) or set(value) != {"x", "y", "width", "height"}:
            raise ValueError("region_of_interest must contain x, y, width, and height")
        return cls(
            x=_integer_value(value["x"], "region_of_interest.x"),
            y=_integer_value(value["y"], "region_of_interest.y"),
            width=_integer_value(value["width"], "region_of_interest.width"),
            height=_integer_value(value["height"], "region_of_interest.height"),
        )

    def as_dict(self) -> dict[str, int]:
        """Return the public source-pixel contract."""
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
        }

    def validate_for(self, frame: VideoFrame) -> None:
        """Require the crop to be inside one decoded frame."""
        if self.x + self.width > frame.width or self.y + self.height > frame.height:
            raise ValueError("region_of_interest is outside the current frame")


@dataclass(frozen=True, slots=True)
class LivePipelineFailure:
    """Safe failure details suitable for a transient API response."""

    category: str
    message: str

    def as_dict(self) -> dict[str, str]:
        """Return only sanitized failure information."""
        return {"category": self.category, "message": self.message}


@dataclass(frozen=True, slots=True)
class LivePipelineMetrics:
    """Bounded current metrics for one live pipeline generation."""

    captured_frames: int = 0
    processed_frames: int = 0
    capture_age_ms: float | None = None
    preprocessing_ms: float | None = None
    model_load_ms: float | None = None
    warmup_ms: float | None = None
    prediction_ms: float | None = None
    postprocessing_ms: float | None = None
    end_to_end_ms: float | None = None
    prediction_p50_ms: float | None = None
    prediction_p95_ms: float | None = None
    processed_fps: float = 0.0
    source_replacement_count: int = 0
    inference_replacement_count: int = 0
    display_replacement_count: int = 0
    stale_frame_count: int = 0
    rejected_candidates: int = 0
    failure_count: int = 0

    def as_dict(self) -> dict[str, int | float | None]:
        """Return the stable metrics payload."""
        return {
            "captured_frames": self.captured_frames,
            "processed_frames": self.processed_frames,
            "capture_age_ms": _safe_float(self.capture_age_ms),
            "preprocessing_ms": _safe_float(self.preprocessing_ms),
            "model_load_ms": _safe_float(self.model_load_ms),
            "warmup_ms": _safe_float(self.warmup_ms),
            "prediction_ms": _safe_float(self.prediction_ms),
            "postprocessing_ms": _safe_float(self.postprocessing_ms),
            "end_to_end_ms": _safe_float(self.end_to_end_ms),
            "prediction_p50_ms": _safe_float(self.prediction_p50_ms),
            "prediction_p95_ms": _safe_float(self.prediction_p95_ms),
            "processed_fps": round(max(0.0, self.processed_fps), 3),
            "source_replacement_count": self.source_replacement_count,
            "inference_replacement_count": self.inference_replacement_count,
            "display_replacement_count": self.display_replacement_count,
            "source_replaced_frames": self.source_replacement_count,
            "inference_replaced_frames": self.inference_replacement_count,
            "display_replaced_frames": self.display_replacement_count,
            "stale_frame_count": self.stale_frame_count,
            "rejected_candidates": self.rejected_candidates,
            "failure_count": self.failure_count,
            # Names used by the later live display contract.
            "frame_age_seconds": _seconds(self.capture_age_ms),
            "replaced_frames": self.source_replacement_count,
        }


@dataclass(frozen=True, slots=True)
class LiveFrameResult:
    """One processed frame with strict source and model provenance."""

    camera_id: str
    model_id: str
    model_checksum: str
    capture_session_id: str
    epoch: str
    frame_sequence: int
    captured_at: datetime
    source_width: int
    source_height: int
    detections: tuple[Detection, ...]
    active_tracks: tuple[ActiveTrack, ...]
    rejected_candidates: int

    @property
    def stream_epoch(self) -> str:
        """Return the WebSocket-facing name for the same epoch."""
        return self.epoch

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-safe metadata without image bytes."""
        return {
            "camera_id": self.camera_id,
            "model_id": self.model_id,
            "model_checksum": self.model_checksum,
            "capture_session_id": self.capture_session_id,
            "epoch": self.epoch,
            "stream_epoch": self.epoch,
            "frame_sequence": self.frame_sequence,
            "captured_at_utc": _timestamp(self.captured_at),
            "source_width": self.source_width,
            "source_height": self.source_height,
            "detections": [_detection_payload(detection) for detection in self.detections],
            "active_tracks": [track.as_dict() for track in self.active_tracks],
            "rejected_candidates": self.rejected_candidates,
        }


@dataclass(frozen=True, slots=True)
class LivePipelineStatus:
    """Immutable public snapshot of coordinator state."""

    state: PipelineState
    camera_id: str | None
    model_id: str | None
    generation_number: int | None
    model_checksum: str | None
    capture_session_id: str | None
    epoch: str | None
    frame_sequence: int | None
    compute_units: str | None
    confidence_threshold: float | None
    region_of_interest: SourcePixelRegionOfInterest | None
    failure: LivePipelineFailure | None
    metrics: LivePipelineMetrics
    last_result: LiveFrameResult | None
    active_tracks: tuple[ActiveTrack, ...]
    camera: dict[str, Any] | None
    state_history: tuple[PipelineState, ...]

    @property
    def stream_epoch(self) -> str | None:
        """Return the WebSocket-facing epoch name."""
        return self.epoch

    def as_dict(self) -> dict[str, Any]:
        """Return a safe state snapshot for the live API."""
        camera = self.camera or {}
        source = camera.get("source") if isinstance(camera.get("source"), dict) else None
        safe_camera = _safe_camera_payload(camera)
        payload: dict[str, Any] = {
            "state": self.state,
            "lifecycle_state": self.state,
            "camera_id": self.camera_id,
            "model_id": self.model_id,
            "generation_number": self.generation_number,
            "model_checksum": self.model_checksum,
            "capture_session_id": self.capture_session_id,
            "epoch": self.epoch,
            "stream_epoch": self.epoch,
            "frame_sequence": self.frame_sequence,
            "compute_units": self.compute_units,
            "confidence_threshold": self.confidence_threshold,
            "region_of_interest": (
                None if self.region_of_interest is None else self.region_of_interest.as_dict()
            ),
            "failure": None if self.failure is None else self.failure.as_dict(),
            "last_error": None if self.failure is None else self.failure.message,
            "metrics": self.metrics.as_dict(),
            "last_result": None if self.last_result is None else self.last_result.as_dict(),
            "active_tracks": [track.as_dict() for track in self.active_tracks],
            "camera_state": camera.get("state"),
            "source": source,
            "camera": safe_camera,
            "state_history": list(self.state_history),
        }
        return payload


@dataclass
class _PipelineGeneration:
    """Mutable worker inputs kept private to one coordinator generation."""

    number: int
    camera_id: str
    camera: CameraConfig
    descriptor: ModelDescriptor
    options: BackendOptions
    stop_requested: threading.Event
    threshold: float
    region_of_interest: SourcePixelRegionOfInterest | None
    captured_frames_baseline: int
    source_replacement_baseline: int


@dataclass
class LivePipelineCoordinator:
    """Own one bounded live inference worker and its capture subscription."""

    camera_runtime: CameraRuntime
    backend_factory: BackendFactory
    clock: Clock = field(default_factory=SystemClock)
    poll_interval_seconds: float = 0.02
    stop_timeout_seconds: float = 5.0
    epoch_factory: EpochFactory = lambda: uuid4().hex
    display_max_fps: float = 10.0
    tracker_factory: TrackerFactory | None = None
    tracking_config: TrackingConfig = field(default_factory=TrackingConfig)
    event_sink: EventSink | None = None

    def __post_init__(self) -> None:
        if self.poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive")
        if self.stop_timeout_seconds <= 0:
            raise ValueError("stop_timeout_seconds must be positive")
        self._condition = threading.Condition()
        self._control_lock = threading.RLock()
        self._state: PipelineState = "stopped"
        self._state_history: deque[PipelineState] = deque(["stopped"], maxlen=32)
        self._generation_number = 0
        self._generation: _PipelineGeneration | None = None
        self._thread: threading.Thread | None = None
        self._subscription: LatestFrameSubscription | None = None
        self._model_checksum: str | None = None
        self._capture_session_id: str | None = None
        self._epoch: str | None = None
        self._frame_sequence: int | None = None
        self._last_result: LiveFrameResult | None = None
        self._active_tracks: tuple[ActiveTrack, ...] = ()
        self._recent_closed_events: deque[ClosedTrackEvent] = deque(maxlen=16)
        self._failure: LivePipelineFailure | None = None
        self._metrics = LivePipelineMetrics()
        self._completed_at: deque[float] = deque(maxlen=120)
        self._prediction_samples_ms: deque[float] = deque(maxlen=120)
        self._display = ProcessedDisplayService(
            max_fps=self.display_max_fps,
            on_error=self._display_failed,
        )
        factory = self.tracker_factory or (
            lambda: default_tracker_factory(
                max_active_tracks=self.tracking_config.max_active_tracks,
            )
        )
        self._tracking = TrackingEventAggregator(
            factory,
            clock=self.clock,
            config=self.tracking_config,
            on_closed_event=self._record_closed_event,
        )

    @property
    def state(self) -> PipelineState:
        """Return the current pipeline lifecycle state."""
        with self._condition:
            return self._state

    @property
    def is_running(self) -> bool:
        """Return whether the pipeline owns a live running generation."""
        with self._condition:
            return self._state in {"starting", "warming", "running", "stopping"}

    @property
    def camera_id(self) -> str | None:
        """Return the camera selected by the current generation."""
        with self._condition:
            return None if self._generation is None else self._generation.camera_id

    @property
    def model_id(self) -> str | None:
        """Return the model selected by the current generation."""
        with self._condition:
            return None if self._generation is None else self._generation.descriptor.model_id

    @property
    def closed_events(self) -> tuple[ClosedTrackEvent, ...]:
        """Return the bounded recent closed-event view for tests and the live UI."""
        with self._condition:
            return tuple(self._recent_closed_events)

    @property
    def tracking(self) -> TrackingEventAggregator:
        """Return the tracking contract owned by this live coordinator."""
        return self._tracking

    def start(
        self,
        camera_id: str,
        camera: CameraConfig,
        model: ModelDescriptor,
        *,
        threshold: float | None = None,
        options: BackendOptions | None = None,
        region_of_interest: SourcePixelRegionOfInterest | None = None,
    ) -> LivePipelineStatus:
        """Start one camera/model generation under one control lock."""
        with self._control_lock:
            return self._start(
                camera_id,
                camera,
                model,
                threshold=threshold,
                options=options,
                region_of_interest=region_of_interest,
            )

    def _start(
        self,
        camera_id: str,
        camera: CameraConfig,
        model: ModelDescriptor,
        *,
        threshold: float | None = None,
        options: BackendOptions | None = None,
        region_of_interest: SourcePixelRegionOfInterest | None = None,
    ) -> LivePipelineStatus:
        """Start one camera/model generation without doing model work inline."""
        _validate_text(camera_id, "camera_id")
        _validate_descriptor(model)
        effective_threshold = _manifest_threshold(model, threshold)
        effective_options = options or BackendOptions()
        if not camera.enabled:
            raise LivePipelineError("camera is disabled; enable it before starting live detection")

        active_camera_id = self.camera_runtime.active_camera_id
        if active_camera_id is not None and active_camera_id != camera_id:
            raise LivePipelineConflict("stop the active live pipeline before switching the camera")
        runtime_was_active = active_camera_id == camera_id
        runtime_metrics = self.camera_runtime.status(camera_id).metrics
        captured_frames_baseline = (
            _metric_value(runtime_metrics, "captured_frames") if runtime_was_active else 0
        )
        source_replacement_baseline = (
            _metric_value(runtime_metrics, "replaced_frames") if runtime_was_active else 0
        )

        with self._condition:
            if self._thread is not None and self._thread.is_alive():
                current = self._generation
                if current is not None and current.camera_id != camera_id:
                    raise LivePipelineConflict(
                        "stop the active live pipeline before switching the camera"
                    )
                if current is not None and current.descriptor.model_id != model.model_id:
                    raise LivePipelineConflict(
                        "stop the active live pipeline before switching the model"
                    )
                if (
                    current is not None
                    and current.options.compute_units != effective_options.compute_units
                ):
                    raise LivePipelineConflict(
                        "stop the active live pipeline before switching the model"
                    )
                return self._status_locked()
            if self._state in {"starting", "warming", "running", "stopping"}:
                raise LivePipelineConflict("stop the active live pipeline before starting again")

            self._tracking.reset()
            self._generation_number += 1
            generation = _PipelineGeneration(
                number=self._generation_number,
                camera_id=camera_id,
                camera=camera,
                descriptor=model,
                options=effective_options,
                stop_requested=threading.Event(),
                threshold=effective_threshold,
                region_of_interest=region_of_interest,
                captured_frames_baseline=captured_frames_baseline,
                source_replacement_baseline=source_replacement_baseline,
            )
            self._generation = generation
            self._model_checksum = model.artifact_sha256
            self._capture_session_id = None
            self._epoch = None
            self._frame_sequence = None
            self._last_result = None
            self._active_tracks = ()
            self._failure = None
            self._metrics = LivePipelineMetrics()
            self._completed_at.clear()
            self._prediction_samples_ms.clear()
            self._state_history = deque(["stopped"], maxlen=32)
            self._transition_locked("starting")
        try:
            self._display.start_generation(generation.number)
            self.camera_runtime.start(camera_id, camera)
            thread = threading.Thread(
                target=self._run,
                args=(generation,),
                name="open-licenseplate-live-inference",
                daemon=True,
            )
            with self._condition:
                if not self._is_current_locked(generation) or self._state != "starting":
                    raise LivePipelineError("live pipeline start was cancelled")
                self._thread = thread
                thread.start()
                return self._status_locked()
        except Exception:
            self._stop_display(reason="failed")
            with suppress(Exception):
                self.camera_runtime.stop(camera_id, timeout=self.stop_timeout_seconds)
            with self._condition:
                if self._is_current_locked(generation):
                    self._generation = None
                    self._transition_locked("stopped")
            raise

    def update_threshold(self, threshold: float) -> LivePipelineStatus:
        """Update confidence filtering without reloading the model."""
        effective_threshold = _validated_threshold(threshold)
        with self._condition:
            generation = self._generation
            if generation is None or self._state not in {"starting", "warming", "running"}:
                raise LivePipelineError("start live detection before changing the threshold")
            generation.threshold = effective_threshold
            return self._status_locked()

    def stop(self, *, timeout: float | None = None) -> LivePipelineStatus:
        """Stop one live generation under one control lock."""
        with self._control_lock:
            return self._stop(timeout=timeout)

    def _stop(
        self,
        *,
        timeout: float | None = None,
        skip_display: bool = False,
    ) -> LivePipelineStatus:
        """Stop inference, then capture, within one bounded deadline."""
        wait_timeout = self.stop_timeout_seconds if timeout is None else max(0.0, timeout)
        started = time.monotonic()
        subscription: LatestFrameSubscription | None = None
        with self._condition:
            thread = self._thread
            generation = self._generation
            if thread is None or not thread.is_alive():
                self._transition_locked("stopped")
                result = self._status_locked()
            else:
                self._transition_locked("stopping")
                if generation is None:
                    raise LivePipelineError("live pipeline generation is not available")
                generation.stop_requested.set()
                subscription = self._subscription
                result = None
        if result is not None:
            display_ok = skip_display or self._stop_display(reason="stopped")
            camera_ok = generation is None or self._stop_camera(
                generation.camera_id,
                wait_timeout,
            )
            self._reset_tracking()
            if not camera_ok:
                self._record_stop_failure()
                raise LivePipelineShutdownError("camera resources did not stop before the deadline")
            if not display_ok:
                raise LivePipelineShutdownError(
                    "processed display resources did not stop before the deadline"
                )
            with self._condition:
                return self._status_locked()

        if subscription is not None:
            subscription.close()
        assert thread is not None
        thread.join(wait_timeout)
        elapsed = time.monotonic() - started
        remaining = max(0.0, wait_timeout - elapsed)
        if thread.is_alive():
            if not skip_display:
                self._stop_display(reason="failed")
            with self._condition:
                self._record_failure_locked(
                    LivePipelineFailure(
                        "shutdown",
                        "live inference did not stop before the deadline",
                    )
                )
                self._transition_locked("failed")
            self._stop_camera(generation.camera_id, 0.0) if generation is not None else None
            raise LivePipelineShutdownError("live inference did not stop before the deadline")

        if generation is not None and not self._stop_camera(
            generation.camera_id,
            remaining,
        ):
            if not skip_display:
                self._stop_display(reason="failed")
            self._record_stop_failure()
            raise LivePipelineShutdownError("camera resources did not stop before the deadline")
        with self._condition:
            if self._state == "stopping":
                self._transition_locked("stopped")
        display_ok = skip_display or self._stop_display(reason="stopped")
        if not display_ok:
            raise LivePipelineShutdownError(
                "processed display resources did not stop before the deadline"
            )
        self._reset_tracking()
        with self._condition:
            return self._status_locked()

    def close(self) -> None:
        """Release the live pipeline during application shutdown."""
        with self._control_lock:
            self._stop_display(reason="shutdown")
            try:
                self._stop(skip_display=True)
            except LivePipelineShutdownError:
                # The daemon worker keeps ownership until its current prediction ends.
                pass
            finally:
                self._stop_display(reason="shutdown")

    def subscribe_display(self) -> ProcessedDisplaySubscription | None:
        """Return a capacity-one processed display subscription."""
        return self._display.subscribe()

    def status(self) -> LivePipelineStatus:
        """Return a safe immutable state snapshot."""
        with self._condition:
            return self._status_locked()

    def wait_for_state(
        self,
        state: PipelineState,
        *,
        timeout: float = 5.0,
    ) -> LivePipelineStatus:
        """Wait for a state transition without polling in the caller."""
        with self._condition:
            reached = self._condition.wait_for(
                lambda: state in self._state_history,
                timeout=max(0.0, timeout),
            )
            if not reached:
                raise TimeoutError(f"live pipeline did not reach {state}")
            return self._status_locked()

    def _run(self, generation: _PipelineGeneration) -> None:
        session: DetectorSession | None = None
        subscription: LatestFrameSubscription | None = None
        failed = False
        try:
            self._set_state(generation, "warming")
            session = DetectorSession(
                backend=self.backend_factory(),
                descriptor=generation.descriptor,
                options=generation.options,
            )
            warmup_started = time.perf_counter()
            warmup = session.detect_timed(
                _warmup_image(generation.descriptor),
                confidence_threshold=generation.threshold,
            )
            warmup_ms = (time.perf_counter() - warmup_started) * 1000
            with self._condition:
                if self._is_current_locked(generation):
                    self._metrics = replace(
                        self._metrics,
                        model_load_ms=_nonnegative(warmup.model_load_ms),
                        warmup_ms=_nonnegative(warmup_ms),
                    )
            while not generation.stop_requested.is_set() and subscription is None:
                subscription = self._ensure_subscription(generation, subscription)
                if subscription is not None:
                    break
                camera_status = self.camera_runtime.status(generation.camera_id)
                if camera_status.state == "failed":
                    raise _WorkerFailure(
                        "capture",
                        "camera capture failed; check the camera settings and connection",
                    )
                generation.stop_requested.wait(self.poll_interval_seconds)
            if generation.stop_requested.is_set():
                return
            self._set_state(generation, "running")

            last_session_id: str | None = None
            last_sequence: int | None = None
            while not generation.stop_requested.is_set():
                subscription = self._ensure_subscription(generation, subscription)
                if subscription is None:
                    camera_status = self.camera_runtime.status(generation.camera_id)
                    if camera_status.state == "failed":
                        raise _WorkerFailure(
                            "capture",
                            "camera capture failed; check the camera settings and connection",
                        )
                    self._tick_tracking(generation)
                    generation.stop_requested.wait(self.poll_interval_seconds)
                    continue

                frame = subscription.get(timeout=self.poll_interval_seconds)
                if frame is None:
                    if subscription.closed:
                        subscription = None
                    self._tick_tracking(generation)
                    continue
                if not self._frame_matches_camera(generation, frame):
                    raise _WorkerFailure("capture", "camera frame provenance is invalid")
                camera_status = self.camera_runtime.status(generation.camera_id)
                source_info = camera_status.source_info
                if (
                    source_info is not None
                    and frame.capture_session_id != source_info.session.capture_session_id
                ) or (
                    source_info is None
                    and last_session_id is not None
                    and frame.capture_session_id != last_session_id
                ):
                    with self._condition:
                        if self._is_current_locked(generation):
                            self._metrics = replace(
                                self._metrics,
                                stale_frame_count=self._metrics.stale_frame_count + 1,
                            )
                    continue
                if frame.capture_session_id != last_session_id:
                    last_session_id = frame.capture_session_id
                    last_sequence = None
                    with self._condition:
                        if self._is_current_locked(generation):
                            if self._capture_session_id != frame.capture_session_id:
                                self._capture_session_id = frame.capture_session_id
                                self._epoch = self.epoch_factory()
                                self._display.set_provenance(
                                    generation.number,
                                    self._epoch,
                                    frame.capture_session_id,
                                )
                            self._frame_sequence = None
                            stream_epoch = self._epoch
                        else:
                            stream_epoch = None
                    if stream_epoch is not None:
                        update = self._tracking.activate_provenance(
                            self._tracking_provenance(
                                generation,
                                frame.capture_session_id,
                                stream_epoch,
                            )
                        )
                        with self._condition:
                            if self._is_current_locked(generation):
                                self._active_tracks = update.active_tracks
                if last_sequence is not None and frame.sequence <= last_sequence:
                    with self._condition:
                        if self._is_current_locked(generation):
                            self._metrics = replace(
                                self._metrics,
                                stale_frame_count=self._metrics.stale_frame_count + 1,
                            )
                    continue

                with self._condition:
                    threshold = generation.threshold
                    region_of_interest = generation.region_of_interest
                crop_started = time.perf_counter()
                image = _frame_image(frame, region_of_interest)
                crop_ms = (time.perf_counter() - crop_started) * 1000
                run = session.detect_timed(
                    image,
                    confidence_threshold=threshold,
                )
                completed_monotonic = self.clock.monotonic()
                detections = _restore_source_coordinates(
                    run.batch,
                    frame=frame,
                    region_of_interest=region_of_interest,
                    descriptor=generation.descriptor,
                )
                last_sequence = frame.sequence
                self._record_result(
                    generation,
                    frame,
                    run,
                    detections,
                    crop_ms=crop_ms,
                    completed_monotonic=completed_monotonic,
                    subscription=subscription,
                    threshold=threshold,
                )
        except _WorkerFailure as error:
            failed = True
            self._fail(generation, error.failure)
        except Exception as error:
            failed = True
            self._fail(generation, _failure_for_exception(error))
        finally:
            if subscription is not None:
                subscription.close()
            if session is not None:
                try:
                    session.close()
                except Exception:
                    failed = True
                    self._fail(
                        generation,
                        LivePipelineFailure(
                            "model_close",
                            "live model resources could not be released",
                        ),
                    )
            if failed and not generation.stop_requested.is_set():
                self._stop_camera(generation.camera_id, self.stop_timeout_seconds)
            with self._condition:
                if self._subscription is subscription:
                    self._subscription = None
                if self._is_current_locked(generation):
                    self._thread = None
                    self._condition.notify_all()

    def _ensure_subscription(
        self,
        generation: _PipelineGeneration,
        current: LatestFrameSubscription | None,
    ) -> LatestFrameSubscription | None:
        if current is not None and not current.closed:
            return current
        if current is not None:
            current.close()
        subscription = self.camera_runtime.subscribe(generation.camera_id)
        if subscription is None or subscription.closed:
            if subscription is not None:
                subscription.close()
            return None
        with self._condition:
            if not self._is_current_locked(generation) or generation.stop_requested.is_set():
                subscription.close()
                return None
            self._subscription = subscription
        return subscription

    def _record_result(
        self,
        generation: _PipelineGeneration,
        frame: VideoFrame,
        run: Any,
        detections: DetectionBatch,
        *,
        crop_ms: float,
        completed_monotonic: float,
        subscription: LatestFrameSubscription,
        threshold: float,
    ) -> None:
        capture_age_ms = max(
            0.0,
            (completed_monotonic - frame.host_received_monotonic) * 1000,
        )
        with self._condition:
            stream_epoch = self._epoch
        if stream_epoch is None:
            raise _WorkerFailure("capture", "live stream provenance is not available")
        tracking_update = self._tracking.consume(
            LiveDetectionFrame(
                provenance=self._tracking_provenance(
                    generation,
                    frame.capture_session_id,
                    stream_epoch,
                ),
                frame_sequence=frame.sequence,
                captured_at=frame.host_received_at,
                frame_width=frame.width,
                frame_height=frame.height,
                detections=detections.detections,
                source_pixels=frame.data,
                pixel_format=frame.pixel_format,
            )
        )
        self._completed_at.append(completed_monotonic)
        processed_fps = _processed_fps(self._completed_at)
        camera_status = self.camera_runtime.status(generation.camera_id)
        source_metrics = camera_status.metrics
        inference_metrics = subscription.metrics(now_monotonic=completed_monotonic)
        with self._condition:
            if not self._is_current_locked(generation):
                return
            session_id = frame.capture_session_id
            epoch = self._epoch or self.epoch_factory()
            self._capture_session_id = session_id
            self._epoch = epoch
            result = LiveFrameResult(
                camera_id=generation.camera_id,
                model_id=generation.descriptor.model_id,
                model_checksum=generation.descriptor.artifact_sha256,
                capture_session_id=session_id,
                epoch=epoch,
                frame_sequence=frame.sequence,
                captured_at=frame.host_received_at,
                source_width=frame.width,
                source_height=frame.height,
                detections=detections.detections,
                active_tracks=tracking_update.active_tracks,
                rejected_candidates=detections.rejected_count,
            )
            self._frame_sequence = frame.sequence
            self._last_result = result
            self._active_tracks = tracking_update.active_tracks
            self._prediction_samples_ms.append(_nonnegative(run.inference_ms))
            prediction_p50_ms = _percentile(self._prediction_samples_ms, 0.50)
            prediction_p95_ms = _percentile(self._prediction_samples_ms, 0.95)
            self._metrics = replace(
                self._metrics,
                captured_frames=_metric_delta(
                    source_metrics,
                    "captured_frames",
                    generation.captured_frames_baseline,
                ),
                processed_frames=self._metrics.processed_frames + 1,
                capture_age_ms=capture_age_ms,
                preprocessing_ms=_nonnegative(crop_ms + run.preprocessing_ms),
                prediction_ms=_nonnegative(run.inference_ms),
                postprocessing_ms=_nonnegative(run.postprocessing_ms),
                end_to_end_ms=capture_age_ms,
                prediction_p50_ms=prediction_p50_ms,
                prediction_p95_ms=prediction_p95_ms,
                processed_fps=processed_fps,
                source_replacement_count=_metric_delta(
                    source_metrics,
                    "replaced_frames",
                    generation.source_replacement_baseline,
                ),
                inference_replacement_count=inference_metrics.replaced_frames,
                rejected_candidates=detections.rejected_count,
            )
            display_metrics = self._display.metrics()
            self._metrics = replace(
                self._metrics,
                display_replacement_count=display_metrics.replaced_units,
            )
            candidate = build_display_candidate(
                generation_number=generation.number,
                frame=frame,
                camera_id=result.camera_id,
                model_id=result.model_id,
                model_checksum=result.model_checksum,
                capture_session_id=result.capture_session_id,
                stream_epoch=result.stream_epoch,
                detections=result.detections,
                active_tracks=result.active_tracks,
                threshold=threshold,
                region_of_interest=(
                    None
                    if generation.region_of_interest is None
                    else generation.region_of_interest.as_dict()
                ),
                metrics=self._metrics.as_dict(),
            )
        self._display.submit(candidate)

    @staticmethod
    def _tracking_provenance(
        generation: _PipelineGeneration,
        capture_session_id: str,
        stream_epoch: str,
    ) -> TrackingProvenance:
        """Build one validated tracking provenance value."""
        return TrackingProvenance(
            camera_id=generation.camera_id,
            capture_session_id=capture_session_id,
            generation_number=generation.number,
            stream_epoch=stream_epoch,
            model_id=generation.descriptor.model_id,
            model_checksum=generation.descriptor.artifact_sha256,
        )

    def _tick_tracking(self, generation: _PipelineGeneration) -> None:
        """Run timeout expiry even when capture temporarily has no frame."""
        update = self._tracking.tick()
        with self._condition:
            if self._is_current_locked(generation):
                self._active_tracks = update.active_tracks

    def _reset_tracking(self) -> None:
        """Close confirmed tracks and clear the bounded state at a boundary."""
        update = self._tracking.reset()
        with self._condition:
            self._active_tracks = update.active_tracks

    def _frame_matches_camera(self, generation: _PipelineGeneration, frame: VideoFrame) -> bool:
        camera_status = self.camera_runtime.status(generation.camera_id)
        source_info = camera_status.source_info
        if source_info is None:
            return True
        return source_info.session.camera_id in {None, generation.camera_id}

    def _stop_camera(self, camera_id: str, timeout: float) -> bool:
        try:
            self.camera_runtime.stop(camera_id, timeout=max(0.0, timeout))
        except Exception:
            return False
        return True

    def _record_stop_failure(self) -> None:
        with self._condition:
            self._record_failure_locked(
                LivePipelineFailure(
                    "shutdown",
                    "camera resources did not stop before the deadline",
                )
            )
            self._transition_locked("failed")

    def _stop_display(self, *, reason: str) -> bool:
        try:
            self._display.stop_generation(
                reason=reason,
                timeout=self.stop_timeout_seconds,
            )
        except DisplayShutdownError:
            self._record_display_shutdown_failure()
            return False
        return True

    def _record_display_shutdown_failure(self) -> None:
        with self._condition:
            self._record_failure_locked(
                LivePipelineFailure(
                    "shutdown",
                    "processed display encoder did not stop before the deadline",
                )
            )
            self._transition_locked("failed")

    def _set_state(self, generation: _PipelineGeneration, state: PipelineState) -> None:
        with self._condition:
            if self._is_current_locked(generation) and self._state not in {"stopping", "failed"}:
                self._transition_locked(state)

    def _fail(self, generation: _PipelineGeneration, failure: LivePipelineFailure) -> bool:
        should_close_display = False
        with self._condition:
            if not self._is_current_locked(generation) or generation.stop_requested.is_set():
                return False
            self._record_failure_locked(failure)
            self._transition_locked("failed")
            should_close_display = True
        if should_close_display:
            self._stop_display(reason="failed")
        return True

    def _display_failed(self) -> None:
        """Fail the active pipeline when processed JPEG encoding is unsafe."""
        with self._condition:
            generation = self._generation
        if generation is not None:
            applied = self._fail(
                generation,
                LivePipelineFailure(
                    "display",
                    "processed preview encoding failed; live detection stopped",
                ),
            )
            if applied:
                generation.stop_requested.set()
                self._stop_camera(generation.camera_id, self.stop_timeout_seconds)

    def _record_failure_locked(self, failure: LivePipelineFailure) -> None:
        self._failure = failure
        self._metrics = replace(
            self._metrics,
            failure_count=self._metrics.failure_count + 1,
        )

    def _record_closed_event(self, event: ClosedTrackEvent) -> None:
        """Commit or forward one event, then release all retained crop pixels."""
        public_event = event.without_crop_candidates()
        with self._condition:
            self._recent_closed_events.append(public_event)
        try:
            if self.event_sink is not None:
                self.event_sink(event)
        except Exception:
            # Event persistence must not interrupt live tracking.
            return
        finally:
            for candidate in event.crop_candidates:
                candidate.release()

    def _is_current_locked(self, generation: _PipelineGeneration) -> bool:
        return self._generation is generation and generation.number == self._generation_number

    def _transition_locked(self, state: PipelineState) -> None:
        if self._state != state:
            self._state = state
            self._state_history.append(state)
            self._condition.notify_all()

    def _status_locked(self) -> LivePipelineStatus:
        generation = self._generation
        camera_id = None if generation is None else generation.camera_id
        camera_payload: dict[str, Any] | None = None
        metrics = self._metrics
        if camera_id is not None and generation is not None:
            runtime_status = self.camera_runtime.status(camera_id)
            camera_payload = runtime_status.as_dict()
            runtime_metrics = runtime_status.metrics
            inference_metrics = None if self._subscription is None else self._subscription.metrics()
            metrics = replace(
                metrics,
                captured_frames=max(
                    metrics.captured_frames,
                    _metric_delta(
                        runtime_metrics,
                        "captured_frames",
                        generation.captured_frames_baseline,
                    ),
                ),
                source_replacement_count=max(
                    metrics.source_replacement_count,
                    _metric_delta(
                        runtime_metrics,
                        "replaced_frames",
                        generation.source_replacement_baseline,
                    ),
                ),
                inference_replacement_count=max(
                    metrics.inference_replacement_count,
                    0 if inference_metrics is None else inference_metrics.replaced_frames,
                ),
                display_replacement_count=max(
                    metrics.display_replacement_count,
                    self._display.metrics().replaced_units,
                ),
            )
        return LivePipelineStatus(
            state=self._state,
            camera_id=camera_id,
            model_id=None if generation is None else generation.descriptor.model_id,
            generation_number=None if generation is None else generation.number,
            model_checksum=self._model_checksum,
            capture_session_id=self._capture_session_id,
            epoch=self._epoch,
            frame_sequence=self._frame_sequence,
            compute_units=(None if generation is None else generation.options.compute_units.value),
            confidence_threshold=None if generation is None else generation.threshold,
            region_of_interest=(None if generation is None else generation.region_of_interest),
            failure=self._failure,
            metrics=metrics,
            last_result=self._last_result,
            active_tracks=self._active_tracks,
            camera=camera_payload,
            state_history=tuple(self._state_history),
        )


@dataclass(frozen=True, slots=True)
class _WorkerFailure(Exception):
    """Private exception that carries only a safe failure payload."""

    category: str
    message: str

    @property
    def failure(self) -> LivePipelineFailure:
        """Return the public safe form."""
        return LivePipelineFailure(self.category, self.message)


def _validate_descriptor(model: ModelDescriptor) -> None:
    if not model.model_id or model.model_id != model.manifest.model_id:
        raise ValueError("model provenance is invalid")
    if model.artifact_sha256 != model.manifest.artifact_sha256:
        raise ValueError("model provenance is invalid")


def _manifest_threshold(model: ModelDescriptor, threshold: float | None) -> float:
    if threshold is not None:
        return _validated_threshold(threshold)
    defaults = model.manifest.raw.get("defaults")
    if not isinstance(defaults, dict):
        raise ValueError("model confidence defaults are invalid")
    return _validated_threshold(defaults.get("confidence_threshold"))


def _validated_threshold(value: object) -> float:
    if isinstance(value, bool):
        raise ValueError("confidence_threshold must be a number from 0 through 1")
    try:
        parsed = float(str(value))
    except (TypeError, ValueError):
        raise ValueError("confidence_threshold must be a number from 0 through 1") from None
    if not math.isfinite(parsed) or not 0 <= parsed <= 1:
        raise ValueError("confidence_threshold must be a number from 0 through 1")
    return parsed


def _integer_value(value: object, field_name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{field_name} must be an integer")
    return int(value)


def _validate_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} is required")


def _warmup_image(model: ModelDescriptor) -> StillImage:
    input_values = model.manifest.raw.get("input")
    if not isinstance(input_values, dict):
        raise PreprocessingError("model input contract is invalid")
    width = input_values.get("width")
    height = input_values.get("height")
    color_space = input_values.get("color_space")
    if type(width) is not int or type(height) is not int or not isinstance(color_space, str):
        raise PreprocessingError("model input contract is invalid")
    shape = (height, width) if color_space == "grayscale" else (height, width, 3)
    return StillImage(np.zeros(shape, dtype=np.uint8), color_space=color_space)


def _frame_image(
    frame: VideoFrame,
    region_of_interest: SourcePixelRegionOfInterest | None,
) -> StillImage:
    if frame.pixel_format.casefold() in {"bgr24", "bgr"}:
        color_space = "bgr"
    elif frame.pixel_format.casefold() in {"rgb24", "rgb"}:
        color_space = "rgb"
    else:
        raise PreprocessingError("decoded frame pixel format is unsupported")
    pixels = np.asarray(frame.data)
    if region_of_interest is not None:
        region_of_interest.validate_for(frame)
        pixels = pixels[
            region_of_interest.y : region_of_interest.y + region_of_interest.height,
            region_of_interest.x : region_of_interest.x + region_of_interest.width,
        ].copy()
    return StillImage(
        np.ascontiguousarray(pixels),
        color_space=color_space,
        frame_sequence=frame.sequence,
        captured_at=frame.host_received_at,
    )


def _restore_source_coordinates(
    batch: DetectionBatch,
    *,
    frame: VideoFrame,
    region_of_interest: SourcePixelRegionOfInterest | None,
    descriptor: ModelDescriptor,
) -> DetectionBatch:
    restored: list[Detection] = []
    offset_x = 0 if region_of_interest is None else region_of_interest.x
    offset_y = 0 if region_of_interest is None else region_of_interest.y
    for detection in batch.detections:
        if (
            detection.model_id != descriptor.model_id
            or detection.model_sha256 != descriptor.artifact_sha256
            or detection.frame_sequence != frame.sequence
        ):
            raise _WorkerFailure("detection", "detector returned invalid frame provenance")
        x1, y1, x2, y2 = detection.box_xyxy
        restored.append(
            replace(
                detection,
                box_xyxy=(
                    max(0.0, min(float(frame.width), x1 + offset_x)),
                    max(0.0, min(float(frame.height), y1 + offset_y)),
                    max(0.0, min(float(frame.width), x2 + offset_x)),
                    max(0.0, min(float(frame.height), y2 + offset_y)),
                ),
            )
        )
    return DetectionBatch(
        detections=tuple(restored),
        rejected_count=batch.rejected_count,
    )


def _failure_for_exception(error: Exception) -> LivePipelineFailure:
    if isinstance(error, BackendUnavailableError):
        return LivePipelineFailure(
            "backend_unavailable",
            "the selected inference backend is not available on this system",
        )
    if isinstance(error, BackendContractError):
        return LivePipelineFailure(
            "model_contract",
            "the selected model is incompatible with the inference backend",
        )
    if isinstance(error, DetectionValidationError):
        return LivePipelineFailure(
            "postprocessing",
            "the inference result did not match the selected model contract",
        )
    if isinstance(error, (PreprocessingError, ValueError)):
        return LivePipelineFailure(
            "preprocessing",
            "the current frame could not be prepared for the selected model",
        )
    return LivePipelineFailure(
        "inference",
        "live inference failed; check the selected camera and model",
    )


def _processed_fps(completed_at: deque[float]) -> float:
    if len(completed_at) < 2:
        return 0.0
    elapsed = completed_at[-1] - completed_at[0]
    if elapsed <= 0:
        return 0.0
    return (len(completed_at) - 1) / elapsed


def _percentile(values: deque[float], percentile: float) -> float | None:
    """Return one bounded nearest-rank percentile."""
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(math.ceil(percentile * len(ordered))) - 1))
    return _safe_float(ordered[index])


def _detection_payload(detection: Detection) -> dict[str, Any]:
    return {
        "box_xyxy": list(detection.box_xyxy),
        "class_id": detection.class_id,
        "label": detection.label,
        "confidence": detection.confidence,
        "model_id": detection.model_id,
        "model_checksum": detection.model_sha256,
        "frame_sequence": detection.frame_sequence,
    }


def _metric_value(metrics: Mapping[str, Any], name: str) -> int:
    try:
        value = int(metrics.get(name, 0))
    except (TypeError, ValueError):
        return 0
    return max(0, value)


def _metric_delta(metrics: Mapping[str, Any], name: str, baseline: int) -> int:
    return max(0, _metric_value(metrics, name) - max(0, baseline))


def _safe_camera_payload(camera: Mapping[str, Any]) -> dict[str, Any] | None:
    """Keep only camera fields that are safe in a live pipeline response."""
    if not camera:
        return None
    safe_fields = (
        "camera_id",
        "camera_name",
        "state",
        "lifecycle_state",
        "reconnect_attempt",
        "next_retry_in_seconds",
        "source",
        "stream_metadata",
        "metrics",
        "active_camera_id",
    )
    return {field_name: camera[field_name] for field_name in safe_fields if field_name in camera}


def _timestamp(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _safe_float(value: float | None) -> float | None:
    return None if value is None else round(_nonnegative(value), 3)


def _nonnegative(value: float) -> float:
    return max(0.0, float(value)) if math.isfinite(float(value)) else 0.0


def _seconds(value: float | None) -> float | None:
    return None if value is None else round(_nonnegative(value) / 1000, 3)


__all__ = [
    "BackendFactory",
    "EpochFactory",
    "LiveFrameResult",
    "LivePipelineConflict",
    "LivePipelineCoordinator",
    "LivePipelineError",
    "LivePipelineFailure",
    "LivePipelineMetrics",
    "LivePipelineShutdownError",
    "LivePipelineStatus",
    "PipelineState",
    "SourcePixelRegionOfInterest",
]
