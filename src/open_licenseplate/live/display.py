"""Bounded, paired processed-frame display delivery."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image

from ..capture.contracts import VideoFrame
from ..capture.preview import encode_jpeg
from ..inference.contract import Detection
from ..tracking.contracts import ActiveTrack

LIVE_PROTOCOL_VERSION = 1
MAX_DISPLAY_METADATA_BYTES = 64 * 1024
MAX_DISPLAY_JPEG_BYTES = 4 * 1024 * 1024
MAX_DISPLAY_DIMENSION = 8192
MAX_DISPLAY_DETECTIONS = 256
MAX_DISPLAY_ACTIVE_TRACKS = 64
MAX_DISPLAY_SUBSCRIBERS = 16
MAX_RETIRED_PROVENANCES = 16
DISPLAY_BUFFER_CAPACITY = 1


class DisplayProtocolError(ValueError):
    """Raised when a processed display unit violates the public protocol."""


class DisplayMessageTooLarge(DisplayProtocolError):
    """Raised when a metadata or JPEG message exceeds its safe limit."""


class DisplayShutdownError(RuntimeError):
    """Raised when the processed display worker misses its stop deadline."""


@dataclass(frozen=True, slots=True)
class ProcessedDisplayUnit:
    """One atomic JSON metadata and binary JPEG pair."""

    metadata: Mapping[str, Any]
    metadata_text: str
    jpeg: bytes
    generation_number: int
    stream_epoch: str
    capture_session_id: str
    frame_sequence: int

    @property
    def jpeg_byte_count(self) -> int:
        """Return the binary message size."""
        return len(self.jpeg)

    def message_pair(self) -> tuple[str, bytes]:
        """Return the ordered WebSocket messages for this unit."""
        return self.metadata_text, self.jpeg


@dataclass(frozen=True, slots=True)
class ProcessedDisplayCandidate:
    """One immutable processed frame waiting for bounded JPEG encoding."""

    generation_number: int
    stream_epoch: str
    frame_sequence: int
    frame: VideoFrame
    camera_id: str
    model_id: str
    model_checksum: str
    capture_session_id: str
    captured_at: datetime
    source_width: int
    source_height: int
    detections: tuple[Detection, ...]
    active_tracks: tuple[ActiveTrack, ...]
    threshold: float
    region_of_interest: Mapping[str, int] | None
    metrics: Mapping[str, int | float | None]


@dataclass(frozen=True, slots=True)
class DisplayBrokerMetrics:
    """Bounded counters for one processed display broker."""

    published_units: int
    consumed_units: int
    replaced_units: int
    subscriber_count: int
    closed: bool


class ProcessedDisplayBroker:
    """Capacity-one broker that stores complete display units only."""

    capacity = DISPLAY_BUFFER_CAPACITY

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._unit: ProcessedDisplayUnit | None = None
        self._closed = False
        self._close_reason: str | None = None
        self._published_units = 0
        self._consumed_units = 0
        self._replaced_units = 0
        self._subscribers: set[ProcessedDisplaySubscription] = set()

    def put(self, unit: ProcessedDisplayUnit) -> bool:
        """Publish one complete unit without waiting for a client."""
        validate_display_unit(unit)
        with self._condition:
            if self._closed:
                return False
            old_unit = self._unit
            self._unit = unit
            self._published_units += 1
            if old_unit is not None:
                self._replaced_units += 1
            subscribers = tuple(self._subscribers)
            self._condition.notify_all()
        for subscriber in subscribers:
            subscriber._publish(unit)
        return True

    publish = put

    def get(self, timeout: float | None = None) -> ProcessedDisplayUnit | None:
        """Consume the newest complete unit."""
        deadline = _deadline(timeout)
        with self._condition:
            while self._unit is None and not self._closed:
                remaining = _remaining(deadline)
                if remaining == 0:
                    return None
                self._condition.wait(remaining)
            if self._unit is None:
                return None
            unit = self._unit
            self._unit = None
            self._consumed_units += 1
            return unit

    consume = get

    def subscribe(self) -> ProcessedDisplaySubscription:
        """Create a capacity-one subscription for one WebSocket client."""
        subscription = ProcessedDisplaySubscription(self)
        with self._condition:
            if self._closed:
                subscription._close_from_parent(self._close_reason or "shutdown")
            elif len(self._subscribers) >= MAX_DISPLAY_SUBSCRIBERS:
                raise DisplayProtocolError("processed display subscriber limit reached")
            else:
                self._subscribers.add(subscription)
                if self._unit is not None:
                    subscription._publish(self._unit)
        return subscription

    def clear(self) -> None:
        """Discard the stored unit without closing the broker."""
        with self._condition:
            self._unit = None

    def close(self, reason: str = "shutdown") -> None:
        """Close the broker and all subscriptions with a safe reason."""
        safe_reason = reason if reason in {"shutdown", "stopped", "failed"} else "failed"
        with self._condition:
            self._unit = None
            self._closed = True
            self._close_reason = safe_reason
            subscribers = tuple(self._subscribers)
            self._subscribers.clear()
            self._condition.notify_all()
        for subscriber in subscribers:
            subscriber._close_from_parent(safe_reason)

    def metrics(self) -> DisplayBrokerMetrics:
        """Return bounded broker counters."""
        with self._condition:
            return DisplayBrokerMetrics(
                published_units=self._published_units,
                consumed_units=self._consumed_units,
                replaced_units=self._replaced_units,
                subscriber_count=len(self._subscribers),
                closed=self._closed,
            )

    @property
    def closed(self) -> bool:
        """Return whether the broker accepts no more units."""
        with self._condition:
            return self._closed

    @property
    def close_reason(self) -> str | None:
        """Return the safe broker close reason."""
        with self._condition:
            return self._close_reason

    def _remove_subscriber(self, subscription: ProcessedDisplaySubscription) -> None:
        with self._condition:
            self._subscribers.discard(subscription)


class ProcessedDisplaySubscription:
    """Capacity-one downstream view of complete display units."""

    capacity = DISPLAY_BUFFER_CAPACITY

    def __init__(self, parent: ProcessedDisplayBroker) -> None:
        self._parent = parent
        self._condition = threading.Condition()
        self._unit: ProcessedDisplayUnit | None = None
        self._closed = False
        self._close_reason: str | None = None
        self._published_units = 0
        self._consumed_units = 0
        self._replaced_units = 0

    def _publish(self, unit: ProcessedDisplayUnit) -> None:
        with self._condition:
            if self._closed:
                return
            old_unit = self._unit
            self._unit = unit
            self._published_units += 1
            if old_unit is not None:
                self._replaced_units += 1
            self._condition.notify_all()

    def _close_from_parent(self, reason: str) -> None:
        with self._condition:
            self._unit = None
            self._closed = True
            self._close_reason = reason if reason in {"shutdown", "stopped", "failed"} else "failed"
            self._condition.notify_all()

    def get(self, timeout: float | None = None) -> ProcessedDisplayUnit | None:
        """Consume the newest complete unit from this subscription."""
        deadline = _deadline(timeout)
        with self._condition:
            while self._unit is None and not self._closed:
                remaining = _remaining(deadline)
                if remaining == 0:
                    return None
                self._condition.wait(remaining)
            if self._unit is None:
                return None
            unit = self._unit
            self._unit = None
            self._consumed_units += 1
            return unit

    consume = get

    def close(self, reason: str = "stopped") -> None:
        """Close this subscription and release its unit."""
        self._parent._remove_subscriber(self)
        self._close_from_parent(reason)

    def metrics(self) -> DisplayBrokerMetrics:
        """Return bounded subscription counters."""
        with self._condition:
            return DisplayBrokerMetrics(
                published_units=self._published_units,
                consumed_units=self._consumed_units,
                replaced_units=self._replaced_units,
                subscriber_count=1,
                closed=self._closed,
            )

    @property
    def closed(self) -> bool:
        """Return whether this subscription is closed."""
        with self._condition:
            return self._closed

    @property
    def close_reason(self) -> str | None:
        """Return the safe subscription close reason."""
        with self._condition:
            return self._close_reason


class _CandidateBroker:
    """Private capacity-one broker for frames waiting for JPEG encoding."""

    capacity = DISPLAY_BUFFER_CAPACITY

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._candidate: ProcessedDisplayCandidate | None = None
        self._closed = False

    def put(self, candidate: ProcessedDisplayCandidate) -> bool:
        with self._condition:
            if self._closed:
                return False
            self._candidate = candidate
            self._condition.notify_all()
            return True

    def get(self, timeout: float | None = None) -> ProcessedDisplayCandidate | None:
        deadline = _deadline(timeout)
        with self._condition:
            while self._candidate is None and not self._closed:
                remaining = _remaining(deadline)
                if remaining == 0:
                    return None
                self._condition.wait(remaining)
            candidate = self._candidate
            self._candidate = None
            return candidate

    def clear(self) -> None:
        with self._condition:
            self._candidate = None

    def close(self) -> None:
        with self._condition:
            self._candidate = None
            self._closed = True
            self._condition.notify_all()

    @property
    def closed(self) -> bool:
        with self._condition:
            return self._closed


class ProcessedDisplayEncoder:
    """Encode processed frames in one bounded worker thread."""

    def __init__(
        self,
        candidate_broker: _CandidateBroker,
        output_broker: ProcessedDisplayBroker,
        *,
        max_fps: float = 10.0,
        encoder: Callable[[VideoFrame], bytes] = encode_jpeg,
        current_identity: Callable[[], tuple[int, str | None, str | None] | None] | None = None,
        on_error: Callable[[], None] | None = None,
    ) -> None:
        if not 0 < max_fps <= 60:
            raise ValueError("processed display max_fps must be between 0 and 60")
        self._candidate_broker = candidate_broker
        self._output_broker = output_broker
        self._interval_seconds = 1.0 / max_fps
        self._encoder = encoder
        self._current_identity = current_identity
        self._on_error = on_error
        self._stop_requested = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the encoding worker."""
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("processed display encoder is already running")
        self._stop_requested.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="open-licenseplate-processed-display",
            daemon=True,
        )
        self._thread.start()

    def stop(self, *, timeout: float = 5.0) -> None:
        """Stop the encoding worker within a bounded timeout."""
        self._stop_requested.set()
        self._candidate_broker.close()
        thread = self._thread
        if thread is not None:
            if thread is threading.current_thread():
                return
            thread.join(max(0.0, timeout))
            if thread.is_alive():
                raise DisplayShutdownError(
                    "processed display encoder did not stop before the deadline"
                )

    @property
    def thread(self) -> threading.Thread | None:
        """Return the worker thread for leak checks."""
        return self._thread

    def _run(self) -> None:
        last_encoded_at: float | None = None
        while not self._stop_requested.is_set():
            candidate = self._candidate_broker.get(timeout=0.25)
            if candidate is None:
                continue
            if last_encoded_at is not None:
                wait_seconds = self._interval_seconds - (time.monotonic() - last_encoded_at)
                if wait_seconds > 0:
                    self._stop_requested.wait(wait_seconds)
                    newer = self._candidate_broker.get(timeout=0)
                    if newer is not None:
                        candidate = newer
            if self._stop_requested.is_set() or not self._is_current(candidate):
                continue
            try:
                jpeg = self._encoder(candidate.frame)
                jpeg_width, jpeg_height = jpeg_dimensions(jpeg)
                unit = build_display_unit(
                    candidate,
                    jpeg=jpeg,
                    jpeg_width=jpeg_width,
                    jpeg_height=jpeg_height,
                )
                if self._is_current(candidate):
                    self._output_broker.put(unit)
                    last_encoded_at = time.monotonic()
            except Exception:
                if self._on_error is not None:
                    self._on_error()

    def _is_current(self, candidate: ProcessedDisplayCandidate) -> bool:
        if self._current_identity is None:
            return True
        identity = self._current_identity()
        return identity is not None and (
            identity[0] == candidate.generation_number
            and identity[1] == candidate.stream_epoch
            and (identity[2] is None or identity[2] == candidate.capture_session_id)
        )


class ProcessedDisplayService:
    """Own one generation of bounded processed display encoding."""

    def __init__(
        self,
        *,
        max_fps: float = 10.0,
        encoder: Callable[[VideoFrame], bytes] = encode_jpeg,
        on_error: Callable[[], None] | None = None,
    ) -> None:
        self._max_fps = max_fps
        self._encoder_function = encoder
        self._on_error = on_error
        self._lock = threading.RLock()
        self._generation_number: int | None = None
        self._stream_epoch: str | None = None
        self._capture_session_id: str | None = None
        self._candidate_broker: _CandidateBroker | None = None
        self._output_broker: ProcessedDisplayBroker | None = None
        self._encoder: ProcessedDisplayEncoder | None = None
        self._shutdown_failure: str | None = None

    def start_generation(self, generation_number: int) -> None:
        """Create a fresh broker pair for a pipeline generation."""
        self.stop_generation(reason="stopped")
        candidate_broker = _CandidateBroker()
        output_broker = ProcessedDisplayBroker()
        encoder = ProcessedDisplayEncoder(
            candidate_broker,
            output_broker,
            max_fps=self._max_fps,
            encoder=self._encoder_function,
            current_identity=self._identity,
            on_error=self._on_error,
        )
        with self._lock:
            self._generation_number = generation_number
            self._stream_epoch = None
            self._capture_session_id = None
            self._candidate_broker = candidate_broker
            self._output_broker = output_broker
            self._encoder = encoder
            self._shutdown_failure = None
        encoder.start()

    def set_epoch(self, generation_number: int, stream_epoch: str) -> None:
        """Set the current epoch and invalidate earlier candidates."""
        with self._lock:
            if self._generation_number != generation_number:
                return
            self._stream_epoch = stream_epoch
            self._capture_session_id = None
            candidate_broker = self._candidate_broker
        if candidate_broker is not None:
            candidate_broker.clear()

    def set_provenance(
        self,
        generation_number: int,
        stream_epoch: str,
        capture_session_id: str,
    ) -> None:
        """Set the current epoch and capture session as one identity."""
        with self._lock:
            if self._generation_number != generation_number:
                return
            self._stream_epoch = stream_epoch
            self._capture_session_id = capture_session_id
            candidate_broker = self._candidate_broker
        if candidate_broker is not None:
            candidate_broker.clear()

    def submit(self, candidate: ProcessedDisplayCandidate) -> bool:
        """Submit one processed frame to the bounded encoder."""
        with self._lock:
            if self._generation_number != candidate.generation_number:
                return False
            if self._stream_epoch != candidate.stream_epoch or (
                self._capture_session_id is not None
                and self._capture_session_id != candidate.capture_session_id
            ):
                return False
            broker = self._candidate_broker
        return broker is not None and broker.put(candidate)

    def subscribe(self) -> ProcessedDisplaySubscription | None:
        """Subscribe to the current generation, if one is active."""
        with self._lock:
            broker = self._output_broker
        return None if broker is None or broker.closed else broker.subscribe()

    def metrics(self) -> DisplayBrokerMetrics:
        """Return current output broker metrics."""
        with self._lock:
            broker = self._output_broker
        if broker is None:
            return DisplayBrokerMetrics(0, 0, 0, 0, True)
        return broker.metrics()

    def stop_generation(self, *, reason: str = "stopped", timeout: float = 5.0) -> None:
        """Stop encoding and close subscribers for the current generation."""
        with self._lock:
            candidate_broker = self._candidate_broker
            output_broker = self._output_broker
            encoder = self._encoder
            self._generation_number = None
            self._stream_epoch = None
            self._capture_session_id = None
            self._candidate_broker = None
            self._output_broker = None
        if output_broker is not None:
            output_broker.close(reason)
        if encoder is not None:
            try:
                encoder.stop(timeout=timeout)
            except DisplayShutdownError as error:
                with self._lock:
                    self._encoder = encoder
                    self._shutdown_failure = str(error)
                raise
            else:
                with self._lock:
                    if self._encoder is encoder:
                        self._encoder = None
        elif candidate_broker is not None:
            candidate_broker.close()

    def close(self) -> None:
        """Close the service during application shutdown."""
        self.stop_generation(reason="shutdown")

    @property
    def encoder_thread(self) -> threading.Thread | None:
        """Return the encoder thread for bounded shutdown inspection."""
        with self._lock:
            return None if self._encoder is None else self._encoder.thread

    @property
    def shutdown_failure(self) -> str | None:
        """Return a safe shutdown failure, if the worker missed its deadline."""
        with self._lock:
            return self._shutdown_failure

    def _identity(self) -> tuple[int, str | None, str | None] | None:
        with self._lock:
            if self._generation_number is None:
                return None
            return (
                self._generation_number,
                self._stream_epoch,
                self._capture_session_id,
            )


def build_display_candidate(
    *,
    generation_number: int,
    frame: VideoFrame,
    camera_id: str,
    model_id: str,
    model_checksum: str,
    capture_session_id: str,
    stream_epoch: str,
    detections: tuple[Detection, ...],
    threshold: float,
    region_of_interest: Mapping[str, int] | None,
    metrics: Mapping[str, int | float | None],
    active_tracks: tuple[ActiveTrack, ...] = (),
) -> ProcessedDisplayCandidate:
    """Copy one processed frame into a bounded display candidate."""
    pixels = np.ascontiguousarray(np.asarray(frame.data)).copy()
    copied_frame = VideoFrame(
        sequence=frame.sequence,
        data=pixels,
        pixel_format=frame.pixel_format,
        host_received_at=frame.host_received_at,
        host_received_monotonic=frame.host_received_monotonic,
        capture_session_id=frame.capture_session_id,
        width=frame.width,
        height=frame.height,
        camera_pts=frame.camera_pts,
        camera_pts_seconds=frame.camera_pts_seconds,
    )
    return ProcessedDisplayCandidate(
        generation_number=generation_number,
        stream_epoch=stream_epoch,
        frame_sequence=frame.sequence,
        frame=copied_frame,
        camera_id=camera_id,
        model_id=model_id,
        model_checksum=model_checksum,
        capture_session_id=capture_session_id,
        captured_at=frame.host_received_at,
        source_width=frame.width,
        source_height=frame.height,
        detections=detections,
        active_tracks=active_tracks,
        threshold=threshold,
        region_of_interest=None if region_of_interest is None else dict(region_of_interest),
        metrics=dict(metrics),
    )


def build_display_unit(
    candidate: ProcessedDisplayCandidate,
    *,
    jpeg: bytes,
    jpeg_width: int,
    jpeg_height: int,
) -> ProcessedDisplayUnit:
    """Build and validate one atomic metadata/JPEG pair."""
    if not isinstance(jpeg, bytes) or not jpeg:
        raise DisplayProtocolError("processed JPEG is invalid")
    if len(jpeg) > MAX_DISPLAY_JPEG_BYTES:
        raise DisplayMessageTooLarge("processed JPEG is too large")
    if not candidate.stream_epoch or candidate.frame_sequence < 0:
        raise DisplayProtocolError("processed frame identity is invalid")
    _validate_dimensions(candidate.source_width, candidate.source_height, "source")
    _validate_dimensions(jpeg_width, jpeg_height, "JPEG")
    if not 0 <= candidate.threshold <= 1:
        raise DisplayProtocolError("confidence threshold is invalid")
    if len(candidate.detections) > MAX_DISPLAY_DETECTIONS:
        raise DisplayMessageTooLarge("processed frame has too many detections")
    detections = [_detection_payload(detection) for detection in candidate.detections]
    if len(candidate.active_tracks) > MAX_DISPLAY_ACTIVE_TRACKS:
        raise DisplayMessageTooLarge("processed frame has too many active tracks")
    active_tracks = [track.as_dict() for track in candidate.active_tracks]
    metadata: dict[str, Any] = {
        "type": "frame_header",
        "message_type": "frame_header",
        "protocol_version": LIVE_PROTOCOL_VERSION,
        "generation_number": candidate.generation_number,
        "camera_id": candidate.camera_id,
        "model_id": candidate.model_id,
        "model_checksum": candidate.model_checksum,
        "capture_session_id": candidate.capture_session_id,
        "stream_epoch": candidate.stream_epoch,
        "frame_sequence": candidate.frame_sequence,
        "captured_at_utc": _timestamp(candidate.captured_at),
        "capture_timestamp": _timestamp(candidate.captured_at),
        "source_width": candidate.source_width,
        "source_height": candidate.source_height,
        "jpeg_width": jpeg_width,
        "jpeg_height": jpeg_height,
        "jpeg_byte_count": len(jpeg),
        "detections": detections,
        "active_tracks": active_tracks,
        "confidence_threshold": candidate.threshold,
        "threshold": candidate.threshold,
        "region_of_interest": candidate.region_of_interest,
        "roi": candidate.region_of_interest,
        "metrics": dict(candidate.metrics),
    }
    metadata_text = json.dumps(metadata, ensure_ascii=True, separators=(",", ":"))
    metadata_size = len(metadata_text.encode("utf-8"))
    if metadata_size > MAX_DISPLAY_METADATA_BYTES:
        raise DisplayMessageTooLarge("processed frame metadata is too large")
    unit = ProcessedDisplayUnit(
        metadata=metadata,
        metadata_text=metadata_text,
        jpeg=bytes(jpeg),
        generation_number=candidate.generation_number,
        stream_epoch=candidate.stream_epoch,
        capture_session_id=candidate.capture_session_id,
        frame_sequence=candidate.frame_sequence,
    )
    validate_display_unit(unit)
    return unit


def validate_display_unit(unit: ProcessedDisplayUnit) -> None:
    """Validate protocol, identity, geometry, and message size."""
    if unit.metadata_text is None or not isinstance(unit.metadata_text, str):
        raise DisplayProtocolError("processed frame metadata is invalid")
    if len(unit.metadata_text.encode("utf-8")) > MAX_DISPLAY_METADATA_BYTES:
        raise DisplayMessageTooLarge("processed frame metadata is too large")
    if not isinstance(unit.jpeg, bytes):
        raise DisplayProtocolError("processed JPEG is invalid")
    if not unit.jpeg:
        raise DisplayProtocolError("processed JPEG is invalid")
    if len(unit.jpeg) > MAX_DISPLAY_JPEG_BYTES:
        raise DisplayMessageTooLarge("processed JPEG is too large")
    try:
        decoded = json.loads(unit.metadata_text)
    except (TypeError, json.JSONDecodeError):
        raise DisplayProtocolError("processed frame metadata is invalid") from None
    if not isinstance(decoded, dict) or decoded != dict(unit.metadata):
        raise DisplayProtocolError("processed frame metadata pairing is invalid")
    if decoded.get("protocol_version") != LIVE_PROTOCOL_VERSION:
        raise DisplayProtocolError("unsupported live WebSocket protocol")
    if decoded.get("type") != "frame_header" or decoded.get("message_type") != "frame_header":
        raise DisplayProtocolError("processed frame message type is invalid")
    if (
        type(decoded.get("generation_number")) is not int
        or decoded["generation_number"] != unit.generation_number
        or decoded["generation_number"] < 0
    ):
        raise DisplayProtocolError("processed frame generation pairing is invalid")
    for name in (
        "camera_id",
        "model_id",
        "model_checksum",
        "capture_session_id",
        "captured_at_utc",
    ):
        if not isinstance(decoded.get(name), str) or not decoded[name]:
            raise DisplayProtocolError("processed frame provenance is invalid")
    if decoded.get("capture_timestamp") != decoded.get("captured_at_utc"):
        raise DisplayProtocolError("processed frame timestamp pairing is invalid")
    if (
        not isinstance(decoded.get("stream_epoch"), str)
        or not decoded["stream_epoch"]
        or decoded.get("stream_epoch") != unit.stream_epoch
    ):
        raise DisplayProtocolError("processed frame epoch pairing is invalid")
    if (
        not isinstance(decoded.get("capture_session_id"), str)
        or not decoded["capture_session_id"]
        or decoded.get("capture_session_id") != unit.capture_session_id
    ):
        raise DisplayProtocolError("processed frame capture session pairing is invalid")
    if (
        type(decoded.get("frame_sequence")) is not int
        or decoded["frame_sequence"] < 0
        or decoded.get("frame_sequence") != unit.frame_sequence
    ):
        raise DisplayProtocolError("processed frame sequence pairing is invalid")
    if decoded.get("jpeg_byte_count") != len(unit.jpeg):
        raise DisplayProtocolError("processed frame JPEG pairing is invalid")
    for name in ("source_width", "source_height", "jpeg_width", "jpeg_height"):
        value = decoded.get(name)
        if type(value) is not int:
            raise DisplayProtocolError("processed frame geometry is invalid")
    _validate_dimensions(decoded["source_width"], decoded["source_height"], "source")
    _validate_dimensions(decoded["jpeg_width"], decoded["jpeg_height"], "JPEG")
    threshold = decoded.get("confidence_threshold")
    if (
        not isinstance(threshold, (int, float))
        or isinstance(threshold, bool)
        or not np.isfinite(float(threshold))
        or not 0 <= float(threshold) <= 1
        or decoded.get("threshold") != threshold
    ):
        raise DisplayProtocolError("processed frame threshold is invalid")
    if decoded.get("roi") != decoded.get("region_of_interest"):
        raise DisplayProtocolError("processed frame ROI pairing is invalid")
    _validate_roi(
        decoded.get("region_of_interest"),
        source_width=decoded["source_width"],
        source_height=decoded["source_height"],
    )
    _validate_metrics(decoded.get("metrics"))
    detections = decoded.get("detections")
    if not isinstance(detections, list) or len(detections) > MAX_DISPLAY_DETECTIONS:
        raise DisplayProtocolError("processed frame detections are invalid")
    for detection in detections:
        _validate_detection(
            detection,
            source_width=decoded["source_width"],
            source_height=decoded["source_height"],
            frame_sequence=decoded["frame_sequence"],
            model_id=decoded["model_id"],
            model_checksum=decoded["model_checksum"],
        )
    _validate_active_tracks(
        decoded.get("active_tracks"),
        source_width=decoded["source_width"],
        source_height=decoded["source_height"],
        camera_id=decoded["camera_id"],
        generation_number=decoded["generation_number"],
        stream_epoch=decoded["stream_epoch"],
        capture_session_id=decoded["capture_session_id"],
        model_id=decoded["model_id"],
        model_checksum=decoded["model_checksum"],
    )


def jpeg_dimensions(jpeg: bytes) -> tuple[int, int]:
    """Read safe JPEG geometry in the encoding worker."""
    if not isinstance(jpeg, bytes) or len(jpeg) > MAX_DISPLAY_JPEG_BYTES:
        raise DisplayMessageTooLarge("processed JPEG is too large")
    try:
        with Image.open(BytesIO(jpeg)) as image:
            if image.format != "JPEG":
                raise DisplayProtocolError("processed preview is not a JPEG")
            image.verify()
        with Image.open(BytesIO(jpeg)) as image:
            width, height = image.size
    except DisplayProtocolError:
        raise
    except Exception:
        raise DisplayProtocolError("processed preview JPEG is invalid") from None
    _validate_dimensions(width, height, "JPEG")
    return int(width), int(height)


def _detection_payload(detection: Detection) -> dict[str, Any]:
    return {
        "box_xyxy": [float(value) for value in detection.box_xyxy],
        "class_id": detection.class_id,
        "label": detection.label,
        "confidence": float(detection.confidence),
        "model_id": detection.model_id,
        "model_checksum": detection.model_sha256,
        "frame_sequence": detection.frame_sequence,
    }


def _validate_detection(
    value: object,
    *,
    source_width: int,
    source_height: int,
    frame_sequence: int,
    model_id: str,
    model_checksum: str,
) -> None:
    if not isinstance(value, dict):
        raise DisplayProtocolError("processed detection is invalid")
    box = value.get("box_xyxy")
    confidence = value.get("confidence")
    label = value.get("label")
    detection_frame_sequence = value.get("frame_sequence")
    detection_model_id = value.get("model_id")
    detection_model_checksum = value.get("model_checksum")
    if (
        not isinstance(box, list)
        or len(box) != 4
        or not isinstance(label, str)
        or not label
        or not isinstance(confidence, (int, float))
        or type(detection_frame_sequence) is not int
        or detection_frame_sequence != frame_sequence
        or detection_model_id != model_id
        or detection_model_checksum != model_checksum
    ):
        raise DisplayProtocolError("processed detection is invalid")
    try:
        x1, y1, x2, y2 = (float(coordinate) for coordinate in box)
        confidence_value = float(confidence)
    except (TypeError, ValueError):
        raise DisplayProtocolError("processed detection is invalid") from None
    if (
        not all(np.isfinite((x1, y1, x2, y2)))
        or not 0 <= x1 <= x2 <= source_width
        or not 0 <= y1 <= y2 <= source_height
        or not 0 <= confidence_value <= 1
    ):
        raise DisplayProtocolError("processed detection geometry is invalid")


def _validate_active_tracks(
    value: object,
    *,
    source_width: int,
    source_height: int,
    camera_id: str,
    generation_number: int,
    stream_epoch: str,
    capture_session_id: str,
    model_id: str,
    model_checksum: str,
) -> None:
    if not isinstance(value, list) or len(value) > MAX_DISPLAY_ACTIVE_TRACKS:
        raise DisplayProtocolError("processed active tracks are invalid")
    track_ids: set[int] = set()
    for track in value:
        if not isinstance(track, dict):
            raise DisplayProtocolError("processed active track is invalid")
        if (
            not isinstance(track.get("camera_id"), str)
            or not track["camera_id"]
            or track.get("camera_id") != camera_id
            or track.get("capture_session_id") != capture_session_id
            or track.get("generation_number") != generation_number
            or track.get("stream_epoch") != stream_epoch
            or track.get("model_id") != model_id
            or track.get("model_checksum") != model_checksum
            or track.get("state") not in {"confirmed", "active"}
            or type(track.get("track_id")) is not int
            or track["track_id"] < 0
            or track["track_id"] in track_ids
            or type(track.get("last_frame_sequence")) is not int
            or track["last_frame_sequence"] < 0
            or type(track.get("observation_count")) is not int
            or track["observation_count"] <= 0
        ):
            raise DisplayProtocolError("processed active track provenance is invalid")
        box = track.get("last_box_xyxy")
        if (
            not isinstance(box, list)
            or len(box) != 4
            or not all(
                isinstance(item, (int, float)) and not isinstance(item, bool) for item in box
            )
        ):
            raise DisplayProtocolError("processed active track geometry is invalid")
        x1, y1, x2, y2 = (float(item) for item in box)
        if (
            not all(np.isfinite((x1, y1, x2, y2)))
            or not 0 <= x1 < x2 <= source_width
            or not 0 <= y1 < y2 <= source_height
        ):
            raise DisplayProtocolError("processed active track geometry is invalid")
        for name in ("last_confidence", "maximum_confidence"):
            confidence = track.get(name)
            if (
                not isinstance(confidence, (int, float))
                or isinstance(confidence, bool)
                or not np.isfinite(float(confidence))
                or not 0 <= float(confidence) <= 1
            ):
                raise DisplayProtocolError("processed active track confidence is invalid")
        for name in ("first_seen_utc", "last_seen_utc"):
            if not isinstance(track.get(name), str) or not track[name]:
                raise DisplayProtocolError("processed active track timestamp is invalid")
        track_ids.add(track["track_id"])


def _validate_roi(value: object, *, source_width: int, source_height: int) -> None:
    if value is None:
        return
    if not isinstance(value, dict) or set(value) != {"x", "y", "width", "height"}:
        raise DisplayProtocolError("processed frame ROI is invalid")
    if any(type(value[key]) is not int for key in value):
        raise DisplayProtocolError("processed frame ROI is invalid")
    if (
        value["x"] < 0
        or value["y"] < 0
        or value["width"] <= 0
        or value["height"] <= 0
        or value["x"] + value["width"] > source_width
        or value["y"] + value["height"] > source_height
    ):
        raise DisplayProtocolError("processed frame ROI is invalid")


def _validate_metrics(value: object) -> None:
    if not isinstance(value, dict) or len(value) > 64:
        raise DisplayProtocolError("processed frame metrics are invalid")
    for metric in value.values():
        if metric is None:
            continue
        if isinstance(metric, bool) or not isinstance(metric, (int, float)):
            raise DisplayProtocolError("processed frame metrics are invalid")
        if not np.isfinite(float(metric)):
            raise DisplayProtocolError("processed frame metrics are invalid")


def _validate_dimensions(width: int, height: int, label: str) -> None:
    if (
        type(width) is not int
        or type(height) is not int
        or width <= 0
        or height <= 0
        or width > MAX_DISPLAY_DIMENSION
        or height > MAX_DISPLAY_DIMENSION
    ):
        raise DisplayProtocolError(f"{label} dimensions are invalid")


def _timestamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _deadline(timeout: float | None) -> float | None:
    return None if timeout is None else time.monotonic() + max(0.0, timeout)


def _remaining(deadline: float | None) -> float | None:
    if deadline is None:
        return None
    remaining = deadline - time.monotonic()
    return 0.0 if remaining <= 0 else remaining


__all__ = [
    "DisplayBrokerMetrics",
    "DISPLAY_BUFFER_CAPACITY",
    "DisplayMessageTooLarge",
    "DisplayProtocolError",
    "DisplayShutdownError",
    "LIVE_PROTOCOL_VERSION",
    "MAX_RETIRED_PROVENANCES",
    "MAX_DISPLAY_JPEG_BYTES",
    "MAX_DISPLAY_METADATA_BYTES",
    "ProcessedDisplayBroker",
    "ProcessedDisplayCandidate",
    "ProcessedDisplayEncoder",
    "ProcessedDisplayService",
    "ProcessedDisplaySubscription",
    "ProcessedDisplayUnit",
    "build_display_candidate",
    "build_display_unit",
    "jpeg_dimensions",
    "validate_display_unit",
]
