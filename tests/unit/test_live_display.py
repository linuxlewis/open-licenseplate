from __future__ import annotations

import json
import threading
import time
from datetime import UTC, datetime
from io import BytesIO
from typing import Any

import numpy as np
import pytest
from PIL import Image

from open_licenseplate.capture import VideoFrame
from open_licenseplate.inference import Detection
from open_licenseplate.live import (
    LIVE_PROTOCOL_VERSION,
    DisplayMessageTooLarge,
    DisplayProtocolError,
    DisplayShutdownError,
    ProcessedDisplayBroker,
    ProcessedDisplayService,
    ProcessedDisplayUnit,
    build_display_candidate,
    build_display_unit,
    validate_display_unit,
)


def _frame(sequence: int, *, width: int = 8, height: int = 6) -> VideoFrame:
    return VideoFrame(
        sequence=sequence,
        data=np.full((height, width, 3), sequence, dtype=np.uint8),
        pixel_format="bgr24",
        host_received_at=datetime(2026, 8, 29, tzinfo=UTC),
        host_received_monotonic=float(sequence),
        capture_session_id="session-1",
        width=width,
        height=height,
    )


def _jpeg(width: int = 8, height: int = 6) -> bytes:
    image = Image.new("RGB", (width, height), color=(40, 80, 120))
    output = BytesIO()
    image.save(output, format="JPEG")
    return output.getvalue()


def _candidate(sequence: int, epoch: str = "epoch-1") -> Any:
    return build_display_candidate(
        generation_number=1,
        frame=_frame(sequence),
        camera_id="camera-1",
        model_id="model-1",
        model_checksum="a" * 64,
        capture_session_id="session-1",
        stream_epoch=epoch,
        detections=(
            Detection(
                box_xyxy=(1.0, 1.0, 5.0, 4.0),
                class_id=0,
                label="license_plate",
                confidence=0.8,
                model_id="model-1",
                model_sha256="a" * 64,
                frame_sequence=sequence,
            ),
        ),
        threshold=0.35,
        region_of_interest={"x": 0, "y": 0, "width": 8, "height": 6},
        metrics={"processed_fps": 4.0, "prediction_ms": 3.0},
    )


def test_processed_display_unit_has_required_provenance_and_ordered_pair() -> None:
    unit = build_display_unit(
        _candidate(7),
        jpeg=_jpeg(),
        jpeg_width=8,
        jpeg_height=6,
    )

    header = json.loads(unit.metadata_text)
    assert unit.message_pair() == (unit.metadata_text, unit.jpeg)
    assert header["protocol_version"] == LIVE_PROTOCOL_VERSION
    assert header["type"] == "frame_header"
    assert header["camera_id"] == "camera-1"
    assert header["model_id"] == "model-1"
    assert header["model_checksum"] == "a" * 64
    assert header["capture_session_id"] == "session-1"
    assert header["stream_epoch"] == "epoch-1"
    assert header["frame_sequence"] == 7
    assert header["source_width"] == 8
    assert header["source_height"] == 6
    assert header["jpeg_width"] == 8
    assert header["jpeg_height"] == 6
    assert header["detections"][0]["box_xyxy"] == [1.0, 1.0, 5.0, 4.0]
    assert header["confidence_threshold"] == 0.35
    assert header["region_of_interest"]["width"] == 8
    assert header["metrics"]["prediction_ms"] == 3.0
    validate_display_unit(unit)


def test_processed_display_broker_replaces_slow_client_units_atomically() -> None:
    broker = ProcessedDisplayBroker()
    subscription = broker.subscribe()
    for sequence in range(1, 4):
        assert broker.put(
            build_display_unit(_candidate(sequence), jpeg=b"jpeg", jpeg_width=8, jpeg_height=6)
        )

    assert subscription.metrics().replaced_units == 2
    unit = subscription.get(timeout=0)
    assert unit is not None
    assert unit.frame_sequence == 3
    assert json.loads(unit.metadata_text)["frame_sequence"] == unit.frame_sequence
    assert unit.jpeg == b"jpeg"
    assert subscription.get(timeout=0) is None
    subscription.close()
    broker.close()


def test_processed_display_protocol_rejects_bad_pair_and_oversized_jpeg() -> None:
    unit = build_display_unit(_candidate(1), jpeg=b"jpeg", jpeg_width=8, jpeg_height=6)
    bad_metadata = dict(unit.metadata)
    bad_metadata["frame_sequence"] = 2
    bad_unit = ProcessedDisplayUnit(
        metadata=bad_metadata,
        metadata_text=json.dumps(bad_metadata, separators=(",", ":")),
        jpeg=unit.jpeg,
        generation_number=unit.generation_number,
        stream_epoch=unit.stream_epoch,
        capture_session_id=unit.capture_session_id,
        frame_sequence=unit.frame_sequence,
    )
    with pytest.raises(DisplayProtocolError, match="sequence pairing"):
        validate_display_unit(bad_unit)

    oversized = ProcessedDisplayUnit(
        metadata=unit.metadata,
        metadata_text=unit.metadata_text,
        jpeg=b"x" * (4 * 1024 * 1024 + 1),
        generation_number=unit.generation_number,
        stream_epoch=unit.stream_epoch,
        capture_session_id=unit.capture_session_id,
        frame_sequence=unit.frame_sequence,
    )
    with pytest.raises(DisplayMessageTooLarge):
        validate_display_unit(oversized)


def test_processed_display_protocol_rejects_detection_provenance_mismatch() -> None:
    unit = build_display_unit(_candidate(1), jpeg=b"jpeg", jpeg_width=8, jpeg_height=6)
    bad_metadata = json.loads(unit.metadata_text)
    bad_metadata["detections"][0]["frame_sequence"] = 2
    bad_unit = ProcessedDisplayUnit(
        metadata=bad_metadata,
        metadata_text=json.dumps(bad_metadata, separators=(",", ":")),
        jpeg=unit.jpeg,
        generation_number=unit.generation_number,
        stream_epoch=unit.stream_epoch,
        capture_session_id=unit.capture_session_id,
        frame_sequence=unit.frame_sequence,
    )
    with pytest.raises(DisplayProtocolError, match="detection is invalid"):
        validate_display_unit(bad_unit)


def test_processed_display_service_discards_stale_epoch_data() -> None:
    service = ProcessedDisplayService(max_fps=60)
    service.start_generation(1)
    subscription = service.subscribe()
    assert subscription is not None
    service.set_epoch(1, "epoch-1")
    assert service.submit(_candidate(1, "epoch-1"))
    first = subscription.get(timeout=1)
    assert first is not None
    assert first.stream_epoch == "epoch-1"

    service.set_epoch(1, "epoch-2")
    assert service.submit(_candidate(2, "epoch-1")) is False
    assert service.submit(_candidate(3, "epoch-2"))
    second = subscription.get(timeout=1)
    assert second is not None
    assert second.stream_epoch == "epoch-2"
    assert second.frame_sequence == 3
    service.stop_generation(reason="stopped")
    assert subscription.closed


def test_processed_display_encoder_rate_is_bounded_and_shutdown_is_clean() -> None:
    encoded_at: list[float] = []

    def encoder(frame: VideoFrame) -> bytes:
        encoded_at.append(time.monotonic())
        return _jpeg(frame.width, frame.height)

    service = ProcessedDisplayService(max_fps=5, encoder=encoder)
    service.start_generation(1)
    subscription = service.subscribe()
    assert subscription is not None
    service.set_epoch(1, "epoch-1")
    for sequence in range(1, 8):
        service.submit(_candidate(sequence))
        time.sleep(0.005)
    time.sleep(0.5)
    service.stop_generation(reason="stopped")

    assert subscription.closed
    assert len(encoded_at) >= 2
    intervals = [
        later - earlier for earlier, later in zip(encoded_at, encoded_at[1:], strict=False)
    ]
    assert min(intervals) >= 0.15


def test_processed_display_service_reports_and_recovers_from_stuck_encoder() -> None:
    encoder_started = threading.Event()
    release_encoder = threading.Event()

    def stuck_encoder(_frame: VideoFrame) -> bytes:
        encoder_started.set()
        release_encoder.wait()
        return _jpeg()

    service = ProcessedDisplayService(max_fps=60, encoder=stuck_encoder)
    service.start_generation(1)
    subscription = service.subscribe()
    assert subscription is not None
    service.set_provenance(1, "epoch-1", "session-1")
    assert service.submit(_candidate(1, "epoch-1"))
    assert encoder_started.wait(1)

    with pytest.raises(DisplayShutdownError, match="did not stop"):
        service.stop_generation(reason="shutdown", timeout=0.01)
    thread = service.encoder_thread
    assert thread is not None
    assert thread.is_alive()
    assert service.shutdown_failure == "processed display encoder did not stop before the deadline"
    assert subscription.closed

    release_encoder.set()
    service.stop_generation(reason="shutdown", timeout=1)
    assert not thread.is_alive()
