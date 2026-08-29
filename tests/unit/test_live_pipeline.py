from __future__ import annotations

import time
from typing import Any

import numpy as np
import pytest

from open_licenseplate.cameras.service import prepare_camera_config
from open_licenseplate.capture import CameraRuntime, FakeFrameSource, LatestFrameBroker
from open_licenseplate.inference import ModelDescriptor
from open_licenseplate.inference.backends import FakeBackend
from open_licenseplate.live import (
    LivePipelineConflict,
    LivePipelineCoordinator,
    SourcePixelRegionOfInterest,
)
from open_licenseplate.models.manifest import parse_manifest


def _descriptor(model_id: str = "model-1") -> ModelDescriptor:
    manifest = parse_manifest(
        {
            "schema_version": 1,
            "id": model_id,
            "display_name": "Test model",
            "task": "object_detection",
            "backend": "coreml",
            "adapter": "ultralytics_yolo_nms",
            "artifact": "model.mlpackage",
            "artifact_sha256": "a" * 64,
            "input": {
                "name": "image",
                "kind": "image",
                "width": 8,
                "height": 8,
                "color_space": "rgb",
            },
            "preprocessing": {"resize": "stretch"},
            "outputs": {
                "boxes": "coordinates",
                "scores": "confidence",
                "box_format": "xyxy",
                "coordinate_space": "model_pixels",
            },
            "labels": ["license_plate"],
            "defaults": {"confidence_threshold": 0.35, "iou_threshold": 0.45},
        }
    )
    return ModelDescriptor(
        model_id=model_id,
        artifact_path="/managed/model.mlpackage",
        artifact_sha256=manifest.artifact_sha256,
        manifest=manifest,
    )


def _camera(name: str = "Fixture") -> Any:
    return prepare_camera_config(
        name=name,
        rtsp_url="rtsp://fixture.local/live",
    )


def _outputs(_prepared: Any) -> dict[str, Any]:
    return {
        "coordinates": np.array([[0, 0, 8, 8]], dtype=np.float32),
        "confidence": np.array([0.8], dtype=np.float32),
    }


def _runtime(sources: list[FakeFrameSource]) -> CameraRuntime:
    def source_factory(camera: Any, camera_id: str) -> FakeFrameSource:
        source = FakeFrameSource(
            [np.full((8, 8, 3), 40, dtype=np.uint8)],
            camera_id=camera_id,
            repeat=True,
            read_interval_seconds=0.002,
        )
        sources.append(source)
        return source

    return CameraRuntime(source_factory, poll_interval_seconds=0.002)


def _wait_processed(
    coordinator: LivePipelineCoordinator,
    *,
    count: int = 1,
    timeout: float = 2.0,
) -> Any:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = coordinator.status()
        if status.metrics.processed_frames >= count:
            return status
        time.sleep(0.005)
    raise AssertionError("live pipeline did not process the expected frame count")


def test_live_pipeline_warms_processes_current_frame_updates_threshold_and_stops() -> None:
    sources: list[FakeFrameSource] = []
    runtime = _runtime(sources)
    backend = FakeBackend(output_factory=_outputs)
    coordinator = LivePipelineCoordinator(
        runtime,
        lambda: backend,
        poll_interval_seconds=0.002,
        epoch_factory=iter(("epoch-1", "epoch-2")).__next__,
    )

    started = coordinator.start("camera-1", _camera(), _descriptor())
    assert started.state == "starting"
    warming = coordinator.wait_for_state("warming")
    assert "warming" in warming.state_history
    running = coordinator.wait_for_state("running")
    assert running.metrics.warmup_ms is not None

    processed = _wait_processed(coordinator)
    assert processed.last_result is not None
    assert processed.last_result.camera_id == "camera-1"
    assert processed.last_result.model_id == "model-1"
    assert processed.last_result.capture_session_id
    assert processed.last_result.epoch == processed.epoch
    assert processed.last_result.frame_sequence >= 1
    assert (
        processed.last_result.detections[0].frame_sequence == processed.last_result.frame_sequence
    )

    updated = coordinator.update_threshold(0.9)
    assert updated.confidence_threshold == 0.9
    hidden = _wait_processed(coordinator, count=processed.metrics.processed_frames + 1)
    assert hidden.last_result is not None
    assert hidden.last_result.detections == ()
    assert hidden.metrics.source_replacement_count >= 0

    stopped = coordinator.stop()
    assert stopped.state == "stopped"
    assert sources and sources[0].closed.is_set()
    assert backend.closes
    assert backend.closes[-1].closed is True


def test_live_pipeline_rejects_camera_and_model_switch_while_running() -> None:
    sources: list[FakeFrameSource] = []
    runtime = _runtime(sources)
    coordinator = LivePipelineCoordinator(
        runtime,
        lambda: FakeBackend(output_factory=_outputs),
        poll_interval_seconds=0.002,
    )
    coordinator.start("camera-1", _camera(), _descriptor("model-1"))
    coordinator.wait_for_state("running")

    with pytest.raises(LivePipelineConflict, match="switching the camera"):
        coordinator.start("camera-2", _camera("Second"), _descriptor("model-1"))
    with pytest.raises(LivePipelineConflict, match="switching the model"):
        coordinator.start("camera-1", _camera(), _descriptor("model-2"))

    coordinator.stop()


def test_live_pipeline_restores_detection_coordinates_from_source_pixel_roi() -> None:
    sources: list[FakeFrameSource] = []
    runtime = _runtime(sources)
    coordinator = LivePipelineCoordinator(
        runtime,
        lambda: FakeBackend(output_factory=_outputs),
        poll_interval_seconds=0.002,
    )
    coordinator.start(
        "camera-1",
        _camera(),
        _descriptor(),
        region_of_interest=SourcePixelRegionOfInterest(2, 1, 4, 4),
    )
    coordinator.wait_for_state("running")

    processed = _wait_processed(coordinator)
    assert processed.last_result is not None
    assert processed.last_result.detections[0].box_xyxy == (2.0, 1.0, 6.0, 5.0)
    coordinator.stop()


def test_live_pipeline_replaces_old_frames_during_slow_inference() -> None:
    sources: list[FakeFrameSource] = []

    def slow_outputs(_prepared: Any) -> dict[str, Any]:
        time.sleep(0.04)
        return _outputs(_prepared)

    runtime = _runtime(sources)
    backend = FakeBackend(output_factory=slow_outputs)
    coordinator = LivePipelineCoordinator(
        runtime,
        lambda: backend,
        poll_interval_seconds=0.002,
    )
    coordinator.start("camera-1", _camera(), _descriptor())
    coordinator.wait_for_state("running")
    processed = _wait_processed(coordinator, count=2, timeout=3)

    assert processed.metrics.inference_replacement_count > 0
    assert processed.metrics.processed_frames >= 2
    assert processed.metrics.capture_age_ms is not None
    assert processed.metrics.end_to_end_ms is not None
    coordinator.stop()


def test_latest_frame_inference_handoff_is_capacity_one() -> None:
    broker = LatestFrameBroker()
    subscription = broker.subscribe()
    frames = [_frame(sequence, received_monotonic=float(sequence)) for sequence in range(1, 4)]
    for frame in frames:
        assert broker.put(frame)

    assert subscription.metrics().replaced_frames == 2
    current = subscription.get(timeout=0)
    assert current is not None
    assert current.sequence == 3
    assert subscription.get(timeout=0) is None
    subscription.close()
    broker.close()


def _frame(sequence: int, *, received_monotonic: float) -> Any:
    from datetime import UTC, datetime

    from open_licenseplate.capture import VideoFrame

    return VideoFrame(
        sequence=sequence,
        data=np.zeros((2, 2, 3), dtype=np.uint8),
        pixel_format="bgr24",
        host_received_at=datetime(2026, 8, 29, tzinfo=UTC),
        host_received_monotonic=received_monotonic,
        capture_session_id="session-1",
        width=2,
        height=2,
    )
