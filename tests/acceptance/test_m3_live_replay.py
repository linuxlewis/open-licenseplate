from __future__ import annotations

import gc
import json
import threading
import tracemalloc
import weakref
from datetime import UTC, datetime
from io import BytesIO
from typing import Any

import numpy as np
import pytest
from PIL import Image

from m3_replay import GeneratedReplaySource, M3ReplayFixture, ReplayClock, ReplayGate
from open_licenseplate.cameras.service import prepare_camera_config
from open_licenseplate.capture import (
    CameraRuntime,
    FakeFrameSource,
    FixtureAttempt,
    LatestFrameBroker,
    LatestFrameSubscription,
    ReconnectBackoff,
    ReconnectFixture,
    VideoFrame,
)
from open_licenseplate.capture.broker import MAX_SUBSCRIBERS
from open_licenseplate.inference import ModelDescriptor
from open_licenseplate.inference.backends import FakeBackend
from open_licenseplate.live import (
    DISPLAY_BUFFER_CAPACITY,
    DisplayProtocolError,
    LivePipelineCoordinator,
    ProcessedDisplayBroker,
    ProcessedDisplayCandidate,
    ProcessedDisplayService,
    ProcessedDisplaySubscription,
    build_display_candidate,
    build_display_unit,
    validate_display_unit,
)
from open_licenseplate.live.display import MAX_DISPLAY_SUBSCRIBERS
from open_licenseplate.models.manifest import parse_manifest


def _fixture() -> M3ReplayFixture:
    return M3ReplayFixture.load()


def _camera() -> Any:
    return prepare_camera_config(
        name="M3 replay camera",
        rtsp_url="rtsp://fixture.local/live",
    )


def _descriptor() -> ModelDescriptor:
    manifest = parse_manifest(
        {
            "schema_version": 1,
            "id": "m3-replay-model",
            "display_name": "M3 replay model",
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
        model_id=manifest.model_id,
        artifact_path="/managed/model.mlpackage",
        artifact_sha256=manifest.artifact_sha256,
        manifest=manifest,
    )


def _outputs_for_replay(prepared: Any) -> dict[str, Any]:
    values = np.asarray(prepared.value)
    center = values[3:5, 2:6]
    if float(center.mean()) < 100:
        return {
            "coordinates": np.empty((0, 4), dtype=np.float32),
            "confidence": np.empty((0,), dtype=np.float32),
        }
    return {
        "coordinates": np.array([[2, 3, 6, 5]], dtype=np.float32),
        "confidence": np.array([0.9], dtype=np.float32),
    }


class NonRetainingFakeBackend(FakeBackend):
    """Keep the fake backend from hiding live-pipeline retention in tests."""

    def predict(self, model: Any, model_input: object) -> Any:
        output = super().predict(model, model_input)
        self.predictions.clear()
        return output


def _jpeg(frame: VideoFrame) -> bytes:
    output = BytesIO()
    Image.fromarray(np.asarray(frame.data), mode="RGB").save(output, format="JPEG")
    return output.getvalue()


def _frame(sequence: int, *, session: str = "session-1") -> VideoFrame:
    data = np.full((6, 8, 3), sequence % 255, dtype=np.uint8)
    return VideoFrame(
        sequence=sequence,
        data=data,
        pixel_format="bgr24",
        host_received_at=datetime(2026, 8, 29, tzinfo=UTC),
        host_received_monotonic=float(sequence),
        capture_session_id=session,
        width=8,
        height=6,
    )


def _candidate(
    sequence: int,
    *,
    generation: int = 1,
    epoch: str = "epoch-1",
    session: str = "session-1",
) -> ProcessedDisplayCandidate:
    frame = _frame(sequence, session=session)
    return build_display_candidate(
        generation_number=generation,
        frame=frame,
        camera_id="camera-1",
        model_id="model-1",
        model_checksum="a" * 64,
        capture_session_id=session,
        stream_epoch=epoch,
        detections=(),
        threshold=0.35,
        region_of_interest=None,
        metrics={"processed_fps": 4.0, "prediction_ms": 3.0},
    )


def _runtime_for_source(source: Any, *, clock: ReplayClock | None = None) -> CameraRuntime:
    source_factory = (
        source
        if callable(source) and not hasattr(source, "read")
        else lambda _camera, _camera_id: source
    )
    return CameraRuntime(
        source_factory,
        clock=clock,
        backoff=ReconnectBackoff(
            base_delay_seconds=0.001,
            cap_seconds=0.001,
            jitter_ratio=0,
            random_value=lambda: 0.5,
        ),
        stable_stream_seconds=0,
        poll_interval_seconds=0.001,
        worker_stop_timeout_seconds=1,
        degraded_hold_seconds=0,
    )


@pytest.mark.m3_acceptance
@pytest.mark.parametrize(
    ("kind", "has_detection"),
    [("no_plate", False), ("plate", True)],
)
def test_m3_replay_proves_no_plate_and_plate_detection_only(
    kind: str,
    has_detection: bool,
) -> None:
    fixture = _fixture()
    source = ReconnectFixture(
        (
            FixtureAttempt(
                frames=(fixture.frame(kind),),
                repeat=True,
            ),
        )
    )
    backend = NonRetainingFakeBackend(output_factory=_outputs_for_replay)
    coordinator = LivePipelineCoordinator(
        _runtime_for_source(source, clock=ReplayClock()),
        lambda: backend,
        poll_interval_seconds=0.001,
        display_max_fps=60,
        epoch_factory=iter(("m3-no-plate", "m3-plate")).__next__,
    )
    subscription: ProcessedDisplaySubscription | None = None
    try:
        coordinator.start("camera-1", _camera(), _descriptor())
        coordinator.wait_for_state("running")
        subscription = coordinator.subscribe_display()
        assert subscription is not None
        unit = subscription.get(timeout=5)
        assert unit is not None
        validate_display_unit(unit)
        detections = json.loads(unit.metadata_text)["detections"]
        assert bool(detections) is has_detection
        if has_detection:
            assert detections[0]["label"] == "license_plate"
            assert detections[0]["frame_sequence"] == unit.frame_sequence
        else:
            assert detections == []
    finally:
        if subscription is not None:
            subscription.close()
        coordinator.close()


@pytest.mark.m3_acceptance
def test_m3_slow_inference_replaces_old_frames_and_keeps_latest_current() -> None:
    fixture = _fixture()
    clock = ReplayClock()
    start_gate = threading.Event()
    inference_started = threading.Event()
    release_inference = threading.Event()
    hold_gate = ReplayGate()
    prediction_calls = 0

    def slow_outputs(prepared: Any) -> dict[str, Any]:
        nonlocal prediction_calls
        prediction_calls += 1
        if prediction_calls == 2:
            inference_started.set()
            assert release_inference.wait(5)
        return _outputs_for_replay(prepared)

    source = GeneratedReplaySource(
        camera_id="camera-1",
        total_frames=96,
        frame_rate=fixture.frame_rate,
        frame_factory=lambda index: fixture.frame("plate", index),
        clock=clock,
        session_id="slow-session",
        start_gate=start_gate,
        hold_after_frames=64,
        hold_gate=hold_gate,
    )
    runtime = _runtime_for_source(source, clock=clock)
    backend = NonRetainingFakeBackend(output_factory=slow_outputs)
    coordinator = LivePipelineCoordinator(
        runtime,
        lambda: backend,
        clock=clock,
        poll_interval_seconds=0.001,
        display_max_fps=60,
        epoch_factory=lambda: "slow-epoch",
    )
    subscription: ProcessedDisplaySubscription | None = None
    try:
        coordinator.start("camera-1", _camera(), _descriptor())
        coordinator.wait_for_state("running")
        subscription = coordinator.subscribe_display()
        assert subscription is not None
        start_gate.set()
        assert inference_started.wait(5)
        hold_gate.wait_until_reached(1)

        status = coordinator.status()
        assert source.frames_emitted >= 64
        assert status.metrics.source_replacement_count > 0
        assert status.metrics.inference_replacement_count > 0

        release_inference.set()
        unit = _get_unit_at_least(subscription, 64)
        assert unit is not None
        validate_display_unit(unit)
        assert unit.frame_sequence >= 64
        assert json.loads(unit.metadata_text)["frame_sequence"] == unit.frame_sequence
        assert coordinator.status().metrics.capture_age_ms is not None
    finally:
        release_inference.set()
        hold_gate.release(1)
        start_gate.set()
        if subscription is not None:
            subscription.close()
        coordinator.close()


@pytest.mark.m3_acceptance
def test_m3_reconnect_starts_new_provenance_and_drops_old_display_units() -> None:
    fixture = _fixture()
    release_failure = threading.Event()
    sources: list[FakeFrameSource] = []
    factory_calls = 0

    class FirstAttemptSource(FakeFrameSource):
        read_calls = 0

        def read(self) -> Any:
            if self.read_calls >= 1:
                release_failure.wait(5)
            self.read_calls += 1
            return super().read()

    def source_factory(_camera: Any, camera_id: str) -> FakeFrameSource:
        nonlocal factory_calls
        if factory_calls == 0:
            source = FirstAttemptSource(
                frames=(fixture.frame("plate"),),
                camera_id=camera_id,
                fail_at=1,
                read_error="controlled replay reconnect",
            )
        else:
            source = FakeFrameSource(
                frames=(fixture.frame("no_plate"),),
                camera_id=camera_id,
                repeat=True,
            )
        factory_calls += 1
        sources.append(source)
        return source

    coordinator = LivePipelineCoordinator(
        _runtime_for_source(source_factory, clock=ReplayClock()),
        lambda: NonRetainingFakeBackend(output_factory=_outputs_for_replay),
        poll_interval_seconds=0.001,
        display_max_fps=60,
        epoch_factory=iter(("reconnect-epoch-1", "reconnect-epoch-2")).__next__,
    )
    subscription: ProcessedDisplaySubscription | None = None
    try:
        coordinator.start("camera-1", _camera(), _descriptor())
        coordinator.wait_for_state("running")
        subscription = coordinator.subscribe_display()
        assert subscription is not None
        first = subscription.get(timeout=5)
        assert first is not None
        assert first.stream_epoch == "reconnect-epoch-1"
        assert sources[0].capture_session_id == first.capture_session_id
        assert first.frame_sequence >= 1
        release_failure.set()

        second = _get_unit_with_epoch(subscription, "reconnect-epoch-2")
        assert second.capture_session_id == sources[1].capture_session_id
        assert second.capture_session_id != first.capture_session_id
        assert second.frame_sequence >= 1
        assert second.generation_number == first.generation_number
        assert second.stream_epoch != first.stream_epoch
        validate_display_unit(second)
    finally:
        release_failure.set()
        if subscription is not None:
            subscription.close()
        coordinator.close()


@pytest.mark.m3_acceptance
def test_m3_source_display_and_websocket_buffers_have_capacity_one() -> None:
    assert LatestFrameBroker.capacity == 1
    assert LatestFrameSubscription.capacity == 1
    assert ProcessedDisplayBroker.capacity == DISPLAY_BUFFER_CAPACITY == 1
    assert ProcessedDisplaySubscription.capacity == DISPLAY_BUFFER_CAPACITY == 1
    assert MAX_SUBSCRIBERS == 16
    assert MAX_DISPLAY_SUBSCRIBERS == 16

    source_broker = LatestFrameBroker()
    source_subscription = source_broker.subscribe()
    for sequence in range(1, 65):
        assert source_broker.put(_frame(sequence))
    assert source_subscription.metrics().replaced_frames == 63
    latest_frame = source_subscription.get(timeout=0)
    assert latest_frame is not None
    assert latest_frame.sequence == 64
    source_subscription.close()
    source_broker.close()

    display_broker = ProcessedDisplayBroker()
    slow_websocket = display_broker.subscribe()
    for sequence in range(1, 65):
        candidate = _candidate(sequence)
        display_broker.put(
            build_display_unit(
                candidate,
                jpeg=_jpeg(candidate.frame),
                jpeg_width=8,
                jpeg_height=6,
            )
        )
    assert display_broker.metrics().replaced_units == 63
    assert slow_websocket.metrics().replaced_units == 63
    latest_unit = slow_websocket.get(timeout=0)
    assert latest_unit is not None
    assert latest_unit.frame_sequence == 64
    validate_display_unit(latest_unit)
    slow_websocket.close()
    display_broker.close()

    limit_broker = ProcessedDisplayBroker()
    subscribers = [limit_broker.subscribe() for _ in range(MAX_DISPLAY_SUBSCRIBERS)]
    with pytest.raises(DisplayProtocolError, match="subscriber limit"):
        limit_broker.subscribe()
    for subscriber in subscribers:
        subscriber.close()
    limit_broker.close()


@pytest.mark.m3_acceptance
def test_m3_one_hour_equivalent_replay_reports_metrics_and_stable_memory(
    record_property: Any,
) -> None:
    fixture = _fixture()
    clock = ReplayClock()
    start_gate = threading.Event()
    checkpoints = ReplayGate()
    source = GeneratedReplaySource(
        camera_id="camera-1",
        total_frames=fixture.one_hour_frames,
        frame_rate=fixture.frame_rate,
        frame_factory=lambda index: fixture.frame(
            "plate" if index // fixture.checkpoint_frames % 2 == 0 else "no_plate",
            index,
        ),
        clock=clock,
        session_id="one-hour-session",
        start_gate=start_gate,
        checkpoint_gate=checkpoints,
        checkpoint_frames=fixture.checkpoint_frames,
        logical_seconds_per_frame=1,
    )
    runtime = _runtime_for_source(source, clock=clock)
    backend = NonRetainingFakeBackend(output_factory=_outputs_for_replay)
    coordinator = LivePipelineCoordinator(
        runtime,
        lambda: backend,
        clock=clock,
        poll_interval_seconds=0.001,
        display_max_fps=60,
        epoch_factory=lambda: "one-hour-epoch",
    )
    observer: ProcessedDisplaySubscription | None = None
    slow_websocket: ProcessedDisplaySubscription | None = None
    memory_samples: list[int] = []
    seen_detection_states: set[bool] = set()
    tracemalloc.start()
    try:
        coordinator.start("camera-1", _camera(), _descriptor())
        coordinator.wait_for_state("running")
        observer = coordinator.subscribe_display()
        slow_websocket = coordinator.subscribe_display()
        assert observer is not None
        assert slow_websocket is not None
        start_gate.set()

        for checkpoint in range(1, fixture.checkpoint_count + 1):
            checkpoints.wait_until_reached(checkpoint)
            gc.collect()
            memory_samples.append(tracemalloc.get_traced_memory()[0])
            unit = observer.get(timeout=5)
            assert unit is not None
            validate_display_unit(unit)
            seen_detection_states.add(bool(json.loads(unit.metadata_text)["detections"]))
            checkpoints.release(checkpoint)

        assert source.exhausted.wait(5)
        final_status = coordinator.status()
        metrics = final_status.metrics
        assert seen_detection_states == {False, True}
        assert metrics.processed_frames >= fixture.checkpoint_count
        assert metrics.processed_fps > 0
        assert metrics.prediction_p50_ms is not None
        assert metrics.prediction_p95_ms is not None
        assert slow_websocket.metrics().replaced_units > 0
        assert slow_websocket.metrics().replaced_units <= metrics.processed_frames
        _assert_stable_memory(
            memory_samples,
            tolerance_bytes=fixture.memory_tolerance_bytes,
            sustained_checkpoints=fixture.sustained_growth_checkpoints,
        )
        evidence = {
            "fixture_id": fixture.fixture_id,
            "logical_duration_seconds": fixture.one_hour_frames,
            "processed_frames": metrics.processed_frames,
            "processed_fps": metrics.processed_fps,
            "prediction_p50_ms": metrics.prediction_p50_ms,
            "prediction_p95_ms": metrics.prediction_p95_ms,
            "source_replacements": metrics.source_replacement_count,
            "inference_replacements": metrics.inference_replacement_count,
            "display_replacements": metrics.display_replacement_count,
            "websocket_replacements": slow_websocket.metrics().replaced_units,
            "memory_current_bytes": memory_samples,
            "memory_tolerance_bytes": fixture.memory_tolerance_bytes,
        }
        print(f"M3 replay evidence: {json.dumps(evidence, sort_keys=True)}")
        for name, value in evidence.items():
            if isinstance(value, (int, float)):
                record_property(name, value)
    finally:
        start_gate.set()
        for checkpoint in range(1, fixture.checkpoint_count + 1):
            checkpoints.release(checkpoint)
        if observer is not None:
            observer.close()
        if slow_websocket is not None:
            slow_websocket.close()
        coordinator.close()
        tracemalloc.stop()


@pytest.mark.m3_acceptance
def test_m3_stop_restart_releases_and_reacquires_all_live_resources() -> None:
    source_factory = ReconnectFixture(
        (
            FixtureAttempt(
                frames=(_fixture().frame("plate"),),
                repeat=True,
            ),
            FixtureAttempt(
                frames=(_fixture().frame("no_plate"),),
                repeat=True,
            ),
        )
    )
    backends: list[NonRetainingFakeBackend] = []

    def backend_factory() -> NonRetainingFakeBackend:
        backend = NonRetainingFakeBackend(output_factory=_outputs_for_replay)
        backends.append(backend)
        return backend

    coordinator = LivePipelineCoordinator(
        _runtime_for_source(source_factory, clock=ReplayClock()),
        backend_factory,
        poll_interval_seconds=0.001,
        display_max_fps=60,
        epoch_factory=iter(("restart-epoch-1", "restart-epoch-2")).__next__,
    )
    first_subscription: ProcessedDisplaySubscription | None = None
    second_subscription: ProcessedDisplaySubscription | None = None
    try:
        coordinator.start("camera-1", _camera(), _descriptor())
        coordinator.wait_for_state("running")
        first_subscription = coordinator.subscribe_display()
        assert first_subscription is not None
        first_unit = first_subscription.get(timeout=5)
        assert first_unit is not None
        first_capture_session = first_unit.capture_session_id
        first_generation = first_unit.generation_number
        first_epoch = first_unit.stream_epoch

        stopped = coordinator.stop()
        assert stopped.state == "stopped"
        assert first_subscription.closed
        assert source_factory.sources[0].closed.is_set()
        assert backends[0].closes and all(model.closed for model in backends[0].closes)

        coordinator.start("camera-1", _camera(), _descriptor())
        coordinator.wait_for_state("running")
        second_subscription = coordinator.subscribe_display()
        assert second_subscription is not None
        second_unit = second_subscription.get(timeout=5)
        assert second_unit is not None
        assert second_unit.generation_number > first_generation
        assert second_unit.stream_epoch != first_epoch
        assert second_unit.capture_session_id != first_capture_session
        assert json.loads(second_unit.metadata_text)["frame_sequence"] == second_unit.frame_sequence
        assert second_unit.frame_sequence >= 1
        assert first_subscription.get(timeout=0) is None
    finally:
        if first_subscription is not None:
            first_subscription.close()
        if second_subscription is not None:
            second_subscription.close()
        coordinator.close()
    assert len(source_factory.sources) >= 2
    assert source_factory.sources[1].closed.is_set()
    assert len(backends) >= 2
    assert all(backend.loads and backend.closes for backend in backends)
    assert all(model.closed for backend in backends for model in backend.closes)


@pytest.mark.m3_acceptance
def test_m3_processed_display_releases_frame_references_on_stop() -> None:
    service = ProcessedDisplayService(max_fps=60, encoder=_jpeg)
    service.start_generation(1)
    service.set_provenance(1, "epoch-1", "session-1")
    subscription = service.subscribe()
    assert subscription is not None
    candidate = _candidate(1)
    frame_reference = weakref.ref(candidate.frame.data)
    assert service.submit(candidate)
    unit = subscription.get(timeout=5)
    assert unit is not None
    service.stop_generation(reason="stopped")
    subscription.close()
    del unit
    del candidate
    gc.collect()
    assert frame_reference() is None


def _assert_stable_memory(
    samples: list[int],
    *,
    tolerance_bytes: int,
    sustained_checkpoints: int,
) -> None:
    assert len(samples) >= sustained_checkpoints + 2
    baseline = max(samples[:2])
    above_tolerance = 0
    for sample in samples[2:]:
        if sample > baseline + tolerance_bytes:
            above_tolerance += 1
        else:
            above_tolerance = 0
        if above_tolerance >= sustained_checkpoints:
            raise AssertionError(
                "traced memory stayed above the documented tolerance for "
                f"{sustained_checkpoints} checkpoints: {samples}"
            )


def _get_unit_at_least(
    subscription: ProcessedDisplaySubscription,
    minimum_sequence: int,
) -> Any:
    """Drain older complete units until the current replay frame arrives."""
    while True:
        unit = subscription.get(timeout=5)
        assert unit is not None
        if unit.frame_sequence >= minimum_sequence:
            return unit


def _get_unit_with_epoch(
    subscription: ProcessedDisplaySubscription,
    epoch: str,
) -> Any:
    """Drain complete units until the requested reconnect epoch arrives."""
    while True:
        unit = subscription.get(timeout=5)
        assert unit is not None
        if unit.stream_epoch == epoch:
            return unit
        assert unit.stream_epoch == "reconnect-epoch-1"
