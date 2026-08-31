from __future__ import annotations

import json
import threading
import time
from datetime import UTC, datetime
from fractions import Fraction
from typing import Any

import numpy as np
import pytest

from open_licenseplate.cameras.service import prepare_camera_config
from open_licenseplate.capture import (
    FakeFrameSource,
    FrameCaptureWorker,
    LatestFrameBroker,
    PyAVRTSPSource,
    RecordedFrame,
    RecordedVideoSource,
    SourceLifecycleError,
    SourceOpenError,
    SourceReadError,
    VideoFrame,
)


class FixedClock:
    def __init__(self) -> None:
        self.wall = datetime(2026, 8, 29, 19, 0, tzinfo=UTC)
        self.ticks = 100.0

    def now(self) -> datetime:
        return self.wall

    def monotonic(self) -> float:
        return self.ticks


def _frame(
    sequence: int,
    *,
    session_id: str = "session-1",
    received_monotonic: float | None = None,
) -> VideoFrame:
    return VideoFrame(
        sequence=sequence,
        data=np.zeros((2, 3, 3), dtype=np.uint8),
        pixel_format="bgr24",
        host_received_at=datetime(2026, 8, 29, 19, 0, tzinfo=UTC),
        host_received_monotonic=(
            float(sequence) if received_monotonic is None else received_monotonic
        ),
        capture_session_id=session_id,
        width=3,
        height=2,
        camera_pts=sequence * 100,
        camera_pts_seconds=sequence / 10,
    )


def test_latest_frame_broker_replaces_unread_frame_and_reports_age() -> None:
    broker = LatestFrameBroker()

    assert broker.put(_frame(1, received_monotonic=10.0))
    assert broker.put(_frame(2, received_monotonic=12.0))

    metrics = broker.metrics(now_monotonic=15.5)
    assert metrics.captured_frames == 2
    assert metrics.consumed_frames == 0
    assert metrics.replaced_frames == 1
    assert metrics.newest_frame_age_seconds == 3.5
    assert metrics.has_frame is True
    assert broker.get(timeout=0) is not None
    assert broker.get(timeout=0) is None


def test_latest_frame_broker_close_releases_frame_and_wakes_waiter() -> None:
    broker = LatestFrameBroker()
    result: list[VideoFrame | None] = []

    waiter = threading.Thread(target=lambda: result.append(broker.get(timeout=2)))
    waiter.start()
    time.sleep(0.02)
    broker.close()
    waiter.join(1)

    assert not waiter.is_alive()
    assert result == [None]
    assert broker.peek() is None
    assert broker.metrics().closed is True
    assert broker.put(_frame(2)) is False


def test_recorded_source_assigns_session_identity_and_separate_timestamps() -> None:
    clock = FixedClock()
    source = RecordedVideoSource(
        [
            RecordedFrame(
                np.ones((4, 5, 3), dtype=np.uint8),
                camera_pts=77,
                camera_pts_seconds=2.5,
            )
        ],
        camera_id="camera-1",
        clock=clock,
        session_id_factory=lambda: "session-77",
    )

    info = source.open()
    frame = source.read()

    assert info.capture_session_id == "session-77"
    assert info.session.camera_id == "camera-1"
    assert info.started_at == clock.wall
    assert frame is not None
    assert frame.capture_session_id == "session-77"
    assert frame.sequence == 1
    assert frame.host_received_at == clock.wall
    assert frame.host_received_monotonic == 100.0
    assert frame.camera_pts == 77
    assert frame.camera_pts_seconds == 2.5
    source.close()


def test_source_lifecycle_rejects_read_before_open_and_second_open() -> None:
    source = FakeFrameSource([np.zeros((2, 2, 3), dtype=np.uint8)])

    with pytest.raises(SourceLifecycleError, match="not open"):
        source.read()
    source.open()
    with pytest.raises(SourceLifecycleError, match="already open"):
        source.open()
    source.close()
    with pytest.raises(SourceLifecycleError, match="not open"):
        source.read()


def test_fake_source_reports_decode_failure_without_secret_output() -> None:
    source = FakeFrameSource(
        [np.zeros((2, 2, 3), dtype=np.uint8)],
        read_error="decode failed for rtsp://user:password@example.test/live",
    )
    source.open()

    with pytest.raises(SourceReadError) as exception:
        source.read()

    assert "password" not in str(exception.value)
    assert "[REDACTED]@" in str(exception.value)


class _FakeDecodedFrame:
    pts = 42
    time = 4.2

    def __init__(self, data: np.ndarray) -> None:
        self.data = data

    def to_ndarray(self, *, format: str) -> np.ndarray:
        assert format == "bgr24"
        return self.data


class _FakeCodecContext:
    name = "h264"


class _FakeStream:
    index = 2
    width = 1920
    height = 1080
    average_rate = Fraction(30, 1)
    time_base = Fraction(1, 90000)
    metadata = {"title": "main"}
    codec_context = _FakeCodecContext()


class _FakeStreams:
    video = [_FakeStream()]


class _FakeContainer:
    def __init__(self, decoded: list[_FakeDecodedFrame]) -> None:
        self.streams = _FakeStreams()
        self.decoded = decoded
        self.decode_arguments: dict[str, Any] | None = None
        self.closed = False

    def decode(self, **kwargs: Any) -> list[_FakeDecodedFrame]:
        self.decode_arguments = kwargs
        return self.decoded

    def close(self) -> None:
        self.closed = True


class _FakeAV:
    def __init__(self, container: _FakeContainer) -> None:
        self.container = container
        self.open_arguments: dict[str, Any] | None = None

    def open(self, endpoint: str, **kwargs: Any) -> _FakeContainer:
        self.open_arguments = {"endpoint": endpoint, **kwargs}
        return self.container


def _camera_with_env_ref() -> Any:
    return prepare_camera_config(
        name="Front gate",
        rtsp_url="rtsp://example.test:554/live",
        credential_ref="env:CAMERA_RTSP_URL",
        connection_options={"open_timeout": 1.5, "read_timeout": 2.5},
    )


def test_pyav_source_uses_tcp_timeouts_video_only_and_redacts_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret_url = "rtsp://operator:secret-value@example.test:554/live"
    monkeypatch.setenv("CAMERA_RTSP_URL", secret_url)
    container = _FakeContainer([_FakeDecodedFrame(np.ones((6, 8, 3), dtype=np.uint8))])
    av_module = _FakeAV(container)
    source = PyAVRTSPSource(
        _camera_with_env_ref(),
        camera_id="camera-1",
        av_module=av_module,
        session_id_factory=lambda: "session-1",
    )

    info = source.open()
    frame = source.read()

    assert av_module.open_arguments is not None
    assert av_module.open_arguments["endpoint"] == secret_url
    assert av_module.open_arguments["options"] == {
        "rtsp_transport": "tcp",
        "rw_timeout": "2500000",
    }
    assert av_module.open_arguments["timeout"] == (1.5, 2.5)
    assert container.decode_arguments == {"video": 2}
    assert info.codec == "h264"
    assert info.width == 1920
    assert info.height == 1080
    assert info.nominal_fps == 30.0
    assert info.has_camera_pts is True
    assert frame is not None
    assert frame.camera_pts == 42
    assert frame.camera_pts_seconds == 4.2
    assert "secret-value" not in json.dumps(
        info.__dict__ if hasattr(info, "__dict__") else str(info)
    )
    source.close()
    assert container.closed is True


def test_pyav_source_redacts_open_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    secret_url = "rtsp://operator:secret-value@example.test/live"
    monkeypatch.setenv("CAMERA_RTSP_URL", secret_url)

    class FailingAV:
        def open(self, endpoint: str, **kwargs: Any) -> None:
            del kwargs
            raise RuntimeError(f"failed to connect to {endpoint}")

    source = PyAVRTSPSource(_camera_with_env_ref(), av_module=FailingAV())

    with pytest.raises(SourceOpenError) as exception:
        source.open()

    assert "secret-value" not in str(exception.value)
    assert "operator" not in str(exception.value)
    assert "[REDACTED]@" in str(exception.value)


def test_capture_worker_stops_a_blocked_source_promptly() -> None:
    gate = threading.Event()
    source = FakeFrameSource(
        [np.zeros((2, 2, 3), dtype=np.uint8)],
        read_gate=gate,
    )
    broker = LatestFrameBroker()
    worker = FrameCaptureWorker(source, broker)

    worker.start()
    assert source.opened.wait(1)
    started = time.monotonic()
    worker.stop(timeout=1)
    elapsed = time.monotonic() - started

    assert elapsed < 1
    assert source.closed.is_set()
    assert worker.capture_session is not None
    assert worker.capture_session.end_reason == "stopped"
    assert broker.closed is True


def test_capture_worker_records_source_failure_and_redacts_error() -> None:
    source = FakeFrameSource(
        [np.zeros((2, 2, 3), dtype=np.uint8)],
        fail_at=0,
        read_error="failure at rtsp://user:secret@example.test/live",
    )
    broker = LatestFrameBroker()
    worker = FrameCaptureWorker(source, broker)
    worker.start()

    deadline = time.monotonic() + 1
    while worker.metrics().running and time.monotonic() < deadline:
        time.sleep(0.01)

    metrics = worker.metrics()
    assert metrics.failed is True
    assert metrics.decode_errors == 1
    assert metrics.error is not None
    assert "secret" not in metrics.error
    assert worker.capture_session is not None
    assert worker.capture_session.end_reason == "failed"
