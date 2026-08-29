from __future__ import annotations

import time

import pytest

from open_licenseplate.cameras.service import prepare_camera_config
from open_licenseplate.capture import (
    ActiveCameraConflict,
    CameraRuntime,
    FixtureAttempt,
    ReconnectBackoff,
    ReconnectFixture,
    disconnect_then_recover_fixture,
    make_preview_frame,
)


def _camera_config():
    return prepare_camera_config(name="Fixture", rtsp_url="rtsp://fixture.local/live")


def _wait_for_state(runtime: CameraRuntime, state: str, timeout: float = 1.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if runtime.status().state == state:
            return
        time.sleep(0.01)
    raise AssertionError(f"runtime did not reach {state}: {runtime.status().as_dict()}")


def test_reconnect_backoff_is_bounded_and_has_deterministic_jitter() -> None:
    policy = ReconnectBackoff(
        base_delay_seconds=1.0,
        cap_seconds=4.0,
        jitter_ratio=0.25,
        random_value=lambda: 1.0,
    )

    assert policy.delay_for(1) == 1.25
    assert policy.delay_for(2) == 2.5
    assert policy.delay_for(3) == 4.0
    assert policy.delay_for(4) == 4.0

    with pytest.raises(ValueError, match="attempt"):
        policy.delay_for(0)


def test_runtime_reconnects_after_disconnect_and_reports_latest_frame_metrics() -> None:
    fixture = disconnect_then_recover_fixture()
    runtime = CameraRuntime(
        fixture,
        backoff=ReconnectBackoff(
            base_delay_seconds=0.01,
            cap_seconds=0.02,
            jitter_ratio=0,
            random_value=lambda: 0.5,
        ),
        stable_stream_seconds=0.05,
        poll_interval_seconds=0.005,
    )

    runtime.start("camera-1", _camera_config())
    try:
        _wait_for_state(runtime, "streaming")
        deadline = time.monotonic() + 1
        while (
            fixture.created_attempts < 2
            or runtime.status().state != "streaming"
            or runtime.status().reconnect_attempt != 0
        ) and time.monotonic() < deadline:
            time.sleep(0.01)
        status = runtime.status("camera-1")
        assert fixture.created_attempts >= 2
        assert status.state == "streaming"
        assert status.reconnect_attempt == 0
        assert status.metrics["reconnect_count"] >= 1
        assert status.metrics["captured_frames"] >= 2
        assert status.metrics["replaced_frames"] >= 1
        assert status.source_info is not None
    finally:
        runtime.stop("camera-1")

    assert runtime.status("camera-1").state == "stopped"
    assert all(source.closed.is_set() for source in fixture.sources)


def test_stop_cancels_a_long_reconnect_wait() -> None:
    fixture = ReconnectFixture((FixtureAttempt(open_error="fixture is disconnected"),))
    runtime = CameraRuntime(
        fixture,
        backoff=ReconnectBackoff(
            base_delay_seconds=30.0,
            cap_seconds=30.0,
            jitter_ratio=0,
            random_value=lambda: 0.5,
        ),
        poll_interval_seconds=0.005,
    )
    runtime.start("camera-1", _camera_config())
    _wait_for_state(runtime, "reconnecting")

    started = time.monotonic()
    runtime.stop("camera-1", timeout=1)

    assert time.monotonic() - started < 1
    assert runtime.status("camera-1").state == "stopped"
    assert fixture.sources[0].closed.is_set()


def test_runtime_rejects_a_second_active_camera_with_actionable_conflict() -> None:
    fixture = ReconnectFixture(
        (
            FixtureAttempt(
                frames=(make_preview_frame(12),),
                repeat=True,
                read_interval_seconds=0.01,
            ),
        )
    )
    runtime = CameraRuntime(fixture, poll_interval_seconds=0.005)
    runtime.start("camera-1", _camera_config())
    try:
        _wait_for_state(runtime, "streaming")
        with pytest.raises(ActiveCameraConflict, match="stop it before starting camera camera-2"):
            runtime.start("camera-2", _camera_config())
    finally:
        runtime.stop("camera-1")
