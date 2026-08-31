from __future__ import annotations

import threading
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from open_licenseplate.cameras.service import prepare_camera_config
from open_licenseplate.capture import (
    ActiveCameraConflict,
    CameraRuntime,
    CaptureSession,
    FrameSource,
    ReconnectBackoff,
    ReconnectStateMachine,
    SourceInfo,
    SourceOpenError,
    SourceReadError,
    VideoFrame,
    WaitScheduler,
)


class FakeClock:
    def __init__(self) -> None:
        self.wall = datetime(2026, 8, 29, 20, 0, tzinfo=UTC)
        self.ticks = 0.0

    def now(self) -> datetime:
        return self.wall

    def monotonic(self) -> float:
        return self.ticks

    def advance(self, seconds: float) -> None:
        self.ticks += seconds
        self.wall += timedelta(seconds=seconds)


class ManualScheduler(WaitScheduler):
    """Advance waits only when the fake clock advances."""

    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock
        self.condition = threading.Condition()
        self.waiter_count = 0
        self.pending_advances = 0
        self.tick = 0
        self.completed_waits = 0

    def wait(self, stop_requested: threading.Event, timeout: float) -> bool:
        deadline = self.clock.monotonic() + max(0.0, timeout)
        with self.condition:
            if self.pending_advances:
                self.pending_advances -= 1
                self.completed_waits += 1
                self.condition.notify_all()
                return stop_requested.is_set()
            starting_tick = self.tick
            self.waiter_count += 1
            self.condition.notify_all()
            try:
                while (
                    not stop_requested.is_set()
                    and self.clock.monotonic() < deadline
                    and self.tick == starting_tick
                ):
                    self.condition.wait()
                return stop_requested.is_set()
            finally:
                self.waiter_count -= 1
                self.completed_waits += 1
                self.condition.notify_all()

    def wake(self) -> None:
        with self.condition:
            self.condition.notify_all()

    def advance(self, seconds: float) -> None:
        with self.condition:
            completed_before = self.completed_waits
            self.clock.advance(seconds)
            self.tick += 1
            if self.waiter_count == 0:
                self.pending_advances += 1
            self.condition.notify_all()
            self.condition.wait_for(
                lambda: self.completed_waits > completed_before,
                timeout=1.0,
            )

    def wait_until_waiting(self, *, timeout: float = 1.0) -> None:
        with self.condition:
            if not self.condition.wait_for(lambda: self.waiter_count > 0, timeout):
                raise AssertionError("runtime did not enter an injected wait")


class ControlledSource(FrameSource):
    def __init__(
        self,
        clock: FakeClock,
        *,
        disconnect_after_first_frame: bool = False,
        disconnect_gate: threading.Event | None = None,
        open_error: str | None = None,
    ) -> None:
        self.clock = clock
        self.disconnect_after_first_frame = disconnect_after_first_frame
        self.disconnect_gate = disconnect_gate
        self.open_error = open_error
        self.opened = threading.Event()
        self.closed = threading.Event()
        self.release_read = threading.Event()
        self.read_count = 0
        self.info: SourceInfo | None = None

    def open(self) -> SourceInfo:
        self.opened.set()
        if self.open_error is not None:
            raise SourceOpenError(self.open_error)
        session = CaptureSession(
            id=f"session-{id(self)}",
            camera_id="camera-1",
            started_at=self.clock.now(),
            started_monotonic=self.clock.monotonic(),
        )
        self.info = SourceInfo(
            source_name="controlled-fixture",
            session=session,
            codec="h264",
            width=8,
            height=6,
            nominal_fps=5.0,
            has_camera_pts=True,
            transport="tcp",
        )
        return self.info

    def read(self) -> VideoFrame | None:
        if self.closed.is_set():
            return None
        if self.read_count == 0:
            self.read_count += 1
            return VideoFrame(
                sequence=1,
                data=np.zeros((6, 8, 3), dtype=np.uint8),
                pixel_format="bgr24",
                host_received_at=self.clock.now(),
                host_received_monotonic=self.clock.monotonic(),
                capture_session_id=self.info.capture_session_id if self.info else "unknown",
                width=8,
                height=6,
                camera_pts=1,
                camera_pts_seconds=0.2,
            )
        if self.disconnect_after_first_frame:
            if self.disconnect_gate is not None:
                self.disconnect_gate.wait()
            raise SourceReadError("controlled fixture disconnected")
        self.release_read.wait()
        return None if self.closed.is_set() else self.read()

    def close(self) -> None:
        self.closed.set()
        self.release_read.set()


class ControlledFactory:
    def __init__(self, sources: list[ControlledSource]) -> None:
        self.sources = sources
        self.created: list[ControlledSource] = []

    def __call__(self, _camera, _camera_id: str) -> ControlledSource:
        source = self.sources[len(self.created)]
        self.created.append(source)
        return source


def _camera_config():
    return prepare_camera_config(name="Fixture", rtsp_url="rtsp://fixture.local/live")


def _advance_until_state(
    runtime: CameraRuntime,
    scheduler: ManualScheduler,
    state: str,
    *,
    seconds: float,
    start_index: int = 0,
    steps: int = 20,
) -> None:
    for _ in range(steps):
        if state in runtime.state_history[start_index:]:
            return
        scheduler.wait_until_waiting()
        scheduler.advance(seconds)
    runtime.wait_for_state(state, after_history_index=start_index)


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


def test_state_machine_has_full_reconnect_sequence_and_resets_after_stable_stream() -> None:
    clock = FakeClock()
    machine = ReconnectStateMachine(
        clock=clock,
        backoff=ReconnectBackoff(
            base_delay_seconds=2.0,
            cap_seconds=8.0,
            jitter_ratio=0,
            random_value=lambda: 0.5,
        ),
        stable_stream_seconds=5.0,
    )

    machine.start()
    machine.opened()
    machine.disconnected("stream ended")
    assert machine.state == "degraded"
    assert machine.schedule_reconnect() == 2.0
    assert machine.state == "reconnecting"
    clock.advance(2.0)
    machine.opened()
    machine.reconnect_attempt = 1
    clock.advance(5.0)
    assert machine.poll_stability() is True
    assert machine.reconnect_attempt == 0
    machine.disconnected("stream ended again")
    assert machine.schedule_reconnect() == 2.0
    machine.stopping()
    machine.stopped()

    assert machine.history == (
        "stopped",
        "connecting",
        "streaming",
        "degraded",
        "reconnecting",
        "streaming",
        "degraded",
        "reconnecting",
        "stopping",
        "stopped",
    )


def test_initial_open_error_is_failed_and_does_not_reconnect() -> None:
    clock = FakeClock()
    machine = ReconnectStateMachine(
        clock=clock,
        backoff=ReconnectBackoff(jitter_ratio=0),
        stable_stream_seconds=5.0,
    )
    machine.start()
    machine.initial_open_failed("could not open rtsp://user:secret@example.test/live")

    assert machine.state == "failed"
    assert machine.reconnect_attempt == 0
    assert machine.next_retry_at is None
    assert "secret" not in (machine.last_error or "")
    assert "[REDACTED]@" in (machine.last_error or "")
    assert machine.history == ("stopped", "connecting", "failed")


def test_runtime_uses_fake_clock_scheduler_for_reconnect() -> None:
    clock = FakeClock()
    scheduler = ManualScheduler(clock)
    disconnect_gate = threading.Event()
    factory = ControlledFactory(
        [
            ControlledSource(
                clock,
                disconnect_after_first_frame=True,
                disconnect_gate=disconnect_gate,
            ),
            ControlledSource(clock),
        ]
    )
    runtime = CameraRuntime(
        factory,
        clock=clock,
        scheduler=scheduler,
        backoff=ReconnectBackoff(
            base_delay_seconds=2.0,
            cap_seconds=8.0,
            jitter_ratio=0,
            random_value=lambda: 0.5,
        ),
        stable_stream_seconds=5.0,
        poll_interval_seconds=0.25,
        degraded_hold_seconds=0.5,
    )

    runtime.start("camera-1", _camera_config())
    try:
        assert factory.sources[0].opened.wait(1)
        _advance_until_state(runtime, scheduler, "streaming", seconds=0.25)
        disconnect_gate.set()
        history_index = len(runtime.state_history)
        _advance_until_state(
            runtime,
            scheduler,
            "degraded",
            seconds=0.25,
            start_index=history_index,
        )
        scheduler.wait_until_waiting()
        history_index = len(runtime.state_history)
        _advance_until_state(
            runtime,
            scheduler,
            "reconnecting",
            seconds=0.5,
            start_index=history_index,
        )
        scheduler.wait_until_waiting()
        scheduler.advance(2.0)
        assert factory.sources[1].opened.wait(1)
        history_index = len(runtime.state_history)
        _advance_until_state(
            runtime,
            scheduler,
            "streaming",
            seconds=0.25,
            start_index=history_index,
        )

        assert runtime.state_history == (
            "stopped",
            "connecting",
            "streaming",
            "degraded",
            "reconnecting",
            "streaming",
        )
        assert runtime.status().reconnect_attempt == 1
    finally:
        runtime.stop("camera-1")

    assert runtime.state_history[-2:] == ("stopping", "stopped")
    assert all(source.closed.is_set() for source in factory.created)


def test_stop_cancels_fake_clock_reconnect_wait_without_advancing_clock() -> None:
    clock = FakeClock()
    scheduler = ManualScheduler(clock)
    disconnect_gate = threading.Event()
    factory = ControlledFactory(
        [
            ControlledSource(
                clock,
                disconnect_after_first_frame=True,
                disconnect_gate=disconnect_gate,
            )
        ]
    )
    runtime = CameraRuntime(
        factory,
        clock=clock,
        scheduler=scheduler,
        backoff=ReconnectBackoff(
            base_delay_seconds=100.0,
            cap_seconds=100.0,
            jitter_ratio=0,
            random_value=lambda: 0.5,
        ),
        poll_interval_seconds=0.25,
        degraded_hold_seconds=0.5,
    )

    runtime.start("camera-1", _camera_config())
    assert factory.sources[0].opened.wait(1)
    _advance_until_state(runtime, scheduler, "streaming", seconds=0.25)
    disconnect_gate.set()
    history_index = len(runtime.state_history)
    _advance_until_state(
        runtime,
        scheduler,
        "degraded",
        seconds=0.25,
        start_index=history_index,
    )
    history_index = len(runtime.state_history)
    _advance_until_state(
        runtime,
        scheduler,
        "reconnecting",
        seconds=0.5,
        start_index=history_index,
    )
    scheduler.wait_until_waiting()
    before_stop = clock.monotonic()

    runtime.stop("camera-1")

    assert clock.monotonic() == before_stop
    assert runtime.state_history[-2:] == ("stopping", "stopped")
    assert factory.created[0].closed.is_set()


def test_runtime_initial_open_error_reports_failed_without_reconnect() -> None:
    clock = FakeClock()
    scheduler = ManualScheduler(clock)
    factory = ControlledFactory(
        [ControlledSource(clock, open_error="camera source is unavailable")]
    )
    runtime = CameraRuntime(
        factory,
        clock=clock,
        scheduler=scheduler,
        backoff=ReconnectBackoff(jitter_ratio=0),
        poll_interval_seconds=0.25,
    )

    runtime.start("camera-1", _camera_config())
    assert factory.sources[0].opened.wait(1)
    for _ in range(10):
        if "failed" in runtime.state_history:
            break
        scheduler.advance(0.25)
    failed = runtime.wait_for_state("failed")

    assert failed.state == "failed"
    assert failed.reconnect_attempt == 0
    assert failed.next_retry_in_seconds is None
    assert failed.last_error == "camera source is unavailable"
    assert runtime.state_history == ("stopped", "connecting", "failed")
    assert factory.created[0].closed.is_set()

    runtime.stop("camera-1")
    assert runtime.state_history[-2:] == ("stopping", "stopped")


def test_runtime_rejects_a_second_active_camera_with_actionable_conflict() -> None:
    clock = FakeClock()
    scheduler = ManualScheduler(clock)
    factory = ControlledFactory([ControlledSource(clock)])
    runtime = CameraRuntime(
        factory,
        clock=clock,
        scheduler=scheduler,
        poll_interval_seconds=0.25,
    )
    runtime.start("camera-1", _camera_config())
    try:
        scheduler.wait_until_waiting()
        _advance_until_state(runtime, scheduler, "streaming", seconds=0.25)
        with pytest.raises(ActiveCameraConflict, match="stop it before starting camera camera-2"):
            runtime.start("camera-2", _camera_config())
    finally:
        runtime.stop("camera-1")
