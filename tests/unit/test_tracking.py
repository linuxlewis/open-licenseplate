from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pytest

from open_licenseplate.inference import Detection
from open_licenseplate.tracking import (
    LiveDetectionFrame,
    TrackedDetection,
    TrackerAdapter,
    TrackingConfig,
    TrackingEventAggregator,
    TrackingProvenance,
)


@dataclass
class FakeClock:
    wall: datetime = datetime(2026, 8, 30, tzinfo=UTC)
    ticks: float = 0.0

    def now(self) -> datetime:
        return self.wall

    def monotonic(self) -> float:
        return self.ticks

    def advance(self, seconds: float) -> None:
        self.ticks += seconds
        self.wall += timedelta(seconds=seconds)


class FixedTrackAdapter:
    """Deterministic tracker double that keeps one ID per input detection."""

    def __init__(self) -> None:
        self.reset_count = 0

    def update(
        self,
        detections: Sequence[Detection],
        *,
        frame_width: int,
        frame_height: int,
    ) -> tuple[TrackedDetection, ...]:
        assert frame_width == 64
        assert frame_height == 48
        return tuple(
            TrackedDetection(track_id=index + 1, detection=detection)
            for index, detection in enumerate(detections)
        )

    def reset(self) -> None:
        self.reset_count += 1


class OneTrackAdapter(FixedTrackAdapter):
    """Deterministic tracker double that maps every detection to track seven."""

    def update(
        self,
        detections: Sequence[Detection],
        *,
        frame_width: int,
        frame_height: int,
    ) -> tuple[TrackedDetection, ...]:
        assert frame_width == 64
        assert frame_height == 48
        return tuple(TrackedDetection(track_id=7, detection=detection) for detection in detections)


def _provenance(
    *,
    session: str = "session-1",
    generation: int = 1,
    epoch: str = "epoch-1",
) -> TrackingProvenance:
    return TrackingProvenance(
        camera_id="camera-1",
        capture_session_id=session,
        generation_number=generation,
        stream_epoch=epoch,
        model_id="model-1",
        model_checksum="a" * 64,
    )


def _frame(
    clock: FakeClock,
    sequence: int,
    *,
    session: str = "session-1",
    generation: int = 1,
    epoch: str = "epoch-1",
    confidence: float = 0.8,
    detections: bool = True,
) -> LiveDetectionFrame:
    captured_at = clock.now()
    values = (
        (
            Detection(
                box_xyxy=(8.0, 8.0, 24.0, 18.0),
                class_id=0,
                label="license_plate",
                confidence=confidence,
                model_id="model-1",
                model_sha256="a" * 64,
                frame_sequence=sequence,
                detected_at=captured_at,
            ),
        )
        if detections
        else ()
    )
    return LiveDetectionFrame(
        provenance=_provenance(session=session, generation=generation, epoch=epoch),
        frame_sequence=sequence,
        captured_at=captured_at,
        frame_width=64,
        frame_height=48,
        detections=values,
    )


def _aggregator(
    clock: FakeClock,
    *,
    adapter: TrackerAdapter | None = None,
    on_closed_event: list | None = None,
) -> TrackingEventAggregator:
    sink = None if on_closed_event is None else on_closed_event.append
    return TrackingEventAggregator(
        lambda: adapter or OneTrackAdapter(),
        clock=clock,
        config=TrackingConfig(
            confirmation_window_seconds=0.75,
            close_timeout_seconds=1.0,
            max_active_tracks=4,
        ),
        on_closed_event=sink,
    )


@pytest.mark.integration
def test_three_matched_observations_confirm_and_timeout_to_one_closed_event() -> None:
    clock = FakeClock()
    events: list = []
    aggregator = _aggregator(clock, on_closed_event=events)

    for sequence in range(1, 4):
        update = aggregator.consume(_frame(clock, sequence, confidence=0.7 + sequence / 100))
        if sequence < 3:
            assert update.active_tracks == ()
        clock.advance(0.1)

    assert len(update.active_tracks) == 1
    active = update.active_tracks[0]
    assert active.state == "active"
    assert active.observation_count == 3
    assert active.maximum_confidence == pytest.approx(0.73)

    clock.advance(1.0)
    closed = aggregator.tick()
    assert len(closed.closed_events) == 1
    event = closed.closed_events[0]
    assert event.event_state == "closed"
    assert event.capture_session_id == "session-1"
    assert event.track_id == 7
    assert event.observation_count == 3
    assert event.maximum_confidence == pytest.approx(0.73)
    assert event.duration_seconds == pytest.approx(0.2)
    assert events == [event]
    assert aggregator.tick().closed_events == ()


@pytest.mark.integration
def test_no_plate_and_short_candidate_do_not_emit_closed_events() -> None:
    clock = FakeClock()
    aggregator = _aggregator(clock)

    aggregator.consume(_frame(clock, 1, detections=False))
    clock.advance(2.0)
    assert aggregator.tick().closed_events == ()

    clock = FakeClock()
    aggregator = _aggregator(clock)
    aggregator.consume(_frame(clock, 1))
    clock.advance(0.2)
    aggregator.consume(_frame(clock, 2))
    clock.advance(1.0)
    update = aggregator.tick()
    assert update.active_tracks == ()
    assert update.closed_events == ()


@pytest.mark.integration
def test_provenance_boundary_closes_old_track_and_rejects_stale_frame() -> None:
    clock = FakeClock()
    aggregator = _aggregator(clock)
    for sequence in range(1, 4):
        aggregator.consume(_frame(clock, sequence))
        clock.advance(0.1)

    new_session = aggregator.consume(_frame(clock, 1, session="session-2"))
    assert len(new_session.closed_events) == 1
    assert new_session.closed_events[0].capture_session_id == "session-1"
    assert new_session.active_tracks == ()

    stale = aggregator.consume(_frame(clock, 99, session="session-1"))
    assert stale.accepted is False
    assert stale.stale is True
    assert stale.closed_events == ()
    assert aggregator.tracker.reset_count == 1


@pytest.mark.integration
def test_reset_and_repeated_ticks_do_not_duplicate_close() -> None:
    clock = FakeClock()
    aggregator = _aggregator(clock)
    for sequence in range(1, 4):
        aggregator.consume(_frame(clock, sequence))
        clock.advance(0.1)

    first = aggregator.reset()
    assert len(first.closed_events) == 1
    assert aggregator.reset().closed_events == ()
    assert aggregator.tick().closed_events == ()


@pytest.mark.integration
def test_active_track_state_is_bounded_by_configured_limit() -> None:
    clock = FakeClock()
    aggregator = _aggregator(clock, adapter=FixedTrackAdapter())
    for sequence in range(1, 4):
        update = aggregator.consume(_frame(clock, sequence))
        clock.advance(0.1)

    assert len(update.active_tracks) == 1
    assert len(aggregator._tracks) <= 4
