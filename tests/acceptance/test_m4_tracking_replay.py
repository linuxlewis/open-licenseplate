from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from open_licenseplate.inference import Detection
from open_licenseplate.tracking import (
    LiveDetectionFrame,
    TrackedDetection,
    TrackingConfig,
    TrackingEventAggregator,
    TrackingProvenance,
)


@dataclass
class ReplayClock:
    wall: datetime
    ticks: float = 0.0

    def now(self) -> datetime:
        return self.wall

    def monotonic(self) -> float:
        return self.ticks

    def move_to(self, offset_seconds: float) -> None:
        self.wall = self.wall.replace() + timedelta(seconds=offset_seconds - self.ticks)
        self.ticks = offset_seconds

    def advance(self, seconds: float) -> None:
        self.ticks += seconds
        self.wall += timedelta(seconds=seconds)


class FixtureTracker:
    def update(
        self,
        detections: Sequence[Detection],
        *,
        frame_width: int,
        frame_height: int,
    ) -> tuple[TrackedDetection, ...]:
        assert (frame_width, frame_height) == (64, 48)
        return tuple(TrackedDetection(7, detection) for detection in detections)

    def reset(self) -> None:
        return None


def _fixture() -> dict[str, Any]:
    path = Path(__file__).parents[1] / "fixtures" / "replay" / "m4_tracking_events.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _run_pass(values: dict[str, Any]) -> tuple:
    base = datetime.fromisoformat(_fixture()["base_utc"].replace("Z", "+00:00"))
    clock = ReplayClock(base)
    aggregator = TrackingEventAggregator(
        lambda: FixtureTracker(),
        clock=clock,
        config=TrackingConfig(confirmation_window_seconds=0.75, close_timeout_seconds=1.0),
    )
    for frame in values["frames"]:
        offset = float(frame["offset_seconds"])
        clock.move_to(offset)
        captured_at = clock.now()
        detections = ()
        if "box_xyxy" in frame:
            detections = (
                Detection(
                    box_xyxy=tuple(float(item) for item in frame["box_xyxy"]),
                    class_id=0,
                    label="license_plate",
                    confidence=float(frame["confidence"]),
                    model_id="fixture-model",
                    model_sha256="a" * 64,
                    frame_sequence=int(frame["frame_sequence"]),
                    detected_at=captured_at,
                ),
            )
        aggregator.consume(
            LiveDetectionFrame(
                provenance=TrackingProvenance(
                    camera_id="fixture-camera",
                    capture_session_id=values["capture_session_id"],
                    generation_number=int(values["generation_number"]),
                    stream_epoch=values["stream_epoch"],
                    model_id="fixture-model",
                    model_checksum="a" * 64,
                ),
                frame_sequence=int(frame["frame_sequence"]),
                captured_at=captured_at,
                frame_width=64,
                frame_height=48,
                detections=detections,
            )
        )
    clock.advance(1.0)
    return aggregator.tick().closed_events


@pytest.mark.m4_a_acceptance
def test_m4_a_replay_has_one_known_event_no_empty_event_and_no_short_candidate() -> None:
    fixture = _fixture()
    passes = {item["name"]: item for item in fixture["passes"]}

    known_events = _run_pass(passes["known_plate_pass"])
    empty_events = _run_pass(passes["no_plate_pass"])
    short_events = _run_pass(passes["short_false_candidate"])

    assert len(known_events) == 1
    assert known_events[0].event_state == "closed"
    assert known_events[0].observation_count == 3
    assert known_events[0].maximum_confidence == pytest.approx(0.93)
    assert empty_events == ()
    assert short_events == ()
