from __future__ import annotations

from datetime import UTC, datetime

import pytest

from open_licenseplate.inference import Detection
from open_licenseplate.tracking import SupervisionByteTrackAdapter

pytestmark = [pytest.mark.integration, pytest.mark.m4_a_acceptance]


def _detection(
    *,
    box: tuple[float, float, float, float],
    confidence: float,
    sequence: int = 1,
) -> Detection:
    return Detection(
        box_xyxy=box,
        class_id=0,
        label="license_plate",
        confidence=confidence,
        model_id="model-1",
        model_sha256="a" * 64,
        frame_sequence=sequence,
        detected_at=datetime(2026, 8, 30, tzinfo=UTC),
    )


def test_supervision_adapter_returns_exact_source_detection_after_filtering() -> None:
    low = _detection(box=(1.0, 1.0, 8.0, 8.0), confidence=0.1)
    high = _detection(box=(20.0, 20.0, 40.0, 32.0), confidence=0.9)
    adapter = SupervisionByteTrackAdapter()

    matches = adapter.update((low, high), frame_width=64, frame_height=48)

    assert len(matches) == 1
    assert matches[0].detection is high
    assert matches[0].detection is not low


def test_supervision_adapter_accepts_empty_input() -> None:
    adapter = SupervisionByteTrackAdapter()

    assert adapter.update((), frame_width=64, frame_height=48) == ()


def test_supervision_adapter_reset_starts_a_new_tracker_identity() -> None:
    detection = _detection(box=(8.0, 8.0, 24.0, 18.0), confidence=0.9)
    adapter = SupervisionByteTrackAdapter()

    first = adapter.update((detection,), frame_width=64, frame_height=48)
    adapter.reset()
    second = adapter.update((detection,), frame_width=64, frame_height=48)

    assert first[0].track_id == 1
    assert second[0].track_id == 1
    assert second[0].detection is detection


def test_supervision_adapter_enforces_configured_input_bound() -> None:
    first = _detection(box=(1.0, 1.0, 8.0, 8.0), confidence=0.8)
    second = _detection(box=(20.0, 20.0, 40.0, 32.0), confidence=0.9)
    adapter = SupervisionByteTrackAdapter(max_active_tracks=1)

    matches = adapter.update((first, second), frame_width=64, frame_height=48)

    assert len(matches) == 1
    assert matches[0].detection is second
