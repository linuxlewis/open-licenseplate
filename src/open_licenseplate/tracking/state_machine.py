"""Bounded track lifecycle and closed-event aggregation."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from ..capture.contracts import Clock, SystemClock
from .contracts import (
    ActiveTrack,
    ClosedTrackEvent,
    LiveDetectionFrame,
    TrackedDetection,
    TrackerAdapter,
    TrackingProvenance,
    TrackingUpdate,
)


@dataclass(frozen=True, slots=True)
class TrackingConfig:
    """Explicit limits and timing defaults for one camera pipeline."""

    confirmation_observations: int = 3
    confirmation_window_seconds: float = 0.75
    close_timeout_seconds: float = 1.0
    max_active_tracks: int = 64
    max_retired_provenances: int = 16
    max_closed_track_keys: int = 128

    def __post_init__(self) -> None:
        if type(self.confirmation_observations) is not int or self.confirmation_observations < 3:
            raise ValueError("confirmation_observations must be at least 3")
        for value, field_name in (
            (self.confirmation_window_seconds, "confirmation_window_seconds"),
            (self.close_timeout_seconds, "close_timeout_seconds"),
        ):
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"{field_name} must be positive and finite")
        for value, field_name in (
            (self.max_active_tracks, "max_active_tracks"),
            (self.max_retired_provenances, "max_retired_provenances"),
            (self.max_closed_track_keys, "max_closed_track_keys"),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"{field_name} must be positive")


@dataclass
class _TrackState:
    """Small mutable aggregate; it never retains observation history."""

    provenance: TrackingProvenance
    track_id: int
    first_seen_at: datetime
    last_seen_at: datetime
    first_seen_monotonic: float
    last_seen_monotonic: float
    last_frame_sequence: int
    last_box_xyxy: tuple[float, float, float, float]
    last_confidence: float
    observation_count: int = 1
    maximum_confidence: float = 0.0
    state: str = "candidate"


TrackerFactory = Callable[[], TrackerAdapter]
ClosedEventSink = Callable[[ClosedTrackEvent], None]


class TrackingEventAggregator:
    """Map validated live detections to bounded active tracks and events."""

    def __init__(
        self,
        tracker_factory: TrackerFactory,
        *,
        clock: Clock | None = None,
        config: TrackingConfig | None = None,
        on_closed_event: ClosedEventSink | None = None,
    ) -> None:
        self.config = config or TrackingConfig()
        self._clock = clock or SystemClock()
        self._tracker = tracker_factory()
        self._on_closed_event = on_closed_event
        self._provenance: TrackingProvenance | None = None
        self._last_frame_sequence: int | None = None
        self._tracks: dict[tuple[str, int], _TrackState] = {}
        self._retired_provenances: deque[TrackingProvenance] = deque(
            maxlen=self.config.max_retired_provenances
        )
        self._closed_track_keys: deque[tuple[str, int]] = deque(
            maxlen=self.config.max_closed_track_keys
        )
        self._closed_track_key_set: set[tuple[str, int]] = set()

    @property
    def active_tracks(self) -> tuple[ActiveTrack, ...]:
        """Return only confirmed tracks in deterministic order."""
        return self._active_track_payloads()

    @property
    def active_track_count(self) -> int:
        """Return the bounded count of confirmed live tracks."""
        return len(self._active_track_payloads())

    @property
    def tracker(self) -> TrackerAdapter:
        """Expose only the internal tracker contract for tests and diagnostics."""
        return self._tracker

    def consume(self, frame: LiveDetectionFrame) -> TrackingUpdate:
        """Process one validated frame and emit any timeout closures."""
        provenance_events = self._switch_provenance(frame.provenance)
        if provenance_events is None:
            return TrackingUpdate(
                active_tracks=self.active_tracks,
                accepted=False,
                stale=True,
            )
        if (
            self._last_frame_sequence is not None
            and frame.frame_sequence <= self._last_frame_sequence
        ):
            return TrackingUpdate(
                active_tracks=self.active_tracks,
                accepted=False,
                stale=True,
            )

        matches = self._tracker.update(
            frame.detections,
            frame_width=frame.frame_width,
            frame_height=frame.frame_height,
        )
        self._last_frame_sequence = frame.frame_sequence
        matched_track_ids: set[int] = set()
        for match in matches:
            if match.track_id in matched_track_ids or not _match_matches_frame(frame, match):
                continue
            matched_track_ids.add(match.track_id)
            self._apply_match(frame, match)
        closed_events = self._expire(self._clock.monotonic())
        return self._update(provenance_events + closed_events)

    def tick(self) -> TrackingUpdate:
        """Expire candidates and confirmed tracks using the injected clock."""
        closed_events = self._expire(self._clock.monotonic())
        return self._update(closed_events)

    def reset(self) -> TrackingUpdate:
        """Close confirmed tracks once and clear all tracker state."""
        if self._provenance is not None:
            self._retired_provenances.append(self._provenance)
        closed_events = tuple(
            self._close_track(key)
            for key, track in tuple(self._tracks.items())
            if track.state in {"confirmed", "active"}
        )
        self._tracks.clear()
        self._last_frame_sequence = None
        self._provenance = None
        self._tracker.reset()
        return self._update(tuple(event for event in closed_events if event is not None))

    def close(self) -> TrackingUpdate:
        """Release tracker state without retaining a closed-event history."""
        return self.reset()

    def _switch_provenance(
        self,
        provenance: TrackingProvenance,
    ) -> tuple[ClosedTrackEvent, ...] | None:
        current = self._provenance
        if current is None:
            self._provenance = provenance
            return ()
        if provenance == current:
            return ()
        if provenance in self._retired_provenances:
            return None
        if provenance.generation_number < current.generation_number:
            return None
        closed_events = tuple(
            event
            for key, track in tuple(self._tracks.items())
            if track.state in {"confirmed", "active"}
            for event in (self._close_track(key),)
            if event is not None
        )
        self._retired_provenances.append(current)
        self._provenance = provenance
        self._last_frame_sequence = None
        self._tracks.clear()
        self._tracker.reset()
        return closed_events

    def _apply_match(self, frame: LiveDetectionFrame, match: TrackedDetection) -> None:
        key = (frame.provenance.capture_session_id, match.track_id)
        if key in self._closed_track_key_set:
            return
        if key not in self._tracks and len(self._tracks) >= self.config.max_active_tracks:
            return
        detection = match.detection
        current_monotonic = self._clock.monotonic()
        track = self._tracks.get(key)
        if track is None:
            track = _TrackState(
                provenance=frame.provenance,
                track_id=match.track_id,
                first_seen_at=frame.captured_at,
                last_seen_at=frame.captured_at,
                first_seen_monotonic=current_monotonic,
                last_seen_monotonic=current_monotonic,
                last_frame_sequence=frame.frame_sequence,
                last_box_xyxy=detection.box_xyxy,
                last_confidence=detection.confidence,
                maximum_confidence=detection.confidence,
            )
            self._tracks[key] = track
            return

        if current_monotonic < track.last_seen_monotonic:
            return
        if frame.captured_at < track.last_seen_at:
            return
        track.last_seen_at = frame.captured_at
        track.last_seen_monotonic = current_monotonic
        track.last_frame_sequence = frame.frame_sequence
        track.last_box_xyxy = detection.box_xyxy
        track.last_confidence = detection.confidence
        track.maximum_confidence = max(track.maximum_confidence, detection.confidence)
        track.observation_count += 1
        if (
            track.state == "candidate"
            and track.observation_count >= self.config.confirmation_observations
        ):
            if (
                current_monotonic - track.first_seen_monotonic
                <= self.config.confirmation_window_seconds
            ):
                track.state = "confirmed"
                track.state = "active"
            else:
                self._tracks.pop(key, None)

    def _expire(self, now_monotonic: float) -> tuple[ClosedTrackEvent, ...]:
        closed: list[ClosedTrackEvent] = []
        for key, track in tuple(self._tracks.items()):
            if track.state == "candidate":
                if (
                    now_monotonic - track.first_seen_monotonic
                    >= self.config.confirmation_window_seconds
                ):
                    self._tracks.pop(key, None)
                continue
            if now_monotonic - track.last_seen_monotonic >= self.config.close_timeout_seconds:
                event = self._close_track(key)
                if event is not None:
                    closed.append(event)
        return tuple(closed)

    def _close_track(self, key: tuple[str, int]) -> ClosedTrackEvent | None:
        track = self._tracks.pop(key, None)
        if track is None or track.state not in {"confirmed", "active"}:
            return None
        if key in self._closed_track_key_set:
            return None
        if len(self._closed_track_keys) == self.config.max_closed_track_keys:
            self._closed_track_key_set.discard(self._closed_track_keys.popleft())
        self._closed_track_keys.append(key)
        self._closed_track_key_set.add(key)
        event = ClosedTrackEvent.create(
            provenance=track.provenance,
            track_id=track.track_id,
            first_seen_at=track.first_seen_at,
            last_seen_at=track.last_seen_at,
            observation_count=track.observation_count,
            maximum_confidence=track.maximum_confidence,
        )
        if self._on_closed_event is not None:
            self._on_closed_event(event)
        return event

    def _update(self, closed_events: tuple[ClosedTrackEvent, ...]) -> TrackingUpdate:
        return TrackingUpdate(
            active_tracks=self.active_tracks,
            closed_events=closed_events,
        )

    def _active_track_payloads(self) -> tuple[ActiveTrack, ...]:
        tracks = [
            ActiveTrack(
                provenance=track.provenance,
                track_id=track.track_id,
                state="active" if track.state == "active" else "confirmed",
                first_seen_at=track.first_seen_at,
                last_seen_at=track.last_seen_at,
                last_frame_sequence=track.last_frame_sequence,
                last_box_xyxy=track.last_box_xyxy,
                last_confidence=track.last_confidence,
                observation_count=track.observation_count,
                maximum_confidence=track.maximum_confidence,
            )
            for track in self._tracks.values()
            if track.state in {"confirmed", "active"}
        ]
        tracks.sort(key=lambda track: (track.track_id, track.first_seen_at))
        return tuple(tracks)


def _match_matches_frame(frame: LiveDetectionFrame, match: TrackedDetection) -> bool:
    detection = match.detection
    return (
        detection.model_id == frame.provenance.model_id
        and detection.model_sha256 == frame.provenance.model_checksum
        and detection.frame_sequence == frame.frame_sequence
        and detection.detected_at is not None
        and detection.detected_at == frame.captured_at
    )


__all__ = [
    "ClosedEventSink",
    "TrackingConfig",
    "TrackingEventAggregator",
    "TrackerFactory",
]
