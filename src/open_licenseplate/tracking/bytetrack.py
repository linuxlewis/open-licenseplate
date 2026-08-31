"""Narrow ByteTrack adapter with third-party types kept private."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..inference.contract import Detection
from .contracts import TrackedDetection, TrackerAdapter

SOURCE_INDEX_KEY = "open_licenseplate_source_index"


class ByteTrackUnavailableError(RuntimeError):
    """Raised when the optional ByteTrack runtime cannot be loaded."""


class SupervisionByteTrackAdapter:
    """Adapt ``supervision.ByteTrack`` to the internal tracker contract."""

    def __init__(
        self,
        *,
        minimum_matching_threshold: float = 0.8,
        minimum_consecutive_frames: int = 1,
        maximum_track_age: int = 30,
        max_active_tracks: int = 64,
    ) -> None:
        if not 0 < minimum_matching_threshold <= 1:
            raise ValueError("minimum_matching_threshold must be between 0 and 1")
        if type(minimum_consecutive_frames) is not int or minimum_consecutive_frames < 1:
            raise ValueError("minimum_consecutive_frames must be positive")
        if type(maximum_track_age) is not int or maximum_track_age < 1:
            raise ValueError("maximum_track_age must be positive")
        if type(max_active_tracks) is not int or max_active_tracks < 1:
            raise ValueError("max_active_tracks must be positive")
        try:
            import supervision as sv
        except ImportError as error:
            raise ByteTrackUnavailableError(
                "the supervision ByteTrack dependency is not installed"
            ) from error

        self._sv = sv
        self._tracker = sv.ByteTrack(
            minimum_matching_threshold=minimum_matching_threshold,
            minimum_consecutive_frames=minimum_consecutive_frames,
            lost_track_buffer=maximum_track_age,
        )
        self._max_active_tracks = max_active_tracks

    def update(
        self,
        detections: Sequence[Detection],
        *,
        frame_width: int,
        frame_height: int,
    ) -> tuple[TrackedDetection, ...]:
        """Associate one validated source-frame detection batch."""
        del frame_width, frame_height
        if not detections:
            empty = self._sv.Detections(
                xyxy=np.empty((0, 4), dtype=np.float32),
                confidence=np.empty((0,), dtype=np.float32),
                class_id=np.empty((0,), dtype=np.int32),
            )
            self._tracker.update_with_detections(empty)
            self._trim_removed_tracks()
            return ()
        if len(detections) > self._max_active_tracks:
            detections = tuple(
                sorted(
                    detections,
                    key=lambda detection: (-detection.confidence, detection.box_xyxy),
                )[: self._max_active_tracks]
            )
        source = self._sv.Detections(
            xyxy=np.asarray([detection.box_xyxy for detection in detections], dtype=np.float32),
            confidence=np.asarray(
                [detection.confidence for detection in detections],
                dtype=np.float32,
            ),
            class_id=np.asarray([detection.class_id for detection in detections], dtype=np.int32),
            data={
                SOURCE_INDEX_KEY: np.arange(
                    len(detections),
                    dtype=np.int32,
                )
            },
        )
        tracked = self._tracker.update_with_detections(source)
        self._trim_removed_tracks()
        tracker_ids = tracked.tracker_id
        source_indices = tracked.data.get(SOURCE_INDEX_KEY)
        if tracker_ids is None or source_indices is None:
            return ()
        results: list[TrackedDetection] = []
        for output_index, raw_track_id in enumerate(tracker_ids):
            if (
                raw_track_id is None
                or output_index >= len(source_indices)
                or source_indices[output_index] is None
            ):
                continue
            source_index = int(source_indices[output_index])
            if source_index < 0 or source_index >= len(detections):
                continue
            track_id = int(raw_track_id)
            if track_id < 0:
                continue
            results.append(
                TrackedDetection(
                    track_id=track_id,
                    detection=detections[source_index],
                )
            )
        return tuple(results)

    def reset(self) -> None:
        """Release the third-party tracker state."""
        self._tracker.reset()

    def _trim_removed_tracks(self) -> None:
        """Prevent the adapter's private removed-track list from growing."""
        removed_tracks = getattr(self._tracker, "removed_tracks", None)
        if not isinstance(removed_tracks, list):
            return
        if len(removed_tracks) > self._max_active_tracks:
            del removed_tracks[: len(removed_tracks) - self._max_active_tracks]


ByteTrackAdapter = SupervisionByteTrackAdapter


def default_tracker_factory(
    *,
    minimum_matching_threshold: float = 0.8,
    minimum_consecutive_frames: int = 1,
    maximum_track_age: int = 30,
    max_active_tracks: int = 64,
) -> TrackerAdapter:
    """Build the production tracker behind the internal contract."""
    return SupervisionByteTrackAdapter(
        minimum_matching_threshold=minimum_matching_threshold,
        minimum_consecutive_frames=minimum_consecutive_frames,
        maximum_track_age=maximum_track_age,
        max_active_tracks=max_active_tracks,
    )


__all__ = [
    "ByteTrackUnavailableError",
    "ByteTrackAdapter",
    "SupervisionByteTrackAdapter",
    "default_tracker_factory",
]
