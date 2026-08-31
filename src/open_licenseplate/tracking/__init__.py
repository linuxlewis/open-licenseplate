"""Tracking contracts, ByteTrack adapter, and event state machine."""

from .bytetrack import (
    ByteTrackAdapter,
    ByteTrackUnavailableError,
    SupervisionByteTrackAdapter,
    default_tracker_factory,
)
from .contracts import (
    ActiveTrack,
    ClosedTrackEvent,
    LiveDetectionFrame,
    TrackedDetection,
    TrackerAdapter,
    TrackingProvenance,
    TrackingUpdate,
)
from .crops import (
    CROP_QUALITY_SCORING_VERSION,
    MAX_CROP_PIXELS,
    CropCandidate,
    CropQuality,
    capture_crop_candidate,
    score_crop_quality,
)
from .state_machine import (
    ClosedEventSink,
    TrackerFactory,
    TrackingConfig,
    TrackingEventAggregator,
)

__all__ = [
    "ActiveTrack",
    "ByteTrackAdapter",
    "ByteTrackUnavailableError",
    "CROP_QUALITY_SCORING_VERSION",
    "ClosedEventSink",
    "ClosedTrackEvent",
    "CropCandidate",
    "CropQuality",
    "LiveDetectionFrame",
    "MAX_CROP_PIXELS",
    "SupervisionByteTrackAdapter",
    "TrackedDetection",
    "TrackerAdapter",
    "TrackerFactory",
    "TrackingConfig",
    "TrackingEventAggregator",
    "TrackingProvenance",
    "TrackingUpdate",
    "capture_crop_candidate",
    "default_tracker_factory",
    "score_crop_quality",
]
