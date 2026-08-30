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
    "ClosedEventSink",
    "ClosedTrackEvent",
    "LiveDetectionFrame",
    "SupervisionByteTrackAdapter",
    "TrackedDetection",
    "TrackerAdapter",
    "TrackerFactory",
    "TrackingConfig",
    "TrackingEventAggregator",
    "TrackingProvenance",
    "TrackingUpdate",
    "default_tracker_factory",
]
