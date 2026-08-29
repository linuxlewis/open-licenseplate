"""Compatibility exports for camera-owned frame sources."""

from ..capture import (
    FakeFrameSource,
    FakeSource,
    FrameSource,
    LatestFrameBroker,
    PyAVFrameSource,
    PyAVRTSPSource,
    PyAVSource,
    RecordedFrame,
    RecordedSource,
    RecordedVideoSource,
    RTSPFrameSource,
    SourceError,
    SourceInfo,
    VideoFrame,
)

__all__ = [
    "FakeFrameSource",
    "FakeSource",
    "FrameSource",
    "LatestFrameBroker",
    "PyAVFrameSource",
    "PyAVSource",
    "PyAVRTSPSource",
    "RTSPFrameSource",
    "RecordedFrame",
    "RecordedSource",
    "RecordedVideoSource",
    "SourceError",
    "SourceInfo",
    "VideoFrame",
]
