"""Frame-source contracts, implementations, and latest-frame delivery."""

from .broker import BrokerMetrics, LatestFrameBroker
from .contracts import (
    CaptureSession,
    CaptureShutdownError,
    Clock,
    FrameSource,
    SourceError,
    SourceInfo,
    SourceLifecycleError,
    SourceOpenError,
    SourceReadError,
    SystemClock,
    VideoFrame,
)
from .sources import (
    FakeFrameSource,
    FakeSource,
    PyAVFrameSource,
    PyAVRTSPSource,
    PyAVSource,
    RecordedFrame,
    RecordedSource,
    RecordedVideoSource,
    RTSPFrameSource,
)
from .worker import CaptureMetrics, CaptureWorker, FrameCaptureWorker

__all__ = [
    "BrokerMetrics",
    "CaptureMetrics",
    "CaptureSession",
    "CaptureShutdownError",
    "CaptureWorker",
    "Clock",
    "FakeSource",
    "FakeFrameSource",
    "FrameCaptureWorker",
    "FrameSource",
    "LatestFrameBroker",
    "PyAVFrameSource",
    "PyAVSource",
    "PyAVRTSPSource",
    "RecordedSource",
    "RTSPFrameSource",
    "RecordedFrame",
    "RecordedVideoSource",
    "SourceError",
    "SourceInfo",
    "SourceLifecycleError",
    "SourceOpenError",
    "SourceReadError",
    "SystemClock",
    "VideoFrame",
]
