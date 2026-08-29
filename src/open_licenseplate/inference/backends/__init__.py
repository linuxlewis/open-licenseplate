"""Portable and platform-specific inference backends."""

from .coreml import CoreMLBackend
from .fake import FakeBackend

__all__ = ["CoreMLBackend", "FakeBackend"]
