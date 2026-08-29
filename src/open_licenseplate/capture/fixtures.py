"""Deterministic capture fixtures for reconnect and preview tests."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..cameras.repository import CameraConfig
from .sources import FakeFrameSource


@dataclass(frozen=True, slots=True)
class FixtureAttempt:
    """One scripted source attempt."""

    frames: tuple[Any, ...] = ()
    open_error: str | None = None
    read_error: str | None = None
    fail_at: int | None = None
    read_interval_seconds: float = 0.01
    repeat: bool = False


class ReconnectFixture:
    """Create a reproducible stream with scripted disconnects."""

    def __init__(self, attempts: Iterable[FixtureAttempt]) -> None:
        self.attempts = tuple(attempts)
        self.created_attempts = 0
        self.sources: list[FakeFrameSource] = []

    def __call__(self, camera: CameraConfig, camera_id: str) -> FakeFrameSource:
        del camera
        if not self.attempts:
            raise RuntimeError("fixture must contain at least one attempt")
        attempt_index = min(self.created_attempts, len(self.attempts) - 1)
        attempt = self.attempts[attempt_index]
        self.created_attempts += 1
        source = FakeFrameSource(
            attempt.frames,
            camera_id=camera_id,
            open_error=attempt.open_error,
            read_error=attempt.read_error,
            fail_at=attempt.fail_at,
            read_interval_seconds=attempt.read_interval_seconds,
            repeat=attempt.repeat,
        )
        self.sources.append(source)
        return source


def make_preview_frame(value: int, *, width: int = 8, height: int = 6) -> np.ndarray:
    """Return a small deterministic BGR frame."""
    return np.full((height, width, 3), value, dtype=np.uint8)


def disconnect_then_recover_fixture() -> ReconnectFixture:
    """Return a source that fails after one frame and then recovers."""
    return ReconnectFixture(
        (
            FixtureAttempt(
                frames=(make_preview_frame(32),),
                fail_at=1,
                read_error="fixture disconnect at frame 1",
                read_interval_seconds=0.01,
            ),
            FixtureAttempt(
                frames=(make_preview_frame(64), make_preview_frame(96)),
                read_interval_seconds=0.01,
                repeat=True,
            ),
        )
    )


__all__ = [
    "FixtureAttempt",
    "ReconnectFixture",
    "disconnect_then_recover_fixture",
    "make_preview_frame",
]
