"""Thread-safe still-image detector session management."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from threading import RLock
from time import perf_counter

from .contract import BackendOptions, InferenceBackend, ModelDescriptor, StillImage
from .detector import DetectionRun, DetectorSession

BackendFactory = Callable[[], InferenceBackend]


class DetectorRegistry:
    """Own one serialized detector session per managed model."""

    def __init__(self, backend_factory: BackendFactory) -> None:
        self._backend_factory = backend_factory
        self._lock = RLock()
        self._sessions: dict[str, DetectorSession] = {}

    def detect(
        self,
        image: StillImage,
        descriptor: ModelDescriptor,
        options: BackendOptions,
        *,
        confidence_threshold: float | None = None,
    ) -> tuple[DetectionRun, bool]:
        """Detect while reloading the session when compute units change."""
        with self._lock:
            session = self._sessions.get(descriptor.model_id)
            reloaded = False
            if session is None or not _same_descriptor(session.descriptor, descriptor):
                if session is not None:
                    session.close()
                session = DetectorSession(
                    backend=self._backend_factory(),
                    descriptor=descriptor,
                    options=options,
                )
                self._sessions[descriptor.model_id] = session
            elif session.options.compute_units != options.compute_units:
                load_started = perf_counter()
                session.set_compute_units(options)
                reload_ms = (perf_counter() - load_started) * 1000
                reloaded = True
                run = session.detect_timed(
                    image,
                    confidence_threshold=confidence_threshold,
                )
                return (
                    replace(
                        run,
                        model_load_ms=run.model_load_ms + reload_ms,
                        total_ms=run.total_ms + reload_ms,
                    ),
                    reloaded,
                )

            return (
                session.detect_timed(
                    image,
                    confidence_threshold=confidence_threshold,
                ),
                reloaded,
            )

    def close(self) -> None:
        """Close all loaded sessions during application shutdown."""
        with self._lock:
            for session in self._sessions.values():
                session.close()
            self._sessions.clear()


def _same_descriptor(first: ModelDescriptor, second: ModelDescriptor) -> bool:
    return (
        first.model_id == second.model_id
        and first.artifact_path == second.artifact_path
        and first.artifact_sha256 == second.artifact_sha256
        and first.manifest.snapshot_json == second.manifest.snapshot_json
    )
