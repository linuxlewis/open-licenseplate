"""High-level still-image detector orchestration."""

from __future__ import annotations

from dataclasses import dataclass

from .adapters import adapter_for_manifest
from .contract import (
    BackendOptions,
    DetectionBatch,
    InferenceBackend,
    LoadedModel,
    ModelDescriptor,
    StillImage,
)
from .coreml import compare_manifest_to_inspection


@dataclass
class DetectorSession:
    """Keep one model instance and reload it when compute units change."""

    backend: InferenceBackend
    descriptor: ModelDescriptor
    options: BackendOptions
    loaded: LoadedModel | None = None

    def load(self) -> LoadedModel:
        """Load and validate a new instance before replacing the current one."""
        loaded = self.backend.load(self.descriptor, self.options)
        try:
            compare_manifest_to_inspection(self.descriptor.manifest, loaded.inspection)
        except Exception:
            self.backend.close(loaded)
            raise
        previous = self.loaded
        self.loaded = loaded
        if previous is not None:
            self.backend.close(previous)
        return self.loaded

    def set_compute_units(self, options: BackendOptions) -> LoadedModel:
        """Reload the model so a compute-unit change cannot reuse old state."""
        previous_options = self.options
        self.options = options
        try:
            return self.load()
        except Exception:
            self.options = previous_options
            raise

    def detect(self, image: StillImage) -> DetectionBatch:
        """Detect one image with the current model instance."""
        loaded = self.loaded or self.load()
        adapter = adapter_for_manifest(self.descriptor.manifest)
        prepared = adapter.preprocess(image, self.descriptor.manifest)
        output = self.backend.predict(loaded, prepared)
        return adapter.decode(output, prepared.transform)

    def close(self) -> None:
        """Close the current model instance."""
        if self.loaded is not None:
            self.backend.close(self.loaded)
            self.loaded = None


class BackendStillImageDetector:
    """Load, run, and close one backend instance for each request."""

    def __init__(self, backend: InferenceBackend) -> None:
        self.backend = backend

    def detect(
        self,
        image: StillImage,
        model: ModelDescriptor,
        options: BackendOptions | None = None,
    ) -> DetectionBatch:
        """Run the stable still-image detector contract."""
        session = DetectorSession(
            backend=self.backend,
            descriptor=model,
            options=options or BackendOptions(),
        )
        try:
            return session.detect(image)
        finally:
            session.close()


StillImageDetector = BackendStillImageDetector
