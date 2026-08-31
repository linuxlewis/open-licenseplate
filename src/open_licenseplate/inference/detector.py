"""High-level still-image detector orchestration."""

from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter

from .adapters import adapter_for_manifest
from .contract import (
    BackendInspection,
    BackendOptions,
    DetectionBatch,
    ImageTransform,
    InferenceBackend,
    LoadedModel,
    ModelDescriptor,
    StillImage,
)
from .coreml import compare_manifest_to_inspection


@dataclass(frozen=True, slots=True)
class DetectionRun:
    """One detection with stage timings and model instance provenance."""

    batch: DetectionBatch
    preprocessing_ms: float
    inference_ms: float
    postprocessing_ms: float
    total_ms: float
    model_load_ms: float
    model_instance_id: str
    inspection: BackendInspection
    transform: ImageTransform


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

    def detect(
        self,
        image: StillImage,
        *,
        confidence_threshold: float | None = None,
    ) -> DetectionBatch:
        """Detect one image with the current model instance."""
        return self.detect_timed(
            image,
            confidence_threshold=confidence_threshold,
        ).batch

    def detect_timed(
        self,
        image: StillImage,
        *,
        confidence_threshold: float | None = None,
    ) -> DetectionRun:
        """Detect one image and measure preprocessing, inference, and decoding."""
        total_started = perf_counter()
        model_load_ms = 0.0
        if self.loaded is None:
            load_started = perf_counter()
            loaded = self.load()
            model_load_ms = (perf_counter() - load_started) * 1000
        else:
            loaded = self.loaded
        adapter = adapter_for_manifest(self.descriptor.manifest)

        preprocessing_started = perf_counter()
        prepared = adapter.preprocess(image, self.descriptor.manifest)
        if confidence_threshold is not None:
            if not 0 <= confidence_threshold <= 1:
                raise ValueError("confidence threshold must be between 0 and 1")
            prepared = replace(
                prepared,
                transform=replace(
                    prepared.transform,
                    confidence_threshold=confidence_threshold,
                ),
            )
        preprocessing_ms = (perf_counter() - preprocessing_started) * 1000

        inference_started = perf_counter()
        output = self.backend.predict(loaded, prepared)
        inference_ms = (perf_counter() - inference_started) * 1000

        postprocessing_started = perf_counter()
        decoded = adapter.decode(output, prepared.transform)
        postprocessing_ms = (perf_counter() - postprocessing_started) * 1000
        total_ms = (perf_counter() - total_started) * 1000
        return DetectionRun(
            batch=decoded,
            preprocessing_ms=preprocessing_ms,
            inference_ms=inference_ms,
            postprocessing_ms=postprocessing_ms,
            total_ms=total_ms,
            model_load_ms=model_load_ms,
            model_instance_id=loaded.instance_id,
            inspection=loaded.inspection,
            transform=prepared.transform,
        )

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
        *,
        confidence_threshold: float | None = None,
    ) -> DetectionBatch:
        """Run the stable still-image detector contract."""
        return self.detect_timed(
            image,
            model,
            options=options,
            confidence_threshold=confidence_threshold,
        ).batch

    def detect_timed(
        self,
        image: StillImage,
        model: ModelDescriptor,
        options: BackendOptions | None = None,
        *,
        confidence_threshold: float | None = None,
    ) -> DetectionRun:
        """Run the still-image detector and return stage timings."""
        session = DetectorSession(
            backend=self.backend,
            descriptor=model,
            options=options or BackendOptions(),
        )
        try:
            return session.detect_timed(
                image,
                confidence_threshold=confidence_threshold,
            )
        finally:
            session.close()


StillImageDetector = BackendStillImageDetector
