"""Backend-neutral contracts for still-image object detection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any, Protocol
from uuid import uuid4

import numpy as np

from ..models.manifest import ModelManifest


class InferenceError(ValueError):
    """Base error for inference contract failures."""


class BackendUnavailableError(InferenceError):
    """Raised when the selected inference backend is not available."""


class BackendContractError(InferenceError):
    """Raised when a model does not match its declared runtime contract."""


class PreprocessingError(InferenceError):
    """Raised when a still image cannot be prepared for a model."""


class DetectionValidationError(InferenceError):
    """Raised when a backend output cannot be decoded safely."""


class ComputeUnit(StrEnum):
    """Supported Core ML compute-unit choices."""

    ALL = "all"
    CPU_ONLY = "cpu_only"
    CPU_AND_GPU = "cpu_and_gpu"
    CPU_AND_NE = "cpu_and_ne"

    @classmethod
    def parse(cls, value: ComputeUnit | str) -> ComputeUnit:
        """Parse stable API values and the displayed compute-unit labels."""
        if isinstance(value, cls):
            return value
        normalized = value.strip().casefold().replace("-", "_")
        aliases = {
            "all": cls.ALL,
            "cpu_only": cls.CPU_ONLY,
            "cpu only": cls.CPU_ONLY,
            "cpu_and_gpu": cls.CPU_AND_GPU,
            "cpu and gpu": cls.CPU_AND_GPU,
            "cpu_and_ne": cls.CPU_AND_NE,
            "cpu_and_neural_engine": cls.CPU_AND_NE,
            "cpu and neural engine": cls.CPU_AND_NE,
        }
        try:
            return aliases[normalized]
        except KeyError as error:
            raise ValueError(
                "compute_units must be all, cpu_only, cpu_and_gpu, or cpu_and_ne"
            ) from error

    @property
    def display_name(self) -> str:
        """Return the exact operator-facing choice name."""
        return {
            self.ALL: "All",
            self.CPU_ONLY: "CPU only",
            self.CPU_AND_GPU: "CPU and GPU",
            self.CPU_AND_NE: "CPU and Neural Engine",
        }[self]


ComputeUnits = ComputeUnit


@dataclass(frozen=True, slots=True)
class BackendOptions:
    """Options selected when a backend model instance is loaded."""

    compute_units: ComputeUnit = ComputeUnit.ALL

    def __post_init__(self) -> None:
        object.__setattr__(self, "compute_units", ComputeUnit.parse(self.compute_units))


@dataclass(frozen=True, slots=True)
class ModelDescriptor:
    """Immutable model provenance passed to a backend."""

    model_id: str
    artifact_path: str
    artifact_sha256: str
    manifest: ModelManifest


@dataclass(frozen=True, slots=True)
class StillImage:
    """A source image with an explicit channel order."""

    pixels: np.ndarray
    color_space: str = "rgb"
    frame_sequence: int | None = None
    captured_at: datetime | None = None

    def __post_init__(self) -> None:
        pixels = np.asarray(self.pixels)
        if pixels.ndim not in {2, 3}:
            raise ValueError("still image pixels must be a 2D or 3D array")
        if pixels.shape[0] <= 0 or pixels.shape[1] <= 0:
            raise ValueError("still image pixels must have positive dimensions")
        color_space = self.color_space.casefold()
        if color_space not in {"rgb", "bgr", "grayscale"}:
            raise ValueError("still image color_space must be rgb, bgr, or grayscale")
        if color_space == "grayscale" and pixels.ndim == 3 and pixels.shape[2] != 1:
            raise ValueError("grayscale still image pixels must have one channel")
        if color_space in {"rgb", "bgr"} and (pixels.ndim != 3 or pixels.shape[2] != 3):
            raise ValueError("rgb and bgr still image pixels must have three channels")
        if not np.issubdtype(pixels.dtype, np.number):
            raise ValueError("still image pixels must be numeric")
        if not np.isfinite(pixels).all():
            raise ValueError("still image pixels must contain finite values")
        object.__setattr__(self, "pixels", pixels)
        object.__setattr__(self, "color_space", color_space)

    @property
    def width(self) -> int:
        """Return the source width in pixels."""
        return int(self.pixels.shape[1])

    @property
    def height(self) -> int:
        """Return the source height in pixels."""
        return int(self.pixels.shape[0])


@dataclass(frozen=True, slots=True)
class ImageTransform:
    """Deterministic model-image transform and inverse source mapping."""

    source_width: int
    source_height: int
    model_width: int
    model_height: int
    resize: str
    scale_x: float
    scale_y: float
    box_format: str
    coordinate_space: str
    pad_left: int = 0
    pad_top: int = 0
    resized_width: int | None = None
    resized_height: int | None = None
    model_id: str = ""
    model_sha256: str = ""
    labels: tuple[str, ...] = ()
    confidence_threshold: float = 0.0
    iou_threshold: float = 0.0
    output_names: tuple[tuple[str, str], ...] = ()
    raw_output_name: str | None = None
    class_output_name: str | None = None
    raw_layout: str | None = None
    raw_has_objectness: bool | None = None
    frame_sequence: int | None = None
    captured_at: datetime | None = None

    def __post_init__(self) -> None:
        if (
            min(
                self.source_width,
                self.source_height,
                self.model_width,
                self.model_height,
            )
            <= 0
        ):
            raise ValueError("image transform dimensions must be positive")
        if self.scale_x <= 0 or self.scale_y <= 0:
            raise ValueError("image transform scales must be positive")
        if self.resize not in {"letterbox", "stretch", "none"}:
            raise ValueError("image transform resize is not supported")
        if self.box_format not in {"xyxy", "xywh"}:
            raise ValueError("image transform box_format must be xyxy or xywh")
        if self.coordinate_space not in {"model_pixels", "normalized"}:
            raise ValueError("image transform coordinate_space must be model_pixels or normalized")
        if not 0 <= self.confidence_threshold <= 1:
            raise ValueError("confidence threshold must be between 0 and 1")
        if not 0 <= self.iou_threshold <= 1:
            raise ValueError("IoU threshold must be between 0 and 1")
        if self.raw_output_name is None and (
            self.raw_layout is not None or self.raw_has_objectness is not None
        ):
            raise ValueError("raw layout values require a raw output name")
        if self.raw_output_name is not None and (
            self.raw_layout not in {"candidates_first", "channels_first", "channels_last"}
            or not isinstance(self.raw_has_objectness, bool)
        ):
            raise ValueError("raw output requires explicit layout and objectness values")

    def model_to_source_box(
        self,
        box_xyxy: Sequence[float],
    ) -> tuple[float, float, float, float]:
        """Map a model-space xyxy box to source-image pixels."""
        if len(box_xyxy) != 4:
            raise ValueError("box must contain four coordinates")
        x1, y1, x2, y2 = (float(value) for value in box_xyxy)
        return (
            (x1 - self.pad_left) / self.scale_x,
            (y1 - self.pad_top) / self.scale_y,
            (x2 - self.pad_left) / self.scale_x,
            (y2 - self.pad_top) / self.scale_y,
        )

    def source_to_model_box(
        self,
        box_xyxy: Sequence[float],
    ) -> tuple[float, float, float, float]:
        """Map a source-image xyxy box to model pixels."""
        if len(box_xyxy) != 4:
            raise ValueError("box must contain four coordinates")
        x1, y1, x2, y2 = (float(value) for value in box_xyxy)
        return (
            x1 * self.scale_x + self.pad_left,
            y1 * self.scale_y + self.pad_top,
            x2 * self.scale_x + self.pad_left,
            y2 * self.scale_y + self.pad_top,
        )

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-safe transform metadata for diagnostics and tests."""
        return {
            "source_width": self.source_width,
            "source_height": self.source_height,
            "model_width": self.model_width,
            "model_height": self.model_height,
            "resize": self.resize,
            "scale_x": self.scale_x,
            "scale_y": self.scale_y,
            "pad_left": self.pad_left,
            "pad_top": self.pad_top,
            "resized_width": self.resized_width,
            "resized_height": self.resized_height,
            "box_format": self.box_format,
            "coordinate_space": self.coordinate_space,
            "raw_layout": self.raw_layout,
            "raw_has_objectness": self.raw_has_objectness,
        }

    @property
    def output_name_map(self) -> dict[str, str]:
        """Return declared output roles to actual backend output names."""
        return dict(self.output_names)


@dataclass(frozen=True, slots=True)
class PreparedInput:
    """Backend-neutral model input and the metadata needed to decode it."""

    value: np.ndarray
    input_name: str
    color_space: str
    transform: ImageTransform


@dataclass(frozen=True, slots=True)
class FeatureDescription:
    """Safe structural description of one backend feature."""

    name: str
    kind: str
    width: int | None = None
    height: int | None = None
    color_space: str | None = None
    shape: tuple[int, ...] = ()
    data_type: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return a safe JSON representation."""
        result: dict[str, Any] = {
            "name": self.name,
            "kind": self.kind,
        }
        if self.width is not None:
            result["width"] = self.width
        if self.height is not None:
            result["height"] = self.height
        if self.color_space is not None:
            result["color_space"] = self.color_space
        if self.shape:
            result["shape"] = list(self.shape)
        if self.data_type is not None:
            result["data_type"] = self.data_type
        return result


@dataclass(frozen=True, slots=True)
class BackendInspection:
    """Actual input and output descriptions read from a loaded model."""

    backend: str
    inputs: tuple[FeatureDescription, ...]
    outputs: tuple[FeatureDescription, ...]

    def as_dict(self) -> dict[str, Any]:
        """Return safe structural validation details."""
        return {
            "backend": self.backend,
            "inputs": [feature.as_dict() for feature in self.inputs],
            "outputs": [feature.as_dict() for feature in self.outputs],
        }


@dataclass(frozen=True, slots=True)
class BackendOutput:
    """Raw named output values returned by an inference backend."""

    values: Mapping[str, Any]


@dataclass(slots=True)
class LoadedModel:
    """One loaded backend instance."""

    descriptor: ModelDescriptor
    options: BackendOptions
    inspection: BackendInspection
    handle: Any
    instance_id: str = ""
    closed: bool = False

    def __post_init__(self) -> None:
        if not self.instance_id:
            self.instance_id = uuid4().hex


@dataclass(frozen=True, slots=True)
class Detection:
    """One validated detection in source-image pixel coordinates."""

    box_xyxy: tuple[float, float, float, float]
    class_id: int
    label: str
    confidence: float
    model_id: str = ""
    model_sha256: str = ""
    frame_sequence: int | None = None
    detected_at: datetime | None = None

    @property
    def x1(self) -> float:
        """Return the left coordinate."""
        return self.box_xyxy[0]

    @property
    def y1(self) -> float:
        """Return the top coordinate."""
        return self.box_xyxy[1]

    @property
    def x2(self) -> float:
        """Return the right coordinate."""
        return self.box_xyxy[2]

    @property
    def y2(self) -> float:
        """Return the bottom coordinate."""
        return self.box_xyxy[3]


@dataclass(frozen=True, slots=True)
class DetectionBatch:
    """Validated detections and count of rejected or filtered candidates."""

    detections: tuple[Detection, ...]
    rejected_count: int = 0


class InferenceBackend(Protocol):
    """Backend contract used by the detector and model registry."""

    def load(self, model: ModelDescriptor, options: BackendOptions) -> LoadedModel:
        """Load a new model instance."""

    def predict(self, model: LoadedModel, model_input: object) -> BackendOutput:
        """Run one prediction and return named raw outputs."""

    def close(self, model: LoadedModel) -> None:
        """Release one model instance."""


class DetectionAdapter(Protocol):
    """Model-family preprocessing and output decoding contract."""

    def preprocess(self, image: StillImage, manifest: ModelManifest) -> PreparedInput:
        """Prepare one source image for the model."""

    def decode(self, output: BackendOutput, transform: ImageTransform) -> DetectionBatch:
        """Decode and validate backend output in source pixels."""


class StillImageDetector(Protocol):
    """Backend-neutral contract for one still-image detection request."""

    def detect(
        self,
        image: StillImage,
        model: ModelDescriptor,
        options: BackendOptions | None = None,
    ) -> DetectionBatch:
        """Run preprocessing, prediction, decoding, and validation."""
