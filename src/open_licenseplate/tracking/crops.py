"""Bounded crop capture and deterministic M4-B quality scoring."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np

from ..inference.contract import Detection

CROP_QUALITY_SCORING_VERSION = "m4b-crop-score-v1"
MAX_CROP_PIXELS = 262_144


@dataclass(frozen=True, slots=True)
class CropQuality:
    """One versioned deterministic crop score with component evidence."""

    score: float
    version: str
    evidence: dict[str, Any]


@dataclass(slots=True)
class CropCandidate:
    """One bounded source-pixel crop retained for a live track."""

    pixels: np.ndarray | None
    color_space: str
    box_xyxy: tuple[float, float, float, float]
    plate_width_px: float
    plate_height_px: float
    source_frame_sequence: int
    source_timestamp: datetime
    detection_confidence: float
    quality: CropQuality

    def __post_init__(self) -> None:
        if self.pixels is not None:
            _validate_crop_pixels(self.pixels)
        if self.color_space not in {"rgb", "bgr", "grayscale"}:
            raise ValueError("crop color_space must be rgb, bgr, or grayscale")
        if type(self.source_frame_sequence) is not int or self.source_frame_sequence < 0:
            raise ValueError("source_frame_sequence must be non-negative")
        if self.source_timestamp.tzinfo is None or self.source_timestamp.utcoffset() is None:
            raise ValueError("source_timestamp must be timezone-aware")
        self.source_timestamp = self.source_timestamp.astimezone(UTC)
        if not math.isfinite(self.detection_confidence) or not 0 <= self.detection_confidence <= 1:
            raise ValueError("detection_confidence must be between 0 and 1")

    @property
    def quality_score(self) -> float:
        """Return the deterministic score used for ranking."""
        return self.quality.score

    @property
    def quality_scoring_version(self) -> str:
        """Return the scorer version stored with committed metadata."""
        return self.quality.version

    def rank_key(self) -> tuple[float, float, float, float, int, str, tuple[float, ...]]:
        """Return a deterministic best-first ordering key."""
        return (
            -self.quality_score,
            -self.detection_confidence,
            -self.plate_width_px,
            -self.plate_height_px,
            self.source_frame_sequence,
            self.source_timestamp.isoformat(),
            self.box_xyxy,
        )

    def release(self) -> None:
        """Release the retained crop pixels after selection or eviction."""
        self.pixels = None


def capture_crop_candidate(
    *,
    source_pixels: object,
    pixel_format: str,
    frame_width: int,
    frame_height: int,
    frame_sequence: int,
    source_timestamp: datetime,
    detection: Detection,
) -> CropCandidate | None:
    """Copy one bounded source crop and score it at observation time."""
    pixels = _source_pixels(source_pixels, pixel_format)
    x1, y1, x2, y2 = detection.box_xyxy
    left = max(0, min(frame_width - 1, math.floor(x1)))
    top = max(0, min(frame_height - 1, math.floor(y1)))
    right = max(left + 1, min(frame_width, math.ceil(x2)))
    bottom = max(top + 1, min(frame_height, math.ceil(y2)))
    if (right - left) * (bottom - top) > MAX_CROP_PIXELS:
        return None
    crop = np.ascontiguousarray(pixels[top:bottom, left:right]).copy()

    color_space = _color_space(pixel_format)
    box_xyxy = (float(x1), float(y1), float(x2), float(y2))
    quality = score_crop_quality(
        crop,
        color_space=color_space,
        box_xyxy=box_xyxy,
        frame_width=frame_width,
        frame_height=frame_height,
        detection_confidence=detection.confidence,
    )
    return CropCandidate(
        pixels=crop,
        color_space=color_space,
        box_xyxy=box_xyxy,
        plate_width_px=max(0.0, float(x2 - x1)),
        plate_height_px=max(0.0, float(y2 - y1)),
        source_frame_sequence=frame_sequence,
        source_timestamp=source_timestamp,
        detection_confidence=float(detection.confidence),
        quality=quality,
    )


def score_crop_quality(
    pixels: np.ndarray,
    *,
    color_space: str,
    box_xyxy: Sequence[float],
    frame_width: int,
    frame_height: int,
    detection_confidence: float,
) -> CropQuality:
    """Score one crop using fixed M4-B components and weights.

    The score is a weighted sum of confidence, plate size, sharpness, exposure,
    contrast, clipping, and distance from image boundaries. Every component is
    included in the JSON-safe evidence so future scorers can be compared.
    """
    array = _as_uint8(pixels)
    grayscale = _grayscale(array, color_space)
    mean = float(np.mean(grayscale))
    contrast_raw = float(np.std(grayscale))
    sharpness_raw = _gradient_energy(grayscale)
    clipped_fraction = float(np.mean((array <= 2) | (array >= 253)))
    width = float(box_xyxy[2] - box_xyxy[0])
    height = float(box_xyxy[3] - box_xyxy[1])
    boundary_distance = max(
        0.0,
        min(
            float(box_xyxy[0]),
            float(box_xyxy[1]),
            float(frame_width - box_xyxy[2]),
            float(frame_height - box_xyxy[3]),
        ),
    )

    components = {
        "detection_confidence": _clamp01(detection_confidence),
        "plate_width": _clamp01(width / 160.0),
        "plate_height": _clamp01(height / 48.0),
        "sharpness": _clamp01(sharpness_raw / 900.0),
        "exposure": _clamp01(1.0 - abs(mean - 127.5) / 127.5),
        "contrast": _clamp01(contrast_raw / 64.0),
        "clipping": _clamp01(1.0 - clipped_fraction / 0.05),
        "boundary_distance": _clamp01(boundary_distance / 32.0),
    }
    weights = {
        "detection_confidence": 0.25,
        "plate_width": 0.12,
        "plate_height": 0.08,
        "sharpness": 0.20,
        "exposure": 0.10,
        "contrast": 0.10,
        "clipping": 0.08,
        "boundary_distance": 0.07,
    }
    score = round(sum(components[name] * weights[name] for name in weights), 6)
    evidence: dict[str, Any] = {
        "frame_width": frame_width,
        "frame_height": frame_height,
        "crop_width": int(array.shape[1]),
        "crop_height": int(array.shape[0]),
        "plate_width_px": round(width, 6),
        "plate_height_px": round(height, 6),
        "mean_luma": round(mean, 6),
        "contrast_stddev": round(contrast_raw, 6),
        "sharpness_gradient_energy": round(sharpness_raw, 6),
        "clipped_fraction": round(clipped_fraction, 6),
        "boundary_distance_px": round(boundary_distance, 6),
        "components": {name: round(value, 6) for name, value in components.items()},
        "weights": weights,
    }
    return CropQuality(score=score, version=CROP_QUALITY_SCORING_VERSION, evidence=evidence)


def _source_pixels(source_pixels: object, pixel_format: str) -> np.ndarray:
    array = np.asarray(source_pixels)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("source pixels must be a three-channel array")
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError("source pixels must be finite numeric values")
    if array.shape[0] <= 0 or array.shape[1] <= 0:
        raise ValueError("source pixels must have positive dimensions")
    if pixel_format.casefold() not in {"bgr24", "rgb24", "bgr", "rgb"}:
        raise ValueError("source pixel format is not supported for crop capture")
    return array


def _color_space(pixel_format: str) -> str:
    return "rgb" if pixel_format.casefold().startswith("rgb") else "bgr"


def _validate_crop_pixels(pixels: np.ndarray) -> None:
    if pixels.ndim != 3 or pixels.shape[2] not in {1, 3}:
        raise ValueError("crop pixels must have one or three channels")
    if pixels.shape[0] <= 0 or pixels.shape[1] <= 0:
        raise ValueError("crop pixels must have positive dimensions")


def _as_uint8(pixels: np.ndarray) -> np.ndarray:
    array = np.asarray(pixels)
    if array.dtype == np.uint8:
        return array
    if np.issubdtype(array.dtype, np.floating) and float(np.max(array)) <= 1.0:
        array = array * 255.0
    return np.asarray(np.clip(array, 0, 255).astype(np.uint8))


def _grayscale(pixels: np.ndarray, color_space: str) -> np.ndarray:
    if pixels.ndim == 2 or pixels.shape[2] == 1:
        return pixels.reshape(pixels.shape[0], pixels.shape[1]).astype(np.float64)
    channels = pixels.astype(np.float64)
    if color_space == "bgr":
        blue, green, red = channels[..., 0], channels[..., 1], channels[..., 2]
    else:
        red, green, blue = channels[..., 0], channels[..., 1], channels[..., 2]
    return 0.114 * blue + 0.587 * green + 0.299 * red


def _gradient_energy(grayscale: np.ndarray) -> float:
    if grayscale.shape[0] < 2 and grayscale.shape[1] < 2:
        return 0.0
    vertical = np.diff(grayscale, axis=0)
    horizontal = np.diff(grayscale, axis=1)
    values = [float(np.mean(np.square(vertical)))] if vertical.size else []
    if horizontal.size:
        values.append(float(np.mean(np.square(horizontal))))
    return float(sum(values) / len(values))


def _clamp01(value: float) -> float:
    if not math.isfinite(float(value)):
        return 0.0
    return min(1.0, max(0.0, float(value)))


__all__ = [
    "CROP_QUALITY_SCORING_VERSION",
    "CropCandidate",
    "CropQuality",
    "MAX_CROP_PIXELS",
    "capture_crop_candidate",
    "score_crop_quality",
]
