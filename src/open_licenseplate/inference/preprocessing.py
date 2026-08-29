"""Deterministic still-image preparation and letterbox geometry."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from PIL import Image

from ..models.manifest import ModelManifest
from .contract import ImageTransform, PreparedInput, PreprocessingError, StillImage

LETTERBOX_PADDING_VALUE = 114


def prepare_image(image: StillImage, manifest: ModelManifest) -> PreparedInput:
    """Convert, resize, and pad an image using the manifest contract."""
    input_values = manifest.raw["input"]
    preprocessing_values = manifest.raw["preprocessing"]
    if not isinstance(input_values, dict) or not isinstance(preprocessing_values, dict):
        raise PreprocessingError("manifest input or preprocessing section is invalid")

    input_name = input_values["name"]
    model_width = int(input_values["width"])
    model_height = int(input_values["height"])
    target_color_space = str(input_values["color_space"]).casefold()
    resize = str(preprocessing_values["resize"]).casefold()
    source = _convert_color_space(image.pixels, image.color_space, target_color_space)

    if resize == "none":
        if image.width != model_width or image.height != model_height:
            raise PreprocessingError(
                "preprocessing.resize=none requires the source image to match model dimensions"
            )
        transformed = source
        transform = ImageTransform(
            source_width=image.width,
            source_height=image.height,
            model_width=model_width,
            model_height=model_height,
            resize=resize,
            scale_x=1.0,
            scale_y=1.0,
            resized_width=model_width,
            resized_height=model_height,
            frame_sequence=image.frame_sequence,
            captured_at=image.captured_at,
        )
    elif resize == "stretch":
        transformed = _resize(source, model_width, model_height, target_color_space)
        transform = ImageTransform(
            source_width=image.width,
            source_height=image.height,
            model_width=model_width,
            model_height=model_height,
            resize=resize,
            scale_x=model_width / image.width,
            scale_y=model_height / image.height,
            resized_width=model_width,
            resized_height=model_height,
            frame_sequence=image.frame_sequence,
            captured_at=image.captured_at,
        )
    elif resize == "letterbox":
        scale = min(model_width / image.width, model_height / image.height)
        resized_width = _scaled_dimension(image.width, scale)
        resized_height = _scaled_dimension(image.height, scale)
        resized = _resize(source, resized_width, resized_height, target_color_space)
        pad_left = (model_width - resized_width) // 2
        pad_top = (model_height - resized_height) // 2
        transformed = _letterbox(
            resized,
            model_width=model_width,
            model_height=model_height,
            pad_left=pad_left,
            pad_top=pad_top,
            color_space=target_color_space,
        )
        transform = ImageTransform(
            source_width=image.width,
            source_height=image.height,
            model_width=model_width,
            model_height=model_height,
            resize=resize,
            scale_x=resized_width / image.width,
            scale_y=resized_height / image.height,
            pad_left=pad_left,
            pad_top=pad_top,
            resized_width=resized_width,
            resized_height=resized_height,
            frame_sequence=image.frame_sequence,
            captured_at=image.captured_at,
        )
    else:
        raise PreprocessingError(f"unsupported preprocessing.resize value: {resize}")

    return PreparedInput(
        value=np.ascontiguousarray(transformed),
        input_name=str(input_name),
        color_space=target_color_space,
        transform=transform,
    )


def enrich_transform(
    prepared: PreparedInput,
    manifest: ModelManifest,
) -> PreparedInput:
    """Attach immutable manifest decoding values to preprocessing metadata."""
    input_values = manifest.raw["input"]
    outputs = manifest.raw["outputs"]
    defaults = manifest.raw["defaults"]
    if not all(isinstance(value, dict) for value in (input_values, outputs, defaults)):
        raise PreprocessingError("manifest sections required for preprocessing are invalid")

    labels = manifest.raw["labels"]
    if not isinstance(labels, list) or not all(isinstance(label, str) for label in labels):
        raise PreprocessingError("manifest labels are invalid")
    output_names = tuple(
        (role, str(name))
        for role, name in outputs.items()
        if role in {"boxes", "scores", "classes"} and isinstance(name, str)
    )
    raw_output_name = outputs.get("raw")
    class_output_name = outputs.get("classes")
    raw_has_objectness = outputs.get("raw_has_objectness", False)
    if not isinstance(raw_has_objectness, bool):
        raise PreprocessingError("manifest outputs.raw_has_objectness must be boolean")
    transform = replace(
        prepared.transform,
        model_id=manifest.model_id,
        model_sha256=manifest.artifact_sha256,
        labels=tuple(labels),
        confidence_threshold=float(defaults["confidence_threshold"]),
        iou_threshold=float(defaults["iou_threshold"]),
        box_format=str(outputs.get("box_format", "xyxy")).casefold(),
        output_names=output_names,
        raw_output_name=raw_output_name if isinstance(raw_output_name, str) else None,
        class_output_name=class_output_name if isinstance(class_output_name, str) else None,
        raw_has_objectness=raw_has_objectness,
    )
    return replace(prepared, transform=transform)


def _scaled_dimension(value: int, scale: float) -> int:
    """Round resized dimensions with one deterministic half-up rule."""
    return max(1, int(np.floor(value * scale + 0.5)))


def _convert_color_space(
    pixels: np.ndarray,
    source_color_space: str,
    target_color_space: str,
) -> np.ndarray:
    source = _as_uint8(pixels)
    if source_color_space == target_color_space:
        if target_color_space == "grayscale" and source.ndim == 3:
            return source[:, :, 0]
        return source
    if source_color_space == "grayscale":
        gray = source[:, :, 0] if source.ndim == 3 else source
        if target_color_space in {"rgb", "bgr"}:
            return np.repeat(gray[:, :, None], 3, axis=2)
    if target_color_space == "grayscale" and source_color_space in {"rgb", "bgr"}:
        return np.rint(source.astype(np.float32).mean(axis=2)).astype(np.uint8)
    if {source_color_space, target_color_space} == {"rgb", "bgr"}:
        return source[:, :, ::-1].copy()
    raise PreprocessingError(
        f"cannot convert image color space from {source_color_space} to {target_color_space}"
    )


def _as_uint8(pixels: np.ndarray) -> np.ndarray:
    if not np.issubdtype(pixels.dtype, np.number):
        raise PreprocessingError("image pixels must be numeric")
    if not np.isfinite(pixels).all():
        raise PreprocessingError("image pixels must contain finite values")
    if pixels.dtype == np.uint8:
        return pixels
    return np.clip(np.rint(pixels), 0, 255).astype(np.uint8)


def _resize(
    pixels: np.ndarray,
    width: int,
    height: int,
    color_space: str,
) -> np.ndarray:
    image = Image.fromarray(pixels, mode=_pil_mode(color_space))
    resized = image.resize((width, height), resample=Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


def _letterbox(
    resized: np.ndarray,
    *,
    model_width: int,
    model_height: int,
    pad_left: int,
    pad_top: int,
    color_space: str,
) -> np.ndarray:
    if color_space == "grayscale":
        canvas = np.full(
            (model_height, model_width),
            LETTERBOX_PADDING_VALUE,
            dtype=np.uint8,
        )
    else:
        canvas = np.full(
            (model_height, model_width, 3),
            LETTERBOX_PADDING_VALUE,
            dtype=np.uint8,
        )
    resized_height, resized_width = resized.shape[:2]
    canvas[pad_top : pad_top + resized_height, pad_left : pad_left + resized_width] = resized
    return canvas


def _pil_mode(color_space: str) -> str:
    return "L" if color_space == "grayscale" else "RGB"
