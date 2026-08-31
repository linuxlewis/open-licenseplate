"""Bounded and validated still-image decoding."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from io import BytesIO

import numpy as np
from PIL import Image

from .contract import StillImage

MAX_STILL_IMAGE_BYTES = 8 * 1024 * 1024
MAX_STILL_IMAGE_PIXELS = 25_000_000
MAX_STILL_IMAGE_DIMENSION = 8192


class StillImageDecodeError(ValueError):
    """Raised when uploaded image data is not safe to decode."""


@dataclass(frozen=True, slots=True)
class DecodedStillImage:
    """Decoded image plus the original bytes used for exact display."""

    image: StillImage
    raw_bytes: bytes
    content_type: str


def decode_still_image(raw_bytes: bytes) -> DecodedStillImage:
    """Decode one bounded JPEG or PNG without storing the upload."""
    if not raw_bytes:
        raise StillImageDecodeError("image upload is empty")
    if len(raw_bytes) > MAX_STILL_IMAGE_BYTES:
        raise StillImageDecodeError("image upload exceeds the size limit")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(raw_bytes)) as probe:
                image_format = probe.format
                width, height = probe.size
                _validate_dimensions(width, height)
                if image_format not in {"JPEG", "PNG"}:
                    raise StillImageDecodeError("image format must be JPEG or PNG")
                _validate_single_frame_count(probe)
                probe.verify()

            with Image.open(BytesIO(raw_bytes)) as decoded:
                if decoded.format != image_format:
                    raise StillImageDecodeError("image format could not be validated")
                _validate_dimensions(*decoded.size)
                _validate_single_frame_geometry(decoded)
                decoded.load()
                pixels = np.asarray(decoded.convert("RGB"), dtype=np.uint8).copy()
    except StillImageDecodeError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise StillImageDecodeError("image dimensions exceed the safe limit") from None
    except (OSError, SyntaxError, ValueError, TypeError):
        raise StillImageDecodeError("image data is malformed or unsupported") from None

    content_type = "image/jpeg" if image_format == "JPEG" else "image/png"
    return DecodedStillImage(
        image=StillImage(pixels=pixels, color_space="rgb"),
        raw_bytes=raw_bytes,
        content_type=content_type,
    )


def _validate_dimensions(width: int, height: int) -> None:
    if (
        width <= 0
        or height <= 0
        or width > MAX_STILL_IMAGE_DIMENSION
        or height > MAX_STILL_IMAGE_DIMENSION
        or width * height > MAX_STILL_IMAGE_PIXELS
    ):
        raise StillImageDecodeError("image dimensions exceed the safe limit")


def _validate_single_frame_geometry(image: Image.Image) -> None:
    """Reject animation and display transforms that change source geometry."""
    _validate_single_frame_count(image)
    try:
        orientation = image.getexif().get(274, 1)
    except (OSError, ValueError, TypeError):
        raise StillImageDecodeError("image EXIF orientation is invalid") from None
    if orientation != 1:
        raise StillImageDecodeError("image EXIF orientation must be identity")


def _validate_single_frame_count(image: Image.Image) -> None:
    """Reject animation before the decoder can select a different frame."""
    frame_count = getattr(image, "n_frames", 1)
    if not isinstance(frame_count, int) or frame_count != 1:
        raise StillImageDecodeError("image must contain exactly one frame")
