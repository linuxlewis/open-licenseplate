from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from open_licenseplate.inference import (
    MAX_STILL_IMAGE_BYTES,
    StillImageDecodeError,
    decode_still_image,
    image_io,
)

FIXTURE_ROOT = Path(__file__).parents[2] / "fixtures" / "still"


def _gif_bytes() -> bytes:
    output = BytesIO()
    Image.new("RGB", (2, 2), (0, 0, 0)).save(output, format="GIF")
    return output.getvalue()


def _animated_png_bytes() -> bytes:
    output = BytesIO()
    first = Image.new("RGB", (2, 2), (0, 0, 0))
    second = Image.new("RGB", (2, 2), (255, 255, 255))
    first.save(
        output,
        format="PNG",
        save_all=True,
        append_images=[second],
        duration=100,
        loop=0,
    )
    return output.getvalue()


def _rotated_jpeg_bytes() -> bytes:
    output = BytesIO()
    exif = Image.Exif()
    exif[274] = 6
    exif[270] = "PRIVATE-EXIF-CONTENT"
    Image.new("RGB", (8, 4), (20, 30, 40)).save(output, format="JPEG", exif=exif)
    return output.getvalue()


def test_decode_preserves_original_bytes_and_source_geometry() -> None:
    raw = (FIXTURE_ROOT / "plate.png").read_bytes()

    decoded = decode_still_image(raw)

    assert decoded.raw_bytes == raw
    assert decoded.content_type == "image/png"
    assert decoded.image.width == 320
    assert decoded.image.height == 180
    assert decoded.image.color_space == "rgb"


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (b"not an image", "malformed or unsupported"),
        (_gif_bytes(), "must be JPEG or PNG"),
    ],
)
def test_decode_rejects_malformed_and_unsupported_data(raw: bytes, message: str) -> None:
    with pytest.raises(StillImageDecodeError, match=message):
        decode_still_image(raw)


def test_decode_rejects_multi_frame_png() -> None:
    with pytest.raises(StillImageDecodeError, match="exactly one frame"):
        decode_still_image(_animated_png_bytes())


def test_decode_rejects_non_identity_exif_orientation_without_transposing() -> None:
    with pytest.raises(StillImageDecodeError, match="orientation must be identity"):
        decode_still_image(_rotated_jpeg_bytes())


def test_decode_rejects_oversized_upload_before_image_parsing() -> None:
    with pytest.raises(StillImageDecodeError, match="size limit"):
        decode_still_image(b"x" * (MAX_STILL_IMAGE_BYTES + 1))


def test_decode_rejects_decompression_bomb_dimensions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(image_io, "MAX_STILL_IMAGE_PIXELS", 100)

    with pytest.raises(StillImageDecodeError, match="dimensions"):
        decode_still_image((FIXTURE_ROOT / "no-plate.png").read_bytes())
