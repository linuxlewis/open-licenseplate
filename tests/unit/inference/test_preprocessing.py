from __future__ import annotations

from pathlib import Path

import numpy as np

from model_helpers import create_model_fixture
from open_licenseplate.inference import StillImage, UltralyticsYoloNmsAdapter
from open_licenseplate.models.manifest import parse_manifest


def test_letterbox_preprocessing_records_deterministic_geometry(tmp_path: Path) -> None:
    manifest_path, _archive_path, _raw_manifest = create_model_fixture(tmp_path)
    manifest = parse_manifest(manifest_path.read_bytes())
    image = StillImage(np.zeros((720, 1280, 3), dtype=np.uint8))

    prepared = UltralyticsYoloNmsAdapter().preprocess(image, manifest)

    assert prepared.value.shape == (640, 640, 3)
    assert prepared.value.dtype == np.uint8
    assert prepared.transform.scale_x == 0.5
    assert prepared.transform.scale_y == 0.5
    assert prepared.transform.pad_left == 0
    assert prepared.transform.pad_top == 140
    assert prepared.transform.resized_width == 640
    assert prepared.transform.resized_height == 360
    assert np.all(prepared.value[:140] == 114)
    assert np.all(prepared.value[500:] == 114)


def test_color_conversion_is_applied_before_resize(tmp_path: Path) -> None:
    manifest_path, _archive_path, raw_manifest = create_model_fixture(tmp_path)
    raw_manifest["input"]["color_space"] = "rgb"
    manifest = parse_manifest(raw_manifest)
    bgr = np.zeros((640, 640, 3), dtype=np.uint8)
    bgr[:, :, 0] = 10
    bgr[:, :, 2] = 240

    prepared = UltralyticsYoloNmsAdapter().preprocess(
        StillImage(bgr, color_space="bgr"),
        manifest,
    )

    assert tuple(prepared.value[10, 10]) == (240, 0, 10)
