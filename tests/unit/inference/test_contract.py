from __future__ import annotations

import numpy as np
import pytest

from open_licenseplate.inference import (
    BackendOptions,
    ComputeUnit,
    ImageTransform,
    StillImage,
)


def test_still_image_requires_explicit_valid_channels() -> None:
    image = StillImage(np.zeros((12, 20, 3), dtype=np.uint8), color_space="BGR")

    assert image.width == 20
    assert image.height == 12
    assert image.color_space == "bgr"

    with pytest.raises(ValueError, match="three channels"):
        StillImage(np.zeros((12, 20), dtype=np.uint8), color_space="rgb")


def test_compute_unit_values_and_display_names_are_stable() -> None:
    assert ComputeUnit.parse("CPU and Neural Engine") is ComputeUnit.CPU_AND_NE
    assert BackendOptions(compute_units="cpu_only").compute_units is ComputeUnit.CPU_ONLY
    assert ComputeUnit.CPU_AND_GPU.display_name == "CPU and GPU"


def test_image_transform_maps_boxes_in_both_directions() -> None:
    transform = ImageTransform(
        source_width=1280,
        source_height=720,
        model_width=640,
        model_height=640,
        resize="letterbox",
        scale_x=0.5,
        scale_y=0.5,
        box_format="xyxy",
        coordinate_space="model_pixels",
        pad_top=140,
        resized_width=640,
        resized_height=360,
    )

    source_box = (200.0, 120.0, 600.0, 320.0)
    model_box = transform.source_to_model_box(source_box)

    assert model_box == (100.0, 200.0, 300.0, 300.0)
    assert transform.model_to_source_box(model_box) == source_box
