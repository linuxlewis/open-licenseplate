from __future__ import annotations

import platform
import sys
from pathlib import Path

import numpy as np
import pytest

from open_licenseplate.inference import (
    BackendOptions,
    CoreMLBackend,
    ModelDescriptor,
    StillImage,
    adapter_for_manifest,
    compare_manifest_to_inspection,
)
from open_licenseplate.models.archive import compute_artifact_sha256
from open_licenseplate.models.manifest import parse_manifest


@pytest.mark.macos
@pytest.mark.skipif(
    sys.platform != "darwin" or platform.machine() not in {"arm64", "aarch64"},
    reason="Apple Silicon macOS is required",
)
def test_coreml_backend_loads_and_predicts_a_generated_package(tmp_path: Path) -> None:
    """Build a small local model so the default macOS acceptance test is self-contained."""
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb

    @mb.program(input_specs=[mb.TensorSpec(shape=(1, 3, 64, 64))])
    def constant_detector(image: object):
        del image
        coordinates = mb.const(val=np.array([[8.0, 12.0, 40.0, 32.0]], dtype=np.float32))
        confidence = mb.const(val=np.array([[0.9]], dtype=np.float32))
        return (
            mb.identity(x=coordinates, name="coordinates"),
            mb.identity(x=confidence, name="confidence"),
        )

    converted = ct.convert(
        constant_detector,
        inputs=[
            ct.ImageType(
                name="image",
                shape=(1, 3, 64, 64),
                color_layout=ct.colorlayout.RGB,
            )
        ],
        convert_to="mlprogram",
    )
    package_path = tmp_path / "deterministic.mlpackage"
    converted.save(str(package_path))
    checksum = compute_artifact_sha256(package_path)
    manifest = parse_manifest(
        {
            "schema_version": 1,
            "id": "generated-coreml-fixture",
            "display_name": "Generated Core ML fixture",
            "task": "object_detection",
            "backend": "coreml",
            "adapter": "ultralytics_yolo_nms",
            "artifact": package_path.name,
            "artifact_sha256": checksum,
            "input": {
                "name": "image",
                "kind": "image",
                "width": 64,
                "height": 64,
                "color_space": "rgb",
            },
            "preprocessing": {"resize": "none"},
            "outputs": {
                "boxes": "coordinates",
                "scores": "confidence",
                "box_format": "xyxy",
                "coordinate_space": "model_pixels",
            },
            "labels": ["license_plate"],
            "defaults": {"confidence_threshold": 0.35, "iou_threshold": 0.45},
        }
    )
    descriptor = ModelDescriptor(
        model_id=manifest.model_id,
        artifact_path=str(package_path),
        artifact_sha256=checksum,
        manifest=manifest,
    )

    backend = CoreMLBackend()
    loaded = backend.load(descriptor, BackendOptions())
    try:
        compare_manifest_to_inspection(manifest, loaded.inspection)
        assert [feature.name for feature in loaded.inspection.inputs] == ["image"]
        assert [feature.name for feature in loaded.inspection.outputs] == [
            "coordinates",
            "confidence",
        ]
        adapter = adapter_for_manifest(manifest)
        prepared = adapter.preprocess(
            StillImage(np.zeros((64, 64, 3), dtype=np.uint8)),
            manifest,
        )
        output = backend.predict(loaded, prepared)
        result = adapter.decode(output, prepared.transform)
    finally:
        backend.close(loaded)

    assert len(result.detections) == 1
    assert result.detections[0].box_xyxy == (8.0, 12.0, 40.0, 32.0)
    assert result.detections[0].confidence == pytest.approx(0.9)
