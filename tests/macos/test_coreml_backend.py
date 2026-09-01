from __future__ import annotations

import os
import platform
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageEnhance

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
def test_coreml_load_and_predict_fixture_when_available() -> None:
    """Run a real package contract check when a developer supplies a fixture."""
    if sys.platform != "darwin" or platform.machine() not in {"arm64", "aarch64"}:
        pytest.skip("Apple Silicon macOS is required")
    package_value = os.environ.get("OPEN_LICENSEPLATE_COREML_PACKAGE")
    manifest_value = os.environ.get("OPEN_LICENSEPLATE_COREML_MANIFEST")
    if not package_value or not manifest_value:
        pytest.skip(
            "Set OPEN_LICENSEPLATE_COREML_PACKAGE and "
            "OPEN_LICENSEPLATE_COREML_MANIFEST to run the fixture test"
        )
    coremltools = pytest.importorskip("coremltools")
    del coremltools

    package_path = Path(package_value)
    manifest_path = Path(manifest_value)
    manifest = parse_manifest(manifest_path.read_bytes())
    if package_path.name != manifest.artifact:
        pytest.fail("Core ML fixture package name does not match manifest artifact")
    descriptor = ModelDescriptor(
        model_id=manifest.model_id,
        artifact_path=str(package_path),
        artifact_sha256=manifest.artifact_sha256,
        manifest=manifest,
    )
    input_values = manifest.raw["input"]
    color_space = str(input_values["color_space"])
    if color_space == "grayscale":
        pixels = np.zeros(
            (int(input_values["height"]), int(input_values["width"])),
            dtype=np.uint8,
        )
    else:
        pixels = np.zeros(
            (int(input_values["height"]), int(input_values["width"]), 3),
            dtype=np.uint8,
        )

    backend = CoreMLBackend()
    loaded = backend.load(descriptor, BackendOptions())
    try:
        adapter = adapter_for_manifest(manifest)
        prepared = adapter.preprocess(
            StillImage(pixels=pixels, color_space=color_space),
            manifest,
        )
        output = backend.predict(loaded, prepared)
        result = adapter.decode(output, prepared.transform)
    finally:
        backend.close(loaded)

    assert result.rejected_count >= 0


@pytest.mark.macos
@pytest.mark.skipif(
    sys.platform != "darwin" or platform.machine() not in {"arm64", "aarch64"},
    reason="Apple Silicon macOS is required",
)
def test_catalog_coreml_receives_application_confidence_before_postprocessing() -> None:
    """Prove a sub-0.25 application threshold changes the raw catalog output."""
    pytest.importorskip("coremltools")
    repository_root = Path(__file__).parents[2]
    package_path = (
        repository_root / "tests" / "fixtures" / "catalog" / "license-plate-yolov11n.mlpackage"
    )
    manifest_path = repository_root / "model-catalog" / "manifests" / "license-plate-yolov11n.json"
    manifest = parse_manifest(manifest_path.read_bytes())
    assert package_path.name == manifest.artifact
    assert compute_artifact_sha256(package_path) == manifest.artifact_sha256

    descriptor = ModelDescriptor(
        model_id=manifest.model_id,
        artifact_path=str(package_path),
        artifact_sha256=manifest.artifact_sha256,
        manifest=manifest,
    )
    backend = CoreMLBackend()
    loaded = backend.load(
        descriptor,
        BackendOptions(compute_units="cpu_only"),
    )
    try:
        compare_manifest_to_inspection(manifest, loaded.inspection)
        adapter = adapter_for_manifest(manifest)
        source = ImageEnhance.Brightness(
            Image.open(repository_root / "tests" / "fixtures" / "still" / "plate.png").convert(
                "RGB"
            )
        ).enhance(1.2)
        source = source.resize((640, 640))
        prepared = adapter.preprocess(
            StillImage(np.asarray(source, dtype=np.uint8)),
            manifest,
        )
        low_confidence_prepared = replace(
            prepared,
            transform=replace(
                prepared.transform,
                confidence_threshold=0.20,
            ),
        )
        default_output = backend.predict(loaded, prepared)
        low_confidence_output = backend.predict(loaded, low_confidence_prepared)
    finally:
        backend.close(loaded)

    score_name = manifest.raw["outputs"]["scores"]
    assert isinstance(score_name, str)
    default_scores = np.asarray(default_output.values[score_name])
    low_confidence_scores = np.asarray(low_confidence_output.values[score_name])
    assert prepared.transform.confidence_threshold == pytest.approx(0.25)
    assert low_confidence_prepared.transform.confidence_threshold == pytest.approx(0.20)
    assert default_scores.shape == (0, 1)
    assert low_confidence_scores.shape == (1, 1)
    assert 0 < float(low_confidence_scores[0, 0]) < 0.25
