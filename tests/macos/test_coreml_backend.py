from __future__ import annotations

import os
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
)
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
