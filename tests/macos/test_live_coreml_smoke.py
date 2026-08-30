from __future__ import annotations

import os
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from open_licenseplate.cameras.service import prepare_camera_config
from open_licenseplate.capture import CameraRuntime, FakeFrameSource
from open_licenseplate.inference import CoreMLBackend, ModelDescriptor
from open_licenseplate.live import LivePipelineCoordinator, validate_display_unit
from open_licenseplate.models.manifest import parse_manifest


@pytest.mark.macos
@pytest.mark.m3_acceptance
@pytest.mark.skipif(
    sys.platform != "darwin" or platform.machine() not in {"arm64", "aarch64"},
    reason="Apple Silicon macOS is required",
)
def test_m3_live_coreml_fixture_smoke() -> None:
    """Run one real live frame only when all explicit fixtures are supplied."""
    package_value = os.environ.get("OPEN_LICENSEPLATE_LIVE_COREML_PACKAGE")
    manifest_value = os.environ.get("OPEN_LICENSEPLATE_LIVE_COREML_MANIFEST")
    frame_value = os.environ.get("OPEN_LICENSEPLATE_LIVE_COREML_FRAME")
    if not package_value or not manifest_value or not frame_value:
        pytest.skip(
            "Set OPEN_LICENSEPLATE_LIVE_COREML_PACKAGE, "
            "OPEN_LICENSEPLATE_LIVE_COREML_MANIFEST, and "
            "OPEN_LICENSEPLATE_LIVE_COREML_FRAME to run the live Core ML fixture"
        )
    pytest.importorskip("coremltools")

    package_path = Path(package_value)
    manifest_path = Path(manifest_value)
    frame_path = Path(frame_value)
    if not package_path.is_dir():
        pytest.fail("live Core ML package fixture is not a directory")
    try:
        manifest = parse_manifest(manifest_path.read_bytes())
        with Image.open(frame_path) as image:
            pixels = np.asarray(image.convert("RGB")).copy()
    except Exception:
        pytest.fail("live Core ML fixture data is invalid", pytrace=False)
    if package_path.name != manifest.artifact:
        pytest.fail("live Core ML package name does not match the manifest")

    descriptor = ModelDescriptor(
        model_id=manifest.model_id,
        artifact_path=str(package_path),
        artifact_sha256=manifest.artifact_sha256,
        manifest=manifest,
    )
    source = FakeFrameSource(
        frames=(pixels,),
        camera_id="live-coreml-fixture",
        pixel_format="rgb24",
        repeat=True,
        read_interval_seconds=0.01,
    )
    runtime = CameraRuntime(
        lambda _camera, _camera_id: source,
        poll_interval_seconds=0.002,
    )
    coordinator = LivePipelineCoordinator(
        runtime,
        CoreMLBackend,
        poll_interval_seconds=0.002,
        display_max_fps=30,
    )
    subscription: Any | None = None
    try:
        coordinator.start(
            "live-coreml-fixture",
            prepare_camera_config(
                name="Live Core ML fixture",
                rtsp_url="rtsp://fixture.local/live",
            ),
            descriptor,
        )
        coordinator.wait_for_state("running")
        subscription = coordinator.subscribe_display()
        assert subscription is not None
        unit = subscription.get(timeout=30)
        assert unit is not None
        validate_display_unit(unit)
        assert unit.metadata["model_id"] == manifest.model_id
        assert unit.metadata["frame_sequence"] == unit.frame_sequence
    finally:
        if subscription is not None:
            subscription.close()
        coordinator.close()
