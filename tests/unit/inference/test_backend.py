from __future__ import annotations

from pathlib import Path

import numpy as np

from model_helpers import create_model_fixture
from open_licenseplate.inference import (
    BackendOptions,
    BackendStillImageDetector,
    ComputeUnit,
    DetectorSession,
    ModelDescriptor,
    StillImage,
)
from open_licenseplate.inference.backends import FakeBackend
from open_licenseplate.models.manifest import parse_manifest


def _descriptor(tmp_path: Path) -> ModelDescriptor:
    manifest_path, _archive_path, _raw_manifest = create_model_fixture(tmp_path)
    manifest = parse_manifest(manifest_path.read_bytes())
    return ModelDescriptor(
        model_id=manifest.model_id,
        artifact_path=str(tmp_path / manifest.artifact),
        artifact_sha256=manifest.artifact_sha256,
        manifest=manifest,
    )


def test_fake_backend_supports_the_still_image_detector_contract(tmp_path: Path) -> None:
    descriptor = _descriptor(tmp_path)
    backend = FakeBackend(
        outputs={
            "coordinates": np.array([[10, 20, 50, 40]], dtype=np.float32),
            "confidence": np.array([0.9], dtype=np.float32),
        }
    )

    detections = BackendStillImageDetector(backend).detect(
        StillImage(np.zeros((640, 640, 3), dtype=np.uint8)),
        descriptor,
    )

    assert len(detections.detections) == 1
    assert detections.detections[0].model_id == descriptor.model_id
    assert len(backend.loads) == 1
    assert len(backend.predictions) == 1
    assert len(backend.closes) == 1


def test_compute_unit_change_closes_and_reloads_a_new_instance(tmp_path: Path) -> None:
    descriptor = _descriptor(tmp_path)
    backend = FakeBackend()
    session = DetectorSession(
        backend=backend,
        descriptor=descriptor,
        options=BackendOptions(),
    )

    first = session.load()
    second = session.set_compute_units(BackendOptions(compute_units=ComputeUnit.CPU_AND_NE))

    assert first.instance_id != second.instance_id
    assert first.closed is True
    assert second.closed is False
    assert second.options.compute_units is ComputeUnit.CPU_AND_NE
    assert [model.options.compute_units for model in backend.loads] == [
        ComputeUnit.ALL,
        ComputeUnit.CPU_AND_NE,
    ]
    session.close()
