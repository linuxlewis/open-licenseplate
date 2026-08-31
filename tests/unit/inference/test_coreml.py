from __future__ import annotations

import pytest

from model_helpers import create_model_fixture
from open_licenseplate.inference import (
    BackendContractError,
    BackendInspection,
    ComputeUnit,
    CoreMLBackend,
    FeatureDescription,
    ModelDescriptor,
    compare_manifest_to_inspection,
    coreml_compute_unit,
)
from open_licenseplate.inference.coreml import inspect_coreml_model
from open_licenseplate.models.manifest import parse_manifest


class _ComputeUnitValues:
    ALL = object()
    CPU_ONLY = object()
    CPU_AND_GPU = object()
    CPU_AND_NE = object()


class _CoreMLToolsStub:
    ComputeUnit = _ComputeUnitValues


class _FeatureType:
    def __init__(self, kind: str, **values: object) -> None:
        self.kind = kind
        for name, value in values.items():
            setattr(self, name, value)

    def HasField(self, name: str) -> bool:
        return name == self.kind


class _Feature:
    def __init__(self, name: str, feature_type: _FeatureType) -> None:
        self.name = name
        self.type = feature_type


class _FakeCoreMLModel:
    def get_spec(self) -> object:
        image = _FeatureType(
            "imageType",
            imageType=type("Image", (), {"width": 640, "height": 640, "colorSpace": 20})(),
        )
        array = _FeatureType(
            "multiArrayType",
            multiArrayType=type(
                "Array",
                (),
                {"shape": [1, 300, 4], "dataType": 65568},
            )(),
        )
        return type(
            "Spec",
            (),
            {
                "description": type(
                    "Description",
                    (),
                    {
                        "input": [_Feature("image", image)],
                        "output": [_Feature("coordinates", array)],
                    },
                )()
            },
        )()


def test_compute_unit_mapping_uses_exact_coreml_enum_members() -> None:
    coremltools = _CoreMLToolsStub()

    assert coreml_compute_unit(ComputeUnit.ALL, coremltools) is coremltools.ComputeUnit.ALL
    assert coreml_compute_unit("cpu_only", coremltools) is coremltools.ComputeUnit.CPU_ONLY
    assert coreml_compute_unit("CPU and GPU", coremltools) is coremltools.ComputeUnit.CPU_AND_GPU
    assert coreml_compute_unit("cpu_and_ne", coremltools) is coremltools.ComputeUnit.CPU_AND_NE


def test_manifest_output_names_are_compared_without_guessing(tmp_path) -> None:
    manifest_path, _archive_path, raw_manifest = create_model_fixture(tmp_path)
    manifest = parse_manifest(manifest_path.read_bytes())
    inspection = BackendInspection(
        backend="coreml",
        inputs=(
            FeatureDescription(
                name="image",
                kind="image",
                width=640,
                height=640,
                color_space="rgb",
            ),
        ),
        outputs=(
            FeatureDescription(name="coordinates", kind="multi_array"),
            FeatureDescription(name="confidence", kind="multi_array"),
        ),
    )
    compare_manifest_to_inspection(manifest, inspection)
    raw_manifest["outputs"]["scores"] = "guessed_score_name"
    mismatched = parse_manifest(raw_manifest)

    with pytest.raises(BackendContractError, match="output name"):
        compare_manifest_to_inspection(mismatched, inspection)


def test_coreml_inspection_reads_actual_feature_properties() -> None:
    inspection = inspect_coreml_model(_FakeCoreMLModel())

    assert inspection.inputs[0].as_dict() == {
        "name": "image",
        "kind": "image",
        "width": 640,
        "height": 640,
        "color_space": "rgb",
    }
    assert inspection.outputs[0].as_dict() == {
        "name": "coordinates",
        "kind": "multi_array",
        "shape": [1, 300, 4],
        "data_type": "float32",
    }


def test_coreml_backend_is_unavailable_off_macos(tmp_path, monkeypatch) -> None:
    import open_licenseplate.inference.coreml as coreml_module

    monkeypatch.setattr(coreml_module.sys, "platform", "linux")
    manifest_path, _archive_path, _raw_manifest = create_model_fixture(tmp_path)
    manifest = parse_manifest(manifest_path.read_bytes())

    descriptor = ModelDescriptor(
        model_id=manifest.model_id,
        artifact_path=str(tmp_path / manifest.artifact),
        artifact_sha256=manifest.artifact_sha256,
        manifest=manifest,
    )

    with pytest.raises(Exception, match="only on macOS"):
        CoreMLBackend().load(descriptor, object())  # type: ignore[arg-type]
