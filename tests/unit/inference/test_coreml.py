from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from model_helpers import create_model_fixture
from open_licenseplate.inference import (
    BackendContractError,
    BackendInspection,
    BackendOptions,
    ComputeUnit,
    CoreMLBackend,
    FeatureDescription,
    LoadedModel,
    ModelDescriptor,
    StillImage,
    adapter_for_manifest,
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


class _PredictingCoreMLModel:
    def __init__(self) -> None:
        self.inputs: list[dict[str, object]] = []

    def predict(self, inputs: dict[str, object]) -> dict[str, object]:
        self.inputs.append(inputs)
        return {"confidence": np.array([[0.9]], dtype=np.float32)}


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


def test_manifest_additional_inputs_are_compared_by_role_and_name(tmp_path) -> None:
    _manifest_path, _archive_path, raw_manifest = create_model_fixture(tmp_path)
    raw_manifest["input"]["additional_inputs"] = [
        {
            "name": "model_confidence",
            "kind": "double",
            "role": "confidence_threshold",
            "optional": True,
            "default": 0.35,
        }
    ]
    manifest = parse_manifest(raw_manifest)
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
            FeatureDescription(name="model_confidence", kind="double"),
        ),
        outputs=(
            FeatureDescription(name="coordinates", kind="multi_array"),
            FeatureDescription(name="confidence", kind="multi_array"),
        ),
    )

    compare_manifest_to_inspection(manifest, inspection)

    raw_manifest["input"]["additional_inputs"][0]["name"] = "missing"
    mismatched = parse_manifest(raw_manifest)
    with pytest.raises(BackendContractError, match="additional input name"):
        compare_manifest_to_inspection(mismatched, inspection)


def test_coreml_backend_passes_thresholds_using_manifest_roles(tmp_path, monkeypatch) -> None:
    import open_licenseplate.inference.coreml as coreml_module

    _manifest_path, _archive_path, raw_manifest = create_model_fixture(tmp_path)
    raw_manifest["input"]["additional_inputs"] = [
        {
            "name": "model_iou",
            "kind": "double",
            "role": "iou_threshold",
            "optional": True,
            "default": 0.45,
        },
        {
            "name": "model_confidence",
            "kind": "double",
            "role": "confidence_threshold",
            "optional": True,
            "default": 0.35,
        },
    ]
    manifest = parse_manifest(raw_manifest)
    descriptor = ModelDescriptor(
        model_id=manifest.model_id,
        artifact_path=str(tmp_path / manifest.artifact),
        artifact_sha256=manifest.artifact_sha256,
        manifest=manifest,
    )
    handle = _PredictingCoreMLModel()
    loaded = LoadedModel(
        descriptor=descriptor,
        options=BackendOptions(),
        inspection=BackendInspection(
            backend="coreml",
            inputs=(
                FeatureDescription(
                    name="image",
                    kind="image",
                    width=640,
                    height=640,
                    color_space="rgb",
                ),
                FeatureDescription(name="model_iou", kind="double"),
                FeatureDescription(name="model_confidence", kind="double"),
            ),
            outputs=(),
        ),
        handle=handle,
    )
    prepared = adapter_for_manifest(manifest).preprocess(
        StillImage(np.zeros((640, 640, 3), dtype=np.uint8)),
        manifest,
    )
    prepared = replace(
        prepared,
        transform=replace(
            prepared.transform,
            confidence_threshold=0.2,
            iou_threshold=0.6,
        ),
    )
    monkeypatch.setattr(coreml_module, "_require_macos", lambda: None)

    output = coreml_module.CoreMLBackend().predict(loaded, prepared)

    assert output.values["confidence"].shape == (1, 1)
    assert isinstance(handle.inputs[0]["model_iou"], float)
    assert isinstance(handle.inputs[0]["model_confidence"], float)
    assert handle.inputs[0]["model_iou"] == pytest.approx(0.6)
    assert handle.inputs[0]["model_confidence"] == pytest.approx(0.2)


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
