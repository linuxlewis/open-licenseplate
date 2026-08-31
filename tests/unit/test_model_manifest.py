from __future__ import annotations

import copy
import json

import pytest

from open_licenseplate.models.manifest import ModelManifestError, parse_manifest


def _manifest(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": 1,
        "id": "test-model",
        "display_name": "Test model",
        "task": "object_detection",
        "backend": "coreml",
        "adapter": "ultralytics_yolo_nms",
        "artifact": "model.mlpackage",
        "artifact_sha256": "a" * 64,
        "input": {
            "name": "image",
            "kind": "image",
            "width": 640,
            "height": 640,
            "color_space": "rgb",
        },
        "preprocessing": {"resize": "letterbox"},
        "outputs": {
            "boxes": "coordinates",
            "scores": "confidence",
            "box_format": "xyxy",
            "coordinate_space": "model_pixels",
        },
        "labels": ["license_plate"],
        "defaults": {"confidence_threshold": 0.35, "iou_threshold": 0.45},
        "compatibility": {"minimum_macos": "14.0"},
        "source": {"url": "https://example.test/model", "license": "MIT"},
        "conversion": {"source_weight": "weights.pt", "tool_versions": {}, "arguments": {}},
    }
    value.update(overrides)
    return value


def test_manifest_accepts_json_and_yaml() -> None:
    manifest = parse_manifest(json.dumps(_manifest()))
    yaml_manifest = parse_manifest(
        """
schema_version: 1
id: test-model
display_name: Test model
task: object_detection
backend: coreml
adapter: ultralytics_yolo_nms
artifact: model.mlpackage
artifact_sha256: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
input:
  name: image
  kind: image
  width: 640
  height: 640
  color_space: rgb
preprocessing:
  resize: letterbox
outputs:
  boxes: coordinates
  scores: confidence
  box_format: xyxy
  coordinate_space: model_pixels
labels:
  - license_plate
defaults:
  confidence_threshold: 0.35
  iou_threshold: 0.45
"""
    )

    assert manifest.model_id == "test-model"
    assert yaml_manifest.backend == "coreml"
    assert yaml_manifest.raw["input"]["width"] == 640


@pytest.mark.parametrize(
    "field",
    ["task", "input", "preprocessing", "outputs", "labels", "defaults"],
)
def test_manifest_requires_contract_sections(field: str) -> None:
    manifest = _manifest()
    del manifest[field]

    with pytest.raises(ModelManifestError, match=field):
        parse_manifest(manifest)


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("input", "name"),
        ("input", "kind"),
        ("input", "width"),
        ("input", "height"),
        ("input", "color_space"),
        ("preprocessing", "resize"),
        ("outputs", "boxes"),
        ("outputs", "scores"),
        ("outputs", "box_format"),
        ("outputs", "coordinate_space"),
        ("defaults", "confidence_threshold"),
        ("defaults", "iou_threshold"),
    ],
)
def test_manifest_requires_contract_subfields(section: str, field: str) -> None:
    manifest = copy.deepcopy(_manifest())
    del manifest[section][field]  # type: ignore[index]

    with pytest.raises(ModelManifestError, match=f"{section}\\.{field}"):
        parse_manifest(manifest)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("backend", "onnx", "backend"),
        ("adapter", "unknown", "adapter"),
        ("id", "../model", "id"),
        ("artifact", "../model.mlpackage", "artifact"),
        ("artifact_sha256", "not-a-checksum", "artifact_sha256"),
    ],
)
def test_manifest_rejects_unsupported_or_unsafe_values(
    field: str,
    value: object,
    message: str,
) -> None:
    with pytest.raises(ModelManifestError, match=message):
        parse_manifest(_manifest(**{field: value}))


def test_manifest_rejects_invalid_nested_values() -> None:
    with pytest.raises(ModelManifestError, match="confidence_threshold"):
        parse_manifest(
            _manifest(
                defaults={"confidence_threshold": 2, "iou_threshold": 0.45},
            )
        )

    with pytest.raises(ModelManifestError, match="source.url"):
        parse_manifest(_manifest(source={"url": "file:///unsafe", "license": "MIT"}))


def test_manifest_rejects_non_object_input() -> None:
    with pytest.raises(ModelManifestError, match="input"):
        parse_manifest(_manifest(input=["not", "an", "object"]))


def test_manifest_accepts_declared_raw_output_contract() -> None:
    manifest = parse_manifest(
        _manifest(
            outputs={
                "raw": "predictions",
                "box_format": "xywh",
                "coordinate_space": "normalized",
                "raw_layout": "channels_first",
                "raw_has_objectness": False,
            }
        )
    )

    assert manifest.raw["outputs"]["raw"] == "predictions"


def test_manifest_rejects_non_boolean_raw_objectness() -> None:
    with pytest.raises(ModelManifestError, match="raw_has_objectness"):
        parse_manifest(
            _manifest(
                outputs={
                    "raw": "predictions",
                    "box_format": "xywh",
                    "coordinate_space": "normalized",
                    "raw_layout": "channels_first",
                    "raw_has_objectness": "false",
                }
            )
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("box_format", "unknown"),
        ("coordinate_space", "source_pixels"),
    ],
)
def test_manifest_rejects_undeclared_output_geometry(field: str, value: str) -> None:
    outputs = _manifest()["outputs"]
    assert isinstance(outputs, dict)
    outputs = dict(outputs)
    outputs[field] = value

    with pytest.raises(ModelManifestError, match=field):
        parse_manifest(_manifest(outputs=outputs))


def test_manifest_requires_all_raw_output_metadata() -> None:
    for missing_field in ("raw_layout", "raw_has_objectness"):
        outputs = {
            "raw": "predictions",
            "box_format": "xywh",
            "coordinate_space": "normalized",
            "raw_layout": "channels_first",
            "raw_has_objectness": False,
        }
        del outputs[missing_field]

        with pytest.raises(ModelManifestError, match=missing_field):
            parse_manifest(_manifest(outputs=outputs))
