"""Optional Core ML backend with a macOS-only import boundary."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np

from ..models.manifest import ModelManifest
from .contract import (
    BackendContractError,
    BackendInspection,
    BackendOptions,
    BackendOutput,
    BackendUnavailableError,
    FeatureDescription,
    LoadedModel,
    ModelDescriptor,
    PreparedInput,
)


class CoreMLBackend:
    """Load and predict Core ML packages only on macOS."""

    name = "coreml"

    def load(self, model: ModelDescriptor, options: BackendOptions) -> LoadedModel:
        """Load a new Core ML model instance for the requested compute units."""
        _require_macos()
        coremltools = _import_coremltools()
        package_path = Path(model.artifact_path)
        if not package_path.is_dir() or package_path.is_symlink():
            raise BackendContractError("Core ML artifact must be a managed .mlpackage directory")

        compute_units = coreml_compute_unit(options.compute_units, coremltools)
        try:
            coreml_model = coremltools.models.MLModel(
                str(package_path),
                compute_units=compute_units,
            )
        except Exception as error:
            raise BackendContractError("Core ML package could not be loaded") from error

        inspection = inspect_coreml_model(coreml_model)
        return LoadedModel(
            descriptor=model,
            options=options,
            inspection=inspection,
            handle=coreml_model,
        )

    def predict(self, model: LoadedModel, model_input: object) -> BackendOutput:
        """Run one prediction using the loaded model instance."""
        _require_macos()
        if model.closed:
            raise BackendContractError("Core ML model instance is closed")
        if not isinstance(model_input, PreparedInput):
            raise BackendContractError("Core ML prediction requires PreparedInput")
        value = _coreml_input_value(model_input)
        try:
            outputs = model.handle.predict({model_input.input_name: value})
        except Exception as error:
            raise BackendContractError("Core ML prediction failed") from error
        if not isinstance(outputs, Mapping):
            raise BackendContractError("Core ML prediction did not return named outputs")
        return BackendOutput(values=dict(outputs))

    def close(self, model: LoadedModel) -> None:
        """Release references held by one Core ML model instance."""
        model.closed = True
        model.handle = None

    def inspect(
        self,
        model: ModelDescriptor,
        options: BackendOptions | None = None,
    ) -> BackendInspection:
        """Inspect an imported package without guessing feature names."""
        loaded = self.load(model, options or BackendOptions())
        inspection = loaded.inspection
        self.close(loaded)
        return inspection


def compare_manifest_to_inspection(
    manifest: ModelManifest,
    inspection: BackendInspection,
) -> None:
    """Require declared input and output names and image properties to match."""
    if inspection.backend != manifest.backend:
        raise BackendContractError(
            f"manifest backend {manifest.backend!r} does not match inspected backend "
            f"{inspection.backend!r}"
        )

    input_values = manifest.raw.get("input")
    output_values = manifest.raw.get("outputs")
    if not isinstance(input_values, dict) or not isinstance(output_values, dict):
        raise BackendContractError("manifest input and outputs sections are invalid")

    input_name = input_values.get("name")
    if not isinstance(input_name, str):
        raise BackendContractError("manifest input name is invalid")
    actual_inputs = {feature.name: feature for feature in inspection.inputs}
    actual_input = actual_inputs.get(input_name)
    if actual_input is None:
        raise BackendContractError(
            f"manifest input name {input_name!r} was not found in the Core ML model"
        )
    if actual_input.kind != "image":
        raise BackendContractError("manifest input must match a Core ML image input")
    if actual_input.width != input_values.get("width"):
        raise BackendContractError("manifest input width does not match the Core ML model")
    if actual_input.height != input_values.get("height"):
        raise BackendContractError("manifest input height does not match the Core ML model")
    if actual_input.color_space != input_values.get("color_space"):
        raise BackendContractError("manifest input color_space does not match the Core ML model")

    actual_outputs = {feature.name: feature for feature in inspection.outputs}
    declared_outputs = {
        key: value
        for key, value in output_values.items()
        if key in {"boxes", "scores", "classes", "raw"} and isinstance(value, str)
    }
    declared_output_names = list(declared_outputs.values())
    if not declared_output_names:
        raise BackendContractError("manifest does not declare any output names")
    missing = [name for name in declared_output_names if name not in actual_outputs]
    if missing:
        raise BackendContractError(
            "manifest output name(s) were not found in the Core ML model: "
            + ", ".join(repr(name) for name in missing)
        )
    non_arrays = [
        (role, name)
        for role, name in declared_outputs.items()
        if actual_outputs[name].kind != "multi_array"
    ]
    if non_arrays:
        role, name = non_arrays[0]
        raise BackendContractError(
            f"manifest output {role!r} name {name!r} is not a Core ML multi-array output"
        )


def coreml_compute_unit(compute_unit: Any, coremltools: Any) -> Any:
    """Map each stable compute choice to its exact Core ML enum."""
    from .contract import ComputeUnit

    choice = ComputeUnit.parse(compute_unit)
    mapping = {
        ComputeUnit.ALL: coremltools.ComputeUnit.ALL,
        ComputeUnit.CPU_ONLY: coremltools.ComputeUnit.CPU_ONLY,
        ComputeUnit.CPU_AND_GPU: coremltools.ComputeUnit.CPU_AND_GPU,
        ComputeUnit.CPU_AND_NE: coremltools.ComputeUnit.CPU_AND_NE,
    }
    return mapping[choice]


def inspect_coreml_model(model: Any) -> BackendInspection:
    """Read feature descriptions from a coremltools model specification."""
    try:
        specification = model.get_spec()
        inputs = tuple(_feature_description(feature) for feature in specification.description.input)
        outputs = tuple(
            _feature_description(feature) for feature in specification.description.output
        )
    except Exception as error:
        raise BackendContractError(
            "Core ML input/output descriptions could not be inspected"
        ) from error
    return BackendInspection(backend="coreml", inputs=inputs, outputs=outputs)


def _feature_description(feature: Any) -> FeatureDescription:
    feature_type = feature.type
    if _has_field(feature_type, "imageType"):
        image_type = feature_type.imageType
        return FeatureDescription(
            name=str(feature.name),
            kind="image",
            width=int(image_type.width),
            height=int(image_type.height),
            color_space=_image_color_space(image_type.colorSpace),
        )
    if _has_field(feature_type, "multiArrayType"):
        array_type = feature_type.multiArrayType
        shape = tuple(int(value) for value in array_type.shape)
        data_type = _enum_name(array_type.dataType)
        return FeatureDescription(
            name=str(feature.name),
            kind="multi_array",
            shape=shape,
            data_type=data_type,
        )
    if _has_field(feature_type, "dictionaryType"):
        return FeatureDescription(name=str(feature.name), kind="dictionary")
    if _has_field(feature_type, "stringType"):
        return FeatureDescription(name=str(feature.name), kind="string")
    if _has_field(feature_type, "int64Type"):
        return FeatureDescription(name=str(feature.name), kind="int64")
    if _has_field(feature_type, "doubleType"):
        return FeatureDescription(name=str(feature.name), kind="double")
    if _has_field(feature_type, "boolType"):
        return FeatureDescription(name=str(feature.name), kind="bool")
    return FeatureDescription(name=str(feature.name), kind="unknown")


def _has_field(value: Any, field_name: str) -> bool:
    try:
        return bool(value.HasField(field_name))
    except (AttributeError, ValueError):
        return getattr(value, field_name, None) is not None


def _enum_name(value: Any) -> str:
    if isinstance(value, str):
        return value.casefold()
    names = {
        0: "invalid",
        65552: "float16",
        65568: "float32",
        65600: "double",
        131104: "int32",
    }
    return names.get(int(value), str(value))


def _image_color_space(value: Any) -> str:
    if isinstance(value, str):
        normalized = value.casefold()
        if normalized in {"rgb", "bgr", "grayscale"}:
            return normalized
    names = {
        10: "grayscale",
        20: "rgb",
        30: "bgr",
        40: "grayscale_float16",
    }
    try:
        return names[int(value)]
    except (KeyError, TypeError, ValueError) as error:
        raise BackendContractError("Core ML image input has an unsupported color space") from error


def _coreml_input_value(prepared: PreparedInput) -> Any:
    """Convert the canonical array to the image object accepted by Core ML."""
    from PIL import Image

    if prepared.color_space == "grayscale":
        return Image.fromarray(np.asarray(prepared.value, dtype=np.uint8), mode="L")
    return Image.fromarray(np.asarray(prepared.value, dtype=np.uint8), mode="RGB")


def _require_macos() -> None:
    if sys.platform != "darwin":
        raise BackendUnavailableError("Core ML backend is available only on macOS")


def _import_coremltools() -> Any:
    if sys.platform != "darwin":
        raise BackendUnavailableError("Core ML backend is available only on macOS")
    try:
        coremltools = import_module("coremltools")
    except ImportError as error:
        raise BackendUnavailableError(
            "Core ML backend requires the optional coremltools package on macOS"
        ) from error
    return coremltools
