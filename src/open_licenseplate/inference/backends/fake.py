"""Deterministic fake inference backend for portable tests."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from ..contract import (
    BackendInspection,
    BackendOptions,
    BackendOutput,
    FeatureDescription,
    LoadedModel,
    ModelDescriptor,
    PreparedInput,
)

FakeOutputFactory = Callable[[PreparedInput], Mapping[str, Any]]


class FakeBackend:
    """Return fixed named outputs without importing platform inference libraries."""

    name = "fake"

    def __init__(
        self,
        outputs: Mapping[str, Any] | None = None,
        *,
        output_factory: FakeOutputFactory | None = None,
        inspection: BackendInspection | None = None,
    ) -> None:
        if outputs is not None and output_factory is not None:
            raise ValueError("fake backend accepts outputs or output_factory, not both")
        self.outputs = dict(outputs or {})
        self.output_factory = output_factory
        self.inspection = inspection
        self.loads: list[LoadedModel] = []
        self.predictions: list[PreparedInput] = []
        self.closes: list[LoadedModel] = []

    def load(self, model: ModelDescriptor, options: BackendOptions) -> LoadedModel:
        """Create a new fake model instance for every load call."""
        inspection = self.inspection or _inspection_from_manifest(model)
        loaded = LoadedModel(
            descriptor=model,
            options=options,
            inspection=inspection,
            handle=object(),
        )
        self.loads.append(loaded)
        return loaded

    def predict(self, model: LoadedModel, model_input: object) -> BackendOutput:
        """Return a copy of fixed outputs or output-factory values."""
        if model.closed:
            raise RuntimeError("fake model instance is closed")
        if not isinstance(model_input, PreparedInput):
            raise TypeError("fake backend prediction requires PreparedInput")
        self.predictions.append(model_input)
        values = (
            self.output_factory(model_input)
            if self.output_factory is not None
            else dict(self.outputs)
        )
        return BackendOutput(values=dict(values))

    def close(self, model: LoadedModel) -> None:
        """Close one fake model instance."""
        model.closed = True
        self.closes.append(model)


def _inspection_from_manifest(model: ModelDescriptor) -> BackendInspection:
    """Build a deterministic structural description for fake contract tests."""
    input_values = model.manifest.raw["input"]
    output_values = model.manifest.raw["outputs"]
    assert isinstance(input_values, dict)
    assert isinstance(output_values, dict)
    input_description = FeatureDescription(
        name=str(input_values["name"]),
        kind="image",
        width=int(input_values["width"]),
        height=int(input_values["height"]),
        color_space=str(input_values["color_space"]),
    )
    additional_inputs = input_values.get("additional_inputs")
    additional_descriptions = (
        tuple(
            FeatureDescription(
                name=str(item["name"]),
                kind=str(item["kind"]),
            )
            for item in additional_inputs
            if isinstance(item, dict)
        )
        if isinstance(additional_inputs, list)
        else ()
    )
    output_descriptions = tuple(
        FeatureDescription(name=str(name), kind="multi_array")
        for role, name in output_values.items()
        if role in {"boxes", "scores", "classes", "raw"} and isinstance(name, str)
    )
    return BackendInspection(
        backend=model.manifest.backend,
        inputs=(input_description, *additional_descriptions),
        outputs=output_descriptions,
    )
