"""Validation for the versioned managed-model manifest."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qsl, urlsplit

import yaml

SUPPORTED_ADAPTERS = frozenset({"ultralytics_yolo_nms"})
SUPPORTED_BACKENDS = frozenset({"coreml"})
MANIFEST_SCHEMA_VERSION = 1
SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
MODEL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class ModelManifestError(ValueError):
    """Raised when a model manifest is not safe or supported."""


@dataclass(frozen=True)
class ModelManifest:
    """Validated manifest values and its immutable JSON snapshot."""

    raw: dict[str, Any]
    model_id: str
    display_name: str
    backend: str
    adapter: str
    artifact: str
    artifact_sha256: str
    source_url: str | None
    source_license: str | None

    @property
    def snapshot_json(self) -> str:
        """Return a deterministic JSON representation for SQLite."""
        return json.dumps(self.raw, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def parse_manifest(value: bytes | str | Mapping[str, Any]) -> ModelManifest:
    """Parse JSON or safe YAML and validate the supported manifest contract."""
    if isinstance(value, Mapping):
        decoded: Any = dict(value)
    else:
        if isinstance(value, bytes):
            try:
                text = value.decode("utf-8")
            except UnicodeDecodeError as error:
                raise ModelManifestError("manifest must be UTF-8 text") from error
        else:
            text = value
        if len(text.encode("utf-8")) > 256 * 1024:
            raise ModelManifestError("manifest is too large")
        try:
            decoded = yaml.safe_load(text)
        except yaml.YAMLError as error:
            raise ModelManifestError("manifest syntax is invalid") from error

    if not isinstance(decoded, dict):
        raise ModelManifestError("manifest must be an object")
    raw = _copy_json_object(decoded)
    if _contains_secret_key(raw):
        raise ModelManifestError("manifest may not contain secret values")

    schema_version = raw.get("schema_version")
    if isinstance(schema_version, bool) or schema_version != MANIFEST_SCHEMA_VERSION:
        raise ModelManifestError("manifest schema_version must be 1")

    model_id = _required_text(raw, "id", max_length=128)
    if MODEL_ID_RE.fullmatch(model_id) is None or model_id in {".", ".."}:
        raise ModelManifestError("manifest id contains unsupported characters")

    display_name = _required_text(raw, "display_name", max_length=255)
    backend = _required_text(raw, "backend", max_length=64).lower()
    if backend not in SUPPORTED_BACKENDS:
        raise ModelManifestError(f"manifest backend must be one of {sorted(SUPPORTED_BACKENDS)}")

    adapter = _required_text(raw, "adapter", max_length=128)
    if adapter not in SUPPORTED_ADAPTERS:
        raise ModelManifestError(f"manifest adapter must be one of {sorted(SUPPORTED_ADAPTERS)}")

    artifact = _required_text(raw, "artifact", max_length=255)
    if (
        "/" in artifact
        or "\\" in artifact
        or ":" in artifact
        or artifact.startswith(".")
        or not artifact.lower().endswith(".mlpackage")
    ):
        raise ModelManifestError("manifest artifact must be one .mlpackage package name")

    artifact_sha256 = _required_text(raw, "artifact_sha256", max_length=64).lower()
    if SHA256_RE.fullmatch(artifact_sha256) is None:
        raise ModelManifestError("manifest artifact_sha256 must be a 64-character SHA-256 value")

    task = _required_text(raw, "task", max_length=64)
    if task != "object_detection":
        raise ModelManifestError("manifest task must be object_detection")

    _validate_input(raw.get("input"))
    _validate_preprocessing(raw.get("preprocessing"))
    _validate_outputs(raw.get("outputs"))
    _validate_labels(raw.get("labels"))
    _validate_defaults(raw.get("defaults"))
    _validate_compatibility(raw.get("compatibility"))
    source_url, source_license = _validate_source(raw.get("source"))
    _validate_conversion(raw.get("conversion"))

    return ModelManifest(
        raw=raw,
        model_id=model_id,
        display_name=display_name,
        backend=backend,
        adapter=adapter,
        artifact=artifact,
        artifact_sha256=artifact_sha256,
        source_url=source_url,
        source_license=source_license,
    )


def _copy_json_object(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        encoded = json.dumps(value, ensure_ascii=True)
        decoded = json.loads(encoded)
    except (TypeError, ValueError) as error:
        raise ModelManifestError("manifest must contain JSON-compatible values") from error
    if not isinstance(decoded, dict):
        raise ModelManifestError("manifest must be an object")
    return decoded


def _required_text(values: Mapping[str, Any], field_name: str, *, max_length: int) -> str:
    value = values.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ModelManifestError(f"manifest {field_name} is required")
    normalized = value.strip()
    if len(normalized) > max_length:
        raise ModelManifestError(f"manifest {field_name} is too long")
    return normalized


def _required_section_text(
    values: Mapping[str, Any],
    section_name: str,
    field_name: str,
    *,
    max_length: int,
) -> str:
    value = values.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ModelManifestError(f"manifest {section_name}.{field_name} is required")
    normalized = value.strip()
    if len(normalized) > max_length:
        raise ModelManifestError(f"manifest {section_name}.{field_name} is too long")
    return normalized


def _optional_mapping(value: Any, field_name: str) -> Mapping[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ModelManifestError(f"manifest {field_name} must be an object")
    return value


def _required_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    values = _optional_mapping(value, field_name)
    if values is None:
        raise ModelManifestError(f"manifest {field_name} is required")
    return values


def _validate_input(value: Any) -> None:
    values = _required_mapping(value, "input")
    _required_section_text(values, "input", "name", max_length=255)
    if _required_section_text(values, "input", "kind", max_length=32) != "image":
        raise ModelManifestError("manifest input.kind must be image")
    for field_name in ("width", "height"):
        if field_name not in values:
            raise ModelManifestError(f"manifest input.{field_name} is required")
        number = values[field_name]
        if isinstance(number, bool) or not isinstance(number, int) or not 1 <= number <= 8192:
            raise ModelManifestError(f"manifest input.{field_name} must be from 1 through 8192")
    color_space = _required_section_text(values, "input", "color_space", max_length=32)
    if color_space not in {"rgb", "bgr", "grayscale"}:
        raise ModelManifestError("manifest input.color_space is not supported")


def _validate_preprocessing(value: Any) -> None:
    values = _required_mapping(value, "preprocessing")
    resize = _required_section_text(values, "preprocessing", "resize", max_length=32)
    if resize not in {"letterbox", "stretch", "none"}:
        raise ModelManifestError("manifest preprocessing.resize is not supported")


def _validate_outputs(value: Any) -> None:
    values = _required_mapping(value, "outputs")
    has_boxes = "boxes" in values
    has_scores = "scores" in values
    has_raw = "raw" in values
    if has_boxes != has_scores:
        missing = "scores" if has_boxes else "boxes"
        raise ModelManifestError(f"manifest outputs.{missing} is required")
    if not (has_raw or (has_boxes and has_scores)):
        raise ModelManifestError("manifest outputs.boxes and outputs.scores are required")
    for field_name in ("boxes", "scores", "raw", "classes"):
        if field_name in values:
            _required_section_text(values, "outputs", field_name, max_length=255)
    box_format = values.get("box_format", "xyxy")
    if not isinstance(box_format, str) or box_format not in {"xyxy", "xywh"}:
        raise ModelManifestError("manifest outputs.box_format must be xyxy or xywh")
    raw_has_objectness = values.get("raw_has_objectness", False)
    if not isinstance(raw_has_objectness, bool):
        raise ModelManifestError("manifest outputs.raw_has_objectness must be boolean")


def _validate_labels(value: Any) -> None:
    if value is None:
        raise ModelManifestError("manifest labels is required")
    if not isinstance(value, list) or not value:
        raise ModelManifestError("manifest labels must be a non-empty list")
    labels: set[str] = set()
    for label in value:
        if not isinstance(label, str) or not label.strip() or len(label.strip()) > 255:
            raise ModelManifestError("manifest labels must contain short text values")
        normalized = label.strip()
        if normalized.casefold() in labels:
            raise ModelManifestError("manifest labels must not contain duplicates")
        labels.add(normalized.casefold())


def _validate_defaults(value: Any) -> None:
    values = _required_mapping(value, "defaults")
    for field_name in ("confidence_threshold", "iou_threshold"):
        if field_name not in values:
            raise ModelManifestError(f"manifest defaults.{field_name} is required")
        number = values[field_name]
        if isinstance(number, bool) or not isinstance(number, (int, float)):
            raise ModelManifestError(f"manifest defaults.{field_name} must be a number")
        if not 0 <= float(number) <= 1:
            raise ModelManifestError(f"manifest defaults.{field_name} must be between 0 and 1")


def _validate_compatibility(value: Any) -> None:
    values = _optional_mapping(value, "compatibility")
    if values is None:
        return
    if "minimum_macos" in values and (
        not isinstance(values["minimum_macos"], str) or not values["minimum_macos"].strip()
    ):
        raise ModelManifestError("manifest compatibility.minimum_macos must be text")


def _validate_source(value: Any) -> tuple[str | None, str | None]:
    values = _optional_mapping(value, "source")
    if values is None:
        return None, None
    source_url = values.get("url")
    if source_url is not None:
        if not isinstance(source_url, str) or len(source_url) > 2048:
            raise ModelManifestError("manifest source.url must be short text")
        parsed = urlsplit(source_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ModelManifestError("manifest source.url must be an HTTP or HTTPS URL")
        if parsed.username is not None or parsed.password is not None:
            raise ModelManifestError("manifest source.url may not contain credentials")
        if any(
            any(
                secret_part in key.lower().replace("-", "_")
                for secret_part in (
                    "authorization",
                    "credential",
                    "password",
                    "passwd",
                    "secret",
                    "token",
                    "api_key",
                )
            )
            for key, _value in parse_qsl(parsed.query, keep_blank_values=True)
        ):
            raise ModelManifestError("manifest source.url may not contain secret query values")
        source_url = source_url.strip()
    source_license = values.get("license")
    if source_license is not None:
        if not isinstance(source_license, str) or not source_license.strip():
            raise ModelManifestError("manifest source.license must be non-empty text")
        source_license = source_license.strip()
    return source_url, source_license


def _validate_conversion(value: Any) -> None:
    values = _optional_mapping(value, "conversion")
    if values is None:
        return
    for field_name in ("source_weight", "tool_versions", "arguments"):
        if (
            field_name in values
            and field_name == "source_weight"
            and not isinstance(values[field_name], str)
        ):
            raise ModelManifestError("manifest conversion.source_weight must be text")
        if (
            field_name in values
            and field_name in {"tool_versions", "arguments"}
            and not isinstance(values[field_name], Mapping)
        ):
            raise ModelManifestError(f"manifest conversion.{field_name} must be an object")


def _contains_secret_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            any(
                secret_part in str(key).lower().replace("-", "_")
                for secret_part in (
                    "authorization",
                    "credential",
                    "password",
                    "passwd",
                    "secret",
                    "token",
                    "api_key",
                )
            )
            or _contains_secret_key(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_secret_key(item) for item in value)
    return False
