"""Validation and safe test operations for camera configuration."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from urllib.parse import SplitResult, parse_qsl, urlsplit

from ..redaction import redact_url
from .credentials import (
    credential_status,
    parse_credential_ref,
    resolve_credential,
)
from .repository import Camera, CameraConfig, camera_config_from_record

SUPPORTED_SCHEMES = frozenset({"rtsp", "rtsps"})
SUPPORTED_TRANSPORTS = frozenset({"tcp", "udp"})
SECRET_QUERY_KEYS = frozenset(
    {"authorization", "credential", "password", "passwd", "secret", "token", "api_key", "api-key"}
)
SECRET_QUERY_KEY_PARTS = (
    "authorization",
    "credential",
    "password",
    "passwd",
    "secret",
    "token",
    "api_key",
)


class CameraConfigurationError(ValueError):
    """Raised when camera configuration cannot be saved safely."""


@dataclass(frozen=True)
class CameraTestResult:
    """Safe result from the configuration-only camera test."""

    status: str
    message: str
    endpoint: str
    credential: dict[str, str | bool]
    details: dict[str, str | bool]

    def as_dict(self, camera: Camera) -> dict[str, object]:
        return {
            "camera_id": camera.id,
            "status": self.status,
            "message": self.message,
            "endpoint": redact_url(self.endpoint),
            "credential": self.credential,
            "details": self.details,
            "tested_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        }


def prepare_camera_config(
    *,
    name: object,
    rtsp_url: object,
    credential_ref: object = None,
    transport: object = "tcp",
    connection_options: object = None,
    preferred_stream: object = "main",
    region_of_interest: object = None,
    enabled: object = True,
    existing_credential_ref: str | None = None,
) -> CameraConfig:
    """Validate input and return only redacted, persistence-safe values."""
    normalized_name = _required_text(name, "name", max_length=255)
    raw_url = _required_text(rtsp_url, "rtsp_url", max_length=4096)
    validate_rtsp_url(raw_url)

    normalized_ref = _normalise_credential_ref(credential_ref)
    if normalized_ref is None:
        normalized_ref = existing_credential_ref
    if contains_secret_parts(raw_url) and normalized_ref is None:
        raise CameraConfigurationError(
            "credential_ref is required when the RTSP URL contains credentials"
        )

    normalized_transport = _normalise_transport(transport)
    normalized_options = _normalise_connection_options(connection_options, normalized_transport)
    normalized_stream = _required_text(
        preferred_stream,
        "preferred_stream",
        max_length=100,
    )
    normalized_roi = _normalise_region_of_interest(region_of_interest)
    normalized_enabled = _normalise_bool(enabled, "enabled")

    return CameraConfig(
        name=normalized_name,
        endpoint=redact_url(raw_url),
        credential_ref=normalized_ref,
        connection_options=normalized_options,
        preferred_stream=normalized_stream,
        region_of_interest=normalized_roi,
        enabled=normalized_enabled,
    )


def validate_rtsp_url(value: str) -> None:
    """Validate an RTSP endpoint without putting its value in an exception."""
    try:
        parsed = _parse_rtsp_url(value)
        if parsed.scheme.lower() not in SUPPORTED_SCHEMES:
            raise CameraConfigurationError("rtsp_url must use the rtsp or rtsps scheme")
        if not parsed.hostname:
            raise CameraConfigurationError("rtsp_url must include a camera host")
        if parsed.port is not None and not 1 <= parsed.port <= 65535:
            raise CameraConfigurationError("rtsp_url port must be from 1 through 65535")
    except CameraConfigurationError:
        raise
    except ValueError as error:
        raise CameraConfigurationError("rtsp_url is not a valid RTSP endpoint") from error


def contains_secret_parts(value: str) -> bool:
    """Return whether an endpoint includes user info or secret query parameters."""
    try:
        parsed = _parse_rtsp_url(value)
        if parsed.username is not None or parsed.password is not None or "@" in parsed.netloc:
            return True
        return any(
            _is_secret_query_key(key) for key, _ in parse_qsl(parsed.query, keep_blank_values=True)
        )
    except ValueError:
        return False


def _parse_rtsp_url(value: str) -> SplitResult:
    """Parse an endpoint, including the safe redaction marker used in storage."""
    return urlsplit(value.replace("[REDACTED]@", "redacted@"))


def test_camera_configuration(camera: Camera) -> CameraTestResult:
    """Check stored configuration and external credential availability.

    This slice does not open a network stream. It validates the endpoint and
    confirms that the configured external credential source can be resolved.
    """
    config = camera_config_from_record(camera)
    reference = parse_credential_ref(config.credential_ref)
    safe_credential = credential_status(reference)
    endpoint = config.endpoint
    effective_endpoint = endpoint

    if reference is not None:
        try:
            resolved = resolve_credential(reference)
        except Exception:
            resolved = None
            safe_credential = {
                "configured": True,
                "kind": reference.kind,
                "status": "unavailable",
            }
        if not resolved:
            return CameraTestResult(
                status="invalid",
                message="The credential reference is configured but its value is not available.",
                endpoint=endpoint,
                credential=safe_credential,
                details={
                    "network_test": False,
                    "reason": "external credential value is missing",
                },
            )
        candidate = _credential_endpoint(resolved)
        if candidate is not None:
            effective_endpoint = candidate

    if "[REDACTED]" in effective_endpoint:
        return CameraTestResult(
            status="invalid",
            message="The endpoint contains credentials that are not available for testing.",
            endpoint=endpoint,
            credential=safe_credential,
            details={
                "network_test": False,
                "reason": "use a credential reference containing the complete RTSP endpoint",
            },
        )

    try:
        validate_rtsp_url(effective_endpoint)
    except CameraConfigurationError as error:
        return CameraTestResult(
            status="invalid",
            message=str(error),
            endpoint=endpoint,
            credential=safe_credential,
            details={"network_test": False, "reason": "endpoint validation failed"},
        )

    transport = str(config.connection_options.get("transport", "tcp"))
    return CameraTestResult(
        status="valid",
        message=(
            "Camera configuration is valid. Network connection testing is not enabled "
            "in this configuration slice."
        ),
        endpoint=effective_endpoint,
        credential=safe_credential,
        details={
            "network_test": False,
            "transport": transport,
            "stream": config.preferred_stream,
        },
    )


def _credential_endpoint(value: str) -> str | None:
    """Accept a full RTSP URL or a JSON object with a URL field from a secret source."""
    candidate = value.strip()
    if candidate.lower().startswith(("rtsp://", "rtsps://")):
        return candidate
    try:
        decoded = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if isinstance(decoded, dict) and isinstance(decoded.get("url"), str):
        return str(decoded["url"])
    return None


def _normalise_credential_ref(value: object) -> str | None:
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if not isinstance(value, str):
        raise CameraConfigurationError("credential_ref must be a string")
    reference = parse_credential_ref(value)
    return None if reference is None else reference.reference


def _normalise_transport(value: object) -> str:
    if not isinstance(value, str) or value.lower() not in SUPPORTED_TRANSPORTS:
        raise CameraConfigurationError("transport must be tcp or udp")
    return value.lower()


def _normalise_connection_options(value: object, transport: str) -> dict[str, object]:
    if value is None:
        return {"transport": transport}
    if not isinstance(value, dict):
        raise CameraConfigurationError("connection_options must be an object")
    if _contains_secret_data(value):
        raise CameraConfigurationError("connection_options cannot contain secret values")
    try:
        json.dumps(value, ensure_ascii=True)
    except (TypeError, ValueError) as error:
        raise CameraConfigurationError("connection_options must contain JSON values") from error
    options = dict(value)
    options["transport"] = transport
    return options


def _normalise_region_of_interest(value: object) -> dict[str, float] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise CameraConfigurationError("region_of_interest must be an object")
    expected = {"x", "y", "width", "height"}
    if set(value) != expected:
        raise CameraConfigurationError("region_of_interest must contain x, y, width, and height")
    result: dict[str, float] = {}
    for key in expected:
        number = value[key]
        if isinstance(number, bool) or not isinstance(number, (int, float)):
            raise CameraConfigurationError(f"region_of_interest.{key} must be a number")
        if not math.isfinite(float(number)) or not 0 <= float(number) <= 1:
            raise CameraConfigurationError(f"region_of_interest.{key} must be between 0 and 1")
        result[key] = float(number)
    if result["width"] <= 0 or result["height"] <= 0:
        raise CameraConfigurationError("region_of_interest width and height must be positive")
    return result


def _normalise_bool(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise CameraConfigurationError(f"{field_name} must be a boolean")
    return value


def _required_text(value: object, field_name: str, *, max_length: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CameraConfigurationError(f"{field_name} is required")
    normalized = value.strip()
    if len(normalized) > max_length:
        raise CameraConfigurationError(f"{field_name} is too long")
    return normalized


def _contains_secret_key(key: object) -> bool:
    normalized = str(key).lower().replace("-", "_")
    return any(
        part in normalized
        for part in (
            "authorization",
            "credential",
            "password",
            "passwd",
            "secret",
            "token",
            "api_key",
        )
    )


def _is_secret_query_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    return normalized in SECRET_QUERY_KEYS or any(
        part in normalized for part in SECRET_QUERY_KEY_PARTS
    )


def _contains_secret_data(value: object) -> bool:
    if isinstance(value, Mapping):
        return any(
            _contains_secret_key(key) or _contains_secret_data(item) for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_secret_data(item) for item in value)
    if isinstance(value, str):
        return contains_secret_parts(value) or redact_url(value) != value
    return False
