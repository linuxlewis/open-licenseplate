"""Camera configuration and credential-reference support."""

from .audit import audit_managed_secrets
from .credentials import CredentialReference, credential_status, parse_credential_ref
from .repository import Camera, CameraConfig, CameraRepository
from .service import (
    CameraConfigurationError,
    CameraTestResult,
    test_camera_configuration,
    test_camera_connection,
)

__all__ = [
    "Camera",
    "CameraConfig",
    "CameraConfigurationError",
    "CameraRepository",
    "CameraTestResult",
    "CredentialReference",
    "audit_managed_secrets",
    "credential_status",
    "parse_credential_ref",
    "test_camera_configuration",
    "test_camera_connection",
]
