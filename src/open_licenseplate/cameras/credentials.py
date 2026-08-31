"""Safe camera credential references.

The application stores a reference such as ``env:CAMERA_RTSP_URL`` or
``keychain:open-licenseplate/camera-1``. It never stores the value resolved
from that reference.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Literal

CredentialKind = Literal["environment", "keychain"]
_ENVIRONMENT_REFERENCE = re.compile(r"^env:(?P<name>[A-Za-z_][A-Za-z0-9_]*)$")
_KEYCHAIN_REFERENCE = re.compile(
    r"^keychain:(?P<service>[A-Za-z0-9._-]+)/(?P<account>[A-Za-z0-9._@+-]+)$"
)


@dataclass(frozen=True)
class CredentialReference:
    """A validated pointer to an external credential value."""

    kind: CredentialKind
    reference: str
    name: str
    account: str | None = None


def parse_credential_ref(value: str | None) -> CredentialReference | None:
    """Validate and parse one environment or Keychain reference."""
    if value is None or not value.strip():
        return None

    normalized = value.strip()
    environment_match = _ENVIRONMENT_REFERENCE.fullmatch(normalized)
    if environment_match is not None:
        return CredentialReference(
            kind="environment",
            reference=normalized,
            name=environment_match.group("name"),
        )

    keychain_match = _KEYCHAIN_REFERENCE.fullmatch(normalized)
    if keychain_match is not None:
        return CredentialReference(
            kind="keychain",
            reference=normalized,
            name=keychain_match.group("service"),
            account=keychain_match.group("account"),
        )

    raise ValueError("credential_ref must use env:VARIABLE_NAME or keychain:SERVICE/ACCOUNT format")


def resolve_credential(reference: CredentialReference) -> str | None:
    """Resolve a credential value without exposing it to application output."""
    if reference.kind == "environment":
        return os.environ.get(reference.name)

    try:
        import importlib

        keyring: Any = importlib.import_module("keyring")
    except ImportError:
        return _resolve_keychain_with_security(reference)

    value = keyring.get_password(reference.name, reference.account or "")
    return value if isinstance(value, str) else None


def credential_status(reference: CredentialReference | None) -> dict[str, str | bool]:
    """Return safe availability information without returning a secret value."""
    if reference is None:
        return {
            "configured": False,
            "kind": "none",
            "status": "not_configured",
        }

    try:
        value = resolve_credential(reference)
    except Exception:
        return {
            "configured": True,
            "kind": reference.kind,
            "status": "unavailable",
        }

    return {
        "configured": True,
        "kind": reference.kind,
        "status": "available" if value else "missing",
    }


def _resolve_keychain_with_security(reference: CredentialReference) -> str | None:
    """Use the macOS command-line Keychain client when keyring is unavailable."""
    if sys.platform != "darwin" or reference.account is None:
        return None

    try:
        result = subprocess.run(
            [
                "security",
                "find-generic-password",
                "-s",
                reference.name,
                "-a",
                reference.account,
                "-w",
            ],
            capture_output=True,
            check=False,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.rstrip("\r\n") or None
