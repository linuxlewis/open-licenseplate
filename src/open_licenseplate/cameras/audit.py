"""Local audit for unredacted camera values in managed files."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from ..paths import ManagedPaths
from ..redaction import redact_text


def audit_managed_secrets(
    paths: ManagedPaths,
    *,
    extra_texts: Mapping[str, str] | None = None,
    secret_values: Iterable[str] = (),
) -> dict[str, Any]:
    """Scan managed files and rendered surfaces without returning their contents."""
    candidates = (
        ("database", paths.database),
        ("settings", paths.settings),
        ("application log", paths.app_log),
        ("worker log", paths.worker_log),
    )
    known_secrets = tuple(value for value in secret_values if value)
    findings: list[str] = []
    scanned = 0
    for label, path in candidates:
        if not path.is_file():
            continue
        scanned += 1
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            findings.append(f"{label}: could not be read")
            continue
        if redact_text(content) != content or _contains_known_secret(content, known_secrets):
            findings.append(f"{label}: unredacted secret pattern found")

    for label, content in (extra_texts or {}).items():
        if redact_text(content) != content or _contains_known_secret(content, known_secrets):
            findings.append(f"{label}: unredacted secret pattern found")

    return {
        "status": "ok" if not findings else "failed",
        "files_scanned": scanned,
        "findings": findings,
    }


def _contains_known_secret(content: str, secret_values: Iterable[str]) -> bool:
    return any(secret and secret in content for secret in secret_values)
