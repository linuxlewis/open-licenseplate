"""Local audit for unredacted camera values in managed files."""

from __future__ import annotations

from typing import Any

from ..paths import ManagedPaths
from ..redaction import redact_text


def audit_managed_secrets(paths: ManagedPaths) -> dict[str, Any]:
    """Scan managed text and SQLite files without returning their contents."""
    candidates = (
        ("database", paths.database),
        ("settings", paths.settings),
        ("application log", paths.app_log),
        ("worker log", paths.worker_log),
    )
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
        if redact_text(content) != content:
            findings.append(f"{label}: unredacted secret pattern found")

    return {
        "status": "ok" if not findings else "failed",
        "files_scanned": scanned,
        "findings": findings,
    }
