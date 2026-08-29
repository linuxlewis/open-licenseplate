"""Secret redaction for logs, diagnostics, and user-facing errors."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

_REDACTED = "[REDACTED]"
_URL_PATTERN = re.compile(r"(?P<url>rtsps?://[^\s\"'<>]+)", re.IGNORECASE)
_KEY_VALUE_PATTERN = re.compile(
    r"(?P<key>\b(?:authorization|credential|password|passwd|secret|token|api[_-]?key)\b)"
    r"(?P<separator>\s*[:=]\s*)"
    r"(?P<quote>[\"']?)(?P<value>[^\s,;}\]\"']+)",
    re.IGNORECASE,
)
_SECRET_KEY_PARTS = (
    "authorization",
    "credential",
    "password",
    "passwd",
    "secret",
    "token",
    "api_key",
    "api-key",
)


def _is_secret_key(key: object) -> bool:
    key_lower = str(key).lower().replace("-", "_")
    return any(part in key_lower for part in _SECRET_KEY_PARTS)


def redact_url(value: str) -> str:
    """Redact RTSP user information and sensitive query parameters."""
    try:
        parsed = urlsplit(value)
    except ValueError:
        return value

    if not parsed.scheme:
        return value

    netloc = parsed.netloc
    if "@" in netloc:
        host = netloc.rsplit("@", 1)[1]
        netloc = f"{_REDACTED}@{host}"

    query_parts = []
    for key, query_value in parse_qsl(parsed.query, keep_blank_values=True):
        query_parts.append((key, _REDACTED if _is_secret_key(key) else query_value))

    return urlunsplit((parsed.scheme, netloc, parsed.path, urlencode(query_parts), parsed.fragment))


def redact_text(value: str) -> str:
    """Redact embedded RTSP credentials and common key-value secrets."""

    def replace_url(match: re.Match[str]) -> str:
        url = match.group("url")
        trailing = ""
        while url and url[-1] in ".,;:)]}":
            trailing = url[-1] + trailing
            url = url[:-1]
        return redact_url(url) + trailing

    redacted = _URL_PATTERN.sub(replace_url, value)

    def replace_key_value(match: re.Match[str]) -> str:
        return f"{match.group('key')}{match.group('separator')}{_REDACTED}"

    return _KEY_VALUE_PATTERN.sub(replace_key_value, redacted)


def redact_value(value: Any) -> Any:
    """Return a recursively redacted copy of a diagnostic value."""
    if isinstance(value, Mapping):
        return {
            str(key): _REDACTED if _is_secret_key(key) else redact_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_value(item) for item in value)
    if isinstance(value, str):
        return redact_text(value)
    return value
