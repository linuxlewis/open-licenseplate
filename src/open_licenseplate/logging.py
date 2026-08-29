"""Structured, local logging with secret redaction."""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .redaction import redact_text, redact_value

_CONTEXT_FIELDS = (
    "worker_id",
    "camera_id",
    "capture_session_id",
    "event_id",
    "job_id",
    "attempt_id",
    "model_id",
    "model_checksum_prefix",
    "error_category",
)


class JsonFormatter(logging.Formatter):
    """Format one log record as one redacted JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.fromtimestamp(record.created, tz=UTC)
        payload: dict[str, Any] = {
            "timestamp": timestamp.isoformat().replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "message": redact_text(record.getMessage()),
            "process_id": record.process,
            "process": record.processName,
        }
        for field in _CONTEXT_FIELDS:
            if field in record.__dict__:
                payload[field] = redact_value(record.__dict__[field])
        return json.dumps(payload, ensure_ascii=True, separators=(",", ":"), default=str)


class RedactionFilter(logging.Filter):
    """Remove secrets from the message before any handler writes it."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact_text(record.getMessage())
        record.args = ()
        return True


def configure_logging(
    *,
    level: str = "INFO",
    log_file: Path | None = None,
) -> None:
    """Configure stderr and optional file output for the current process."""
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level.upper())

    formatter = JsonFormatter()
    redaction_filter = RedactionFilter()

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(redaction_filter)
    root.addHandler(console_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.addFilter(redaction_filter)
        root.addHandler(file_handler)
