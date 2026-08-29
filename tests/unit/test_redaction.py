import json

from open_licenseplate.logging import JsonFormatter, RedactionFilter
from open_licenseplate.redaction import redact_text, redact_url, redact_value


def test_redact_url_removes_rtsp_user_information_and_sensitive_query_values() -> None:
    value = "rtsp://camera-user:camera-password@example.test:554/live?password=query-password&transport=tcp"

    redacted = redact_url(value)

    assert "camera-user" not in redacted
    assert "camera-password" not in redacted
    assert "query-password" not in redacted
    assert "transport=tcp" in redacted
    assert "[REDACTED]@" in redacted


def test_redact_text_handles_urls_and_key_value_secrets() -> None:
    value = "open rtsp://user:secret@example.test/stream, password=another-secret"

    redacted = redact_text(value)

    assert "secret" not in redacted
    assert "another-secret" not in redacted
    assert "[REDACTED]@" in redacted
    assert "password=[REDACTED]" in redacted


def test_redact_text_handles_json_and_authorization_secrets() -> None:
    value = '{"password":"json-secret","authorization":"Bearer bearer-secret"}'

    redacted = redact_text(value)

    assert redacted == '{"password":"[REDACTED]","authorization":"[REDACTED]"}'
    assert "json-secret" not in redacted
    assert "bearer-secret" not in redacted


def test_redact_text_keeps_all_rtsp_query_parameters() -> None:
    value = "rtsp://user:secret@example.test/stream?password=query-secret&token=token-secret"

    redacted = redact_text(value)

    assert "password=[REDACTED]" in redacted
    assert "token=[REDACTED]" in redacted
    assert "query-secret" not in redacted
    assert "token-secret" not in redacted


def test_redact_value_recurses_through_diagnostics() -> None:
    value = {
        "url": "rtsp://user:secret@example.test/stream",
        "password": "secret",
        "nested": [{"token": "secret"}, "safe"],
    }

    redacted = redact_value(value)

    assert redacted["password"] == "[REDACTED]"
    assert redacted["nested"] == [{"token": "[REDACTED]"}, "safe"]
    assert "secret" not in json.dumps(redacted)


def test_json_formatter_emits_structured_redacted_context() -> None:
    import logging

    record = logging.LogRecord(
        "test",
        logging.INFO,
        __file__,
        1,
        "connect %s",
        ("rtsp://user:secret@example.test/stream",),
        None,
    )
    record.job_id = "job-1"
    record.password = "secret"
    RedactionFilter().filter(record)

    payload = json.loads(JsonFormatter().format(record))

    assert payload["message"] == "connect rtsp://[REDACTED]@example.test/stream"
    assert payload["job_id"] == "job-1"
    assert "secret" not in json.dumps(payload)
