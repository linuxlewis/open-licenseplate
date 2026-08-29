import json
from pathlib import Path

import pytest

from open_licenseplate.cameras.repository import CameraRepository
from open_licenseplate.cameras.service import prepare_camera_config
from open_licenseplate.cli import main
from open_licenseplate.config import SettingsError
from open_licenseplate.database import Database
from open_licenseplate.paths import ManagedPaths

SECRET_RTSP_URL = (
    "rtsp://camera-user:camera-password@example.test/stream"
    "?password=query-password&token=query-token"
)


def test_db_upgrade_command_runs_first_migration(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = main(
        [
            "db",
            "upgrade",
            "--data-dir",
            str(tmp_path / "data"),
            "--log-dir",
            str(tmp_path / "logs"),
        ]
    )

    assert result == 0
    assert "0002_cameras" in capsys.readouterr().out
    assert (tmp_path / "data" / "open-licenseplate.sqlite3").is_file()


def test_dev_fixture_command_creates_only_managed_directories(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data_dir = tmp_path / "data"
    log_dir = tmp_path / "logs"

    result = main(
        [
            "dev",
            "fixture",
            "--data-dir",
            str(data_dir),
            "--log-dir",
            str(log_dir),
        ]
    )

    assert result == 0
    output = capsys.readouterr().out
    assert "Empty development fixture ready" in output
    assert "No camera, model, plate, event, job, or OCR data was created." in output
    assert (data_dir / "models").is_dir()
    assert (data_dir / "artifacts").is_dir()
    assert (data_dir / "staging").is_dir()
    assert not (data_dir / "open-licenseplate.sqlite3").exists()
    assert not (data_dir / "settings.json").exists()


def test_doctor_reports_uninitialized_database(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    result = main(
        ["doctor", "--data-dir", str(tmp_path / "data"), "--log-dir", str(tmp_path / "logs")]
    )

    assert result == 1
    output = capsys.readouterr().out
    assert "result: not ready" in output
    assert "database: not_initialized" in output


def test_doctor_json_reports_uninitialized_database(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = main(
        [
            "doctor",
            "--data-dir",
            str(tmp_path / "data"),
            "--log-dir",
            str(tmp_path / "logs"),
            "--json",
        ]
    )

    assert result == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ready"] is False
    assert payload["database"]["status"] == "not_initialized"


def test_doctor_reports_ready_after_database_upgrade(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    arguments = [
        "--data-dir",
        str(tmp_path / "data"),
        "--log-dir",
        str(tmp_path / "logs"),
    ]

    assert main(["db", "upgrade", *arguments]) == 0
    capsys.readouterr()

    result = main(["doctor", *arguments])

    assert result == 0
    output = capsys.readouterr().out
    assert "database: ok" in output
    assert "result: ready" in output


def test_doctor_audit_secrets_reports_managed_files_are_safe(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    arguments = [
        "--data-dir",
        str(tmp_path / "data"),
        "--log-dir",
        str(tmp_path / "logs"),
    ]

    assert main(["db", "upgrade", *arguments]) == 0
    capsys.readouterr()

    result = main(["doctor", "--audit-secrets", "--json", *arguments])

    assert result == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["secret_audit"] == {
        "files_scanned": 1,
        "findings": [],
        "status": "ok",
    }


def test_doctor_audit_secrets_checks_resolved_values_in_logs_and_surfaces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    data_dir = tmp_path / "data"
    log_dir = tmp_path / "logs"
    arguments = ["--data-dir", str(data_dir), "--log-dir", str(log_dir)]
    assert main(["db", "upgrade", *arguments]) == 0
    capsys.readouterr()

    monkeypatch.setenv("CAMERA_RTSP_URL", SECRET_RTSP_URL)
    paths = ManagedPaths.from_roots(data_dir, log_dir)
    database = Database(paths.database)
    try:
        CameraRepository(database).create(
            prepare_camera_config(
                name="Front gate",
                rtsp_url="rtsp://example.test/stream",
                credential_ref="env:CAMERA_RTSP_URL",
            )
        )
    finally:
        database.dispose()
    paths.app_log.write_text(SECRET_RTSP_URL, encoding="utf-8")

    result = main(["doctor", "--audit-secrets", "--json", *arguments])

    assert result == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["secret_audit"]["status"] == "failed"
    assert any("application log" in finding for finding in payload["secret_audit"]["findings"])
    assert "camera-password" not in json.dumps(payload)


def test_serve_command_passes_explicit_options_to_uvicorn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import uvicorn

    called: dict[str, object] = {}

    def fake_run(application: object, **kwargs: object) -> None:
        called["application"] = application
        called.update(kwargs)

    monkeypatch.setattr(uvicorn, "run", fake_run)
    result = main(
        [
            "serve",
            "--host",
            "127.0.0.1",
            "--port",
            "9000",
            "--data-dir",
            str(tmp_path / "data"),
            "--log-dir",
            str(tmp_path / "logs"),
        ]
    )

    assert result == 0
    assert called["host"] == "127.0.0.1"
    assert called["port"] == 9000
    assert called["access_log"] is False


def test_cli_redacts_settings_errors_before_writing_stderr(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def raise_settings_error(_arguments: object) -> int:
        raise SettingsError(SECRET_RTSP_URL)

    monkeypatch.setattr("open_licenseplate.cli._run_settings_set", raise_settings_error)

    assert main(["settings", "set", "server.port", "9000"]) == 2

    output = capsys.readouterr().err
    assert "camera-user" not in output
    assert "camera-password" not in output
    assert "query-password" not in output
    assert "query-token" not in output
    assert "[REDACTED]@" in output
    assert "password=[REDACTED]" in output


def test_cli_redacts_unexpected_errors_before_writing_stderr(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def raise_unexpected_error(_arguments: object) -> int:
        raise RuntimeError(SECRET_RTSP_URL)

    monkeypatch.setattr("open_licenseplate.cli._run_doctor", raise_unexpected_error)

    assert main(["doctor"]) == 1

    output = capsys.readouterr().err
    assert "camera-user" not in output
    assert "camera-password" not in output
    assert "query-password" not in output
    assert "query-token" not in output
    assert "[REDACTED]@" in output
    assert "password=[REDACTED]" in output
