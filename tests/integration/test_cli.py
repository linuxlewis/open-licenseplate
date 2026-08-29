import json
from pathlib import Path

import pytest

from open_licenseplate.cli import main


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
    assert "0001_initial" in capsys.readouterr().out
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
