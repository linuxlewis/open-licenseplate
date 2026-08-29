import json
from pathlib import Path

import pytest

from open_licenseplate.cli import main


def test_db_upgrade_command_is_a_p01_placeholder(
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

    assert result == 1
    assert "Database support arrives in P01" in capsys.readouterr().err
    assert not (tmp_path / "data").exists()


def test_doctor_reports_unmigrated_data_directory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    result = main(
        ["doctor", "--data-dir", str(tmp_path / "data"), "--log-dir", str(tmp_path / "logs")]
    )

    assert result == 1
    output = capsys.readouterr().out
    assert "result: not ready" in output
    assert "Database support arrives in P01" in output


def test_doctor_json_reports_truthful_pre_database_state(
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
    assert payload["database"]["status"] == "not_implemented"


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
