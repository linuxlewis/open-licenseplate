import json
from datetime import UTC
from pathlib import Path

import pytest
from alembic.script import ScriptDirectory
from sqlalchemy import text

from open_licenseplate.cli import main
from open_licenseplate.config import load_settings
from open_licenseplate.database import Database, database_status, upgrade_database
from open_licenseplate.settings_store import ApplicationSetting, SettingsStore


def _database_path(tmp_path: Path) -> Path:
    return tmp_path / "data" / "open-licenseplate.sqlite3"


def test_fresh_database_migration_creates_settings_table(tmp_path: Path) -> None:
    database_path = _database_path(tmp_path)

    upgrade_database(database_path)
    assert database_status(database_path)["status"] == "ok"

    database = Database(database_path)
    try:
        with database.connection() as connection:
            tables = {
                row[0]
                for row in connection.execute(
                    text(
                        "SELECT name FROM sqlite_master "
                        "WHERE type = 'table' AND name IN "
                        "('alembic_version', 'application_settings')"
                    )
                )
            }
    finally:
        database.dispose()

    assert tables == {"alembic_version", "application_settings"}

    # An already current database must accept a second upgrade.
    upgrade_database(database_path)
    assert database_status(database_path)["current_revision"] == "0001_initial"


def test_database_status_discovers_head_from_alembic_for_missing_database(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_head = "head-from-alembic"

    monkeypatch.setattr(
        ScriptDirectory,
        "get_current_head",
        lambda _self: expected_head,
    )

    status = database_status(_database_path(tmp_path))

    assert status["status"] == "not_initialized"
    assert status["head_revision"] == expected_head


def test_non_secret_setting_is_available_after_restart(tmp_path: Path) -> None:
    database_path = _database_path(tmp_path)
    upgrade_database(database_path)
    settings = load_settings(cli_overrides={"storage.data_dir": tmp_path / "data"})
    database = Database(database_path)

    try:
        SettingsStore(database).set("server.port", 9007)
    finally:
        database.dispose()

    restarted_settings = load_settings(
        cli_overrides={"storage.data_dir": tmp_path / "data"},
    )

    assert restarted_settings.server.port == 9007
    assert restarted_settings.sources["server.port"] == "persisted"
    assert settings.server.port == 8421


def test_persisted_unsafe_development_setting_is_rejected(tmp_path: Path) -> None:
    database_path = _database_path(tmp_path)
    upgrade_database(database_path)
    database = Database(database_path)

    try:
        with pytest.raises(ValueError, match="not persistable"):
            SettingsStore(database).set("server.unsafe_development", True)
    finally:
        database.dispose()


def test_setting_updated_at_round_trip_is_aware_utc(tmp_path: Path) -> None:
    database_path = _database_path(tmp_path)
    upgrade_database(database_path)
    database = Database(database_path)

    try:
        SettingsStore(database).set("server.port", 9011)
    finally:
        database.dispose()

    restarted_database = Database(database_path)
    try:
        with restarted_database.session() as session:
            row = session.get(ApplicationSetting, "server.port")
            assert row is not None
            assert row.updated_at.tzinfo is UTC
    finally:
        restarted_database.dispose()


def test_settings_cli_writes_a_setting_for_the_next_process(tmp_path: Path) -> None:
    database_path = _database_path(tmp_path)
    upgrade_database(database_path)
    arguments = [
        "--data-dir",
        str(tmp_path / "data"),
        "--log-dir",
        str(tmp_path / "logs"),
    ]

    assert main(["settings", "set", "server.port", "9010", *arguments]) == 0

    settings = load_settings(cli_overrides={"storage.data_dir": tmp_path / "data"})

    assert settings.server.port == 9010
    assert settings.sources["server.port"] == "persisted"


def test_configuration_source_reporting_keeps_precedence_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = _database_path(tmp_path)
    upgrade_database(database_path)
    database = Database(database_path)
    try:
        SettingsStore(database).set("server.port", 9001)
    finally:
        database.dispose()

    monkeypatch.setenv("OPEN_LICENSEPLATE_SERVER__PORT", "9002")
    environment_settings = load_settings(
        cli_overrides={"storage.data_dir": tmp_path / "data"},
    )
    cli_settings = load_settings(
        cli_overrides={
            "storage.data_dir": tmp_path / "data",
            "server.port": 9003,
        },
    )
    monkeypatch.delenv("OPEN_LICENSEPLATE_SERVER__PORT")
    persisted_settings = load_settings(
        cli_overrides={"storage.data_dir": tmp_path / "data"},
    )

    assert environment_settings.server.port == 9002
    assert environment_settings.sources["server.port"] == "environment"
    assert cli_settings.server.port == 9003
    assert cli_settings.sources["server.port"] == "cli"
    assert persisted_settings.server.port == 9001
    assert persisted_settings.sources["server.port"] == "persisted"


def test_secret_setting_is_rejected_and_not_written(tmp_path: Path) -> None:
    database_path = _database_path(tmp_path)
    upgrade_database(database_path)
    database = Database(database_path)

    try:
        with pytest.raises(ValueError, match="not persistable"):
            SettingsStore(database).set("server.password", "do-not-store")
        with pytest.raises(ValueError, match="secret values"):
            SettingsStore(database).set("app_name", {"password": "do-not-store"})
        with pytest.raises(ValueError, match="secret values"):
            SettingsStore(database).set("app_name", {"nested": {"token": "do-not-store"}})

        with database.connection() as connection:
            rows = connection.execute(
                text("SELECT setting_key, value_json FROM application_settings")
            ).all()
    finally:
        database.dispose()

    assert rows == []
    assert "do-not-store" not in json.dumps(rows)
