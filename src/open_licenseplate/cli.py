"""Command-line entry points for the application foundation."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .app import create_app
from .cameras.audit import audit_managed_secrets
from .config import SettingsError, load_settings
from .database import Database, database_status, upgrade_database
from .logging import configure_logging
from .paths import ManagedPaths
from .redaction import redact_text
from .settings_store import SettingsStore, validate_setting_key

logger = logging.getLogger("open_licenseplate.cli")


def _add_runtime_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--host", default=argparse.SUPPRESS)
    parser.add_argument("--port", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--data-dir", type=Path, default=argparse.SUPPRESS)
    parser.add_argument("--log-dir", type=Path, default=argparse.SUPPRESS)
    parser.add_argument("--log-level", default=argparse.SUPPRESS)
    parser.add_argument("--unsafe-development", action="store_true", default=argparse.SUPPRESS)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="open-licenseplate",
        description="Local-first license plate detection application.",
    )
    _add_runtime_options(parser)
    commands = parser.add_subparsers(dest="command")

    serve_parser = commands.add_parser("serve", help="start the local web server")
    _add_runtime_options(serve_parser)

    db_parser = commands.add_parser("db", help="database commands")
    db_commands = db_parser.add_subparsers(dest="db_command")
    upgrade_parser = db_commands.add_parser(
        "upgrade",
        help="upgrade the database to the current migration",
    )
    _add_runtime_options(upgrade_parser)

    settings_parser = commands.add_parser(
        "settings",
        help="persist non-secret application settings",
    )
    settings_commands = settings_parser.add_subparsers(dest="settings_command")
    set_parser = settings_commands.add_parser("set", help="persist one setting")
    set_parser.add_argument("setting_key")
    set_parser.add_argument("value")
    _add_runtime_options(set_parser)

    doctor_parser = commands.add_parser("doctor", help="check local application readiness")
    doctor_parser.add_argument("--json", action="store_true", help="write diagnostics as JSON")
    doctor_parser.add_argument(
        "--audit-secrets",
        action="store_true",
        help="scan managed files for unredacted secret patterns",
    )
    _add_runtime_options(doctor_parser)

    dev_parser = commands.add_parser("dev", help="development-only commands")
    dev_commands = dev_parser.add_subparsers(dest="dev_command")
    fixture_parser = dev_commands.add_parser(
        "fixture",
        help="create empty managed directories without application data",
    )
    _add_runtime_options(fixture_parser)
    return parser


def _cli_overrides(arguments: argparse.Namespace) -> dict[str, Any]:
    values: dict[str, Any] = {}
    if hasattr(arguments, "host"):
        values["server.host"] = arguments.host
    if hasattr(arguments, "port"):
        values["server.port"] = arguments.port
    if hasattr(arguments, "data_dir"):
        values["storage.data_dir"] = arguments.data_dir
    if hasattr(arguments, "log_dir"):
        values["storage.log_dir"] = arguments.log_dir
    if hasattr(arguments, "log_level"):
        values["log_level"] = arguments.log_level
    if hasattr(arguments, "unsafe_development"):
        values["server.unsafe_development"] = arguments.unsafe_development
    return values


def _load_cli_settings(arguments: argparse.Namespace) -> tuple[Any, ManagedPaths]:
    settings = load_settings(cli_overrides=_cli_overrides(arguments))
    return settings, ManagedPaths.from_settings(settings)


def _run_serve(arguments: argparse.Namespace) -> int:
    settings, paths = _load_cli_settings(arguments)
    paths.ensure_directories()
    configure_logging(level=settings.log_level, log_file=paths.app_log)
    logger.info(
        "starting server",
        extra={
            "host": settings.server.host,
            "port": settings.server.port,
        },
    )

    import uvicorn

    uvicorn.run(
        create_app(settings),
        host=settings.server.host,
        port=settings.server.port,
        log_config=None,
        access_log=False,
    )
    return 0


def _run_database_upgrade(arguments: argparse.Namespace) -> int:
    settings = load_settings(
        cli_overrides=_cli_overrides(arguments),
        include_persisted=False,
    )
    paths = ManagedPaths.from_settings(settings)
    paths.ensure_directories()
    upgrade_database(paths.database)
    status = database_status(paths.database)
    print(
        f"Database upgraded to {status['current_revision']} at {paths.database}",
    )
    return 0


def _parse_setting_value(raw_value: str) -> Any:
    try:
        return json.loads(raw_value)
    except json.JSONDecodeError:
        return raw_value


def _run_settings_set(arguments: argparse.Namespace) -> int:
    settings, paths = _load_cli_settings(arguments)
    setting_key = validate_setting_key(arguments.setting_key)
    value = _parse_setting_value(arguments.value)

    candidate = settings.model_dump(mode="python")
    candidate = _merge_for_cli(candidate, setting_key, value)
    type(settings).model_validate(candidate)

    status = database_status(paths.database)
    if status["status"] != "ok":
        raise SettingsError(
            "database is not ready; run `open-licenseplate db upgrade` before saving settings"
        )

    database = Database(paths.database)
    try:
        SettingsStore(database).set(setting_key, value)
    finally:
        database.dispose()
    print(f"Saved setting {setting_key}.")
    return 0


def _merge_for_cli(base: dict[str, Any], setting_key: str, value: Any) -> dict[str, Any]:
    cursor = base
    parts = setting_key.split(".")
    for part in parts[:-1]:
        child = cursor.get(part)
        if not isinstance(child, dict):
            raise SettingsError(f"setting path is not an object: {setting_key}")
        cursor = child
    cursor[parts[-1]] = value
    return base


def _doctor_payload(
    settings: Any,
    paths: ManagedPaths,
    *,
    audit_secrets: bool = False,
) -> tuple[dict[str, Any], bool]:
    directories = paths.directory_checks()
    database = database_status(paths.database)
    payload = {
        "application": settings.app_name,
        "configuration": {
            "status": "ok",
            "sources": settings.sources,
        },
        "paths": {
            "data_dir": str(paths.data_dir),
            "database": str(paths.database),
            "models": str(paths.models),
            "artifacts": str(paths.artifacts),
            "staging": str(paths.staging),
            "settings": str(paths.settings),
            "log_dir": str(paths.log_dir),
        },
        "directories": directories,
        "database": database,
        "ready": all(directories.values()) and database["status"] == "ok",
    }
    if audit_secrets:
        payload["secret_audit"] = audit_managed_secrets(paths)
        payload["ready"] = bool(payload["ready"] and payload["secret_audit"]["status"] == "ok")
    return payload, bool(payload["ready"])


def _run_doctor(arguments: argparse.Namespace) -> int:
    settings, paths = _load_cli_settings(arguments)
    paths.ensure_directories()
    payload, ready = _doctor_payload(
        settings,
        paths,
        audit_secrets=arguments.audit_secrets,
    )
    if arguments.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(f"{settings.app_name} doctor")
        print(f"configuration: {payload['configuration']['status']}")
        for name, value in payload["directories"].items():
            print(f"{name}: {'ok' if value else 'not ready'}")
        database = payload["database"]
        print(f"database: {database['status']}")
        print(f"database detail: {database['detail']}")
        if database["current_revision"] is not None:
            print(f"database revision: {database['current_revision']}")
        if "secret_audit" in payload:
            audit = payload["secret_audit"]
            print(f"secret audit: {audit['status']}")
            for finding in audit["findings"]:
                print(f"secret audit detail: {finding}")
        print(f"result: {'ready' if ready else 'not ready'}")
    return 0 if ready else 1


def _run_dev_fixture(arguments: argparse.Namespace) -> int:
    """Create an empty managed directory layout without creating application data."""
    settings = load_settings(
        cli_overrides=_cli_overrides(arguments),
        include_persisted=False,
    )
    paths = ManagedPaths.from_settings(settings)
    paths.ensure_directories()
    print(f"Empty development fixture ready at {paths.data_dir}")
    print("No camera, model, plate, event, job, or OCR data was created.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run one CLI command and return its process exit code."""
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "serve":
            return _run_serve(arguments)
        if arguments.command == "db" and arguments.db_command == "upgrade":
            return _run_database_upgrade(arguments)
        if arguments.command == "settings" and arguments.settings_command == "set":
            return _run_settings_set(arguments)
        if arguments.command == "doctor":
            return _run_doctor(arguments)
        if arguments.command == "dev" and arguments.dev_command == "fixture":
            return _run_dev_fixture(arguments)
    except (SettingsError, ValueError) as error:
        print(f"error: {redact_text(str(error))}", file=sys.stderr)
        return 2
    except Exception as error:
        print(f"error: command failed: {redact_text(str(error))}", file=sys.stderr)
        return 1

    parser.print_help()
    return 2
