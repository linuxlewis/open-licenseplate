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
from .config import SettingsError, load_settings
from .logging import configure_logging
from .paths import ManagedPaths

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
        help="database upgrade placeholder",
    )
    _add_runtime_options(upgrade_parser)

    doctor_parser = commands.add_parser("doctor", help="check local application readiness")
    doctor_parser.add_argument("--json", action="store_true", help="write diagnostics as JSON")
    _add_runtime_options(doctor_parser)
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
    _load_cli_settings(arguments)
    print(
        "Database support arrives in P01; `db upgrade` is not available in P00.",
        file=sys.stderr,
    )
    return 1


def _doctor_payload(settings: Any, paths: ManagedPaths) -> tuple[dict[str, Any], bool]:
    directories = paths.directory_checks()
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
            "log_dir": str(paths.log_dir),
        },
        "directories": directories,
        "database": {
            "status": "not_implemented",
            "detail": "Database support arrives in P01.",
        },
        "ready": False,
    }
    return payload, bool(payload["ready"])


def _run_doctor(arguments: argparse.Namespace) -> int:
    settings, paths = _load_cli_settings(arguments)
    paths.ensure_directories()
    payload, ready = _doctor_payload(settings, paths)
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
        print(f"result: {'ready' if ready else 'not ready'}")
    return 0 if ready else 1


def main(argv: Sequence[str] | None = None) -> int:
    """Run one CLI command and return its process exit code."""
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "serve":
            return _run_serve(arguments)
        if arguments.command == "db" and arguments.db_command == "upgrade":
            return _run_database_upgrade(arguments)
        if arguments.command == "doctor":
            return _run_doctor(arguments)
    except (SettingsError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except Exception as error:
        print(f"error: command failed: {error}", file=sys.stderr)
        return 1

    parser.print_help()
    return 2
