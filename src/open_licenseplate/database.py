"""SQLite engine, connection settings, and Alembic migration helpers."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from alembic import command
from alembic.config import Config
from alembic.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import Connection, create_engine, event
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

DATABASE_PRAGMAS = {
    "journal_mode": "wal",
    "synchronous": 2,
    "foreign_keys": 1,
    "busy_timeout": 5000,
}
"""Required SQLite pragma values.

SQLite reports ``synchronous = FULL`` as the integer value ``2``.
"""

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ALEMBIC_CONFIG_PATH = _REPOSITORY_ROOT / "alembic.ini"


class Database:
    """Own one SQLAlchemy engine and its short-lived sessions."""

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser()
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)

        connect_args: dict[str, Any] = {
            "check_same_thread": False,
            "timeout": 5.0,
        }
        if str(self.path) == ":memory:":
            self.engine = create_engine(
                "sqlite+pysqlite:///:memory:",
                connect_args=connect_args,
            )
        else:
            url = f"sqlite+pysqlite:///{self.path.resolve().as_posix()}"
            self.engine = create_engine(
                url,
                connect_args=connect_args,
                pool_pre_ping=True,
            )

        self._install_sqlite_pragmas()
        self.session_factory = sessionmaker(
            bind=self.engine,
            autoflush=False,
            expire_on_commit=False,
        )

    def _install_sqlite_pragmas(self) -> None:
        @event.listens_for(self.engine, "connect")
        def set_pragmas(dbapi_connection: Any, _connection_record: Any) -> None:
            cursor = dbapi_connection.cursor()
            try:
                cursor.execute("PRAGMA journal_mode = WAL")
                cursor.fetchone()
                cursor.execute("PRAGMA synchronous = FULL")
                cursor.execute("PRAGMA foreign_keys = ON")
                cursor.execute("PRAGMA busy_timeout = 5000")
            finally:
                cursor.close()

    @contextmanager
    def connection(self) -> Iterator[Connection]:
        """Yield one connection and close it when the operation finishes."""
        with self.engine.connect() as connection:
            yield connection

    @contextmanager
    def session(self) -> Iterator[Session]:
        """Yield one transactional session with rollback on errors."""
        with self.session_factory() as session:
            try:
                yield session
            except Exception:
                session.rollback()
                raise
            else:
                session.commit()

    def pragma_values(self, connection: Connection | None = None) -> dict[str, Any]:
        """Read the required SQLite pragma values from one connection."""
        if connection is not None:
            return _read_pragma_values(connection)
        with self.connection() as owned_connection:
            return _read_pragma_values(owned_connection)

    def dispose(self) -> None:
        """Close pooled connections owned by this database."""
        self.engine.dispose()


def _read_pragma_values(connection: Connection) -> dict[str, Any]:
    return {
        "journal_mode": str(connection.exec_driver_sql("PRAGMA journal_mode").scalar()).lower(),
        "synchronous": int(connection.exec_driver_sql("PRAGMA synchronous").scalar_one()),
        "foreign_keys": int(connection.exec_driver_sql("PRAGMA foreign_keys").scalar_one()),
        "busy_timeout": int(connection.exec_driver_sql("PRAGMA busy_timeout").scalar_one()),
    }


def _alembic_config() -> Config:
    config = Config(str(ALEMBIC_CONFIG_PATH))
    config.set_main_option("script_location", str(_REPOSITORY_ROOT / "migrations"))
    return config


def _migration_head(config: Config | None = None) -> str:
    migration_config = config or _alembic_config()
    head = ScriptDirectory.from_config(migration_config).get_current_head()
    if head is None:
        raise RuntimeError("Alembic has no migration head")
    return head


def migration_revisions(connection: Connection) -> tuple[str | None, str]:
    """Return the current database revision and the application head."""
    config = _alembic_config()
    current = MigrationContext.configure(connection).get_current_revision()
    return current, _migration_head(config)


def upgrade_database(path: Path) -> None:
    """Upgrade a SQLite database to the current Alembic head."""
    database = Database(path)
    try:
        config = _alembic_config()
        with database.engine.begin() as connection:
            config.attributes["connection"] = connection
            command.upgrade(config, "head")
    finally:
        database.dispose()


def database_status(path: Path) -> dict[str, Any]:
    """Return migration and pragma state without creating a new database file."""
    database_path = path.expanduser()
    head_revision = _migration_head()
    if not database_path.is_file():
        return {
            "status": "not_initialized",
            "detail": "Database file does not exist; run `open-licenseplate db upgrade`.",
            "current_revision": None,
            "head_revision": head_revision,
            "pragmas": {},
        }

    database = Database(database_path)
    try:
        with database.connection() as connection:
            current_revision, head_revision = migration_revisions(connection)
            pragmas = database.pragma_values(connection)
    except SQLAlchemyError as error:
        return {
            "status": "error",
            "detail": f"Database check failed: {error}",
            "current_revision": None,
            "head_revision": head_revision,
            "pragmas": {},
        }
    finally:
        database.dispose()

    if current_revision != head_revision:
        return {
            "status": "not_migrated",
            "detail": (
                f"Database revision is {current_revision or 'none'}; "
                f"expected {head_revision}. Run `open-licenseplate db upgrade`."
            ),
            "current_revision": current_revision,
            "head_revision": head_revision,
            "pragmas": pragmas,
        }

    if pragmas != DATABASE_PRAGMAS:
        return {
            "status": "invalid",
            "detail": "Database pragmas do not match the required safety settings.",
            "current_revision": current_revision,
            "head_revision": head_revision,
            "pragmas": pragmas,
        }

    return {
        "status": "ok",
        "detail": "Database is migrated and its required pragmas are active.",
        "current_revision": current_revision,
        "head_revision": head_revision,
        "pragmas": pragmas,
    }
