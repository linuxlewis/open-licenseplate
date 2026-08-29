from pathlib import Path

from open_licenseplate.database import DATABASE_PRAGMAS, Database


def test_database_connection_enables_required_sqlite_pragmas(tmp_path: Path) -> None:
    database = Database(tmp_path / "database.sqlite3")

    try:
        with database.connection() as connection:
            assert database.pragma_values(connection) == DATABASE_PRAGMAS
    finally:
        database.dispose()
