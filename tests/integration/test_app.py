from pathlib import Path

from fastapi.testclient import TestClient

from open_licenseplate.app import create_app
from open_licenseplate.config import load_settings


def _settings(tmp_path: Path):
    return load_settings(
        cli_overrides={
            "storage.data_dir": tmp_path / "data",
            "storage.log_dir": tmp_path / "logs",
        }
    )


def test_health_endpoints_distinguish_liveness_from_readiness(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    app = create_app(settings)

    with TestClient(app) as client:
        live_response = client.get("/api/v1/health/live")
        ready_response = client.get("/api/v1/health/ready")

    assert live_response.status_code == 200
    assert live_response.json() == {"status": "ok"}
    assert ready_response.status_code == 503
    assert ready_response.json()["status"] == "not_ready"
    assert ready_response.json()["checks"]["database"]["status"] == "not_initialized"
    assert "db upgrade" in ready_response.json()["checks"]["database"]["detail"]


def test_health_readiness_is_ready_after_database_upgrade(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    from open_licenseplate.database import upgrade_database

    upgrade_database(settings.storage.data_dir / "open-licenseplate.sqlite3")

    with TestClient(create_app(settings)) as client:
        response = client.get("/api/v1/health/ready")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    assert response.json()["checks"]["directories"]["status"] == "ok"
    assert response.json()["checks"]["database"]["status"] == "ok"
