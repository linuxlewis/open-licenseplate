from pathlib import Path

from fastapi.testclient import TestClient

from open_licenseplate.app import create_app
from open_licenseplate.config import load_settings
from open_licenseplate.database import upgrade_database


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
    upgrade_database(settings.storage.data_dir / "open-licenseplate.sqlite3")

    with TestClient(create_app(settings)) as client:
        response = client.get("/api/v1/health/ready")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    assert response.json()["checks"]["directories"]["status"] == "ok"
    assert response.json()["checks"]["database"]["status"] == "ok"


def test_shell_routes_render_with_navigation_and_security_headers(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    routes = {
        "/live": "Live view",
        "/events": "Plate events",
        "/jobs": "Processing jobs",
        "/cameras": "Camera sources",
        "/models": "Detection models",
        "/system": "System status",
    }

    with TestClient(create_app(settings)) as client:
        root_response = client.get("/", follow_redirects=False)
        assert root_response.status_code == 307
        assert root_response.headers["location"] == "/live"

        for route, title in routes.items():
            response = client.get(route)
            assert response.status_code == 200
            assert f"<h1>{title}</h1>" in response.text
            assert 'href="/system"' in response.text
            assert "/static/vendor/htmx.min.js" in response.text
            assert response.headers["content-security-policy"].startswith("default-src 'self'")
            assert response.headers["x-content-type-options"] == "nosniff"
            assert response.headers["x-frame-options"] == "DENY"

        css_response = client.get("/static/app.css")
        htmx_response = client.get("/static/vendor/htmx.min.js")
        assert css_response.status_code == 200
        assert "--canvas:" in css_response.text
        assert htmx_response.status_code == 200
        assert "htmx" in htmx_response.text


def test_system_page_shows_versions_paths_database_and_setting_sources(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    database_path = settings.storage.data_dir / "open-licenseplate.sqlite3"
    upgrade_database(database_path)

    from open_licenseplate.database import Database
    from open_licenseplate.settings_store import SettingsStore

    database = Database(database_path)
    try:
        SettingsStore(database).set("server.port", 9017)
    finally:
        database.dispose()

    persisted_settings = _settings(tmp_path)
    with TestClient(create_app(persisted_settings)) as client:
        response = client.get("/system")

    assert response.status_code == 200
    assert "0.1.0" in response.text
    assert "0003_models" in response.text
    assert str(database_path) in response.text
    assert "9017" in response.text
    assert "persisted" in response.text
    assert "Not configured" in response.text
    assert "Core ML support is not part of M0." in response.text


def test_system_density_preference_persists_and_applies_after_restart(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    database_path = settings.storage.data_dir / "open-licenseplate.sqlite3"
    upgrade_database(database_path)

    with TestClient(create_app(settings)) as client:
        before_response = client.get("/system")
        save_response = client.post(
            "/system/preferences",
            data={"density": "compact"},
            follow_redirects=False,
        )
        after_response = client.get("/system")

    assert before_response.status_code == 200
    assert 'class="density-comfortable"' in before_response.text
    assert save_response.status_code == 303
    assert save_response.headers["location"] == "/system"
    assert after_response.status_code == 200
    assert 'class="density-compact"' in after_response.text
    assert 'value="compact" selected' in after_response.text
    assert "UI density" in after_response.text
    assert "persisted" in after_response.text

    restarted_settings = _settings(tmp_path)
    assert restarted_settings.ui.density == "compact"
    assert restarted_settings.sources["ui.density"] == "persisted"

    with TestClient(create_app(restarted_settings)) as client:
        restarted_response = client.get("/system")

    assert restarted_response.status_code == 200
    assert 'class="density-compact"' in restarted_response.text


def test_shell_escapes_rendered_setting_values(tmp_path: Path) -> None:
    settings = load_settings(
        cli_overrides={
            "app_name": "<script>alert(1)</script>",
            "storage.data_dir": tmp_path / "data",
            "storage.log_dir": tmp_path / "logs",
        }
    )

    with TestClient(create_app(settings)) as client:
        response = client.get("/system")

    assert response.status_code == 200
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in response.text
    assert "<script>alert(1)</script>" not in response.text
