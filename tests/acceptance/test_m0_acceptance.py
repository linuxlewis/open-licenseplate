from __future__ import annotations

import json
import os
import shutil
import socket
import sqlite3
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from ipaddress import ip_address
from pathlib import Path

import httpx
import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
KEEP_ENVIRONMENT_KEYS = {"OPEN_LICENSEPLATE_CHROMIUM"}


def _clean_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for key in tuple(environment):
        if key.startswith("OPEN_LICENSEPLATE_") and key not in KEEP_ENVIRONMENT_KEYS:
            del environment[key]
    return environment


def _cli_path() -> str:
    cli_path = shutil.which("open-licenseplate")
    if cli_path is None:
        raise AssertionError("open-licenseplate is not available in the test environment")
    return cli_path


def _run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [_cli_path(), *arguments],
        cwd=REPOSITORY_ROOT,
        env=_clean_environment(),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def _assert_cli_success(result: subprocess.CompletedProcess[str]) -> None:
    assert result.returncode == 0, (
        f"command failed with exit code {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


def _table_names(database_path: Path) -> set[str]:
    with sqlite3.connect(database_path) as connection:
        rows = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
        ).fetchall()
    return {str(row[0]) for row in rows}


def _setting_keys(database_path: Path) -> list[str]:
    with sqlite3.connect(database_path) as connection:
        rows = connection.execute(
            "SELECT setting_key FROM application_settings ORDER BY setting_key"
        ).fetchall()
    return [str(row[0]) for row in rows]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as socket_instance:
        socket_instance.bind(("127.0.0.1", 0))
        return int(socket_instance.getsockname()[1])


@contextmanager
def _running_server(data_dir: Path, log_dir: Path, port: int) -> Iterator[str]:
    process = subprocess.Popen(
        [
            _cli_path(),
            "serve",
            "--port",
            str(port),
            "--data-dir",
            str(data_dir),
            "--log-dir",
            str(log_dir),
        ],
        cwd=REPOSITORY_ROOT,
        env=_clean_environment(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    base_url = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 10
    try:
        while time.monotonic() < deadline:
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                raise AssertionError(
                    f"server exited with code {process.returncode}\n"
                    f"stdout:\n{stdout}\n"
                    f"stderr:\n{stderr}"
                )
            try:
                response = httpx.get(
                    f"{base_url}/api/v1/health/live",
                    timeout=0.5,
                    trust_env=False,
                )
            except httpx.HTTPError:
                time.sleep(0.05)
            else:
                if response.status_code == 200:
                    break
                time.sleep(0.05)
        else:
            raise AssertionError("server did not become ready within 10 seconds")

        yield base_url
    finally:
        if process.poll() is None:
            process.terminate()
        try:
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
        if process.returncode not in {0, -15}:
            raise AssertionError(
                f"server exited with code {process.returncode}\n"
                f"stdout:\n{stdout}\n"
                f"stderr:\n{stderr}"
            )


def _local_non_loopback_addresses() -> set[str]:
    addresses: set[str] = set()
    try:
        results = socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)
    except OSError:
        results = []
    for _family, _kind, _protocol, _canonical_name, address in results:
        candidate = str(address[0])
        parsed = ip_address(candidate)
        if not parsed.is_loopback and not parsed.is_unspecified:
            addresses.add(candidate)

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as socket_instance:
        with suppress(OSError):
            socket_instance.connect(("192.0.2.1", 9))
        candidate = socket_instance.getsockname()[0]
    parsed_candidate = ip_address(candidate)
    if not parsed_candidate.is_loopback and not parsed_candidate.is_unspecified:
        addresses.add(candidate)
    return addresses


def _assert_loopback_only(port: int) -> None:
    loopback_response = httpx.get(
        f"http://127.0.0.1:{port}/api/v1/health/live",
        timeout=1,
        trust_env=False,
    )
    assert loopback_response.status_code == 200

    addresses = _local_non_loopback_addresses()
    assert addresses, "the test host has no non-loopback IPv4 address for binding validation"
    for address in addresses:
        try:
            response = httpx.get(
                f"http://{address}:{port}/api/v1/health/live",
                timeout=1,
                trust_env=False,
            )
        except httpx.HTTPError:
            continue
        raise AssertionError(
            f"default server was reachable through non-loopback address {address}: "
            f"HTTP {response.status_code}"
        )


@pytest.fixture
def chromium():
    from playwright.sync_api import sync_playwright

    candidates = [
        os.environ.get("OPEN_LICENSEPLATE_CHROMIUM"),
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary",
    ]
    executable = next(
        (candidate for candidate in candidates if candidate and Path(candidate).is_file()),
        None,
    )
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=True,
            **({"executable_path": executable} if executable else {}),
        )
        yield browser
        browser.close()


@pytest.mark.browser
@pytest.mark.m0_acceptance
def test_m0_acceptance_from_fresh_fixture(tmp_path: Path, chromium) -> None:
    data_dir = tmp_path / "data"
    log_dir = tmp_path / "logs"
    database_path = data_dir / "open-licenseplate.sqlite3"

    assert not data_dir.exists()
    fixture_result = _run_cli(
        "dev",
        "fixture",
        "--data-dir",
        str(data_dir),
        "--log-dir",
        str(log_dir),
    )
    _assert_cli_success(fixture_result)
    assert "No camera, model, plate, event, job, or OCR data was created." in fixture_result.stdout
    assert not database_path.exists()

    migration_result = _run_cli(
        "db",
        "upgrade",
        "--data-dir",
        str(data_dir),
        "--log-dir",
        str(log_dir),
    )
    _assert_cli_success(migration_result)
    assert database_path.is_file()
    assert _table_names(database_path) == {
        "alembic_version",
        "application_settings",
        "cameras",
        "models",
        "capture_sessions",
        "detection_events",
        "event_artifacts",
    }
    assert _setting_keys(database_path) == []

    first_port = _free_port()
    with _running_server(data_dir, log_dir, first_port) as base_url:
        _assert_loopback_only(first_port)
        page = chromium.new_page(viewport={"width": 1280, "height": 900})
        page.goto(f"{base_url}/", wait_until="domcontentloaded")
        assert page.url == f"{base_url}/live"

        pages = {
            "Live": ("/live", "Live view"),
            "Events": ("/events", "Plate events"),
            "Jobs": ("/jobs", "Processing jobs"),
            "Cameras": ("/cameras", "Camera sources"),
            "Models": ("/models", "Detection models"),
            "System": ("/system", "System status"),
        }
        for label, (path, title) in pages.items():
            link = page.get_by_role("link", name=label, exact=True)
            assert link.get_attribute("href") == path
            link.click()
            page.wait_for_url(f"{base_url}{path}")
            assert page.get_by_role("heading", name=title, exact=True).is_visible()

        page.goto(f"{base_url}/system", wait_until="domcontentloaded")
        assert "density-comfortable" in (page.locator("body").get_attribute("class") or "")
        page.locator("#ui-density").select_option("compact")
        page.get_by_role("button", name="Save preference", exact=True).click()
        page.wait_for_url(f"{base_url}/system")
        assert "density-compact" in (page.locator("body").get_attribute("class") or "")
        assert page.locator("#ui-density").input_value() == "compact"
        page.close()

    second_port = _free_port()
    with _running_server(data_dir, log_dir, second_port) as base_url:
        page = chromium.new_page(viewport={"width": 1280, "height": 900})
        page.goto(f"{base_url}/system", wait_until="domcontentloaded")
        assert "density-compact" in (page.locator("body").get_attribute("class") or "")
        assert page.locator("#ui-density").input_value() == "compact"
        page.close()

    doctor_result = _run_cli(
        "doctor",
        "--json",
        "--data-dir",
        str(data_dir),
        "--log-dir",
        str(log_dir),
    )
    _assert_cli_success(doctor_result)
    doctor = json.loads(doctor_result.stdout)
    assert doctor["ready"] is True
    assert doctor["database"]["status"] == "ok"
    assert all(doctor["directories"].values())
    assert _table_names(database_path) == {
        "alembic_version",
        "application_settings",
        "cameras",
        "models",
        "capture_sessions",
        "detection_events",
        "event_artifacts",
    }
    assert _setting_keys(database_path) == ["ui.density"]
