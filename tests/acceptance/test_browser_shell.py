from __future__ import annotations

import os
import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
import uvicorn

from open_licenseplate.app import create_app
from open_licenseplate.config import load_settings
from open_licenseplate.database import upgrade_database


def _free_port() -> int:
    with socket.socket() as socket_instance:
        socket_instance.bind(("127.0.0.1", 0))
        return int(socket_instance.getsockname()[1])


@pytest.fixture
def browser_base_url(tmp_path: Path) -> Iterator[str]:
    settings = load_settings(
        cli_overrides={
            "storage.data_dir": tmp_path / "data",
            "storage.log_dir": tmp_path / "logs",
        }
    )
    upgrade_database(settings.storage.data_dir / "open-licenseplate.sqlite3")
    server = uvicorn.Server(
        uvicorn.Config(
            create_app(settings),
            host="127.0.0.1",
            port=_free_port(),
            log_config=None,
            access_log=False,
        )
    )
    server.install_signal_handlers = lambda: None
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.config.port}"
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            response = httpx.get(f"{base_url}/live", timeout=0.5)
            if response.status_code == 200:
                break
        except httpx.HTTPError:
            pass
        time.sleep(0.05)
    else:
        server.should_exit = True
        thread.join(timeout=5)
        raise RuntimeError("browser test server did not start")

    yield base_url
    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture
def chromium():
    pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import Error, sync_playwright

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
        try:
            browser = playwright.chromium.launch(
                headless=True,
                **({"executable_path": executable} if executable else {}),
            )
        except Error as error:
            pytest.skip(f"Chromium is not available: {error}")
        yield browser
        browser.close()


@pytest.mark.browser
def test_browser_can_visit_every_page_and_use_primary_navigation(
    browser_base_url: str,
    chromium,
) -> None:
    page = chromium.new_page(viewport={"width": 1280, "height": 900})
    page.goto(browser_base_url, wait_until="domcontentloaded")
    assert page.url == f"{browser_base_url}/live"

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
        page.wait_for_url(f"{browser_base_url}{path}")
        assert page.get_by_role("heading", name=title, exact=True).is_visible()
        assert page.locator('a[aria-current="page"] > span').first.inner_text() == label

    page.set_viewport_size({"width": 420, "height": 800})
    page.goto(f"{browser_base_url}/system", wait_until="domcontentloaded")
    assert page.locator(".primary-nav").is_visible()
    assert page.locator(".system-grid").is_visible()
