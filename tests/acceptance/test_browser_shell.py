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

from model_helpers import create_model_fixture
from open_licenseplate.app import create_app
from open_licenseplate.capture import FixtureAttempt, ReconnectFixture, make_preview_frame
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
    source_factory = ReconnectFixture(
        (FixtureAttempt(frames=(make_preview_frame(16),), repeat=True),)
    )
    server = uvicorn.Server(
        uvicorn.Config(
            create_app(settings, source_factory=source_factory),
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


@pytest.mark.browser
def test_browser_can_save_and_test_a_camera_without_secret_output(
    browser_base_url: str,
    chromium,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "CAMERA_RTSP_URL",
        "rtsp://operator:secret-value@example.test:554/live",
    )
    page = chromium.new_page(viewport={"width": 1280, "height": 900})
    page.goto(f"{browser_base_url}/cameras", wait_until="domcontentloaded")

    page.get_by_label("Name", exact=True).fill("Front gate")
    page.get_by_label("RTSP endpoint", exact=True).fill(
        "rtsp://operator:secret-value@example.test:554/live"
    )
    page.get_by_label("Credential reference", exact=True).fill("env:CAMERA_RTSP_URL")
    page.get_by_role("button", name="Save camera", exact=True).click()
    page.wait_for_url(f"{browser_base_url}/cameras?notice=created")

    assert page.get_by_role("heading", name="Front gate", exact=True).is_visible()
    assert "secret-value" not in page.content()
    assert "operator" not in page.content()

    page.get_by_role("button", name="Test configuration", exact=True).click()
    page.wait_for_url(f"{browser_base_url}/cameras?notice=test&status=valid*")
    assert page.get_by_role("status").inner_text().startswith("Camera configuration is valid")
    assert "secret-value" not in page.content()


@pytest.mark.browser
def test_browser_can_start_stop_preview_and_show_safe_runtime_state(
    browser_base_url: str,
    chromium,
) -> None:
    page = chromium.new_page(viewport={"width": 1280, "height": 900})
    page.goto(f"{browser_base_url}/cameras", wait_until="domcontentloaded")
    page.get_by_label("Name", exact=True).fill("Fixture camera")
    page.get_by_label("RTSP endpoint", exact=True).fill("rtsp://fixture.local/live")
    page.get_by_role("button", name="Save camera", exact=True).click()
    page.wait_for_url(f"{browser_base_url}/cameras?notice=created")

    page.goto(f"{browser_base_url}/live", wait_until="domcontentloaded")
    page.get_by_role("button", name="Start preview", exact=True).click()
    page.get_by_text("Streaming", exact=True).wait_for(timeout=3000)
    page.locator("#live-resolution").filter(has_text="8x6").wait_for(timeout=3000)
    page.locator("#live-preview").wait_for(state="visible", timeout=3000)
    assert page.locator("canvas").count() == 0
    assert page.locator("#live-replaced").is_visible()

    page.get_by_role("button", name="Stop", exact=True).click()
    page.get_by_text("Stopped", exact=True).wait_for(timeout=3000)
    assert page.locator("#live-preview").is_hidden()


@pytest.mark.browser
def test_browser_can_import_and_manage_a_model_package(
    browser_base_url: str,
    chromium,
    tmp_path: Path,
) -> None:
    manifest_path, archive_path, _manifest = create_model_fixture(
        tmp_path,
        model_id="browser-model",
    )
    page = chromium.new_page(viewport={"width": 1280, "height": 900})
    page.goto(f"{browser_base_url}/models", wait_until="domcontentloaded")

    page.locator("#model-manifest").set_input_files(str(manifest_path))
    page.locator("#model-archive").set_input_files(str(archive_path))
    page.get_by_role("button", name="Import model", exact=True).click()
    page.wait_for_url(f"{browser_base_url}/models?notice=imported")

    assert page.get_by_role("heading", name="Test model", exact=True).is_visible()
    assert page.get_by_role("button", name="Activation pending P08", exact=True).is_visible()
    assert "Runtime validation was not run" in page.content()
    page.get_by_role("button", name="Delete", exact=True).click()
    page.wait_for_url(f"{browser_base_url}/models?notice=deleted")
    assert page.get_by_text("No managed model packages yet.").is_visible()
