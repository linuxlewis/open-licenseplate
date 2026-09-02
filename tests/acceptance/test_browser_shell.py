from __future__ import annotations

import json
import os
import socket
import threading
import time
from collections.abc import Iterator
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
import uvicorn
from PIL import Image

from model_helpers import create_model_fixture
from open_licenseplate.app import create_app
from open_licenseplate.capture import FixtureAttempt, ReconnectFixture, make_preview_frame
from open_licenseplate.config import load_settings
from open_licenseplate.database import Database, upgrade_database
from open_licenseplate.inference.backends import FakeBackend
from open_licenseplate.models.repository import ModelRepository
from open_licenseplate.models.service import import_model
from open_licenseplate.paths import ManagedPaths


def _free_port() -> int:
    with socket.socket() as socket_instance:
        socket_instance.bind(("127.0.0.1", 0))
        return int(socket_instance.getsockname()[1])


def _seed_managed_model(tmp_path: Path, model_id: str) -> None:
    fixture_root = tmp_path / "catalog-install-fixture"
    manifest_path, archive_path, _manifest = create_model_fixture(
        fixture_root,
        model_id=model_id,
    )
    settings = load_settings(
        cli_overrides={
            "storage.data_dir": tmp_path / "data",
            "storage.log_dir": tmp_path / "logs",
        }
    )
    database = Database(settings.storage.data_dir / "open-licenseplate.sqlite3")
    try:
        import_model(
            manifest_value=manifest_path.read_bytes(),
            source_path=archive_path,
            paths=ManagedPaths.from_settings(settings),
            repository=ModelRepository(database),
        )
    finally:
        database.dispose()


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


@pytest.fixture
def fake_browser_base_url(tmp_path: Path) -> Iterator[str]:
    settings = load_settings(
        cli_overrides={
            "storage.data_dir": tmp_path / "data",
            "storage.log_dir": tmp_path / "logs",
        }
    )
    upgrade_database(settings.storage.data_dir / "open-licenseplate.sqlite3")

    def outputs(prepared: Any) -> dict[str, Any]:
        region = np.asarray(prepared.value)[280:360, 160:480]
        if float(region.mean()) < 100:
            return {
                "coordinates": np.empty((0, 4), dtype=np.float32),
                "confidence": np.empty((0,), dtype=np.float32),
            }
        return {
            "coordinates": np.array([[160, 280, 480, 360]], dtype=np.float32),
            "confidence": np.array([0.9], dtype=np.float32),
        }

    backend = FakeBackend(output_factory=outputs)
    server = uvicorn.Server(
        uvicorn.Config(
            create_app(settings, inference_backend_factory=lambda: backend),
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
        raise RuntimeError("fake browser test server did not start")

    yield base_url
    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture
def fake_live_browser_base_url(tmp_path: Path) -> Iterator[str]:
    settings = load_settings(
        cli_overrides={
            "storage.data_dir": tmp_path / "data",
            "storage.log_dir": tmp_path / "logs",
        }
    )
    upgrade_database(settings.storage.data_dir / "open-licenseplate.sqlite3")
    source_factory = ReconnectFixture(
        (
            FixtureAttempt(
                frames=(make_preview_frame(180, width=640, height=360),),
                repeat=True,
                read_interval_seconds=0.01,
            ),
        )
    )

    def outputs(_prepared: Any) -> dict[str, Any]:
        return {
            "coordinates": np.array([[100, 240, 500, 440]], dtype=np.float32),
            "confidence": np.array([0.99], dtype=np.float32),
        }

    backend = FakeBackend(output_factory=outputs)
    server = uvicorn.Server(
        uvicorn.Config(
            create_app(
                settings,
                source_factory=source_factory,
                inference_backend_factory=lambda: backend,
            ),
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
        raise RuntimeError("fake live browser test server did not start")

    yield base_url
    server.should_exit = True
    thread.join(timeout=5)


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
        if label == "Models":
            page.locator("[data-catalog-card]").first.wait_for()

    page.set_viewport_size({"width": 420, "height": 800})
    page.goto(f"{browser_base_url}/system", wait_until="domcontentloaded")
    assert page.locator(".primary-nav").is_visible()
    assert page.locator(".system-grid").is_visible()


@pytest.mark.browser
def test_browser_models_page_shows_only_the_recommended_catalog_model(
    browser_base_url: str,
    chromium,
) -> None:
    page = chromium.new_page(viewport={"width": 1280, "height": 900})
    catalog_url = f"{browser_base_url}/api/v1/models/catalog"
    catalog_payload = page.request.get(catalog_url).json()
    assert len(catalog_payload["models"]) == 3
    target = next(
        entry for entry in catalog_payload["models"] if entry["recommendation"] == "fast_default"
    )
    other_entries = [
        entry for entry in catalog_payload["models"] if entry["catalog_id"] != target["catalog_id"]
    ]
    assert other_entries
    html_like_value = '<img src=x onerror="window.__catalogXss = true">'
    target["catalog_id"] = html_like_value
    target["display_name"] = html_like_value
    target["license"] = html_like_value
    target["source"]["repository"] = html_like_value
    target["source"]["revision"] = html_like_value
    catalog_payload["models"] = [other_entries[0], target, *other_entries[1:]]
    assert catalog_payload["models"][0]["recommendation"] != "fast_default"
    assert catalog_payload["models"][1]["recommendation"] == "fast_default"

    def catalog_route(route: Any, request: Any) -> None:
        del request
        route.fulfill(
            status=200,
            headers={"content-type": "application/json"},
            body=json.dumps(catalog_payload),
        )

    page.route(f"{catalog_url}**", catalog_route)
    page.goto(f"{browser_base_url}/models", wait_until="domcontentloaded")

    catalog = page.locator("[data-model-catalog]")
    cards = page.locator("[data-catalog-card]")
    cards.first.wait_for()
    assert cards.count() == 1
    assert cards.first.get_attribute("data-catalog-id") == html_like_value
    assert cards.first.locator(".catalog-item-name").text_content() == "YOLO license plate model"
    assert cards.first.locator(".catalog-recommendation").text_content() == "Recommended"
    assert cards.first.locator("img, script").count() == 0
    assert page.evaluate("window.__catalogXss === true") is False
    catalog_text = catalog.text_content() or ""
    assert all(label not in catalog_text for label in ("Nano", "Small", "Medium"))
    assert page.get_by_role("button", name="Install", exact=True).count() == 1
    assert cards.first.locator(".catalog-installed").text_content() == "Not installed"
    assert page.locator("details.custom-model").is_visible()
    assert page.get_by_text("Custom model", exact=True).is_visible()
    assert page.locator("#model-manifest").is_visible()
    assert page.locator("#model-archive").is_visible()
    custom_summary = page.locator("details.custom-model > summary")
    custom_summary.focus()
    custom_summary.press("Enter")
    assert page.locator("details.custom-model").get_attribute("open") is None
    custom_summary.press("Enter")
    assert page.locator("details.custom-model").get_attribute("open") == ""
    page.set_viewport_size({"width": 420, "height": 800})
    install_box = page.get_by_role("button", name="Install", exact=True).bounding_box()
    assert install_box is not None
    assert install_box["x"] + install_box["width"] <= 420
    assert page.locator("#model-manifest").is_visible()


@pytest.mark.browser
def test_browser_models_page_shows_initially_installed_recommended_catalog_model(
    browser_base_url: str,
    chromium,
) -> None:
    page = chromium.new_page(viewport={"width": 1280, "height": 900})
    catalog_url = f"{browser_base_url}/api/v1/models/catalog"
    catalog_payload = page.request.get(catalog_url).json()
    target = next(
        entry for entry in catalog_payload["models"] if entry["recommendation"] == "fast_default"
    )
    target["installed"] = True
    target["install_available"] = False

    def catalog_route(route: Any, request: Any) -> None:
        del request
        route.fulfill(
            status=200,
            headers={"content-type": "application/json"},
            body=json.dumps(catalog_payload),
        )

    page.route(f"{catalog_url}**", catalog_route)
    page.goto(f"{browser_base_url}/models", wait_until="domcontentloaded")

    card = page.locator("[data-catalog-card]")
    card.first.wait_for()
    assert card.count() == 1
    assert card.locator(".catalog-installed").text_content() == "Installed"
    assert page.get_by_role("button", name="Install", exact=True).count() == 0


@pytest.mark.browser
def test_browser_catalog_install_updates_server_rendered_model_list_and_prevents_duplicate_clicks(
    browser_base_url: str,
    chromium,
    tmp_path: Path,
) -> None:
    page = chromium.new_page(viewport={"width": 1280, "height": 900})
    catalog_url = f"{browser_base_url}/api/v1/models/catalog"
    initial_payload = page.request.get(catalog_url).json()
    target = next(
        entry for entry in initial_payload["models"] if entry["recommendation"] == "fast_default"
    )
    install_state = {"installed": False}
    managed_model_seeded = {"value": False}
    post_urls: list[str] = []

    def catalog_route(route: Any, request: Any) -> None:
        if request.method == "POST":
            post_urls.append(request.url)
            if not managed_model_seeded["value"]:
                _seed_managed_model(tmp_path, target["catalog_id"])
                managed_model_seeded["value"] = True
            install_state["installed"] = True
            time.sleep(0.25)
            route.fulfill(
                status=201,
                headers={"content-type": "application/json"},
                body=json.dumps({"id": target["catalog_id"]}),
            )
            return
        payload = deepcopy(initial_payload)
        if install_state["installed"]:
            installed_entry = next(
                entry for entry in payload["models"] if entry["catalog_id"] == target["catalog_id"]
            )
            installed_entry["installed"] = True
            installed_entry["install_available"] = False
        route.fulfill(
            status=200,
            headers={"content-type": "application/json"},
            body=json.dumps(payload),
        )

    page.route(f"{catalog_url}**", catalog_route)
    page.goto(f"{browser_base_url}/models", wait_until="domcontentloaded")
    card = page.locator("[data-catalog-card]").first
    button = card.get_by_role("button", name="Install", exact=True)
    button_state = button.evaluate(
        """element => {
          element.click();
          element.dispatchEvent(new MouseEvent("click", { bubbles: true }));
          element.dispatchEvent(new MouseEvent("click", { bubbles: true }));
          return { disabled: element.disabled, text: element.textContent };
        }"""
    )
    assert button_state == {"disabled": True, "text": "Installing..."}

    page.locator(".model-list-card .model-item").first.wait_for(timeout=5000)
    page.wait_for_function(
        """() => document.querySelector(
          "[data-catalog-card] .catalog-installed"
        )?.textContent === "Installed" """,
        timeout=5000,
    )
    installed_card = page.locator("[data-catalog-card]").first
    installed_card.locator(".catalog-installed-true").wait_for(timeout=5000)
    assert installed_card.locator(".catalog-installed").text_content() == "Installed"
    assert installed_card.get_by_role("button", name="Install", exact=True).count() == 0
    assert page.get_by_text("Test model", exact=True).is_visible()
    assert page.get_by_role("button", name="Validate package", exact=True).count() == 1
    assert page.get_by_role("button", name="Delete", exact=True).count() == 1
    assert post_urls == [f"{catalog_url}/{target['catalog_id']}/install"]


@pytest.mark.browser
def test_browser_catalog_install_failure_allows_retry(
    browser_base_url: str,
    chromium,
) -> None:
    page = chromium.new_page(viewport={"width": 1280, "height": 900})
    catalog_url = f"{browser_base_url}/api/v1/models/catalog"
    initial_payload = page.request.get(catalog_url).json()
    target = next(
        entry for entry in initial_payload["models"] if entry["recommendation"] == "fast_default"
    )
    post_urls: list[str] = []
    install_state = {"installed": False}

    def catalog_route(route: Any, request: Any) -> None:
        if request.method == "POST":
            post_urls.append(request.url)
            if len(post_urls) == 1:
                route.fulfill(status=502)
            else:
                install_state["installed"] = True
                route.fulfill(
                    status=201,
                    headers={"content-type": "application/json"},
                    body=json.dumps({"id": target["catalog_id"]}),
                )
            return
        payload = deepcopy(initial_payload)
        if install_state["installed"]:
            installed_entry = next(
                entry for entry in payload["models"] if entry["catalog_id"] == target["catalog_id"]
            )
            installed_entry["installed"] = True
            installed_entry["install_available"] = False
        route.fulfill(
            status=200,
            headers={"content-type": "application/json"},
            body=json.dumps(payload),
        )

    page.route(f"{catalog_url}**", catalog_route)
    page.goto(f"{browser_base_url}/models", wait_until="domcontentloaded")
    card = page.locator("[data-catalog-card]").first
    button = card.get_by_role("button", name="Install", exact=True)
    button.evaluate("element => element.click()")
    status = card.locator(".catalog-item-status")
    page.wait_for_function(
        """element => element.textContent ===
          "Install failed. Try again." """,
        arg=status.element_handle(),
        timeout=5000,
    )
    assert status.text_content() == "Install failed. Try again."
    assert button.is_enabled()
    assert button.text_content() == "Install"
    button.evaluate("element => element.click()")
    page.wait_for_function(
        """() => document.querySelector(
          "[data-catalog-card] .catalog-installed"
        )?.textContent === "Installed" """,
        timeout=5000,
    )
    assert post_urls == [
        f"{catalog_url}/{target['catalog_id']}/install",
        f"{catalog_url}/{target['catalog_id']}/install",
    ]


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
def test_browser_can_run_detection_change_threshold_resize_overlay_and_stop(
    fake_live_browser_base_url: str,
    chromium,
    tmp_path: Path,
) -> None:
    manifest_path, archive_path, _manifest = create_model_fixture(
        tmp_path,
        model_id="browser-live-model",
    )
    page = chromium.new_page(viewport={"width": 1280, "height": 1000})
    page.goto(f"{fake_live_browser_base_url}/cameras", wait_until="domcontentloaded")
    page.get_by_label("Name", exact=True).fill("Live fixture")
    page.get_by_label("RTSP endpoint", exact=True).fill("rtsp://fixture.local/live")
    page.get_by_role("button", name="Save camera", exact=True).click()
    page.wait_for_url(f"{fake_live_browser_base_url}/cameras?notice=created")

    page.goto(f"{fake_live_browser_base_url}/models", wait_until="domcontentloaded")
    page.locator("#model-manifest").set_input_files(str(manifest_path))
    page.locator("#model-archive").set_input_files(str(archive_path))
    page.get_by_role("button", name="Import model", exact=True).click()
    page.wait_for_url(f"{fake_live_browser_base_url}/models?notice=imported")
    page.get_by_role("button", name="Validate package", exact=True).click()
    page.wait_for_url(f"{fake_live_browser_base_url}/models?notice=validated")

    page.goto(f"{fake_live_browser_base_url}/live", wait_until="domcontentloaded")
    page.get_by_role("button", name="Start detection", exact=True).click()
    page.locator("#live-processed-preview").wait_for(state="visible", timeout=5000)
    page.locator("#live-overlay").wait_for(state="visible", timeout=5000)
    page.locator("#live-processed-sequence").filter(has_text="1").wait_for(timeout=5000)
    assert page.locator("#live-processed-prediction").inner_text().endswith(" ms")
    assert page.locator("#live-processed-p50").inner_text().endswith(" ms")
    assert page.locator("#live-processed-p95").inner_text().endswith(" ms")

    image_box = page.locator("#live-processed-preview").bounding_box()
    canvas_box = page.locator("#live-overlay").bounding_box()
    assert image_box is not None
    assert canvas_box is not None
    assert abs(image_box["width"] - canvas_box["width"]) < 1
    assert abs(image_box["height"] - canvas_box["height"]) < 1
    box_check = page.evaluate(
        """() => {
          const image = document.querySelector("#live-processed-preview");
          const canvas = document.querySelector("#live-overlay");
          const bounds = image.getBoundingClientRect();
          const ratio = window.devicePixelRatio || 1;
          const context = canvas.getContext("2d");
          const points = [
            [bounds.width * 100 / 640, bounds.height * 200 / 360],
            [bounds.width * 500 / 640, bounds.height * 200 / 360],
            [bounds.width * 300 / 640, bounds.height * 300 / 360],
          ];
          const hasOverlayColor = (x, y) => {
            const centerX = Math.round(x * ratio);
            const centerY = Math.round(y * ratio);
            for (let dx = -4; dx <= 4; dx += 1) {
              for (let dy = -4; dy <= 4; dy += 1) {
                const pixel = context.getImageData(centerX + dx, centerY + dy, 1, 1).data;
                if (pixel[0] > 180 && pixel[1] > 140 && pixel[2] < 160 && pixel[3] > 0) {
                  return true;
                }
              }
            }
            return false;
          };
          return {
            expected: points,
            found: points.map(([x, y]) => hasOverlayColor(x, y)),
          };
        }"""
    )
    assert box_check["expected"][0][0] > 0
    assert box_check["found"] == [True, True, True]

    page.locator("#live-threshold").fill("0.95")
    page.locator("#live-threshold").press("Tab")
    page.get_by_text(
        "Confidence threshold updated without reloading the model.",
        exact=True,
    ).wait_for(timeout=3000)

    page.set_viewport_size({"width": 520, "height": 900})
    page.wait_for_timeout(200)
    resized_image_box = page.locator("#live-processed-preview").bounding_box()
    resized_canvas_box = page.locator("#live-overlay").bounding_box()
    assert resized_image_box is not None
    assert resized_canvas_box is not None
    assert abs(resized_image_box["width"] - resized_canvas_box["width"]) < 1
    assert abs(resized_image_box["height"] - resized_canvas_box["height"]) < 1
    resized_box_check = page.evaluate(
        """() => {
          const image = document.querySelector("#live-processed-preview");
          const canvas = document.querySelector("#live-overlay");
          const bounds = image.getBoundingClientRect();
          const ratio = window.devicePixelRatio || 1;
          const context = canvas.getContext("2d");
          const points = [
            [bounds.width * 100 / 640, bounds.height * 200 / 360],
            [bounds.width * 500 / 640, bounds.height * 200 / 360],
            [bounds.width * 300 / 640, bounds.height * 300 / 360],
          ];
          const hasOverlayColor = (x, y) => {
            const centerX = Math.round(x * ratio);
            const centerY = Math.round(y * ratio);
            for (let dx = -4; dx <= 4; dx += 1) {
              for (let dy = -4; dy <= 4; dy += 1) {
                const pixel = context.getImageData(centerX + dx, centerY + dy, 1, 1).data;
                if (pixel[0] > 180 && pixel[1] > 140 && pixel[2] < 160 && pixel[3] > 0) {
                  return true;
                }
              }
            }
            return false;
          };
          return {
            expected: points,
            found: points.map(([x, y]) => hasOverlayColor(x, y)),
          };
        }"""
    )
    assert resized_box_check["found"] == [True, True, True]

    page.get_by_role("button", name="Stop detection", exact=True).click()
    page.get_by_text("Detection is stopped", exact=True).wait_for(timeout=5000)
    assert page.locator("#live-processed-preview").is_hidden()
    assert page.locator("#live-overlay").count() == 0


@pytest.mark.browser
@pytest.mark.parametrize(
    "message_order",
    [
        "header_header_binary",
        "unexpected_binary",
        "wrong_track_camera",
        "too_many_active_tracks",
    ],
)
def test_browser_rejects_invalid_processed_message_order(
    fake_live_browser_base_url: str,
    chromium,
    tmp_path: Path,
    message_order: str,
) -> None:
    manifest_path, archive_path, _manifest = create_model_fixture(
        tmp_path,
        model_id=f"browser-order-{message_order}",
    )
    page = chromium.new_page(viewport={"width": 1280, "height": 1000})
    page.add_init_script(
        """
        class MockWebSocket {
          static OPEN = 1;
          constructor(url) {
            this.url = url;
            this.readyState = MockWebSocket.OPEN;
            window.__mockSocket = this;
          }
          close(code, reason) {
            this.readyState = 3;
            window.__mockClose = { code, reason };
            if (this.onclose) this.onclose();
          }
          emit(data) {
            if (this.onmessage) this.onmessage({ data });
          }
        }
        window.WebSocket = MockWebSocket;
        """
    )
    page.goto(f"{fake_live_browser_base_url}/cameras", wait_until="domcontentloaded")
    page.get_by_label("Name", exact=True).fill("Order fixture")
    page.get_by_label("RTSP endpoint", exact=True).fill("rtsp://fixture.local/live")
    page.get_by_role("button", name="Save camera", exact=True).click()
    page.wait_for_url(f"{fake_live_browser_base_url}/cameras?notice=created")
    page.goto(f"{fake_live_browser_base_url}/models", wait_until="domcontentloaded")
    page.locator("#model-manifest").set_input_files(str(manifest_path))
    page.locator("#model-archive").set_input_files(str(archive_path))
    page.get_by_role("button", name="Import model", exact=True).click()
    page.wait_for_url(f"{fake_live_browser_base_url}/models?notice=imported")
    page.get_by_role("button", name="Validate package", exact=True).click()
    page.wait_for_url(f"{fake_live_browser_base_url}/models?notice=validated")
    page.goto(f"{fake_live_browser_base_url}/live", wait_until="domcontentloaded")
    page.get_by_role("button", name="Start detection", exact=True).click()
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if page.evaluate("window.__mockSocket !== undefined"):
            break
        page.wait_for_timeout(50)
    else:
        raise AssertionError("mock WebSocket did not open")

    header = {
        "type": "frame_header",
        "message_type": "frame_header",
        "protocol_version": 1,
        "generation_number": 1,
        "camera_id": "camera-1",
        "model_id": "model-1",
        "model_checksum": "a" * 64,
        "capture_session_id": "session-1",
        "stream_epoch": "epoch-1",
        "frame_sequence": 1,
        "captured_at_utc": "2026-08-29T00:00:00Z",
        "capture_timestamp": "2026-08-29T00:00:00Z",
        "source_width": 8,
        "source_height": 6,
        "jpeg_width": 8,
        "jpeg_height": 6,
        "jpeg_byte_count": 1,
        "detections": [],
        "confidence_threshold": 0.35,
        "threshold": 0.35,
        "region_of_interest": None,
        "roi": None,
        "metrics": {},
    }
    active_track = {
        "camera_id": "camera-2",
        "capture_session_id": "session-1",
        "generation_number": 1,
        "stream_epoch": "epoch-1",
        "model_id": "model-1",
        "model_checksum": "a" * 64,
        "track_id": 1,
        "state": "active",
        "first_seen_utc": "2026-08-29T00:00:00Z",
        "last_seen_utc": "2026-08-29T00:00:00Z",
        "last_frame_sequence": 1,
        "last_box_xyxy": [1, 1, 5, 4],
        "last_confidence": 0.9,
        "observation_count": 3,
        "maximum_confidence": 0.9,
    }
    header_text = json.dumps(header)
    if message_order == "header_header_binary":
        page.evaluate(
            """(header) => {
              window.__mockSocket.emit(header);
              window.__mockSocket.emit(header);
              window.__mockSocket.emit(new Uint8Array([1]).buffer);
            }""",
            header_text,
        )
    elif message_order == "wrong_track_camera":
        header["active_tracks"] = [active_track]
        page.evaluate(
            "(header) => window.__mockSocket.emit(JSON.stringify(header))",
            json.dumps(header),
        )
    elif message_order == "too_many_active_tracks":
        header["active_tracks"] = [
            {**active_track, "camera_id": "camera-1", "track_id": index} for index in range(65)
        ]
        page.evaluate(
            "(header) => window.__mockSocket.emit(JSON.stringify(header))",
            json.dumps(header),
        )
    else:
        page.evaluate(
            "() => window.__mockSocket.emit(new Uint8Array([1]).buffer)",
        )
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if page.evaluate("window.__mockClose !== undefined"):
            break
        page.wait_for_timeout(50)
    else:
        raise AssertionError("mock WebSocket did not close")
    assert page.evaluate("window.__mockClose.code") == 1008
    assert "protocol" in page.evaluate("window.__mockClose.reason")
    page.get_by_role("button", name="Stop detection", exact=True).click()


@pytest.mark.browser
def test_browser_accepts_new_reconnect_epoch_and_discards_old_epoch(
    fake_live_browser_base_url: str,
    chromium,
    tmp_path: Path,
) -> None:
    manifest_path, archive_path, _manifest = create_model_fixture(
        tmp_path,
        model_id="browser-reconnect-model",
    )
    page = chromium.new_page(viewport={"width": 1280, "height": 1000})
    page.add_init_script(
        """
        class MockWebSocket {
          static OPEN = 1;
          constructor(url) {
            this.url = url;
            this.readyState = MockWebSocket.OPEN;
            window.__mockSocket = this;
          }
          close(code, reason) {
            this.readyState = 3;
            window.__mockClose = { code, reason };
            if (this.onclose) this.onclose();
          }
          emit(data) {
            if (this.onmessage) this.onmessage({ data });
          }
        }
        window.WebSocket = MockWebSocket;
        """
    )
    page.goto(f"{fake_live_browser_base_url}/cameras", wait_until="domcontentloaded")
    page.get_by_label("Name", exact=True).fill("Reconnect fixture")
    page.get_by_label("RTSP endpoint", exact=True).fill("rtsp://fixture.local/live")
    page.get_by_role("button", name="Save camera", exact=True).click()
    page.wait_for_url(f"{fake_live_browser_base_url}/cameras?notice=created")
    page.goto(f"{fake_live_browser_base_url}/models", wait_until="domcontentloaded")
    page.locator("#model-manifest").set_input_files(str(manifest_path))
    page.locator("#model-archive").set_input_files(str(archive_path))
    page.get_by_role("button", name="Import model", exact=True).click()
    page.wait_for_url(f"{fake_live_browser_base_url}/models?notice=imported")
    page.get_by_role("button", name="Validate package", exact=True).click()
    page.wait_for_url(f"{fake_live_browser_base_url}/models?notice=validated")
    page.goto(f"{fake_live_browser_base_url}/live", wait_until="domcontentloaded")
    page.get_by_role("button", name="Start detection", exact=True).click()
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if page.evaluate("window.__mockSocket !== undefined"):
            break
        page.wait_for_timeout(50)
    else:
        raise AssertionError("mock WebSocket did not open")

    jpeg_image = Image.new("RGB", (8, 6), color=(40, 80, 120))
    jpeg_output = BytesIO()
    jpeg_image.save(jpeg_output, format="JPEG")
    jpeg = list(jpeg_output.getvalue())

    def header(epoch: str, session: str, sequence: int) -> dict[str, object]:
        return {
            "type": "frame_header",
            "message_type": "frame_header",
            "protocol_version": 1,
            "generation_number": 1,
            "camera_id": "camera-1",
            "model_id": "model-1",
            "model_checksum": "a" * 64,
            "capture_session_id": session,
            "stream_epoch": epoch,
            "frame_sequence": sequence,
            "captured_at_utc": "2026-08-29T00:00:00Z",
            "capture_timestamp": "2026-08-29T00:00:00Z",
            "source_width": 8,
            "source_height": 6,
            "jpeg_width": 8,
            "jpeg_height": 6,
            "jpeg_byte_count": len(jpeg),
            "detections": [],
            "confidence_threshold": 0.35,
            "threshold": 0.35,
            "region_of_interest": None,
            "roi": None,
            "metrics": {},
        }

    def emit_frame(epoch: str, session: str, sequence: int) -> None:
        page.evaluate(
            """({header, jpeg}) => {
              window.__mockSocket.emit(JSON.stringify(header));
              window.__mockSocket.emit(new Uint8Array(jpeg).buffer);
            }""",
            {"header": header(epoch, session, sequence), "jpeg": jpeg},
        )

    emit_frame("epoch-1", "session-1", 1)
    page.locator("#live-processed-sequence").filter(has_text="1").wait_for(timeout=3000)
    assert page.locator("#live-processed-epoch").inner_text() == "epoch-1"
    assert page.locator("#live-processed-session").inner_text() == "session-1"
    emit_frame("epoch-2", "session-2", 2)
    page.locator("#live-processed-sequence").filter(has_text="2").wait_for(timeout=3000)
    assert page.locator("#live-processed-epoch").inner_text() == "epoch-2"
    assert page.locator("#live-processed-session").inner_text() == "session-2"
    emit_frame("epoch-1", "session-1", 3)
    page.wait_for_timeout(100)
    assert page.locator("#live-processed-sequence").inner_text() == "2"
    assert page.evaluate("window.__mockClose === undefined")
    page.get_by_role("button", name="Stop detection", exact=True).click()


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
    assert page.get_by_role("button", name="Runtime validation required", exact=True).is_visible()
    assert "Runtime validation was not run" in page.content()
    page.get_by_role("button", name="Delete", exact=True).click()
    page.wait_for_url(f"{browser_base_url}/models?notice=deleted")
    assert page.get_by_role("heading", name="No models", exact=True).is_visible()


@pytest.mark.browser
@pytest.mark.m2_acceptance
def test_browser_can_validate_and_detect_plate_and_no_plate_images(
    fake_browser_base_url: str,
    chromium,
    tmp_path: Path,
) -> None:
    manifest_path, archive_path, _manifest = create_model_fixture(
        tmp_path,
        model_id="browser-still-model",
    )
    fixture_root = Path(__file__).parents[1] / "fixtures" / "still"
    page = chromium.new_page(viewport={"width": 1280, "height": 1000})
    page.goto(f"{fake_browser_base_url}/models", wait_until="domcontentloaded")

    page.locator("#model-manifest").set_input_files(str(manifest_path))
    page.locator("#model-archive").set_input_files(str(archive_path))
    page.get_by_role("button", name="Import model", exact=True).click()
    page.wait_for_url(f"{fake_browser_base_url}/models?notice=imported")

    page.get_by_role("button", name="Validate package", exact=True).click()
    page.wait_for_url(f"{fake_browser_base_url}/models?notice=validated")
    page.get_by_text("Inspect runtime validation", exact=True).click()
    assert page.get_by_text("coordinates", exact=True).is_visible()
    assert page.get_by_text("confidence", exact=True).is_visible()
    page.locator("#model-image-browser-still-model").set_input_files(
        str(fixture_root / "plate.png")
    )
    page.get_by_role("button", name="Detect image", exact=True).click()
    page.get_by_text(
        "Detection complete. The displayed boxes use source-image pixels.",
        exact=True,
    ).wait_for()

    assert (
        page.locator("[data-detection-image]")
        .get_attribute("src")
        .startswith("data:image/png;base64,")
    )
    assert page.get_by_text("license_plate 90.0%", exact=True).is_visible()
    assert (
        page.locator('[data-metric="model_checksum"]').inner_text() == _manifest["artifact_sha256"]
    )
    assert page.locator('[data-metric="preprocessing_ms"]').inner_text().endswith(" ms")
    assert page.locator('[data-metric="inference_ms"]').inner_text().endswith(" ms")
    assert page.locator('[data-metric="postprocessing_ms"]').inner_text().endswith(" ms")
    assert page.locator('[data-metric="total_ms"]').inner_text().endswith(" ms")

    page.locator("#model-image-browser-still-model").set_input_files(
        str(fixture_root / "no-plate.png")
    )
    page.get_by_role("button", name="Detect image", exact=True).click()
    page.get_by_text(
        "Detection complete. The displayed boxes use source-image pixels.",
        exact=True,
    ).wait_for()
    assert page.locator('[data-metric="detections"]').inner_text() == "0"
    assert not page.get_by_text("license_plate 90.0%", exact=True).is_visible()

    page.locator("#model-compute-browser-still-model").select_option("cpu_only")
    page.locator("#model-image-browser-still-model").set_input_files(
        str(fixture_root / "plate.png")
    )
    page.get_by_role("button", name="Detect image", exact=True).click()
    page.get_by_text(
        "Detection complete. The model reloaded for the new compute units.",
        exact=True,
    ).wait_for()
    assert page.locator('[data-metric="compute_units"]').inner_text() == "CPU only"
    assert page.locator('[data-metric="detections"]').inner_text() == "1"
