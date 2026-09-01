from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urlsplit

from open_licenseplate.models.manifest import parse_manifest

CATALOG_ROOT = Path(__file__).parents[2] / "model-catalog"
LOCK_PATH = CATALOG_ROOT / "model-catalog-lock.json"
EXPECTED_IDS = (
    "license-plate-yolov11n",
    "license-plate-yolov11s",
    "license-plate-yolov11m",
)
SOURCE_REVISION = "251a30d7daedca065f56e04b0af04052c907c68f"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _load_lock() -> dict[str, object]:
    value = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_catalog_lock_has_a_bounded_schema_and_fixed_release_urls() -> None:
    lock = _load_lock()
    assert lock["schema_version"] == 1
    assert lock["catalog_id"] == "open-licenseplate-model-catalog-v1"
    release = lock["release"]
    assert isinstance(release, dict)
    assert release == {
        "repository": "linuxlewis/open-licenseplate",
        "tag": "model-catalog-v1",
        "prerelease": True,
    }

    models = lock["models"]
    assert isinstance(models, list)
    assert [model["id"] for model in models if isinstance(model, dict)] == list(EXPECTED_IDS)
    assert len(models) == len(EXPECTED_IDS)

    archive_names: set[str] = set()
    manifest_names: set[str] = set()
    archive_urls: set[str] = set()
    for model in models:
        assert isinstance(model, dict)
        assert set(model) == {
            "id",
            "manifest",
            "manifest_asset",
            "archive_asset",
            "archive_url",
            "manifest_sha256",
            "package_sha256",
            "archive_sha256",
            "archive_size",
            "source",
            "license",
            "release_tag",
        }
        archive_name = model["archive_asset"]
        manifest_name = model["manifest_asset"]
        archive_url = model["archive_url"]
        assert isinstance(archive_name, str)
        assert isinstance(manifest_name, str)
        assert isinstance(archive_url, str)
        assert archive_name not in archive_names
        assert manifest_name not in manifest_names
        assert archive_url not in archive_urls
        archive_names.add(archive_name)
        manifest_names.add(manifest_name)
        archive_urls.add(archive_url)

        parsed_archive_url = urlsplit(archive_url)
        assert parsed_archive_url.scheme == "https"
        assert parsed_archive_url.netloc == "github.com"
        assert parsed_archive_url.query == ""
        assert parsed_archive_url.fragment == ""
        assert parsed_archive_url.path.startswith(
            "/linuxlewis/open-licenseplate/releases/download/model-catalog-v1/"
        )
        assert model["release_tag"] == "model-catalog-v1"
        assert isinstance(model["archive_size"], int)
        assert model["archive_size"] > 0
        assert isinstance(model["package_sha256"], str)
        assert isinstance(model["archive_sha256"], str)
        assert isinstance(model["manifest_sha256"], str)
        assert SHA256_RE.fullmatch(model["package_sha256"])
        assert SHA256_RE.fullmatch(model["archive_sha256"])
        assert SHA256_RE.fullmatch(model["manifest_sha256"])

        source = model["source"]
        assert isinstance(source, dict)
        assert source["license"] == "AGPL-3.0"
        assert source["revision"] == SOURCE_REVISION
        assert source["repository"] == (
            "https://huggingface.co/morsetechlab/yolov11-license-plate-detection"
        )
        parsed_source_url = urlsplit(source["url"])
        assert parsed_source_url.scheme == "https"
        assert parsed_source_url.netloc == "huggingface.co"
        assert parsed_source_url.path.startswith(
            "/morsetechlab/yolov11-license-plate-detection/resolve/"
            f"{SOURCE_REVISION}/license-plate-finetune-v1"
        )
        assert SHA256_RE.fullmatch(source["sha256"])


def test_catalog_manifests_match_the_lock_and_inspected_coreml_contract() -> None:
    lock = _load_lock()
    models = lock["models"]
    assert isinstance(models, list)
    for model in models:
        assert isinstance(model, dict)
        manifest_path = CATALOG_ROOT / str(model["manifest"])
        manifest_value = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = parse_manifest(manifest_value)
        manifest_bytes = manifest_path.read_bytes()

        assert manifest.model_id == model["id"]
        assert manifest.artifact == f"{model['id']}.mlpackage"
        assert model["manifest_sha256"] == hashlib.sha256(manifest_bytes).hexdigest()
        assert manifest.artifact_sha256 == model["package_sha256"]
        assert manifest.source_license == "AGPL-3.0"
        assert manifest.raw["source"]["revision"] == SOURCE_REVISION
        assert manifest.raw["source"]["file"] == manifest.raw["conversion"]["source_weight"]
        assert manifest.raw["distribution"]["archive"] == model["archive_asset"]
        assert manifest.raw["distribution"]["archive_sha256"] == model["archive_sha256"]
        assert manifest.raw["distribution"]["archive_size"] == model["archive_size"]

        input_values = manifest.raw["input"]
        assert input_values == {
            "name": "image",
            "kind": "image",
            "width": 640,
            "height": 640,
            "color_space": "rgb",
            "additional_inputs": [
                {
                    "name": "iouThreshold",
                    "kind": "double",
                    "role": "iou_threshold",
                    "optional": True,
                    "default": 0.7,
                },
                {
                    "name": "confidenceThreshold",
                    "kind": "double",
                    "role": "confidence_threshold",
                    "optional": True,
                    "default": 0.25,
                },
            ],
        }
        tool_versions = manifest.raw["conversion"]["tool_versions"]
        assert tool_versions == {
            "coremltools": "8.3.0",
            "numpy": "1.26.4",
            "python": "3.12.14",
            "torch": "2.5.0",
            "torchvision": "0.20.0",
            "ultralytics": "8.3.200",
        }
        build_platform = manifest.raw["conversion"]["build_platform"]
        assert build_platform["operating_system"] == "macOS"
        assert build_platform["architecture"] == "arm64"
        assert re.fullmatch(r"\d+\.\d+", build_platform["macos_version"])
        assert build_platform["kernel_release"]
        outputs = manifest.raw["outputs"]
        assert outputs["boxes"] == "coordinates"
        assert outputs["scores"] == "confidence"
        assert outputs["box_format"] == "xywh"
        assert outputs["coordinate_space"] == "normalized"
        assert outputs["geometry"] == {
            "confidence": {
                "name": "confidence",
                "kind": "multi_array",
                "data_type": "float32",
                "shape_range": [
                    {"lower_bound": 0, "upper_bound": -1},
                    {"lower_bound": 1, "upper_bound": 1},
                ],
            },
            "coordinates": {
                "name": "coordinates",
                "kind": "multi_array",
                "data_type": "float32",
                "shape_range": [
                    {"lower_bound": 0, "upper_bound": -1},
                    {"lower_bound": 4, "upper_bound": 4},
                ],
            },
        }

        inspection = manifest.raw["inspection"]
        assert [item["name"] for item in inspection["inputs"]] == [
            "image",
            "iouThreshold",
            "confidenceThreshold",
        ]
        assert [item["name"] for item in inspection["outputs"]] == [
            "confidence",
            "coordinates",
        ]
        assert inspection["inputs"][1:] == [
            {"name": "iouThreshold", "kind": "double"},
            {"name": "confidenceThreshold", "kind": "double"},
        ]
        assert inspection["outputs"] == [
            outputs["geometry"]["confidence"],
            outputs["geometry"]["coordinates"],
        ]


def test_catalog_notice_contains_license_attribution_and_custom_import_statement() -> None:
    notice = (CATALOG_ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    assert "morsetechlab/yolov11-license-plate-detection" in notice
    assert SOURCE_REVISION in notice
    assert "AGPL-3.0" in notice
    assert "Upstream metrics are not product performance claims" in notice
    assert "custom model import option remains required" in notice

    license_text = (CATALOG_ROOT / "licenses/AGPL-3.0.txt").read_text(encoding="utf-8")
    assert license_text.startswith("                    GNU AFFERO GENERAL PUBLIC LICENSE")
    assert "END OF TERMS AND CONDITIONS" in license_text


def test_conversion_requirements_are_exact_and_hash_locked() -> None:
    requirements = (
        (CATALOG_ROOT.parent / "tools/model_catalog/requirements.in")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    lock = (
        (CATALOG_ROOT.parent / "tools/model_catalog/requirements-macos-arm64.lock")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    direct = {
        line.split("==", 1)[0].strip().lower(): line.split("==", 1)[1].strip()
        for line in requirements
        if line.strip() and not line.lstrip().startswith("#")
    }
    package_starts = [
        index
        for index, line in enumerate(lock)
        if "==" in line and line and not line[0].isspace() and not line.startswith("#")
    ]
    locked = {
        lock[index].split("==", 1)[0].strip().lower(): lock[index]
        .split("==", 1)[1]
        .split("\\", 1)[0]
        .strip()
        for index in package_starts
    }
    assert direct == {name: locked[name] for name in direct}
    assert len(locked) == len(package_starts)
    for position, start in enumerate(package_starts):
        end = package_starts[position + 1] if position + 1 < len(package_starts) else len(lock)
        block = "\n".join(lock[start:end])
        assert re.search(r"--hash=sha256:[0-9a-f]{64}\b", block)
