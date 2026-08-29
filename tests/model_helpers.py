from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

from open_licenseplate.models.archive import compute_artifact_sha256


def create_model_fixture(
    root: Path,
    *,
    model_id: str = "test-model",
) -> tuple[Path, Path, dict[str, Any]]:
    """Create a small deterministic package, archive, and matching manifest."""
    package = root / "model.mlpackage"
    (package / "Data").mkdir(parents=True)
    (package / "Manifest.json").write_text("{}", encoding="utf-8")
    (package / "Data" / "weights.bin").write_bytes(b"test model bytes")
    checksum = compute_artifact_sha256(package)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "id": model_id,
        "display_name": "Test model",
        "task": "object_detection",
        "backend": "coreml",
        "adapter": "ultralytics_yolo_nms",
        "artifact": "model.mlpackage",
        "artifact_sha256": checksum,
        "input": {
            "name": "image",
            "kind": "image",
            "width": 640,
            "height": 640,
            "color_space": "rgb",
        },
        "preprocessing": {"resize": "letterbox"},
        "outputs": {"boxes": "coordinates", "scores": "confidence"},
        "labels": ["license_plate"],
        "defaults": {"confidence_threshold": 0.35, "iou_threshold": 0.45},
        "compatibility": {"minimum_macos": "14.0"},
        "source": {"url": "https://example.test/model", "license": "MIT"},
        "conversion": {"source_weight": "weights.pt", "tool_versions": {}, "arguments": {}},
    }
    archive = root / f"{model_id}.zip"
    with zipfile.ZipFile(archive, "w") as output:
        for path in package.rglob("*"):
            if path.is_file():
                output.write(path, path.relative_to(root).as_posix())
    manifest_path = root / f"{model_id}.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, archive, manifest
