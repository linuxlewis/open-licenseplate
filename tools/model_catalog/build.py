"""Build, inspect, normalize, and checksum the model catalog assets.

This script is intentionally outside the application package. Run it from a
temporary Python 3.12 environment that contains the pinned conversion tools.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote
from urllib.request import urlopen

SOURCE_REPOSITORY = "morsetechlab/yolov11-license-plate-detection"
SOURCE_REVISION = "251a30d7daedca065f56e04b0af04052c907c68f"
SOURCE_REPOSITORY_URL = f"https://huggingface.co/{SOURCE_REPOSITORY}"
SOURCE_LICENSE = "AGPL-3.0"
RELEASE_TAG = "model-catalog-v1"
DEFAULT_RELEASE_BASE_URL = (
    "https://github.com/linuxlewis/open-licenseplate/releases/download/model-catalog-v1"
)
EXPECTED_PYTHON = "3.12.14"
ULTRALYTICS_VERSION = "8.3.200"
COREMLTOOLS_VERSION = "8.3.0"
TORCH_VERSION = "2.5.0"
TORCHVISION_VERSION = "0.20.0"
NUMPY_VERSION = "1.26.4"
IMAGE_SIZE = 640
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.70


@dataclass(frozen=True)
class Variant:
    suffix: str
    model_id: str
    display_name: str
    source_weight: str
    recommendation: str


VARIANTS = (
    Variant(
        suffix="n",
        model_id="license-plate-yolov11n",
        display_name="YOLOv11 Nano License Plate Detector",
        source_weight="license-plate-finetune-v1n.pt",
        recommendation="fast_default",
    ),
    Variant(
        suffix="s",
        model_id="license-plate-yolov11s",
        display_name="YOLOv11 Small License Plate Detector",
        source_weight="license-plate-finetune-v1s.pt",
        recommendation="balanced",
    ),
    Variant(
        suffix="m",
        model_id="license-plate-yolov11m",
        display_name="YOLOv11 Medium License Plate Detector",
        source_weight="license-plate-finetune-v1m.pt",
        recommendation="higher_capacity",
    ),
)


def main() -> int:
    args = _parse_args()
    if sys.version_info[:3] != tuple(int(part) for part in EXPECTED_PYTHON.split(".")):
        raise SystemExit(
            f"Python {EXPECTED_PYTHON} is required for this build; found {sys.version.split()[0]}"
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    source_dir = output_dir / "source"
    package_dir = output_dir / "packages"
    archive_dir = output_dir / "archives"
    manifest_dir = output_dir / "manifests"
    for directory in (source_dir, package_dir, archive_dir, manifest_dir):
        directory.mkdir()

    _check_tool_versions()
    for variant in VARIANTS:
        source_path = source_dir / variant.source_weight
        _download_source_weight(variant, source_path)
        package_path = _export_variant(variant, source_path, package_dir)
        inspection = _inspect_and_normalize_package(variant, package_path)
        package_sha256 = compute_package_sha256(package_path)
        archive_name = f"open-licenseplate-model-catalog-{variant.model_id}.zip"
        archive_path = archive_dir / archive_name
        create_reproducible_archive(package_path, archive_path)
        archive_sha256 = sha256_file(archive_path)
        manifest = _make_manifest(
            variant=variant,
            source_path=source_path,
            package_path=package_path,
            package_sha256=package_sha256,
            archive_name=archive_name,
            archive_path=archive_path,
            archive_sha256=archive_sha256,
            inspection=inspection,
        )
        manifest_path = manifest_dir / f"{variant.model_id}.json"
        write_json(manifest_path, manifest)

    lock = _make_lock(
        output_dir=output_dir,
        release_base_url=args.release_base_url.rstrip("/"),
    )
    write_json(output_dir / "model-catalog-lock.json", lock)
    _write_checksums(output_dir, lock)
    print(f"Built model catalog assets in {output_dir}")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="new directory for source weights and generated catalog assets",
    )
    parser.add_argument(
        "--release-base-url",
        default=DEFAULT_RELEASE_BASE_URL,
        help="fixed HTTPS release URL prefix used in the generated lock file",
    )
    return parser.parse_args()


def _check_tool_versions() -> None:
    import coremltools
    import numpy
    import torch
    import torchvision
    import ultralytics

    actual = {
        "ultralytics": ultralytics.__version__,
        "coremltools": coremltools.__version__,
        "torch": torch.__version__.split("+", 1)[0],
        "torchvision": torchvision.__version__.split("+", 1)[0],
        "numpy": numpy.__version__,
    }
    expected = {
        "ultralytics": ULTRALYTICS_VERSION,
        "coremltools": COREMLTOOLS_VERSION,
        "torch": TORCH_VERSION,
        "torchvision": TORCHVISION_VERSION,
        "numpy": NUMPY_VERSION,
    }
    mismatches = [
        f"{name}={actual[name]} (expected {expected[name]})"
        for name in expected
        if actual[name] != expected[name]
    ]
    if mismatches:
        raise RuntimeError("conversion tool version mismatch: " + ", ".join(mismatches))


def _download_source_weight(variant: Variant, destination: Path) -> None:
    source_url = _source_weight_url(variant.source_weight)
    with urlopen(source_url) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output)
    expected_size, expected_sha256 = _expected_source_digest(variant.source_weight)
    if destination.stat().st_size != expected_size:
        raise RuntimeError(
            f"{variant.source_weight} size mismatch: "
            f"{destination.stat().st_size} != {expected_size}"
        )
    actual_sha256 = sha256_file(destination)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"{variant.source_weight} SHA-256 mismatch: {actual_sha256} != {expected_sha256}"
        )


def _expected_source_digest(source_weight: str) -> tuple[int, str]:
    expected = {
        "license-plate-finetune-v1n.pt": (
            5465235,
            "0aec75976c56eb6f26dfb274c430620ec65137915ff1ae47c3a48c7af8afb7b2",
        ),
        "license-plate-finetune-v1s.pt": (
            19173715,
            "95e50c25ab7066dd0ca5aec18fa80349676db08697780d1149576461174d2381",
        ),
        "license-plate-finetune-v1m.pt": (
            40507749,
            "d691f8d5e7709d2065b18a0e2bd75deaf08f6115cc0295cc3de7d0fd2eabcdae",
        ),
    }
    try:
        return expected[source_weight]
    except KeyError as error:
        raise RuntimeError(f"unexpected source weight: {source_weight}") from error


def _source_weight_url(source_weight: str) -> str:
    return f"{SOURCE_REPOSITORY_URL}/resolve/{SOURCE_REVISION}/{quote(source_weight, safe='')}"


def _export_variant(variant: Variant, source_path: Path, package_dir: Path) -> Path:
    from ultralytics import YOLO

    export_root = Path(tempfile.mkdtemp(prefix=f"export-{variant.suffix}-"))
    try:
        export_source = export_root / source_path.name
        shutil.copyfile(source_path, export_source)
        model = YOLO(str(export_source))
        if model.task != "detect" or model.names != {0: "License_Plate"}:
            raise RuntimeError(
                f"{variant.source_weight} has an unexpected task or class mapping: "
                f"task={model.task!r}, names={model.names!r}"
            )
        exported = model.export(
            format="coreml",
            imgsz=IMAGE_SIZE,
            nms=True,
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            half=False,
            int8=False,
            device="cpu",
        )
        exported_path = Path(str(exported))
        if not exported_path.is_dir() or exported_path.suffix != ".mlpackage":
            raise RuntimeError(f"Core ML export did not produce an mlpackage: {exported_path}")
        destination = package_dir / f"{variant.model_id}.mlpackage"
        shutil.move(str(exported_path), str(destination))
        return destination
    finally:
        shutil.rmtree(export_root, ignore_errors=True)


def _inspect_and_normalize_package(variant: Variant, package_path: Path) -> dict[str, Any]:
    import coremltools as ct

    model_spec_path = package_path / "Data" / "com.apple.CoreML" / "model.mlmodel"
    package_manifest_path = package_path / "Manifest.json"
    if not model_spec_path.is_file() or not package_manifest_path.is_file():
        raise RuntimeError(f"{package_path} is missing the expected Core ML package files")

    spec = ct.utils.load_spec(str(package_path))
    _remove_volatile_metadata(spec)
    model_spec_path.write_bytes(spec.SerializeToString(deterministic=True))
    _write_stable_package_manifest(package_manifest_path, variant.model_id)

    model = ct.models.MLModel(str(package_path), compute_units=ct.ComputeUnit.CPU_ONLY)
    inspection = _inspect_spec(model.get_spec())
    expected_input = {
        "name": "image",
        "kind": "image",
        "width": IMAGE_SIZE,
        "height": IMAGE_SIZE,
        "color_space": "rgb",
    }
    if inspection["inputs"][0] != expected_input:
        raise RuntimeError(f"unexpected image input for {variant.model_id}: {inspection}")
    output_names = {item["name"] for item in inspection["outputs"]}
    if output_names != {"coordinates", "confidence"}:
        raise RuntimeError(f"unexpected output names for {variant.model_id}: {inspection}")
    coordinate = next(item for item in inspection["outputs"] if item["name"] == "coordinates")
    confidence = next(item for item in inspection["outputs"] if item["name"] == "confidence")
    if coordinate["shape_range"] != [
        {"lower_bound": 0, "upper_bound": -1},
        {"lower_bound": 4, "upper_bound": 4},
    ]:
        raise RuntimeError(f"unexpected coordinates geometry for {variant.model_id}: {coordinate}")
    if confidence["shape_range"] != [
        {"lower_bound": 0, "upper_bound": -1},
        {"lower_bound": 1, "upper_bound": 1},
    ]:
        raise RuntimeError(f"unexpected confidence geometry for {variant.model_id}: {confidence}")
    return inspection


def _remove_volatile_metadata(spec: Any) -> None:
    user_defined = spec.description.metadata.userDefined
    user_defined.pop("date", None)


def _write_stable_package_manifest(path: Path, model_id: str) -> None:
    values = json.loads(path.read_text(encoding="utf-8"))
    model_identifier = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{model_id}:model")).upper()
    weights_identifier = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{model_id}:weights")).upper()
    entries = values["itemInfoEntries"]
    model_entry = next(
        entry for entry in entries.values() if entry["path"] == "com.apple.CoreML/model.mlmodel"
    )
    weights_entry = next(
        entry for entry in entries.values() if entry["path"] == "com.apple.CoreML/weights"
    )
    values["itemInfoEntries"] = {
        model_identifier: {
            "author": model_entry["author"],
            "description": model_entry["description"],
            "name": model_entry["name"],
            "path": model_entry["path"],
        },
        weights_identifier: {
            "author": weights_entry["author"],
            "description": weights_entry["description"],
            "name": weights_entry["name"],
            "path": weights_entry["path"],
        },
    }
    values["rootModelIdentifier"] = model_identifier
    write_json(path, values)


def _inspect_spec(spec: Any) -> dict[str, Any]:
    inputs = [_inspect_feature(feature) for feature in spec.description.input]
    outputs = [_inspect_feature(feature) for feature in spec.description.output]
    return {"inputs": inputs, "outputs": outputs}


def _inspect_feature(feature: Any) -> dict[str, Any]:
    feature_type = feature.type
    if feature_type.HasField("imageType"):
        image_type = feature_type.imageType
        color_spaces = {10: "grayscale", 20: "rgb", 30: "bgr", 40: "grayscale_float16"}
        return {
            "name": feature.name,
            "kind": "image",
            "width": int(image_type.width),
            "height": int(image_type.height),
            "color_space": color_spaces[int(image_type.colorSpace)],
        }
    if feature_type.HasField("multiArrayType"):
        array_type = feature_type.multiArrayType
        return {
            "name": feature.name,
            "kind": "multi_array",
            "data_type": {65552: "float16", 65568: "float32", 65600: "double"}.get(
                int(array_type.dataType), str(array_type.dataType)
            ),
            "shape_range": [
                {
                    "lower_bound": int(size_range.lowerBound),
                    "upper_bound": int(size_range.upperBound),
                }
                for size_range in array_type.shapeRange.sizeRanges
            ],
        }
    type_name = feature_type.WhichOneof("Type")
    type_names = {
        "dictionaryType": "dictionary",
        "stringType": "string",
        "int64Type": "int64",
        "doubleType": "double",
        "boolType": "bool",
    }
    return {"name": feature.name, "kind": type_names.get(type_name, "unknown")}


def _make_manifest(
    *,
    variant: Variant,
    source_path: Path,
    package_path: Path,
    package_sha256: str,
    archive_name: str,
    archive_path: Path,
    archive_sha256: str,
    inspection: dict[str, Any],
) -> dict[str, Any]:
    source_weight = variant.source_weight
    return {
        "schema_version": 1,
        "id": variant.model_id,
        "display_name": variant.display_name,
        "task": "object_detection",
        "backend": "coreml",
        "adapter": "ultralytics_yolo_nms",
        "artifact": package_path.name,
        "artifact_sha256": package_sha256,
        "input": {
            "name": "image",
            "kind": "image",
            "width": IMAGE_SIZE,
            "height": IMAGE_SIZE,
            "color_space": "rgb",
            "additional_inputs": [
                {
                    "name": "iouThreshold",
                    "kind": "double",
                    "optional": True,
                    "default": IOU_THRESHOLD,
                },
                {
                    "name": "confidenceThreshold",
                    "kind": "double",
                    "optional": True,
                    "default": CONFIDENCE_THRESHOLD,
                },
            ],
        },
        "preprocessing": {"resize": "letterbox"},
        "outputs": {
            "boxes": "coordinates",
            "scores": "confidence",
            "box_format": "xywh",
            "coordinate_space": "normalized",
            "geometry": {
                "coordinates": next(
                    item for item in inspection["outputs"] if item["name"] == "coordinates"
                ),
                "confidence": next(
                    item for item in inspection["outputs"] if item["name"] == "confidence"
                ),
            },
        },
        "labels": ["license_plate"],
        "defaults": {
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "iou_threshold": IOU_THRESHOLD,
        },
        "compatibility": {"minimum_macos": "14.0"},
        "source": {
            "url": _source_weight_url(source_weight),
            "repository": SOURCE_REPOSITORY_URL,
            "revision": SOURCE_REVISION,
            "file": source_weight,
            "license": SOURCE_LICENSE,
            "sha256": sha256_file(source_path),
        },
        "conversion": {
            "source_weight": source_weight,
            "tool_versions": {
                "python": EXPECTED_PYTHON,
                "ultralytics": ULTRALYTICS_VERSION,
                "coremltools": COREMLTOOLS_VERSION,
                "torch": TORCH_VERSION,
                "torchvision": TORCHVISION_VERSION,
                "numpy": NUMPY_VERSION,
            },
            "arguments": {
                "format": "coreml",
                "imgsz": IMAGE_SIZE,
                "nms": True,
                "conf": CONFIDENCE_THRESHOLD,
                "iou": IOU_THRESHOLD,
                "half": False,
                "int8": False,
                "device": "cpu",
            },
        },
        "inspection": inspection,
        "distribution": {
            "archive": archive_name,
            "archive_sha256": archive_sha256,
            "archive_size": archive_path.stat().st_size,
            "release_tag": RELEASE_TAG,
            "recommendation": variant.recommendation,
        },
    }


def _make_lock(*, output_dir: Path, release_base_url: str) -> dict[str, Any]:
    models = []
    for variant in VARIANTS:
        manifest_path = output_dir / "manifests" / f"{variant.model_id}.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        archive_name = manifest["distribution"]["archive"]
        models.append(
            {
                "id": manifest["id"],
                "manifest": f"manifests/{manifest_path.name}",
                "manifest_asset": manifest_path.name,
                "archive_asset": archive_name,
                "archive_url": f"{release_base_url}/{archive_name}",
                "package_sha256": manifest["artifact_sha256"],
                "archive_sha256": manifest["distribution"]["archive_sha256"],
                "archive_size": manifest["distribution"]["archive_size"],
                "source": manifest["source"],
                "license": manifest["source"]["license"],
                "release_tag": manifest["distribution"]["release_tag"],
            }
        )
    return {
        "schema_version": 1,
        "catalog_id": "open-licenseplate-model-catalog-v1",
        "release": {
            "repository": "linuxlewis/open-licenseplate",
            "tag": RELEASE_TAG,
            "prerelease": True,
        },
        "models": models,
    }


def _write_checksums(output_dir: Path, lock: dict[str, Any]) -> None:
    lines = [
        "# Package tree SHA-256 values are hashes of the unpacked .mlpackage trees.",
    ]
    for model in lock["models"]:
        lines.append(f"{model['package_sha256']}  package-tree/{model['id']}.mlpackage")
        lines.append(f"{model['archive_sha256']}  {model['archive_asset']}")
        manifest_path = output_dir / model["manifest"]
        lines.append(f"{sha256_file(manifest_path)}  {model['manifest_asset']}")
    lines.append(f"{sha256_file(output_dir / 'model-catalog-lock.json')}  model-catalog-lock.json")
    catalog_dir = Path(__file__).resolve().parents[2] / "model-catalog"
    for file_name in ("THIRD_PARTY_NOTICES.md", "licenses/AGPL-3.0.txt"):
        source_path = catalog_dir / file_name
        if source_path.is_file():
            lines.append(f"{sha256_file(source_path)}  {Path(file_name).name}")
    (output_dir / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def compute_package_sha256(package_path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(
        (
            path.relative_to(package_path).as_posix(),
            path,
        )
        for path in package_path.rglob("*")
        if path.is_file() and not path.is_symlink()
    )
    for relative_name, path in files:
        encoded_name = relative_name.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(8, "big"))
        digest.update(encoded_name)
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def create_reproducible_archive(package_path: Path, archive_path: Path) -> None:
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as archive:
        paths = sorted(path for path in package_path.rglob("*") if path.is_file())
        for path in paths:
            relative_name = path.relative_to(package_path.parent).as_posix()
            info = zipfile.ZipInfo(relative_name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o100600 << 16
            info.compress_type = zipfile.ZIP_STORED
            archive.writestr(info, path.read_bytes())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
