from __future__ import annotations

import stat
import zipfile
from pathlib import Path

import pytest

from open_licenseplate.models import archive
from open_licenseplate.models.archive import (
    ModelArchiveError,
    validate_and_extract_archive,
    validate_package_directory,
)


def _write_zip(path: Path, entries: list[tuple[str, bytes, int | None]]) -> None:
    with zipfile.ZipFile(path, "w") as output:
        for name, content, external_attr in entries:
            info = zipfile.ZipInfo(name)
            if external_attr is not None:
                info.create_system = 3
                info.external_attr = external_attr
            output.writestr(info, content)


def test_archive_extracts_one_safe_package(tmp_path: Path) -> None:
    archive_path = tmp_path / "model.zip"
    _write_zip(
        archive_path,
        [
            ("model.mlpackage/", b"", None),
            ("model.mlpackage/Manifest.json", b"{}", None),
            ("model.mlpackage/Data/weights.bin", b"weights", None),
        ],
    )

    package = validate_and_extract_archive(
        archive_path,
        tmp_path / "staging",
        expected_artifact="model.mlpackage",
    )

    assert package.is_dir()
    assert (package / "Data" / "weights.bin").read_bytes() == b"weights"


def test_package_directory_validator_returns_checked_files(tmp_path: Path) -> None:
    package = tmp_path / "model.mlpackage"
    (package / "Data").mkdir(parents=True)
    (package / "Manifest.json").write_text("{}", encoding="utf-8")
    (package / "Data" / "weights.bin").write_bytes(b"weights")

    checked_files = validate_package_directory(
        package,
        expected_artifact="model.mlpackage",
    )

    assert sorted(relative.as_posix() for _path, relative in checked_files) == [
        "Data/weights.bin",
        "Manifest.json",
    ]


@pytest.mark.parametrize(
    "entry_name",
    [
        "../outside",
        "/outside",
        "model.mlpackage/../../outside",
        "other.mlpackage/file",
        r"model.mlpackage\..\outside",
    ],
)
def test_archive_rejects_path_traversal_and_wrong_roots(tmp_path: Path, entry_name: str) -> None:
    archive_path = tmp_path / "model.zip"
    _write_zip(archive_path, [(entry_name, b"unsafe", None)])

    with pytest.raises(ModelArchiveError):
        validate_and_extract_archive(
            archive_path,
            tmp_path / "staging",
            expected_artifact="model.mlpackage",
        )
    assert not (tmp_path / "outside").exists()


def test_archive_rejects_symlink_duplicate_and_executable_entries(tmp_path: Path) -> None:
    symlink_archive = tmp_path / "symlink.zip"
    _write_zip(
        symlink_archive,
        [
            (
                "model.mlpackage/link",
                b"outside",
                (stat.S_IFLNK | 0o777) << 16,
            )
        ],
    )
    with pytest.raises(ModelArchiveError, match="symlinks"):
        validate_and_extract_archive(
            symlink_archive,
            tmp_path / "symlink-stage",
            expected_artifact="model.mlpackage",
        )

    duplicate_archive = tmp_path / "duplicate.zip"
    _write_zip(
        duplicate_archive,
        [
            ("model.mlpackage/file.bin", b"one", None),
            ("model.mlpackage/file.bin", b"two", None),
        ],
    )
    with pytest.raises(ModelArchiveError, match="duplicate"):
        validate_and_extract_archive(
            duplicate_archive,
            tmp_path / "duplicate-stage",
            expected_artifact="model.mlpackage",
        )

    executable_archive = tmp_path / "executable.zip"
    _write_zip(
        executable_archive,
        [("model.mlpackage/run.sh", b"#!/bin/sh", None)],
    )
    with pytest.raises(ModelArchiveError, match="executable"):
        validate_and_extract_archive(
            executable_archive,
            tmp_path / "executable-stage",
            expected_artifact="model.mlpackage",
        )


def test_archive_enforces_file_size_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive_path = tmp_path / "large.zip"
    _write_zip(archive_path, [("model.mlpackage/file.bin", b"1234", None)])
    monkeypatch.setattr(archive, "MAX_FILE_BYTES", 3)

    with pytest.raises(ModelArchiveError, match="oversized"):
        validate_and_extract_archive(
            archive_path,
            tmp_path / "stage",
            expected_artifact="model.mlpackage",
        )
