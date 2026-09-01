"""Secure archive and package validation for managed models."""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import stat
import zipfile
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

MAX_ARCHIVE_BYTES = 512 * 1024 * 1024
MAX_UNCOMPRESSED_BYTES = 2 * 1024 * 1024 * 1024
MAX_FILE_BYTES = 512 * 1024 * 1024
MAX_ARCHIVE_ENTRIES = 4096
MAX_PATH_LENGTH = 512
_WINDOWS_ABSOLUTE_RE = re.compile(r"^[A-Za-z]:([/\\\\]|$)")
_EXECUTABLE_SUFFIXES = frozenset(
    {
        ".app",
        ".bat",
        ".bash",
        ".command",
        ".dll",
        ".dylib",
        ".exe",
        ".fish",
        ".py",
        ".pyc",
        ".pyo",
        ".sh",
        ".so",
        ".zsh",
    }
)


class ModelArchiveError(ValueError):
    """Raised when a model archive or package is unsafe."""


def validate_and_extract_archive(
    archive_path: Path,
    destination: Path,
    *,
    expected_artifact: str,
) -> Path:
    """Validate a ZIP archive and extract it into a new staging directory."""
    if archive_path.is_symlink() or not archive_path.is_file():
        raise ModelArchiveError("model archive was not found")
    if archive_path.stat().st_size > MAX_ARCHIVE_BYTES:
        raise ModelArchiveError("model archive exceeds the compressed size limit")
    try:
        with zipfile.ZipFile(archive_path) as archive:
            infos = _validated_zip_infos(archive, expected_artifact)
            package_root = destination / expected_artifact
            package_root.mkdir(parents=True, exist_ok=False)
            for info, relative_name in infos:
                if info.is_dir():
                    continue
                target = destination / relative_name
                target.parent.mkdir(parents=True, exist_ok=True)
                _extract_file(archive, info, target)
    except zipfile.BadZipFile as error:
        raise ModelArchiveError("model archive is not a valid ZIP file") from error
    except OSError as error:
        raise ModelArchiveError("model archive could not be extracted safely") from error
    return package_root


def copy_and_validate_package(
    package_path: Path,
    destination: Path,
    *,
    expected_artifact: str,
) -> Path:
    """Validate a local .mlpackage directory and copy it into staging."""
    if not package_path.is_dir() or package_path.is_symlink():
        raise ModelArchiveError("model package must be a real directory")
    if package_path.name != expected_artifact:
        raise ModelArchiveError("model package name does not match manifest artifact")

    files = list(_validated_directory_files(package_path))
    if not files:
        raise ModelArchiveError("model package must contain at least one file")
    target = destination / expected_artifact
    target.mkdir(parents=True, exist_ok=False)
    try:
        for source, relative_name in files:
            target_file = target / relative_name
            target_file.parent.mkdir(parents=True, exist_ok=True)
            with source.open("rb") as source_file, target_file.open("xb") as target_handle:
                shutil.copyfileobj(source_file, target_handle, length=1024 * 1024)
            target_file.chmod(0o600)
    except OSError as error:
        raise ModelArchiveError("model package could not be copied safely") from error
    return target


def validate_package_directory(
    package_path: Path,
    *,
    expected_artifact: str,
) -> tuple[tuple[Path, Path], ...]:
    """Validate a local .mlpackage directory and return its checked files."""
    if not package_path.is_dir() or package_path.is_symlink():
        raise ModelArchiveError("model package must be a real directory")
    if package_path.name != expected_artifact:
        raise ModelArchiveError("model package name does not match manifest artifact")
    return tuple(_validated_directory_files(package_path))


def compute_artifact_sha256(package_path: Path) -> str:
    """Hash sorted relative paths and file bytes for deterministic provenance."""
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


def make_package_immutable(package_path: Path) -> None:
    """Make an imported package read-only after its atomic move."""
    for path in sorted(package_path.rglob("*"), reverse=True):
        if path.is_symlink():
            raise ModelArchiveError("model package may not contain symlinks")
        if path.is_dir():
            path.chmod(0o500)
        elif path.is_file():
            path.chmod(0o400)
    package_path.chmod(0o500)


def _validated_zip_infos(
    archive: zipfile.ZipFile,
    expected_artifact: str,
) -> list[tuple[zipfile.ZipInfo, str]]:
    infos = archive.infolist()
    if not infos:
        raise ModelArchiveError("model archive is empty")
    if len(infos) > MAX_ARCHIVE_ENTRIES:
        raise ModelArchiveError("model archive contains too many entries")

    seen: set[str] = set()
    files = 0
    uncompressed_bytes = 0
    validated: list[tuple[zipfile.ZipInfo, str]] = []
    for info in infos:
        normalized = _normalize_archive_name(info.filename)
        if normalized.casefold() in seen:
            raise ModelArchiveError("model archive contains duplicate paths")
        seen.add(normalized.casefold())

        if normalized != expected_artifact and not normalized.startswith(f"{expected_artifact}/"):
            raise ModelArchiveError("model archive must contain one top-level .mlpackage")
        if normalized == expected_artifact and not info.is_dir():
            raise ModelArchiveError("model archive package root must be a directory")
        _reject_unsafe_entry(info, normalized)
        if not info.is_dir():
            files += 1
            if files > MAX_ARCHIVE_ENTRIES:
                raise ModelArchiveError("model archive contains too many files")
            if info.file_size > MAX_FILE_BYTES:
                raise ModelArchiveError("model archive contains an oversized file")
            uncompressed_bytes += info.file_size
            if uncompressed_bytes > MAX_UNCOMPRESSED_BYTES:
                raise ModelArchiveError("model archive exceeds the uncompressed size limit")
        validated.append((info, normalized))

    if files == 0:
        raise ModelArchiveError("model package must contain at least one file")
    if expected_artifact not in {name.rstrip("/") for _info, name in validated} and not any(
        name.startswith(f"{expected_artifact}/") for _info, name in validated
    ):
        raise ModelArchiveError("model archive has an invalid package root")
    return validated


def _normalize_archive_name(name: str) -> str:
    if "\x00" in name or len(name) > MAX_PATH_LENGTH:
        raise ModelArchiveError("model archive contains an invalid path")
    replaced = name.replace("\\", "/")
    if replaced.startswith("/") or _WINDOWS_ABSOLUTE_RE.match(replaced):
        raise ModelArchiveError("model archive contains an absolute path")
    if replaced.endswith("/"):
        replaced = replaced.rstrip("/")
    parts = replaced.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ModelArchiveError("model archive contains a traversal path")
    normalized = PurePosixPath(*parts).as_posix()
    if normalized in {".", ""}:
        raise ModelArchiveError("model archive contains an empty path")
    return normalized


def _reject_unsafe_entry(info: zipfile.ZipInfo, normalized: str) -> None:
    mode = (info.external_attr >> 16) & 0xFFFF
    if stat.S_ISLNK(mode):
        raise ModelArchiveError("model archive may not contain symlinks")
    if mode and stat.S_IMODE(mode) & 0o111:
        raise ModelArchiveError("model archive may not contain executable files")
    if not info.is_dir() and _looks_executable(normalized):
        raise ModelArchiveError("model archive contains executable content")
    if info.flag_bits & 0x1:
        raise ModelArchiveError("encrypted model archives are not supported")


def _looks_executable(relative_name: str) -> bool:
    path = PurePosixPath(relative_name)
    return path.name.startswith("__pycache__") or path.suffix.lower() in _EXECUTABLE_SUFFIXES


def _extract_file(archive: zipfile.ZipFile, info: zipfile.ZipInfo, target: Path) -> None:
    written = 0
    try:
        with archive.open(info, "r") as source, target.open("xb") as destination:
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > MAX_FILE_BYTES:
                    raise ModelArchiveError("model archive contains an oversized file")
                destination.write(chunk)
        if written != info.file_size:
            raise ModelArchiveError("model archive file size is inconsistent")
    except OSError as error:
        raise ModelArchiveError("model archive file could not be read") from error
    target.chmod(0o600)


def _validated_directory_files(package_path: Path) -> Iterable[tuple[Path, Path]]:
    files = 0
    total_bytes = 0
    for root, directories, names in os.walk(package_path, followlinks=False):
        root_path = Path(root)
        safe_directories: list[str] = []
        for directory in directories:
            candidate = root_path / directory
            if candidate.is_symlink():
                raise ModelArchiveError("model package may not contain symlinks")
            if not candidate.is_dir():
                raise ModelArchiveError("model package contains a non-directory entry")
            safe_directories.append(directory)
        directories[:] = safe_directories
        for name in names:
            source = root_path / name
            if source.is_symlink():
                raise ModelArchiveError("model package may not contain symlinks")
            if not source.is_file():
                raise ModelArchiveError("model package contains a non-file entry")
            relative = source.relative_to(package_path)
            _reject_local_entry(source, relative)
            size = source.stat().st_size
            if size > MAX_FILE_BYTES:
                raise ModelArchiveError("model package contains an oversized file")
            files += 1
            if files > MAX_ARCHIVE_ENTRIES:
                raise ModelArchiveError("model package contains too many files")
            total_bytes += size
            if total_bytes > MAX_UNCOMPRESSED_BYTES:
                raise ModelArchiveError("model package exceeds the uncompressed size limit")
            yield source, relative

    if files == 0:
        raise ModelArchiveError("model package must contain at least one file")


def _reject_local_entry(path: Path, relative: Path) -> None:
    if len(relative.as_posix()) > MAX_PATH_LENGTH:
        raise ModelArchiveError("model package contains an invalid path")
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o111:
        raise ModelArchiveError("model package may not contain executable files")
    if _looks_executable(relative.as_posix()):
        raise ModelArchiveError("model package contains executable content")
