"""Secure model import and package-only validation operations."""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..paths import ManagedPaths
from .archive import (
    ModelArchiveError,
    compute_artifact_sha256,
    copy_and_validate_package,
    make_package_immutable,
    validate_and_extract_archive,
    validate_package_directory,
)
from .manifest import ModelManifestError, parse_manifest
from .repository import Model, ModelCreate, ModelRepository, manifest_from_record


class ModelImportError(ValueError):
    """Raised when a model cannot be imported without changing managed state."""


class ModelConflictError(ModelImportError):
    """Raised when an import would replace an existing managed model."""


class ModelDeletionError(ModelImportError):
    """Raised when a model deletion cannot finish or restore its prior state."""


PENDING_RUNTIME_VALIDATION = "pending_runtime_validation"
RUNTIME_VALID = "runtime_valid"


@dataclass(frozen=True)
class ModelValidation:
    """Package validation result safe to return to an operator."""

    structural_valid: bool
    runtime_valid: bool | None
    state: str
    details: dict[str, Any]


def import_model(
    *,
    manifest_value: bytes | str | dict[str, Any],
    source_path: Path,
    paths: ManagedPaths,
    repository: ModelRepository,
) -> Model:
    """Import one ZIP archive or .mlpackage directory with failure cleanup."""
    try:
        manifest = parse_manifest(manifest_value)
    except ModelManifestError as error:
        raise ModelImportError(str(error)) from error

    paths.ensure_directories()
    model_directory = paths.models / manifest.model_id
    if model_directory.exists() or model_directory.is_symlink():
        raise ModelConflictError("a managed model with this manifest id already exists")

    stage_root = Path(tempfile.mkdtemp(prefix="model-", dir=paths.staging))
    stage_package: Path | None = None
    model_directory_created = False
    committed = False
    try:
        model_directory.mkdir(mode=0o700)
        model_directory_created = True
        final_path = model_directory / manifest.artifact
        try:
            if source_path.is_dir():
                stage_package = copy_and_validate_package(
                    source_path,
                    stage_root,
                    expected_artifact=manifest.artifact,
                )
            else:
                stage_package = validate_and_extract_archive(
                    source_path,
                    stage_root,
                    expected_artifact=manifest.artifact,
                )
        except ModelArchiveError as error:
            raise ModelImportError(str(error)) from error

        checksum = compute_artifact_sha256(stage_package)
        if checksum != manifest.artifact_sha256:
            raise ModelImportError("model artifact SHA-256 does not match manifest artifact_sha256")

        details = {
            "manifest": "valid",
            "archive": "valid",
            "artifact_sha256": checksum,
            "runtime_validation": "not_run",
            "structural_validation": "passed",
        }
        try:
            stage_package.chmod(0o700)
            os.replace(stage_package, final_path)
        except FileExistsError as error:
            raise ModelConflictError(
                "a managed model with this manifest id already exists"
            ) from error
        stage_package = None

        make_package_immutable(final_path)
        model = repository.create(
            ModelCreate(
                manifest=manifest,
                artifact_path=model_directory.relative_to(paths.models)
                .joinpath(manifest.artifact)
                .as_posix(),
                validation_state=PENDING_RUNTIME_VALIDATION,
                validation_details=details,
            )
        )
        committed = True
        return model
    except ModelImportError:
        raise
    except OSError as error:
        raise ModelImportError("model import failed before it was committed") from error
    finally:
        if stage_package is not None:
            _remove_path(stage_package)
        _remove_path(stage_root)
        if model_directory_created and not committed:
            _remove_path(model_directory)


def validate_model(
    *,
    model: Model,
    paths: ManagedPaths,
    repository: ModelRepository,
) -> ModelValidation:
    """Recheck manifest, package structure, and checksum without loading the model."""
    details: dict[str, Any] = {
        "runtime_validation": "not_run",
        "structural_validation": "not_run",
    }
    try:
        manifest = manifest_from_record(model)
        if manifest.model_id != model.id:
            raise ModelImportError("stored manifest id does not match model record")
        artifact_path = ManagedPaths.validate_contained_path(
            paths.models / model.artifact_path,
            paths.models,
        )
        if artifact_path.name != manifest.artifact:
            raise ModelImportError("stored artifact path does not match manifest")
        if not artifact_path.is_dir() or artifact_path.is_symlink():
            raise ModelImportError("managed model artifact is missing")
        validate_package_directory(artifact_path, expected_artifact=manifest.artifact)
        checksum = compute_artifact_sha256(artifact_path)
        details.update(
            {
                "manifest": "valid",
                "archive": "valid",
                "artifact_sha256": checksum,
                "structural_validation": "passed",
            }
        )
        if checksum != manifest.artifact_sha256 or checksum != model.artifact_sha256:
            raise ModelImportError("managed model artifact SHA-256 does not match provenance")
    except (ModelImportError, ModelManifestError, ValueError, OSError) as error:
        details["error"] = str(error)
        details["structural_validation"] = "failed"
        stored = repository.update_validation(model.id, state="invalid", details=details)
        if stored is None:
            raise ModelImportError("model was removed during validation") from error
        return ModelValidation(
            structural_valid=False,
            runtime_valid=None,
            state=stored.validation_state,
            details=details,
        )

    stored = repository.update_validation(
        model.id,
        state=PENDING_RUNTIME_VALIDATION,
        details=details,
    )
    if stored is None:
        raise ModelImportError("model was removed during validation")
    return ModelValidation(
        structural_valid=True,
        runtime_valid=None,
        state=stored.validation_state,
        details=details,
    )


def delete_model(
    *,
    model: Model,
    paths: ManagedPaths,
    repository: ModelRepository,
) -> None:
    """Delete one inactive model with quarantine and rollback protection."""
    if model.active:
        raise ModelImportError("active models cannot be deleted")
    artifact_path = ManagedPaths.validate_contained_path(
        paths.models / model.artifact_path,
        paths.models,
    )
    if artifact_path == paths.models:
        raise ModelImportError("model artifact path is invalid")
    if (artifact_path.exists() or artifact_path.is_symlink()) and (
        artifact_path.is_symlink() or not artifact_path.is_dir()
    ):
        raise ModelImportError("managed model artifact path is invalid")

    model_directory = artifact_path.parent
    if model_directory.parent != paths.models or model_directory.name != model.id:
        raise ModelImportError("managed model artifact path is invalid")
    if artifact_path.exists() and {child.name for child in model_directory.iterdir()} != {
        artifact_path.name
    }:
        raise ModelImportError("managed model directory contains unexpected files")

    paths.ensure_directories()
    quarantine_root = Path(tempfile.mkdtemp(prefix="delete-", dir=paths.staging))
    quarantine_directory = quarantine_root / model_directory.name
    moved = False
    deleted: Model | None = None
    try:
        try:
            os.replace(model_directory, quarantine_directory)
            moved = True
        except OSError as error:
            raise ModelDeletionError("model artifact could not be quarantined") from error

        try:
            deleted = repository.delete(model.id)
        except Exception as error:
            _restore_deleted_model(
                model=model,
                repository=repository,
                model_directory=model_directory,
                quarantine_directory=quarantine_directory,
                quarantine_root=quarantine_root,
            )
            raise ModelDeletionError(
                "model registry deletion failed; model was restored"
            ) from error
        if deleted is None:
            _restore_deleted_model(
                model=model,
                repository=repository,
                model_directory=model_directory,
                quarantine_directory=quarantine_directory,
                quarantine_root=quarantine_root,
            )
            raise ModelDeletionError("model was not found; model was restored")

        try:
            _remove_path(quarantine_root)
        except OSError as error:
            try:
                os.replace(quarantine_directory, model_directory)
                repository.restore(deleted)
                _remove_path(quarantine_root)
            except Exception as restore_error:
                raise ModelDeletionError(
                    "model deletion cleanup failed and model restoration failed"
                ) from restore_error
            raise ModelDeletionError("model deletion cleanup failed; model was restored") from error
    finally:
        if not moved:
            _remove_path(quarantine_root)


def _restore_deleted_model(
    *,
    model: Model,
    repository: ModelRepository,
    model_directory: Path,
    quarantine_directory: Path,
    quarantine_root: Path,
) -> None:
    """Restore both sides of a deletion before returning an error."""
    restore_errors: list[Exception] = []
    try:
        if quarantine_directory.exists():
            os.replace(quarantine_directory, model_directory)
    except Exception as error:
        restore_errors.append(error)
    try:
        repository.restore(model)
    except Exception as error:
        restore_errors.append(error)
    if not restore_errors:
        try:
            if quarantine_root.exists():
                _remove_path(quarantine_root)
        except Exception as error:
            restore_errors.append(error)
    if restore_errors:
        raise ModelDeletionError("model deletion failed and model restoration failed") from (
            restore_errors[0]
        )


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.is_dir():
        for directory, directories, files in os.walk(path):
            for name in files:
                candidate = Path(directory) / name
                if candidate.is_symlink():
                    candidate.unlink(missing_ok=True)
                else:
                    candidate.chmod(0o600)
            for name in list(directories):
                candidate = Path(directory) / name
                if candidate.is_symlink():
                    candidate.unlink(missing_ok=True)
                    directories.remove(name)
                else:
                    candidate.chmod(0o700)
        path.chmod(0o700)
        shutil.rmtree(path)
