"""Live pipeline JSON API."""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from typing import Any, cast

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from ..cameras.repository import CameraConfig, CameraRepository, camera_config_from_record
from ..capture import ActiveCameraConflict
from ..database import Database, database_status
from ..inference import BackendOptions, ComputeUnit, ModelDescriptor
from ..models.archive import compute_artifact_sha256
from ..models.manifest import ModelManifestError
from ..models.repository import (
    Model,
    ModelRepository,
    manifest_from_record,
)
from ..models.service import RUNTIME_VALID
from ..paths import ManagedPaths
from .pipeline import (
    LivePipelineConflict,
    LivePipelineError,
    LivePipelineShutdownError,
    SourcePixelRegionOfInterest,
)

router = APIRouter(prefix="/api/v1/live", tags=["live"])


class LiveResourceError(ValueError):
    """Raised when a camera or model cannot be used for live detection."""


class LiveResourceNotFound(LiveResourceError):
    """Raised when a requested camera or model does not exist."""


@dataclass(frozen=True, slots=True)
class LiveResources:
    """Validated camera and model values passed to the runtime worker."""

    camera_id: str
    camera: CameraConfig
    descriptor: ModelDescriptor


@router.post("/start")
async def start_live(request: Request) -> JSONResponse:
    try:
        body = await _read_json(request)
        camera_id = _required_text(body.get("camera_id"), "camera_id")
        model_id = _optional_text(body.get("model_id"), "model_id")
        threshold = _threshold(body.get("confidence_threshold"))
        options = BackendOptions(compute_units=_compute_units(body.get("compute_units", "all")))
        region_of_interest = SourcePixelRegionOfInterest.from_value(body.get("region_of_interest"))
        resources = await asyncio.to_thread(
            _load_resources,
            request.app.state.paths,
            camera_id,
            model_id,
        )
        status = await asyncio.to_thread(
            request.app.state.live_pipeline.start,
            resources.camera_id,
            resources.camera,
            resources.descriptor,
            threshold=threshold,
            options=options,
            region_of_interest=region_of_interest,
        )
        return JSONResponse(content=status.as_dict())
    except LiveResourceNotFound as error:
        return _error(str(error), status_code=404)
    except LiveResourceError as error:
        return _error(str(error), status_code=409)
    except LivePipelineConflict as error:
        return _error(str(error), status_code=409)
    except ActiveCameraConflict as error:
        return _error(str(error), status_code=409)
    except (LivePipelineError, ValueError) as error:
        return _error(str(error), status_code=422)


@router.get("/state")
async def live_state(request: Request) -> dict[str, Any]:
    """Return the current live pipeline state without opening SQLite."""
    return cast(dict[str, Any], request.app.state.live_pipeline.status().as_dict())


@router.patch("/settings")
async def update_live_settings(request: Request) -> JSONResponse:
    try:
        body = await _read_json(request)
        value = body.get("confidence_threshold", body.get("threshold"))
        if value is None:
            raise ValueError("confidence_threshold is required")
        threshold = _threshold(value)
        status = await asyncio.to_thread(
            request.app.state.live_pipeline.update_threshold,
            threshold,
        )
        return JSONResponse(content=status.as_dict())
    except (LivePipelineError, ValueError) as error:
        return _error(str(error), status_code=409 if isinstance(error, LivePipelineError) else 422)


@router.post("/stop")
async def stop_live(request: Request) -> JSONResponse:
    try:
        status = await asyncio.to_thread(request.app.state.live_pipeline.stop)
        return JSONResponse(content=status.as_dict())
    except LivePipelineShutdownError as error:
        return _error(str(error), status_code=409)
    except LivePipelineError as error:
        return _error(str(error), status_code=409)


def _load_resources(
    paths: ManagedPaths,
    camera_id: str,
    model_id: str | None,
) -> LiveResources:
    if database_status(paths.database)["status"] != "ok":
        raise LiveResourceError("database is not ready; run `open-licenseplate db upgrade` first")
    database = Database(paths.database)
    try:
        camera_record = CameraRepository(database).get(camera_id)
        if camera_record is None:
            raise LiveResourceNotFound("camera was not found")
        model_record = _selected_model(ModelRepository(database), model_id)
        if model_record.validation_state != RUNTIME_VALID:
            raise LiveResourceError("runtime model validation is required before live detection")
        try:
            manifest = manifest_from_record(model_record)
            artifact_path = ManagedPaths.validate_contained_path(
                paths.models / model_record.artifact_path,
                paths.models,
            )
            if (
                not artifact_path.is_dir()
                or artifact_path.is_symlink()
                or artifact_path.name != manifest.artifact
                or compute_artifact_sha256(artifact_path) != model_record.artifact_sha256
            ):
                raise LiveResourceError("managed model artifact is missing or invalid")
        except (ModelManifestError, OSError, ValueError) as error:
            if isinstance(error, LiveResourceError):
                raise
            raise LiveResourceError("managed model artifact is missing or invalid") from None
        descriptor = ModelDescriptor(
            model_id=model_record.id,
            artifact_path=str(artifact_path),
            artifact_sha256=model_record.artifact_sha256,
            manifest=manifest,
        )
        return LiveResources(
            camera_id=camera_record.id,
            camera=camera_config_from_record(camera_record),
            descriptor=descriptor,
        )
    finally:
        database.dispose()


def _selected_model(repository: ModelRepository, model_id: str | None) -> Model:
    if model_id is not None:
        model = repository.get(model_id)
        if model is None:
            raise LiveResourceNotFound("model was not found")
        return model
    models = repository.list()
    active = next((model for model in models if model.active), None)
    if active is None:
        raise LiveResourceError("model_id is required when no model is active")
    return active


async def _read_json(request: Request) -> dict[str, Any]:
    try:
        body = await request.json()
    except ValueError:
        raise ValueError("request body must be valid JSON") from None
    if not isinstance(body, dict):
        raise ValueError("request body must be a JSON object")
    return body


def _required_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > 128:
        raise ValueError(f"{field_name} is required")
    return value.strip()


def _optional_text(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    return _required_text(value, field_name)


def _threshold(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("confidence_threshold must be a number from 0 through 1")
    try:
        result = float(str(value))
    except (TypeError, ValueError):
        raise ValueError("confidence_threshold must be a number from 0 through 1") from None
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise ValueError("confidence_threshold must be a number from 0 through 1")
    return result


def _compute_units(value: object) -> ComputeUnit:
    if not isinstance(value, str):
        raise ValueError("compute_units must be a supported text value")
    try:
        return ComputeUnit.parse(value)
    except ValueError:
        raise ValueError(
            "compute_units must be all, cpu_only, cpu_and_gpu, or cpu_and_ne"
        ) from None


def _error(message: str, *, status_code: int) -> JSONResponse:
    return JSONResponse(content={"detail": message}, status_code=status_code)


__all__ = [
    "LiveResourceError",
    "LiveResourceNotFound",
    "LiveResources",
    "router",
]
