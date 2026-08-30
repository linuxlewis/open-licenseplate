"""Managed crop artifacts, interruption cleanup, and M4-B file/DB seams."""

from __future__ import annotations

import hashlib
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path, PurePosixPath
from uuid import NAMESPACE_URL, uuid5

import numpy as np
from PIL import Image

from .. import __version__
from ..database import Database
from ..paths import ManagedPaths
from ..tracking import CROP_QUALITY_SCORING_VERSION, ClosedTrackEvent, CropCandidate
from .repository import CaptureSessionCreate, CommittedArtifact, DetectionEvent, EventRepository

MAX_COMMITTED_ARTIFACTS_PER_EVENT = 3
ARTIFACT_MIME_TYPE = "image/jpeg"
ARTIFACT_EXTENSION = ".jpg"
JPEG_QUALITY = 90
JPEG_SUBSAMPLING = 0


class ArtifactCommitError(RuntimeError):
    """Raised when a crop artifact event cannot be committed safely."""


@dataclass(frozen=True, slots=True)
class ReconciliationReport:
    """Safe startup cleanup counts."""

    stale_staging_entries_removed: int
    orphan_final_files_removed: int
    database_available: bool


@dataclass(slots=True)
class _PreparedArtifact:
    record: CommittedArtifact
    payload: bytes
    staging_path: Path
    final_path: Path
    staged: bool = False
    final_renamed: bool = False


DatabaseFactory = Callable[[Path], Database]


class ManagedArtifactService:
    """Own managed crop files and the short event closure transaction.

    M4-B writes fixed JPEG files under ``artifacts/events``. It stages each
    file under ``staging``, renames only within the managed data directory,
    then asks the repository to commit provenance, event, and artifact rows in
    one SQLite transaction.
    """

    def __init__(
        self,
        paths: ManagedPaths,
        *,
        application_version: str = __version__,
        database_factory: DatabaseFactory = Database,
    ) -> None:
        self.paths = paths
        self.application_version = application_version
        self._database_factory = database_factory
        self._commit_lock = threading.Lock()

    def commit_closed_event(self, event: ClosedTrackEvent) -> DetectionEvent:
        """Commit one closed event and at most three ranked crop artifacts."""
        with self._commit_lock:
            return self._commit_closed_event_locked(event)

    def reconcile(self) -> ReconciliationReport:
        """Remove stale staging and uncommitted final files at startup."""
        self.paths.ensure_directories()
        self._validate_root(self.paths.artifacts)
        self._validate_root(self.paths.staging)
        staging_removed = self._remove_staging_entries()
        committed_paths, database_available = self._committed_paths()
        orphan_removed = 0
        if database_available:
            orphan_removed = self._remove_orphan_final_files(committed_paths)
        return ReconciliationReport(
            stale_staging_entries_removed=staging_removed,
            orphan_final_files_removed=orphan_removed,
            database_available=database_available,
        )

    def _commit_closed_event_locked(self, event: ClosedTrackEvent) -> DetectionEvent:
        database: Database | None = None
        prepared: list[_PreparedArtifact] = []
        keep_final_files = False
        try:
            self.paths.ensure_directories()
            self._validate_root(self.paths.artifacts)
            self._validate_root(self.paths.staging)
            database = self._database_factory(self.paths.database)
            repository = EventRepository(database)
            existing = repository.get_by_durable_key(
                event.capture_session_id,
                event.track_id,
            )
            if existing is not None:
                return existing

            selected = sorted(event.crop_candidates, key=CropCandidate.rank_key)[
                :MAX_COMMITTED_ARTIFACTS_PER_EVENT
            ]
            prepared = self._prepare_artifacts(event, selected)
            self._stage_artifacts(prepared)
            self._rename_artifacts(prepared, repository)
            result = repository.commit_closed_event(
                event,
                artifacts=[item.record for item in prepared],
                crop_ranking_version=CROP_QUALITY_SCORING_VERSION,
                capture_session=self._capture_session_values(event),
            )
            keep_final_files = result.id == event.event_id
            return result
        except ArtifactCommitError:
            recovered = self._recover_failed_commit(event, prepared, database)
            if recovered is not None:
                keep_final_files = recovered.id == event.event_id
                return recovered
            raise
        except Exception:
            recovered = self._recover_failed_commit(event, prepared, database)
            if recovered is not None:
                keep_final_files = recovered.id == event.event_id
                return recovered
            raise ArtifactCommitError("closed event artifacts could not be committed") from None
        finally:
            if not keep_final_files:
                self._remove_uncommitted_paths(prepared)
            else:
                for item in prepared:
                    self._remove_empty_parents(item.staging_path.parent, self.paths.staging)
            if database is not None:
                database.dispose()
            for candidate in event.crop_candidates:
                candidate.release()

    def _capture_session_values(self, event: ClosedTrackEvent) -> CaptureSessionCreate:
        return CaptureSessionCreate(
            id=event.capture_session_id,
            camera_id=event.camera_id,
            model_id=event.model_id,
            model_checksum=event.model_checksum,
            started_at=event.first_seen_at,
            compute_configuration={
                "capture": "live",
                "crop_artifact_format": ARTIFACT_MIME_TYPE,
                "jpeg_quality": JPEG_QUALITY,
            },
            application_version=self.application_version,
        )

    def _prepare_artifacts(
        self,
        event: ClosedTrackEvent,
        candidates: list[CropCandidate],
    ) -> list[_PreparedArtifact]:
        prepared: list[_PreparedArtifact] = []
        for rank, candidate in enumerate(candidates):
            if candidate.pixels is None:
                continue
            payload, width, height = _encode_jpeg(candidate)
            sha256 = hashlib.sha256(payload).hexdigest()
            artifact_id = str(
                uuid5(
                    NAMESPACE_URL,
                    f"open-licenseplate:m4b:{event.event_id}:{rank}:{sha256}",
                )
            )
            relative_path = PurePosixPath("events") / f"{artifact_id}{ARTIFACT_EXTENSION}"
            record = CommittedArtifact(
                id=artifact_id,
                event_id=event.event_id,
                artifact_kind="crop",
                managed_relative_path=relative_path.as_posix(),
                sha256=sha256,
                mime_type=ARTIFACT_MIME_TYPE,
                byte_size=len(payload),
                width=width,
                height=height,
                source_frame_sequence=candidate.source_frame_sequence,
                source_timestamp=candidate.source_timestamp,
                detection_confidence=candidate.detection_confidence,
                quality_score=candidate.quality_score,
                quality_scoring_version=candidate.quality_scoring_version,
                quality_evidence=dict(candidate.quality.evidence),
            )
            final_path = self._artifact_path(record.managed_relative_path)
            staging_path = self._staging_path(event, artifact_id)
            prepared.append(
                _PreparedArtifact(
                    record=record,
                    payload=payload,
                    staging_path=staging_path,
                    final_path=final_path,
                )
            )
        return prepared

    def _stage_artifacts(self, prepared: list[_PreparedArtifact]) -> None:
        for item in prepared:
            self._validate_file_path(item.staging_path, self.paths.staging)
            self._validate_directory(item.staging_path.parent, self.paths.staging)
            item.staging_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            self._validate_directory(item.staging_path.parent, self.paths.staging)
            try:
                with item.staging_path.open("xb") as handle:
                    handle.write(item.payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.chmod(item.staging_path, 0o600)
                item.staged = True
            except OSError:
                raise ArtifactCommitError("crop artifact staging failed") from None

    def _rename_artifacts(
        self,
        prepared: list[_PreparedArtifact],
        repository: EventRepository,
    ) -> None:
        final_root = self.paths.artifacts / "events"
        self._validate_directory(final_root, self.paths.artifacts)
        final_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            staging_device = os.stat(self.paths.staging).st_dev
            final_device = os.stat(final_root).st_dev
        except OSError:
            raise ArtifactCommitError("crop artifact storage is not available") from None
        if staging_device != final_device:
            raise ArtifactCommitError(
                "crop artifact staging and storage are on different filesystems"
            )

        committed_paths = repository.managed_relative_paths()
        for item in prepared:
            if item.record.managed_relative_path in committed_paths:
                raise ArtifactCommitError("crop artifact path is already committed")
            self._validate_file_path(item.final_path, self.paths.artifacts)
            if item.final_path.exists() or item.final_path.is_symlink():
                self._unlink_file(item.final_path, self.paths.artifacts)
            try:
                os.replace(item.staging_path, item.final_path)
                item.final_renamed = True
                os.chmod(item.final_path, 0o600)
            except OSError:
                raise ArtifactCommitError("crop artifact rename failed") from None
            _verify_payload(item.final_path, item.payload, item.record)

    def _recover_failed_commit(
        self,
        event: ClosedTrackEvent,
        prepared: list[_PreparedArtifact],
        database: Database | None,
    ) -> DetectionEvent | None:
        del database
        try:
            probe = self._database_factory(self.paths.database)
            try:
                existing = EventRepository(probe).get_by_durable_key(
                    event.capture_session_id,
                    event.track_id,
                )
            finally:
                probe.dispose()
        except Exception:
            existing = None
        if existing is None:
            return None
        return existing

    def _remove_uncommitted_paths(self, prepared: list[_PreparedArtifact]) -> None:
        for item in prepared:
            paths = []
            if item.staged:
                paths.append((item.staging_path, self.paths.staging))
            if item.final_renamed:
                paths.append((item.final_path, self.paths.artifacts))
            for path, root in paths:
                try:
                    if path.exists() or path.is_symlink():
                        self._unlink_file(path, root)
                except (OSError, ValueError):
                    continue
        for item in prepared:
            self._remove_empty_parents(item.staging_path.parent, self.paths.staging)

    def _remove_staging_entries(self) -> int:
        removed = 0
        for entry in tuple(self.paths.staging.iterdir()):
            if self._remove_tree_entry(entry, self.paths.staging):
                removed += 1
        return removed

    def _remove_orphan_final_files(self, committed_paths: set[str]) -> int:
        events_root = self.paths.artifacts / "events"
        if events_root.is_symlink():
            self._unlink_file(events_root, self.paths.artifacts)
            return 1
        if not events_root.exists():
            return 0
        self._validate_directory(events_root, self.paths.artifacts)
        removed = 0
        for path in self._walk_files(events_root):
            relative = path.relative_to(self.paths.artifacts).as_posix()
            if relative in committed_paths:
                continue
            self._unlink_file(path, self.paths.artifacts)
            removed += 1
        self._remove_empty_parents(events_root, self.paths.artifacts, keep_root=True)
        return removed

    def _committed_paths(self) -> tuple[set[str], bool]:
        if not self.paths.database.is_file():
            return set(), False
        database: Database | None = None
        try:
            database = self._database_factory(self.paths.database)
            paths = EventRepository(database).managed_relative_paths()
            safe_paths = {
                relative for relative in paths if self._is_safe_relative_artifact_path(relative)
            }
            return safe_paths, True
        except Exception:
            return set(), False
        finally:
            if database is not None:
                database.dispose()

    def _artifact_path(self, relative: str) -> Path:
        if not self._is_safe_relative_artifact_path(relative):
            raise ArtifactCommitError("managed artifact path is invalid")
        path = self.paths.artifacts / Path(relative)
        self._validate_file_path(path, self.paths.artifacts)
        return path

    def _staging_path(self, event: ClosedTrackEvent, artifact_id: str) -> Path:
        directory_id = uuid5(
            NAMESPACE_URL,
            f"open-licenseplate:m4b-staging:{event.event_id}",
        )
        path = self.paths.staging / directory_id.hex / f"{artifact_id}{ARTIFACT_EXTENSION}"
        self._validate_file_path(path, self.paths.staging)
        return path

    @staticmethod
    def _is_safe_relative_artifact_path(relative: str) -> bool:
        if not isinstance(relative, str) or not relative:
            return False
        path = PurePosixPath(relative)
        return (
            not path.is_absolute()
            and path.parts[0] == "events"
            and all(part not in {"", ".", ".."} for part in path.parts)
            and len(path.parts) >= 2
            and path.suffix == ARTIFACT_EXTENSION
        )

    @staticmethod
    def _validate_root(path: Path) -> None:
        if path.is_symlink() or not path.is_dir():
            raise ArtifactCommitError("managed artifact root is not safe")

    @staticmethod
    def _validate_directory(path: Path, root: Path) -> None:
        resolved_root = root.expanduser().resolve()
        resolved_path = path.expanduser().resolve()
        try:
            relative = resolved_path.relative_to(resolved_root)
        except ValueError:
            raise ValueError("managed directory is outside its root") from None
        if not relative or path.is_symlink():
            raise ValueError("managed directory is not an exact child")

    @staticmethod
    def _validate_file_path(path: Path, root: Path) -> None:
        resolved_root = root.expanduser().resolve()
        resolved_path = path.expanduser().resolve()
        try:
            relative = resolved_path.relative_to(resolved_root)
        except ValueError:
            raise ValueError("managed file is outside its root") from None
        if not relative:
            raise ValueError("managed file path must be below its root")
        if path.is_symlink():
            raise ValueError("managed file path cannot be a symlink")

    @classmethod
    def _unlink_file(cls, path: Path, root: Path) -> None:
        lexical_root = root.expanduser().absolute()
        lexical_path = path.expanduser().absolute()
        try:
            lexical_path.relative_to(lexical_root)
        except ValueError:
            raise ValueError("cleanup path is outside its managed root") from None
        if path.is_symlink():
            path.unlink()
            return
        cls._validate_file_path(path, root)
        path.unlink()

    def _remove_tree_entry(self, path: Path, root: Path) -> bool:
        lexical_root = root.expanduser().absolute()
        try:
            path.expanduser().absolute().relative_to(lexical_root)
        except ValueError:
            raise ValueError("cleanup path is outside its managed root") from None
        if path.is_symlink() or path.is_file():
            self._unlink_file(path, root)
            return True
        if not path.is_dir():
            return False
        self._validate_directory(path, root)
        for child in tuple(path.iterdir()):
            self._remove_tree_entry(child, root)
        path.rmdir()
        return True

    @classmethod
    def _remove_empty_parents(cls, path: Path, root: Path, *, keep_root: bool = False) -> None:
        root_resolved = root.expanduser().resolve()
        current = path
        while current != root_resolved:
            try:
                current.resolve().relative_to(root_resolved)
            except ValueError:
                return
            try:
                current.rmdir()
            except OSError:
                return
            current = current.parent
        if not keep_root:
            return

    @classmethod
    def _walk_files(cls, root: Path) -> list[Path]:
        files: list[Path] = []
        for entry in tuple(root.iterdir()):
            if entry.is_symlink() or entry.is_file():
                files.append(entry)
            elif entry.is_dir():
                files.extend(cls._walk_files(entry))
        return files


def _encode_jpeg(candidate: CropCandidate) -> tuple[bytes, int, int]:
    if candidate.pixels is None:
        raise ArtifactCommitError("crop pixels are no longer available")
    pixels = np.asarray(candidate.pixels)
    if pixels.dtype != np.uint8:
        pixels = np.clip(pixels, 0, 255).astype(np.uint8)
    if pixels.ndim != 3 or pixels.shape[2] != 3:
        raise ArtifactCommitError("crop pixels have an invalid shape")
    if candidate.color_space == "bgr":
        pixels = pixels[..., ::-1]
    image = Image.fromarray(np.ascontiguousarray(pixels), mode="RGB")
    output = BytesIO()
    try:
        image.save(
            output,
            format="JPEG",
            quality=JPEG_QUALITY,
            subsampling=JPEG_SUBSAMPLING,
            optimize=False,
            progressive=False,
        )
    except (OSError, ValueError):
        raise ArtifactCommitError("crop JPEG encoding failed") from None
    payload = output.getvalue()
    if not payload:
        raise ArtifactCommitError("crop JPEG encoding returned no data")
    return payload, int(image.width), int(image.height)


def _verify_payload(path: Path, payload: bytes, record: CommittedArtifact) -> None:
    try:
        actual = path.read_bytes()
        with Image.open(BytesIO(actual)) as image:
            image.load()
            valid_dimensions = image.format == "JPEG" and image.size == (
                record.width,
                record.height,
            )
    except (OSError, ValueError):
        raise ArtifactCommitError("committed crop metadata could not be verified") from None
    if (
        actual != payload
        or hashlib.sha256(actual).hexdigest() != record.sha256
        or len(actual) != record.byte_size
        or not valid_dimensions
    ):
        raise ArtifactCommitError("committed crop checksum or metadata did not verify")


EventArtifactService = ManagedArtifactService


__all__ = [
    "ARTIFACT_EXTENSION",
    "ARTIFACT_MIME_TYPE",
    "ArtifactCommitError",
    "CROP_QUALITY_SCORING_VERSION",
    "EventArtifactService",
    "JPEG_QUALITY",
    "JPEG_SUBSAMPLING",
    "MAX_COMMITTED_ARTIFACTS_PER_EVENT",
    "ManagedArtifactService",
    "ReconciliationReport",
]
