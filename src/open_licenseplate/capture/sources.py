"""PyAV, recorded, and fake decoded-frame sources."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from urllib.parse import quote, urlsplit, urlunsplit

import numpy as np

from ..cameras.credentials import parse_credential_ref, resolve_credential
from ..cameras.repository import CameraConfig
from ..cameras.service import CameraConfigurationError, validate_rtsp_url
from ..redaction import redact_text, redact_url, redact_value
from .contracts import (
    CaptureSession,
    Clock,
    FrameSource,
    SourceInfo,
    SourceLifecycleError,
    SourceOpenError,
    SourceReadError,
    SystemClock,
    VideoFrame,
    new_capture_session_id,
)

SessionIdFactory = Callable[[], str]


@dataclass(frozen=True, slots=True)
class RecordedFrame:
    """Optional metadata wrapper for deterministic recorded input."""

    data: Any
    camera_pts: int | None = None
    camera_pts_seconds: float | None = None


class _SourceBase:
    """Shared lifecycle, timestamp, and frame conversion behavior."""

    def __init__(
        self,
        *,
        source_name: str,
        camera_id: str | None,
        clock: Clock | None,
        session_id_factory: SessionIdFactory | None,
        pixel_format: str,
    ) -> None:
        self._source_name = source_name
        self._camera_id = camera_id
        self._clock = clock or SystemClock()
        self._session_id_factory = session_id_factory or new_capture_session_id
        self._pixel_format = pixel_format
        self._condition = threading.Condition()
        self._stop_requested = threading.Event()
        self._state = "new"
        self._session: CaptureSession | None = None
        self._info: SourceInfo | None = None
        self._sequence = 0

    def _begin_open(self) -> CaptureSession:
        with self._condition:
            if self._state == "opening" or self._state == "open":
                raise SourceLifecycleError("source is already open")
            self._state = "opening"
            self._stop_requested.clear()
            self._sequence = 0
            self._info = None

        session = CaptureSession(
            id=self._session_id_factory(),
            camera_id=self._camera_id,
            started_at=_utc(self._clock.now()),
            started_monotonic=self._clock.monotonic(),
        )
        self._session = session
        return session

    def _finish_open(self, info: SourceInfo) -> SourceInfo:
        with self._condition:
            if self._stop_requested.is_set():
                self._state = "closed"
                raise SourceOpenError("source was closed during open")
            self._info = info
            self._state = "open"
        return info

    def _open_failed(self) -> None:
        with self._condition:
            self._state = "closed" if self._stop_requested.is_set() else "failed"

    def _require_open(self) -> CaptureSession:
        with self._condition:
            if self._state != "open" or self._session is None:
                raise SourceLifecycleError("source is not open")
            return self._session

    def _next_frame(
        self,
        data: Any,
        *,
        camera_pts: int | None = None,
        camera_pts_seconds: float | None = None,
    ) -> VideoFrame:
        session = self._require_open()
        array = np.asarray(data)
        if array.ndim != 3 or array.shape[2] != 3:
            raise SourceReadError("decoded video frame must have three color channels")
        self._sequence += 1
        return VideoFrame(
            sequence=self._sequence,
            data=array,
            pixel_format=self._pixel_format,
            host_received_at=_utc(self._clock.now()),
            host_received_monotonic=self._clock.monotonic(),
            capture_session_id=session.id,
            width=int(array.shape[1]),
            height=int(array.shape[0]),
            camera_pts=camera_pts,
            camera_pts_seconds=camera_pts_seconds,
        )

    def _end_session(self, reason: str) -> None:
        with self._condition:
            if self._session is not None and self._session.ended_at is None:
                self._session = replace(
                    self._session,
                    ended_at=_utc(self._clock.now()),
                    end_reason=reason,
                )

    @property
    def state(self) -> str:
        """Return the source lifecycle state."""
        with self._condition:
            return self._state

    @property
    def info(self) -> SourceInfo | None:
        """Return safe source metadata after open."""
        with self._condition:
            return self._info

    @property
    def capture_session_id(self) -> str | None:
        """Return the current runtime session identifier."""
        with self._condition:
            return None if self._session is None else self._session.id

    @property
    def capture_session(self) -> CaptureSession | None:
        """Return the current runtime session, including end timestamps."""
        with self._condition:
            return self._session

    def _close_state(self, *, reason: str = "stopped") -> None:
        self._stop_requested.set()
        self._end_session(reason)
        with self._condition:
            self._state = "closed"


class PyAVRTSPSource(_SourceBase):
    """Decode one configured RTSP video stream through PyAV.

    The source stores only the redacted camera configuration. A credential
    reference is resolved inside open(), and the resolved endpoint is passed
    directly to PyAV without being retained or returned in SourceInfo.
    """

    def __init__(
        self,
        camera: CameraConfig,
        *,
        camera_id: str | None = None,
        clock: Clock | None = None,
        session_id_factory: SessionIdFactory | None = None,
        av_module: Any | None = None,
    ) -> None:
        super().__init__(
            source_name="rtsp",
            camera_id=camera_id,
            clock=clock,
            session_id_factory=session_id_factory,
            pixel_format="bgr24",
        )
        self._camera = replace(
            camera,
            endpoint=redact_url(camera.endpoint),
            connection_options=redact_value(camera.connection_options),
        )
        self._av_module = av_module
        self._container: Any | None = None
        self._video_stream: Any | None = None
        self._frame_iterator: Iterator[Any] | None = None

    def open(self) -> SourceInfo:
        """Open RTSP over the configured transport with bounded I/O timeouts."""
        session = self._begin_open()
        container: Any | None = None
        try:
            endpoint = _resolve_camera_endpoint(self._camera)
            av_module = self._av_module or _load_av()
            options = _av_options(self._camera.connection_options)
            timeout = _timeouts(self._camera.connection_options)
            container = av_module.open(
                endpoint,
                mode="r",
                options=options,
                timeout=timeout,
            )
            stream = _select_video_stream(
                container,
                self._camera.preferred_stream,
            )
            iterator = iter(container.decode(video=stream.index))
            info = SourceInfo(
                source_name="rtsp",
                session=session,
                codec=_codec_name(stream),
                width=_positive_int(getattr(stream, "width", None)),
                height=_positive_int(getattr(stream, "height", None)),
                nominal_fps=_frame_rate(stream),
                has_camera_pts=getattr(stream, "time_base", None) is not None,
                transport=str(options["rtsp_transport"]),
                endpoint=redact_url(self._camera.endpoint),
            )
            with self._condition:
                if self._stop_requested.is_set():
                    raise SourceOpenError("source was closed during open")
                self._container = container
                self._video_stream = stream
                self._frame_iterator = iterator
            return self._finish_open(info)
        except SourceOpenError:
            self.close()
            _close_quietly(container)
            self._open_failed()
            raise
        except (CameraConfigurationError, ValueError) as error:
            self.close()
            _close_quietly(container)
            self._open_failed()
            raise SourceOpenError(str(error)) from None
        except Exception as error:
            self.close()
            _close_quietly(container)
            self._open_failed()
            detail = redact_text(str(error))
            message = "could not open the RTSP video source"
            raise SourceOpenError(f"{message}: {detail}" if detail else message) from None

    def read(self) -> VideoFrame | None:
        """Decode one video frame without touching audio streams."""
        self._require_open()
        iterator = self._frame_iterator
        if iterator is None:
            raise SourceLifecycleError("RTSP video decoder is not ready")
        try:
            decoded = next(iterator)
        except StopIteration:
            self._end_session("end_of_input")
            return None
        except Exception as error:
            if self._stop_requested.is_set():
                return None
            self._mark_failed()
            detail = redact_text(str(error))
            message = "RTSP video decode failed"
            raise SourceReadError(f"{message}: {detail}" if detail else message) from None

        try:
            data = decoded.to_ndarray(format="bgr24")
            pts = _optional_int(getattr(decoded, "pts", None))
            pts_seconds = _frame_time_seconds(decoded)
            return self._next_frame(
                data,
                camera_pts=pts,
                camera_pts_seconds=pts_seconds,
            )
        except SourceReadError:
            raise
        except Exception:
            self._mark_failed()
            raise SourceReadError("decoded RTSP frame is invalid") from None

    def close(self) -> None:
        """Signal decode cancellation and close the PyAV container."""
        self._close_state()
        with self._condition:
            container = self._container
            self._container = None
            self._video_stream = None
            self._frame_iterator = None
        _close_quietly(container)

    def _mark_failed(self) -> None:
        self._close_state(reason="failed")
        with self._condition:
            container = self._container
            self._container = None
            self._video_stream = None
            self._frame_iterator = None
        _close_quietly(container)


class RecordedVideoSource(_SourceBase):
    """Read a local video file or an in-memory deterministic frame sequence."""

    def __init__(
        self,
        source: str | Path | Iterable[Any] | None = None,
        *,
        frames: Iterable[Any] | None = None,
        path: str | Path | None = None,
        camera_id: str | None = "recorded",
        frame_rate: float = 30.0,
        clock: Clock | None = None,
        session_id_factory: SessionIdFactory | None = None,
        pixel_format: str = "bgr24",
        av_module: Any | None = None,
    ) -> None:
        if sum(value is not None for value in (source, frames, path)) != 1:
            raise ValueError("provide exactly one recorded source")
        selected = source if source is not None else frames if frames is not None else path
        assert selected is not None
        super().__init__(
            source_name="recorded-video",
            camera_id=camera_id,
            clock=clock,
            session_id_factory=session_id_factory,
            pixel_format=pixel_format,
        )
        self._frame_rate_value = frame_rate
        self._av_module = av_module
        self._path: Path | None
        self._frames: list[Any] | None
        if isinstance(selected, (str, Path)):
            self._path = Path(selected)
            self._frames = None
        else:
            self._path = None
            if isinstance(selected, np.ndarray):
                self._frames = [selected]
            else:
                self._frames = list(selected)
        self._frame_index = 0
        self._container: Any | None = None
        self._video_stream: Any | None = None
        self._frame_iterator: Iterator[Any] | None = None

    def open(self) -> SourceInfo:
        """Open a local file or start the in-memory sequence."""
        session = self._begin_open()
        container: Any | None = None
        try:
            if self._path is not None:
                av_module = self._av_module or _load_av()
                container = av_module.open(str(self._path), mode="r")
                stream = _select_video_stream(container, None)
                iterator = iter(container.decode(video=stream.index))
                with self._condition:
                    if self._stop_requested.is_set():
                        raise SourceOpenError("source was closed during open")
                    self._container = container
                    self._video_stream = stream
                    self._frame_iterator = iterator
                info = SourceInfo(
                    source_name="recorded-video",
                    session=session,
                    codec=_codec_name(stream),
                    width=_positive_int(getattr(stream, "width", None)),
                    height=_positive_int(getattr(stream, "height", None)),
                    nominal_fps=_frame_rate(stream),
                    has_camera_pts=getattr(stream, "time_base", None) is not None,
                )
            else:
                self._frame_index = 0
                first = self._frames[0] if self._frames else None
                array = np.asarray(_recorded_data(first)) if first is not None else None
                info = SourceInfo(
                    source_name="recorded-video",
                    session=session,
                    width=int(array.shape[1]) if array is not None else None,
                    height=int(array.shape[0]) if array is not None else None,
                    nominal_fps=self._frame_rate_value,
                    has_camera_pts=_recorded_pts(first) is not None,
                )
            return self._finish_open(info)
        except SourceOpenError:
            self.close()
            _close_quietly(container)
            self._open_failed()
            raise
        except Exception:
            self.close()
            _close_quietly(container)
            self._open_failed()
            raise SourceOpenError("could not open the recorded video source") from None

    def read(self) -> VideoFrame | None:
        """Return the next deterministic frame, or None at end of input."""
        self._require_open()
        if self._path is not None:
            iterator = self._frame_iterator
            if iterator is None:
                raise SourceLifecycleError("recorded video decoder is not ready")
            try:
                decoded = next(iterator)
            except StopIteration:
                self._end_session("end_of_input")
                return None
            except Exception as error:
                self._mark_failed()
                detail = redact_text(str(error))
                message = "recorded video decode failed"
                raise SourceReadError(f"{message}: {detail}" if detail else message) from None
            try:
                return self._next_frame(
                    decoded.to_ndarray(format=self._pixel_format),
                    camera_pts=_optional_int(getattr(decoded, "pts", None)),
                    camera_pts_seconds=_frame_time_seconds(decoded),
                )
            except SourceReadError:
                raise
            except Exception:
                self._mark_failed()
                raise SourceReadError("recorded video frame is invalid") from None

        if self._frames is None or self._frame_index >= len(self._frames):
            return None
        value = self._frames[self._frame_index]
        self._frame_index += 1
        return self._next_frame(
            _recorded_data(value),
            camera_pts=_recorded_pts(value),
            camera_pts_seconds=_recorded_pts_seconds(value),
        )

    def close(self) -> None:
        """Release local decode resources and in-memory frame references."""
        self._close_state()
        with self._condition:
            container = self._container
            self._container = None
            self._video_stream = None
            self._frame_iterator = None
        _close_quietly(container)

    def _mark_failed(self) -> None:
        self._close_state(reason="failed")
        with self._condition:
            container = self._container
            self._container = None
            self._video_stream = None
            self._frame_iterator = None
        _close_quietly(container)


class FakeFrameSource(RecordedVideoSource):
    """Deterministic source with injectable open, read, and blocking failures."""

    def __init__(
        self,
        frames: Iterable[Any] | None = None,
        *,
        open_error: str | BaseException | None = None,
        read_error: str | BaseException | None = None,
        fail_at: int | None = None,
        read_gate: threading.Event | None = None,
        read_interval_seconds: float = 0.0,
        repeat: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(frames if frames is not None else [], **kwargs)
        self._open_error = open_error
        self._read_error = read_error
        self._fail_at = fail_at
        self._read_gate = read_gate
        self._read_interval_seconds = max(0.0, read_interval_seconds)
        self._repeat = repeat
        self.opened = threading.Event()
        self.closed = threading.Event()

    def open(self) -> SourceInfo:
        if self._open_error is not None:
            self._open_failed()
            raise SourceOpenError(_safe_failure_message(self._open_error))
        info = super().open()
        self.opened.set()
        return info

    def read(self) -> VideoFrame | None:
        self._require_open()
        if self._read_gate is not None:
            while not self._read_gate.wait(0.02):
                if self._stop_requested.is_set():
                    return None
        if self._fail_at is not None and self._frame_index == self._fail_at:
            self._mark_failed()
            raise SourceReadError(_safe_failure_message(self._read_error or "fake source failure"))
        if self._read_error is not None and self._fail_at is None:
            self._mark_failed()
            raise SourceReadError(_safe_failure_message(self._read_error))
        if self._read_interval_seconds and self._stop_requested.wait(self._read_interval_seconds):
            return None
        if self._repeat and self._frames and self._frame_index >= len(self._frames):
            self._frame_index = 0
        return super().read()

    def close(self) -> None:
        super().close()
        self.closed.set()


def _resolve_camera_endpoint(camera: CameraConfig) -> str:
    """Resolve a credential reference only at the RTSP source boundary."""
    endpoint = camera.endpoint
    reference = parse_credential_ref(camera.credential_ref)
    if reference is not None:
        try:
            secret_value = resolve_credential(reference)
        except Exception:
            raise SourceOpenError("camera credential could not be resolved") from None
        if not secret_value:
            raise SourceOpenError("camera credential is not available")
        endpoint = _endpoint_from_secret(endpoint, secret_value)

    if "[REDACTED]" in endpoint:
        raise SourceOpenError("camera endpoint requires a complete credential reference")
    try:
        validate_rtsp_url(endpoint)
    except CameraConfigurationError as error:
        raise SourceOpenError(str(error)) from None
    return endpoint


def _endpoint_from_secret(redacted_endpoint: str, secret_value: str) -> str:
    candidate = secret_value.strip()
    if candidate.lower().startswith(("rtsp://", "rtsps://")):
        return candidate
    try:
        decoded = json.loads(candidate)
    except json.JSONDecodeError:
        decoded = None
    if isinstance(decoded, Mapping):
        url = decoded.get("url")
        if isinstance(url, str):
            return url
        username = decoded.get("username", decoded.get("user"))
        password = decoded.get("password", decoded.get("pass"))
        if isinstance(username, str) and isinstance(password, str):
            return _insert_user_info(redacted_endpoint, username, password)
    raise SourceOpenError(
        "camera credential must contain a complete RTSP URL or username and password"
    )


def _insert_user_info(endpoint: str, username: str, password: str) -> str:
    parsed = urlsplit(endpoint.replace("[REDACTED]@", "redacted@"))
    if "@" not in parsed.netloc:
        raise SourceOpenError("camera endpoint does not define a credential location")
    host = parsed.netloc.rsplit("@", 1)[1]
    user_info = f"{quote(username, safe='')}:{quote(password, safe='')}"
    return urlunsplit(
        (parsed.scheme, f"{user_info}@{host}", parsed.path, parsed.query, parsed.fragment)
    )


def _av_options(options: Mapping[str, Any]) -> dict[str, str]:
    transport = str(options.get("transport", "tcp")).lower()
    if transport not in {"tcp", "udp"}:
        transport = "tcp"
    result = {"rtsp_transport": transport}
    read_timeout = _timeout_value(options, "read_timeout", default=5.0)
    result["rw_timeout"] = str(int(read_timeout * 1_000_000))
    return result


def _timeouts(options: Mapping[str, Any]) -> tuple[float, float]:
    return (
        _timeout_value(options, "open_timeout", default=5.0),
        _timeout_value(options, "read_timeout", default=5.0),
    )


def _timeout_value(options: Mapping[str, Any], key: str, *, default: float) -> float:
    value = options.get(key, default)
    if value is None:
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        result = default
    if not 0.1 <= result <= 300:
        return default
    return result


def _select_video_stream(container: Any, preferred_stream: str | None) -> Any:
    streams = list(getattr(container.streams, "video", ()))
    if not streams:
        raise SourceOpenError("source has no video stream")
    if preferred_stream:
        normalized = preferred_stream.strip().lower()
        for stream in streams:
            if str(getattr(stream, "index", "")).lower() == normalized:
                return stream
            metadata = getattr(stream, "metadata", {}) or {}
            title = metadata.get("title") or metadata.get("name")
            if isinstance(title, str) and title.lower() == normalized:
                return stream
    return streams[0]


def _codec_name(stream: Any) -> str | None:
    context = getattr(stream, "codec_context", None)
    value = getattr(context, "name", None) or getattr(stream, "name", None)
    return str(value) if value else None


def _frame_rate(stream: Any) -> float | None:
    value = getattr(stream, "average_rate", None) or getattr(stream, "base_rate", None)
    try:
        result = float(cast(Any, value))
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None


def _positive_int(value: Any) -> int | None:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _frame_time_seconds(frame: Any) -> float | None:
    value = getattr(frame, "time", None)
    try:
        return None if value is None else float(cast(Any, value))
    except (TypeError, ValueError):
        return None


def _recorded_data(value: Any) -> Any:
    if isinstance(value, RecordedFrame):
        return value.data
    if isinstance(value, VideoFrame):
        return value.data
    return value


def _recorded_pts(value: Any) -> int | None:
    if isinstance(value, RecordedFrame):
        return value.camera_pts
    if isinstance(value, VideoFrame):
        return value.camera_pts
    return None


def _recorded_pts_seconds(value: Any) -> float | None:
    if isinstance(value, RecordedFrame):
        return value.camera_pts_seconds
    if isinstance(value, VideoFrame):
        return value.camera_pts_seconds
    return None


def _load_av() -> Any:
    try:
        import av
    except ImportError:
        raise SourceOpenError("PyAV is not installed") from None
    return av


def _close_quietly(container: Any | None) -> None:
    if container is None:
        return
    from contextlib import suppress

    with suppress(Exception):
        container.close()


def _safe_failure_message(error: str | BaseException) -> str:
    return redact_text(str(error)) if str(error) else "fake source failure"


def _utc(value: Any) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError("clock returned an invalid timestamp")
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


PyAVFrameSource = PyAVRTSPSource
RTSPFrameSource = PyAVRTSPSource
PyAVSource = PyAVRTSPSource
RecordedSource = RecordedVideoSource
FakeSource = FakeFrameSource

__all__ = [
    "FakeFrameSource",
    "FrameSource",
    "PyAVFrameSource",
    "PyAVSource",
    "PyAVRTSPSource",
    "FakeSource",
    "RecordedSource",
    "RTSPFrameSource",
    "RecordedFrame",
    "RecordedVideoSource",
]
