"""Bounded JPEG encoding and multipart preview output."""

from __future__ import annotations

from collections.abc import Iterator
from fractions import Fraction
from typing import Any

from .contracts import VideoFrame


def encode_jpeg(frame: VideoFrame, *, av_module: Any | None = None) -> bytes:
    """Encode one BGR frame as a JPEG without retaining encoder state."""
    av = av_module or _load_av()
    video_frame = av.VideoFrame.from_ndarray(frame.data, format="bgr24")
    codec = av.CodecContext.create("mjpeg", "w")
    codec.width = frame.width
    codec.height = frame.height
    codec.pix_fmt = "yuvj420p"
    codec.time_base = Fraction(1, 1)
    packets = codec.encode(video_frame)
    packets.extend(codec.encode(None))
    if not packets:
        raise ValueError("JPEG encoder returned no data")
    return b"".join(bytes(packet) for packet in packets)


def multipart_chunk(jpeg: bytes) -> bytes:
    """Build one inspectable MJPEG multipart chunk."""
    header = (
        b"--frame\r\n"
        b"Content-Type: image/jpeg\r\n" + f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii")
    )
    return header + jpeg + b"\r\n"


def preview_chunks(
    frames: Iterator[VideoFrame],
    *,
    encoder: Any = encode_jpeg,
) -> Iterator[bytes]:
    """Convert a bounded frame iterator into MJPEG chunks."""
    for frame in frames:
        try:
            jpeg = encoder(frame)
            yield multipart_chunk(jpeg)
        finally:
            del frame


def _load_av() -> Any:
    try:
        import av
    except ImportError:
        raise RuntimeError("PyAV is not installed") from None
    return av


__all__ = ["encode_jpeg", "multipart_chunk", "preview_chunks"]
