from __future__ import annotations

import os
import sys
from pathlib import Path

import av
import numpy as np
import pytest

from open_licenseplate.cameras.repository import CameraConfig
from open_licenseplate.capture import PyAVRTSPSource, RecordedVideoSource


def _write_recorded_stream(path: Path) -> None:
    container = av.open(str(path), mode="w", format="matroska")
    stream = container.add_stream("mpeg4", rate=5)
    stream.width = 8
    stream.height = 6
    stream.pix_fmt = "yuv420p"
    try:
        for value in (20, 80, 140):
            data = np.full((6, 8, 3), value, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(data, format="bgr24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()


def test_repeatable_recorded_stream_fixture_opens_and_reports_metadata(tmp_path: Path) -> None:
    path = tmp_path / "m1-preview-fixture.mkv"
    _write_recorded_stream(path)
    source = RecordedVideoSource(path, camera_id="fixture-camera")

    info = source.open()
    frames = []
    try:
        while True:
            frame = source.read()
            if frame is None:
                break
            frames.append(frame)
    finally:
        source.close()

    assert info.codec == "mpeg4"
    assert info.width == 8
    assert info.height == 6
    assert info.nominal_fps == 5.0
    assert info.has_camera_pts is True
    assert len(frames) == 3
    assert [frame.camera_pts for frame in frames] == [0, 200, 400]


@pytest.mark.skipif(
    sys.platform not in {"darwin", "linux"} or not os.getenv("OPEN_LICENSEPLATE_RTSP_URL"),
    reason="set OPEN_LICENSEPLATE_RTSP_URL for an optional local RTSP fixture",
)
def test_optional_local_rtsp_fixture_opens() -> None:
    endpoint = os.environ["OPEN_LICENSEPLATE_RTSP_URL"]
    camera = CameraConfig(
        name="local-rtsp-fixture",
        endpoint="rtsp://127.0.0.1:8554/fixture",
        credential_ref="env:OPEN_LICENSEPLATE_RTSP_URL",
        connection_options={"transport": "tcp", "open_timeout": 3.0, "read_timeout": 3.0},
        preferred_stream="main",
        region_of_interest=None,
        enabled=True,
    )
    assert endpoint.lower().startswith(("rtsp://", "rtsps://"))
    source = PyAVRTSPSource(camera, camera_id="local-rtsp-fixture")
    info = source.open()
    try:
        assert info.codec
        assert info.width and info.height
        assert info.nominal_fps
    finally:
        source.close()
