# open-licenseplate

Local-first license plate detection for Apple Silicon macOS.

## Development

Install Python 3.12 and `uv`, then run:

```bash
uv sync --locked
uv run open-licenseplate dev fixture
uv run open-licenseplate db upgrade
uv run open-licenseplate serve
```

The server binds to `127.0.0.1` by default. Open `http://127.0.0.1:8421/` in a
browser. The M0 and M1 slices provide startup, health checks, managed paths,
diagnostics, SQLite persistence, migrations, a Jinja/HTMX shell, camera
configuration, source testing, a live MJPEG preview, and reconnect recovery.
The shell includes Live, Events, Jobs, Cameras, Models, and System pages. The
Cameras page saves safe profiles and opens a source during a connection test.
The Live page starts and stops one camera at a time. Other future product pages
use clear empty states until their milestone is complete. The Models page
provides secure package import and registry operations. The
P08 inference package provides a backend-neutral still-image detection
contract, deterministic letterbox preprocessing, a fake backend for portable
tests, and an optional Core ML backend for macOS. The existing Models page does
not yet provide the later still-image browser workflow.
The System page can save a comfortable or compact display density in the local
settings table. Camera streaming, preview, model, tracking, and processing
features arrive in later milestones.

`db upgrade` creates or upgrades the managed SQLite database. The database uses
WAL mode, full synchronous writes, foreign keys, and a 5-second busy timeout.

Persist one safe application setting with:

```bash
uv run open-licenseplate settings set server.port 9000
```

Configuration precedence is CLI, environment, persisted setting, then built-in
default. Database settings contain only non-secret application values. Camera
credentials and other secrets are not supported by this generic settings store.

### Camera configuration

Save a camera with an endpoint and an external credential reference. Supported
reference formats are:

```text
env:CAMERA_RTSP_URL
keychain:service/account
```

For an environment reference, the variable can contain the complete RTSP URL,
including credentials. The application stores only the reference and a
redacted endpoint description in SQLite. It does not return the resolved value
to the API or browser. The camera test opens the source and reports safe codec,
resolution, nominal FPS, transport, and camera PTS availability. It does not
return the resolved URL or password.

The same controls are available on the Cameras page at `/cameras`, and the
JSON API uses `/api/v1/cameras`.

Camera lifecycle endpoints are:

```text
POST /api/v1/cameras/{camera_id}/start
POST /api/v1/cameras/{camera_id}/stop
GET  /api/v1/cameras/{camera_id}/status
GET  /api/v1/cameras/{camera_id}/preview.mjpeg
GET  /api/v1/cameras/{camera_id}/snapshot.jpg
```

The lifecycle states are `stopped`, `connecting`, `streaming`, `degraded`,
`reconnecting`, `stopping`, and `failed`. An initial source-open error enters
`failed` with an action message. It does not retry. A disconnect after a
successful session enters `degraded`, then `reconnecting`. Reconnect uses
bounded exponential backoff with jitter. A stable stream resets the retry
delay. Stop enters `stopping`, cancels a reconnect wait, and closes the source,
broker, and capture worker.

### Model import and registry

Import a model with a manifest and a local ZIP archive:

```bash
uv run open-licenseplate models import \
  --manifest model-manifest.yaml \
  --archive model.zip \
  --data-dir /tmp/open-licenseplate-m2/data \
  --log-dir /tmp/open-licenseplate-m2/logs
```

The archive must contain one top-level `.mlpackage` directory. The importer
rejects absolute paths, traversal, symlinks, duplicate paths, executable
content, and oversized archives or files. It extracts into a unique staging
directory, computes a deterministic SHA-256 tree checksum, and atomically
moves the package into the managed models directory before writing the
registry row. A failed import removes staging and any final package created
by that import.

The manifest is stored as an immutable JSON snapshot. The registry records the
manifest ID, backend, adapter, source, license, checksum, structural validation
state, and active state. Imported packages use the
`pending_runtime_validation` state. On macOS, the Models validate operation
loads the package, reads its actual Core ML input and output descriptions,
compares them with the manifest, and runs one deterministic validation
prediction. It records safe inspection and prediction details before it marks
the model `runtime_valid`. On other platforms, Core ML validation remains
pending and the portable fake backend is available for tests.

The browser workflow is available at `/models`. The JSON registry API uses
`/api/v1/models` with import, read, package validation, activation,
deactivation, and safe deletion operations.

Audit managed files for unredacted secret patterns with:

```bash
uv run open-licenseplate doctor --audit-secrets
```

The audit reports file names and safe status only. It does not print file
contents or resolved credential values.

### Frame sources and latest frame delivery

The capture package provides the M1 runtime:

- `FrameSource.open()`, `read()`, and `close()` own one capture session.
- `VideoFrame` keeps host UTC time, host monotonic time, camera PTS, sequence,
  dimensions, pixel format, and capture-session identity separate.
- `PyAVRTSPSource` resolves a credential reference only while opening the
  source, uses TCP by default, selects video only, and applies bounded I/O
  timeouts.
- `LatestFrameBroker` has capacity one. A new frame replaces an unread frame.
- `RecordedVideoSource` and `FakeFrameSource` support deterministic tests.
- `FrameCaptureWorker` runs blocking source operations on a dedicated thread.
- `CameraRuntime` owns one source, one worker, one capacity-one broker, and
  the reconnect state machine.
- `ReconnectFixture` provides a reproducible disconnect and recovery sequence.

The preview encoder creates bounded MJPEG output from the newest decoded frame.
The snapshot endpoint returns the newest frame as one JPEG.

### Deterministic reconnect fixture

The test suite keeps an in-memory fixture. It sends one frame, injects a decode
disconnect, waits through a short reconnect delay, and then repeats new frames.
It is safe to run on Linux, macOS, or another host without camera access:

```bash
uv run pytest tests/unit/test_lifecycle.py tests/integration/test_preview_lifecycle.py
```

The fixture uses `ReconnectFixture` and `FixtureAttempt` from
`open_licenseplate.capture`. Production code still uses `PyAVRTSPSource` for
real RTSP cameras. No fixture password or complete secret URL is written to
logs, SQLite, HTML, API output, or diagnostics.

The recorded-stream integration fixture creates a small deterministic Matroska
file with PyAV, then opens it through `RecordedVideoSource`:

```bash
uv run pytest tests/integration/test_stream_fixture.py -q
```

This is the default local stream check because this repository environment does
not include an RTSP server. On macOS or Linux with a local RTSP server, set the
fixture URL and run the optional test:

```bash
OPEN_LICENSEPLATE_RTSP_URL=rtsp://127.0.0.1:8554/fixture \
  uv run pytest tests/integration/test_stream_fixture.py -q
```

The optional test expects the local server to publish the same URL. It uses
TCP and does not print the URL.

### M1 manual validation

Use a configured RTSP URL through an environment reference:

```bash
M1_ROOT="$(mktemp -d)"
M1_DATA="$M1_ROOT/data"
M1_LOGS="$M1_ROOT/logs"
export CAMERA_RTSP_URL='rtsp://user:password@camera.example/live'
uv run open-licenseplate dev fixture --data-dir "$M1_DATA" --log-dir "$M1_LOGS"
uv run open-licenseplate db upgrade --data-dir "$M1_DATA" --log-dir "$M1_LOGS"
uv run open-licenseplate serve --data-dir "$M1_DATA" --log-dir "$M1_LOGS"
```

Open `/cameras`, save an endpoint with `env:CAMERA_RTSP_URL`, and run these
checks:

1. The camera test opens the source and shows safe stream metadata.
2. Live starts one camera and shows the current MJPEG frame.
3. A second camera start returns a conflict with a stop action.
4. A source interruption shows `degraded` and `reconnecting`.
5. Source recovery returns to `streaming`.
6. Stop closes the preview and cancels a pending reconnect.
7. `doctor --audit-secrets` reports safe output.

### Still-image inference contracts

Install the project on an Apple Silicon Mac with:

```bash
uv sync --locked
```

The supported compute choices are `all`, `cpu_only`, `cpu_and_gpu`, and
`cpu_and_ne`. Core ML maps these choices to `ct.ComputeUnit.ALL`,
`CPU_ONLY`, `CPU_AND_GPU`, and `CPU_AND_NE`. A compute-unit change always
closes the old model instance and loads a new one.

The manifest must declare `outputs.box_format` as `xyxy` or `xywh` and
`outputs.coordinate_space` as `model_pixels` or `normalized`. Normalized
coordinates are scaled by the declared model width and height before inverse
letterbox mapping. A raw output must also declare `raw_layout` as
`candidates_first` (`[N,A]`), `channels_first` (`[1,A,N]`), or `channels_last`
(`[1,N,A]`), and must declare `raw_has_objectness`. The adapter does not infer
box geometry, coordinate space, objectness, or matrix orientation.

The adapter returns source-image pixel boxes only. It applies manifest-defined
confidence and IoU thresholds, rejects non-finite or invalid candidates,
clips valid boxes to the source image, and applies deterministic class-wise
NMS. The manifest output names must match names found in the inspected Core ML
model. The backend does not infer or guess output names.

To run the optional real Core ML fixture test, set both variables to a local
managed fixture and run:

```bash
OPEN_LICENSEPLATE_COREML_PACKAGE=/path/to/model.mlpackage \
OPEN_LICENSEPLATE_COREML_MANIFEST=/path/to/model-manifest.yaml \
uv run pytest -m macos
```

### Empty development fixture

Create a clean directory layout for local M0 work with:

```bash
uv run open-licenseplate dev fixture \
  --data-dir /tmp/open-licenseplate-m0/data \
  --log-dir /tmp/open-licenseplate-m0/logs
```

This command creates only the managed directories. It does not run migrations,
create the SQLite database, or create camera, model, plate, event, job, or OCR
data. It does not change an existing database. Run `db upgrade` as a separate
step.

### M0 acceptance

The M0 acceptance starts a fresh temporary fixture. It runs the fixture,
migration, and server commands. It checks all shell pages, default loopback
binding, the persisted display density after a server restart, and `doctor`
readiness:

```bash
uv run pytest -m m0_acceptance
```

Install a Playwright browser before the first local run when needed:

```bash
uv run playwright install chromium
```

For manual M0 validation, use a new temporary root:

```bash
M0_ROOT="$(mktemp -d)"
M0_DATA="$M0_ROOT/data"
M0_LOGS="$M0_ROOT/logs"

uv run open-licenseplate dev fixture --data-dir "$M0_DATA" --log-dir "$M0_LOGS"
uv run open-licenseplate db upgrade --data-dir "$M0_DATA" --log-dir "$M0_LOGS"
uv run open-licenseplate serve \
  --data-dir "$M0_DATA" \
  --log-dir "$M0_LOGS"
```

With the server running, open `http://127.0.0.1:8421/` and confirm:

1. The root route opens the Live page.
2. Live, Events, Jobs, Cameras, Models, and System open from the navigation.
3. The server uses loopback by default and is not reachable through the Mac LAN address.
4. The System page saves the Compact display density.
5. After a server restart, the System page still shows Compact.
6. `uv run open-licenseplate doctor --json --data-dir "$M0_DATA" --log-dir "$M0_LOGS"` reports `"ready": true`.

### Clean verification

Use these commands from a clean development environment:

```bash
uv sync --locked
uv lock --check
uv run pytest
uv run pytest -m browser
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv build
```

The browser test command includes the M0 acceptance. To run the original shell
smoke test without the full M0 process workflow, use:

```bash
uv run pytest -m "browser and not m0_acceptance"
```

The browser tests use Playwright Chromium or a local Chromium executable. Set
`OPEN_LICENSEPLATE_CHROMIUM` when Chromium is not installed in a standard
location. CI installs Playwright Chromium before the browser test steps.
