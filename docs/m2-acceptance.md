# M2 acceptance

M2 provides one complete still-image workflow. It does not start a camera,
store events, or require Core ML on a non-Mac test host.

## Automated checks

Run the full M2 acceptance set:

```bash
uv run pytest -m m2_acceptance
```

Run the complete applicable verification set:

```bash
uv sync --locked
uv lock --check
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest
uv run pytest -m browser
uv run pytest -m m2_acceptance
```

The browser checks use a deterministic fake backend. The Core ML checks remain
marked `macos` and use a real managed package only when the package and
manifest environment variables are set.

## Human acceptance

Use a fresh local data directory and open `/models`.

1. Import one converted MorseTech YOLO11n `.mlpackage` ZIP and its manifest.
2. Run model validation. Confirm the result shows the checksum, input and
   output inspection, and the declared adapter.
3. Select `tests/fixtures/still/plate.png`. Run detection. Confirm the image
   displayed in the result is the submitted image, one `license_plate` box is
   visible, and the box follows the plate after letterbox resize.
4. Select `tests/fixtures/still/no-plate.png`. Run detection. Confirm the
   result shows zero detections and no stale box.
5. Change the confidence threshold to `0.95` and run the plate fixture.
   Confirm the lower-confidence fake detection is hidden.
6. Change compute units to `CPU only` and run the plate fixture. Confirm the
   status reports a model reload and the selected compute units.
7. Import an incompatible manifest. Confirm validation fails with an actionable
   compatibility message. The response and page must not show local paths,
   uploaded file names, credentials, or raw backend tracebacks.
8. Upload malformed, unsupported, oversized, and decompression-bomb image data.
   Confirm each request is rejected without creating an artifact or retaining
   the original upload.

Record the date, host, Python version, model checksum, selected compute units,
fixture names, observed timings, and any Core ML availability limitation.
