# M3 acceptance

M3-C validates the complete live capture, inference, processed display, and
WebSocket delivery path. It uses synthetic frames and a fake detector on every
platform. It does not create tracks, events, crops, artifacts, jobs, or OCR
results.

## Portable replay

Run the M3 replay acceptance tests with:

```bash
uv run pytest -m m3_acceptance -q -s
```

The replay fixture is
`tests/fixtures/replay/m3_live_replay.json`. It contains a synthetic road
scene with no-plate and plate frames. The fixture has no camera URL, model
path, secret, or real plate image.

The one-hour-equivalent test uses 3,600 logical frames. One logical frame is
one logical second, so the test covers 3,600 logical seconds without waiting
one hour. The source stops at 12 barriers of 300 frames. The test records:

- Processed FPS.
- Prediction latency P50 and P95.
- Source, inference, display, and slow-WebSocket replacement counts.
- Current traced memory at each barrier.

The documented memory rule is a 2 MiB tolerance. The first two checkpoints
set the baseline. The test fails when current traced memory stays more than
2 MiB above that baseline for 3 consecutive later checkpoints. This is a
Python allocation check. It is not a hardware memory benchmark.

The slow-inference test uses barriers, not a timed sleep. It holds one
prediction while the source publishes 64 frames. It checks that replacement
counters rise and the processed frame is current.

## Browser regression

Run the browser checks separately:

```bash
uv run pytest -m browser -q
```

The P11 resize and overlay test remains in the browser suite. It checks that
the processed JPEG and canvas overlay keep the same displayed geometry after a
browser resize.

## Optional Apple Silicon smoke test

The live Core ML test skips when any fixture variable is absent. It requires
Apple Silicon macOS and a real local fixture:

```bash
OPEN_LICENSEPLATE_LIVE_COREML_PACKAGE=/path/to/model.mlpackage \
OPEN_LICENSEPLATE_LIVE_COREML_MANIFEST=/path/to/model-manifest.yaml \
OPEN_LICENSEPLATE_LIVE_COREML_FRAME=/path/to/frame.png \
  uv run pytest -m "macos and m3_acceptance" -q -s
```

The smoke test runs one real frame through the live coordinator, validates the
paired JSON/JPEG unit, and stops the camera, display, and Core ML resources.
It does not print the fixture paths or image data. It does not claim a
specific CPU, GPU, or Neural Engine performance.

## Acceptance evidence

Use the replay output as the portable evidence record. It contains only
fixture names, counts, timings, replacement counts, and memory values. For a
Mac run, record the output with the host model, macOS version, model checksum,
and fixture names in the M3 validation record from SPEC.md.
