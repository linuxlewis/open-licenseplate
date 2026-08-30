# M4-B acceptance

M4-B captures source-pixel crops while a confirmed track is observed. The
tracker retains no full frame and keeps at most three ranked crop candidates
per track. The scorer is `m4b-crop-score-v1`. It uses fixed weights for
detection confidence, plate width and height, sharpness, exposure, contrast,
clipping, and boundary distance. The component values are stored as JSON
evidence with each artifact row.

Crop artifacts use JPEG with quality `90`, `4:4:4` subsampling, and disabled
optimization and progressive output. Files are written below the managed
`artifacts/events` directory. Staging uses the managed `staging` directory and
an atomic same-filesystem rename. The final artifact directory and the staging
directory are fsynced before the SQLite transaction starts. SQLite stores only
the relative path. Each artifact stores an explicit zero-based rank so the
read seam uses the same order as selection.

## Transaction checks

The closure seam stages and verifies all selected files first. One short
SQLite transaction then creates capture-session provenance when needed,
inserts the event, inserts all artifact rows, sets `best_artifact_id`, and
commits. A database or rename failure rolls back rows and removes files owned by
that attempt. A duplicate `(capture_session_id, track_id)` returns the existing
event without creating new rows or files.

## Interruption checks

Startup reconciliation removes every entry below the managed staging root.
It removes only unreferenced files below `artifacts/events`; committed rows are
used as the allow-list. Symlinks are unlinked as links and are never followed
for cleanup. The application-support root is never a cleanup target.

Focused tests:

```text
uv run pytest tests/integration/test_m4_b_artifacts.py
uv run pytest tests/unit/test_tracking.py tests/integration/test_m4_migration.py
```

The focused tests cover stable scorer fixtures, candidate limits and release,
staging and rename failure cleanup, directory durability ordering, JPEG
metadata and checksum verification, one-transaction persistence, tied-score
ordering, duplicate-close idempotence, database failure cleanup, and startup
reconciliation.
