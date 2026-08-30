# M4-C acceptance

M4-C adds the operator review workflow for closed events. The review API reads
newest-first event summaries and one bounded event detail payload. Crop bytes
are returned only by an event ID plus an artifact ID. The server uses the
database artifact row as the allow-list and serves only verified JPEG content.
File paths are not part of the public API or the review pages.

The event detail page shows camera and model provenance, capture-session and
track identity, event timing, confidence, ranked crop metadata, and an explicit
no-OCR state. Missing, corrupt, deleted, or unsafe artifacts show as unavailable
without exposing a path or raw exception.

Run the portable M4-C checks with:

```bash
uv run pytest tests/integration/test_m4_c_events.py -q
uv run pytest tests/acceptance/test_m4_c_events_workflow.py -q
uv run pytest -m m4_c_acceptance -q
```

The acceptance tests use deterministic fake camera, detector, tracker, and
clock inputs. They check one event for one known pass, no event for a no-plate
pass, service restart access, bounded list ordering, event/artifact ownership,
safe JPEG responses, metadata and checksum checks, traversal rejection,
symlink rejection, and browser navigation through ranked crops and no OCR.

Browser tests need a local Chromium installation:

```bash
uv run playwright install chromium
uv run pytest -m browser -q
```
