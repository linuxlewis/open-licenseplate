# M4-A acceptance

M4-A maps validated live detections to bounded ByteTrack state. It confirms
three matched observations, shows confirmed tracks in live metadata, closes a
track after one second without a match, and emits one immutable closed-event
aggregate. It does not write an event to SQLite. The event and artifact
transaction belongs to the next persistence milestone.

## Replay check

Run the portable replay:

```bash
uv run pytest -m m4_a_acceptance -q -s
```

The fixture is
`tests/fixtures/replay/m4_tracking_events.json`. It contains:

- One known pass with three observations.
- One no-plate pass.
- One false candidate with two observations.

Expected result:

- The known pass emits one `closed` aggregate.
- The no-plate pass emits no aggregate.
- The false candidate expires before persistence and emits no aggregate.

## Fake-clock checks

Run the state-machine and migration checks:

```bash
uv run pytest \
  tests/unit/test_tracking.py \
  tests/integration/test_m4_migration.py \
  -q -s
```

The tests use a fake UTC and monotonic clock. They check:

- Confirmation after three matches in the configured window.
- One-second no-match timeout.
- Candidate expiration.
- Reconnect provenance reset.
- Stale frame rejection.
- Duplicate-close prevention on repeated ticks and reset calls.
- `(capture_session_id, track_id)` and managed-path uniqueness.
- M4 migration upgrade and downgrade.

The tracker stores only aggregate fields for each active track. Retired
provenance entries, duplicate-close keys, recent closed events, and live
metadata have explicit limits. It stores no observation history.
