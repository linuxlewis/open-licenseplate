# open-licenseplate

Local-first license plate detection for Apple Silicon macOS.

## Development

Install Python 3.12 and `uv`, then run:

```bash
uv sync
uv run open-licenseplate serve
```

The server binds to `127.0.0.1` by default. Open `http://127.0.0.1:8421/` in a
browser. The P00 slice provides startup, health checks, managed paths, and
diagnostics. Database support arrives in P01. Camera, model, tracking, and
processing features arrive in later milestones.

The `db upgrade` command is present as a command-shell placeholder. It reports
that database support arrives in P01 and does not modify local data.

Run focused checks with:

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src
```
