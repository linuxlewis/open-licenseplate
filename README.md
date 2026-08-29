# open-licenseplate

Local-first license plate detection for Apple Silicon macOS.

## Development

Install Python 3.12 and `uv`, then run:

```bash
uv sync --locked
uv run open-licenseplate db upgrade
uv run open-licenseplate serve
```

The server binds to `127.0.0.1` by default. Open `http://127.0.0.1:8421/` in a
browser. The M0 slice provides startup, health checks, managed paths,
diagnostics, SQLite persistence, the first migration, and a Jinja/HTMX shell.
The shell includes Live, Events, Jobs, Cameras, Models, and System pages. The
future product pages use clear empty states until their milestone is complete.
The System page can save a comfortable or compact display density in the local
settings table. Camera, model, tracking, and processing features arrive in
later milestones.

`db upgrade` creates or upgrades the managed SQLite database. The database uses
WAL mode, full synchronous writes, foreign keys, and a 5-second busy timeout.

Persist one safe application setting with:

```bash
uv run open-licenseplate settings set server.port 9000
```

Configuration precedence is CLI, environment, persisted setting, then built-in
default. Database settings contain only non-secret application values. Camera
credentials and other secrets are not supported by this generic settings store.

Run focused checks with:

```bash
uv run pytest
uv run pytest -m browser
uv run ruff check .
uv run ruff format --check .
uv run mypy src
```

The browser smoke test uses Playwright Chromium or a local Chromium executable.
Set `OPEN_LICENSEPLATE_CHROMIUM` when Chromium is not installed in a standard
location. CI installs Playwright Chromium before the browser test step.
