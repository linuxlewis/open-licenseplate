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
