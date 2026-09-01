from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).parents[2]


def test_built_wheel_contains_catalog_and_starts(tmp_path: Path) -> None:
    distribution_directory = tmp_path / "dist"
    subprocess.run(
        ["uv", "build", "--out-dir", str(distribution_directory)],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    wheel_paths = tuple(distribution_directory.glob("*.whl"))
    assert len(wheel_paths) == 1

    installed_directory = tmp_path / "installed"
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--target",
            str(installed_directory),
            "--no-deps",
            str(wheel_paths[0]),
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    dependency_paths = tuple(
        path for path in sys.path if "site-packages" in path and Path(path).is_dir()
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((str(installed_directory), *dependency_paths))
    script = """
from pathlib import Path
from tempfile import TemporaryDirectory

from open_licenseplate.app import create_app
from open_licenseplate.config import load_settings
from open_licenseplate.models.catalog import load_model_catalog

with TemporaryDirectory() as temporary_directory:
    root = Path(temporary_directory)
    settings = load_settings(
        cli_overrides={
            "storage.data_dir": root / "data",
            "storage.log_dir": root / "logs",
        }
    )
    app = create_app(settings)
    assert app.title == "open-licenseplate"
print(len(load_model_catalog().entries))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "3"
