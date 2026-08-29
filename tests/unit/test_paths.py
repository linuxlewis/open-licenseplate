from pathlib import Path

import pytest

from open_licenseplate.config import load_settings
from open_licenseplate.paths import ManagedPaths


def test_managed_paths_have_expected_layout(tmp_path: Path) -> None:
    settings = load_settings(
        cli_overrides={
            "storage.data_dir": tmp_path / "data",
            "storage.log_dir": tmp_path / "logs",
        },
    )
    paths = ManagedPaths.from_settings(settings)

    paths.ensure_directories()

    assert paths.database == tmp_path / "data" / "open-licenseplate.sqlite3"
    assert paths.models == tmp_path / "data" / "models"
    assert paths.artifacts == tmp_path / "data" / "artifacts"
    assert paths.staging == tmp_path / "data" / "staging"
    assert paths.app_log == tmp_path / "logs" / "app.log"
    assert all(directory.is_dir() for directory in paths.all_directories)


def test_managed_path_validation_rejects_paths_outside_root(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    inside = root / "event" / "crop.jpg"
    outside = tmp_path / "other" / "crop.jpg"

    assert ManagedPaths.validate_contained_path(inside, root) == inside.resolve()
    with pytest.raises(ValueError, match="outside managed root"):
        ManagedPaths.validate_contained_path(outside, root)
