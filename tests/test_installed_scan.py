r"""Apps with no Start Menu shortcut at all.

Modrinth is installed at `%LOCALAPPDATA%\Modrinth App\Modrinth App.exe` and
has no shortcut in either Start Menu, and `Get-StartApps` does not list it. Both
index sources missed it, so "open modrinth" could not work however clearly it
was said - the app was not a candidate to choose from.

The rule that keeps this from becoming noise: only take an executable whose
name MATCHES ITS FOLDER. "Modrinth App\Modrinth App.exe" is the application;
"Modrinth App\resources\crashpad_handler.exe" is not, and a folder scan that
grabs everything would bury the shortlist in updaters and helpers.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from jevflow import apps


def test_an_exe_matching_its_folder_is_the_app(tmp_path):
    d = tmp_path / "Modrinth App"
    d.mkdir()
    (d / "Modrinth App.exe").write_bytes(b"MZ")
    found = apps.scan_install_dirs([tmp_path])
    assert [a.name for a in found] == ["Modrinth App"]


def test_helpers_and_updaters_are_not_apps(tmp_path):
    d = tmp_path / "Modrinth App"
    (d / "resources").mkdir(parents=True)
    (d / "Modrinth App.exe").write_bytes(b"MZ")
    for junk in ("crashpad_handler.exe", "Update.exe", "unins000.exe",
                 "vcredist_x64.exe", "setup.exe", "MyApp Helper.exe"):
        (d / junk).write_bytes(b"MZ")
    names = [a.name for a in apps.scan_install_dirs([tmp_path])]
    assert names == ["Modrinth App"], names


def test_a_folder_with_no_matching_exe_yields_nothing(tmp_path):
    d = tmp_path / "Some Library"
    d.mkdir()
    (d / "libthing.exe").write_bytes(b"MZ")
    assert apps.scan_install_dirs([tmp_path]) == []


def test_a_missing_root_is_not_an_error(tmp_path):
    assert apps.scan_install_dirs([tmp_path / "does not exist"]) == []


def test_the_scan_marks_where_it_came_from(tmp_path):
    d = tmp_path / "Thing"
    d.mkdir()
    (d / "Thing.exe").write_bytes(b"MZ")
    assert apps.scan_install_dirs([tmp_path])[0].source == "installed"


def test_it_has_a_real_target_so_launching_can_be_verified(tmp_path):
    d = tmp_path / "Thing"
    d.mkdir()
    (d / "Thing.exe").write_bytes(b"MZ")
    app = apps.scan_install_dirs([tmp_path])[0]
    assert app.exe == "thing.exe"
    assert Path(app.launch).is_file()


# -- on this machine ----------------------------------------------------------
def test_modrinth_is_reachable_now():
    """A real app, on a real machine."""
    names = {a.name.lower() for a in apps.index()}
    if not any("modrinth" in n for n in names):
        pytest.skip("Modrinth is not installed on this machine")
    hits = [a.name for a in apps.shortlist("modrinth", limit=3)]
    assert hits, "modrinth matched nothing"
    assert "modrinth" in hits[0].lower(), hits


def test_a_shortcut_still_wins_over_a_scanned_exe():
    """A .lnk carries arguments and a working directory that a bare exe does
    not, so it must keep priority when both exist."""
    by_name = {}
    for a in apps.index():
        by_name.setdefault(a.name.lower(), a)
    for name, app in by_name.items():
        if app.source == "installed":
            others = [b for b in apps.index()
                      if b.name.lower() == name and b.source != "installed"]
            assert not others, f"{name} was taken from a scan despite a shortcut"


def test_the_index_did_not_explode_in_size():
    """A folder scan that grabs everything buries the shortlist in helpers."""
    assert len(apps.index()) < 700
