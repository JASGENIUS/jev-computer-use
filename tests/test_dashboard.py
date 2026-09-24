"""The dashboard, checked without opening a window.

What matters here is not how it looks: it is that every control it offers maps
to a real setting, and that the command line it builds actually starts the
tool. A settings panel whose Apply button produces an invalid command line is
worse than no panel, because it fails at the moment you are trying to fix
something.
"""
from __future__ import annotations

import pytest

import dashboard
import main as entry
from jevflow import settings


def test_every_control_maps_to_a_real_setting():
    """A control for a setting that does not exist silently does nothing."""
    for tool, fields in dashboard.FIELDS.items():
        known = set(settings.DEFAULTS[tool])
        for key, _label, _kind, _extra in fields:
            assert key in known, f"{tool}.{key} is not a real setting"


def test_both_tools_have_a_panel():
    assert set(dashboard.TOOLS) == set(settings.DEFAULTS)


def test_the_command_line_each_panel_builds_actually_parses():
    """The check that matters. An Apply that produces an invalid command line
    fails exactly when someone is trying to fix something."""
    for tool in dashboard.TOOLS:
        entry.build_parser().parse_args(settings.to_args(tool))


def test_changing_a_control_changes_what_gets_run(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "STORE", tmp_path / "s.json")
    settings.reload()
    settings.set_value("jcu", "dry_run", True)
    assert "--dry-run" in settings.to_args("jcu")
    settings.set_value("jcu", "dry_run", False)
    assert "--dry-run" not in settings.to_args("jcu")


def test_the_choices_offered_are_ones_the_parser_accepts():
    """A dropdown offering a value the tool rejects is a trap."""
    parser = entry.build_parser()
    for tool, fields in dashboard.FIELDS.items():
        for key, _label, kind, extra in fields:
            if kind != "choice" or key != "wake":
                continue
            for option in extra:
                parser.parse_args(["listen", "--command", "--wake", option])


def test_number_ranges_are_sane():
    for tool, fields in dashboard.FIELDS.items():
        for key, _label, kind, extra in fields:
            if kind == "number":
                lo, hi = extra
                assert lo < hi
                default = settings.DEFAULTS[tool][key]
                assert lo <= default <= hi, f"{tool}.{key} default is outside its own range"


def test_it_can_find_processes_without_crashing():
    """Used on every status poll; a throw here would freeze the panel."""
    assert isinstance(dashboard.running_pids("NoSuchToolName"), list)
