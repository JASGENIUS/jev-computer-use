"""What this machine can actually open.

The index is the wide pass, and its failures are silent: if code never finds an
app, Jev cannot choose it however clearly it was named, and the whole chain
reports "nothing installed matches" for something sitting on the taskbar.
"""
from __future__ import annotations

import pytest

from jevflow import apps


def test_the_index_is_not_empty():
    assert len(apps.index()) > 20


def test_store_apps_are_indexed_too():
    """Windows 11 Notepad, Calculator and Terminal ship as Store apps with no
    Start Menu .lnk at all. A shortcut-only index cannot see them, so "open
    notepad" - about as basic a request as exists - found nothing."""
    names = {a.name.lower() for a in apps.index()}
    assert any("notepad" == n or n.startswith("notepad") and "++" not in n
               for n in names), "Notepad is still invisible"


def test_notepad_shortlists_ahead_of_notepad_plus_plus():
    """Both exist. Asking for one and getting the other is worse than nothing,
    because it looks like it worked."""
    hits = [a.name for a in apps.shortlist("notepad")]
    assert hits, "nothing matched notepad"
    assert hits[0].lower().startswith("notepad")
    assert "++" not in hits[0], f"asked for Notepad, got {hits[0]}"


def test_a_store_app_knows_it_cannot_be_verified_by_exe():
    """No target exe means window-counting cannot confirm it launched. That has
    to be reported as unknown, never quietly as success."""
    store = [a for a in apps.index() if a.source == "apps-folder"]
    if not store:
        pytest.skip("no Store apps on this machine")
    assert all(a.exe == "" for a in store)


def test_every_indexed_app_has_something_to_launch():
    for a in apps.index():
        assert a.launch, a.name


def test_the_index_is_deduplicated():
    names = [a.name.lower() for a in apps.index()]
    assert len(names) == len(set(names))


def test_running_apps_can_be_identified(monkeypatch):
    """switch_to is unselectable unless code tells Jev what is already open -
    the model cannot see the taskbar."""
    from jevflow import actions
    fake = apps.App(name="Obsidian", launch="x.lnk", target=r"C:\obsidian.exe")
    monkeypatch.setattr(actions, "windows_for_exe",
                        lambda exe: [(1, "Vault - Obsidian")] if exe == "obsidian.exe" else [])
    other = apps.App(name="Steam", launch="y.lnk", target=r"C:\steam.exe")
    assert apps.running([fake, other]) == ["Obsidian"]


def test_running_is_empty_when_nothing_is_open(monkeypatch):
    from jevflow import actions
    monkeypatch.setattr(actions, "windows_for_exe", lambda exe: [])
    fake = apps.App(name="Obsidian", launch="x.lnk", target=r"C:\obsidian.exe")
    assert apps.running([fake]) == []


def test_running_survives_a_broken_lookup(monkeypatch):
    """An unknown answer must not become a confident 'nothing is running'."""
    from jevflow import actions

    def boom(exe):
        raise OSError("enumeration failed")
    monkeypatch.setattr(actions, "windows_for_exe", boom)
    fake = apps.App(name="Obsidian", launch="x.lnk", target=r"C:\obsidian.exe")
    assert apps.running([fake]) == []
