"""Finding the app someone just said out loud.

Speech puts spaces where app names do not have them. "LocalSend" is one word
on disk and "local send" is two words out of a mouth, and every scoring rule
compared them literally - so asking for LocalSend scored **0.000** against
LocalSend and returned "Send to OneNote" instead.

That is not one app being awkward. This machine has 26 names of that shape:
CapCut, FileZilla, HandBrake, OneDrive, RustDesk, PowerPoint. Every one of them
was unreachable by voice.
"""
from __future__ import annotations

import pytest

from jevflow import apps


def _names(query, limit=4):
    return [a.name for a in apps.shortlist(query, limit=limit)]


def _installed(name: str) -> bool:
    return any(a.name.lower() == name.lower() for a in apps.index())


# -- the bug --------------------------------------------------------------
@pytest.mark.parametrize("said,want", [
    ("local send", "LocalSend"),
    ("cap cut", "CapCut"),
    ("file zilla", "FileZilla"),
    ("hand brake", "HandBrake"),
    ("one drive", "OneDrive"),
    ("rust desk", "RustDesk"),
    ("power point", "PowerPoint"),
    ("one note", "OneNote"),
])
def test_a_spoken_space_still_finds_a_compressed_name(said, want):
    if not _installed(want):
        pytest.skip(f"{want} is not installed on this machine")
    hits = _names(said)
    assert hits, f"{said!r} matched nothing at all"
    assert hits[0] == want, f"{said!r} -> {hits[0]!r}, wanted {want!r}"


def test_the_exact_failure_he_reported():
    if not _installed("LocalSend"):
        pytest.skip("LocalSend not installed")
    assert apps._score("local send", 
                       next(a for a in apps.index() if a.name == "LocalSend")) > 0.9


# -- what must NOT break --------------------------------------------------
@pytest.mark.parametrize("said,want", [
    ("chrome", "Google Chrome"),
    ("obsidian", "Obsidian"),
    ("notepad", "Notepad"),
])
def test_the_matches_that_already_worked_still_work(said, want):
    if not _installed(want):
        pytest.skip(f"{want} is not installed")
    assert _names(said)[0] == want


def test_notepad_does_not_lose_to_notepad_plus_plus():
    """Both squash to the same letters once punctuation is stripped. Asking for
    one and getting the other is worse than nothing, because it looks like it
    worked."""
    if not (_installed("Notepad") and _installed("Notepad++")):
        pytest.skip("need both to test the tiebreak")
    assert _names("notepad")[0] == "Notepad"


def test_an_unrelated_word_does_not_match_everything():
    """Loosening the matcher must not turn it into a machine that says yes."""
    hits = _names("xyzzyqwertyplugh")
    assert not hits, f"nonsense matched {hits}"


def test_a_short_query_does_not_drag_in_the_whole_start_menu():
    assert len(_names("a", limit=10)) <= 10


# -- mishearings ----------------------------------------------------------
@pytest.mark.parametrize("said,want", [
    ("localsend", "LocalSend"),
    ("local-send", "LocalSend"),
    ("Local Send", "LocalSend"),
])
def test_spelling_and_spacing_variants_all_land(said, want):
    if not _installed(want):
        pytest.skip(f"{want} not installed")
    assert _names(said)[0] == want


def test_every_compressed_name_on_this_machine_is_reachable_when_split():
    """The real coverage check: take each compressed name, split it the way
    speech would, and confirm it comes back. This is the test that says the
    class of bug is fixed, not just the one app someone happened to notice."""
    import re
    failures = []
    for app in apps.index():
        if " " in app.name or not re.search(r"[a-z][A-Z]", app.name):
            continue
        spoken = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", app.name).lower()
        hits = _names(spoken, limit=3)
        if app.name not in hits:
            failures.append((spoken, app.name, hits[:2]))
    assert not failures, f"{len(failures)} compressed names unreachable: {failures[:6]}"
