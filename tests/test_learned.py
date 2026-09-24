"""Teaching it, once, when it gets an app wrong.

Matching will always have a gap somewhere: a nickname nobody could guess, a
name the recogniser mangles the same way every time. The cure is not a cleverer
scorer, it is being able to say "no, I meant X" and have that stick.

Two rules matter more than the mechanism:

* A correction must be an EXPLICIT correction. Learning from ordinary speech
  would poison the store with things nobody meant to teach.
* A corrupt or missing store must behave as an empty one. A learning feature
  that can brick app-opening is worse than no learning feature.
"""
from __future__ import annotations

import json

import pytest

from jevflow import learned, parse


@pytest.fixture
def store(tmp_path, monkeypatch):
    path = tmp_path / "learned.json"
    monkeypatch.setattr(learned, "STORE", path)
    learned.forget_all()
    return path


# -- remembering -------------------------------------------------------------
def test_a_correction_is_remembered(store):
    learned.remember("local send", "LocalSend")
    assert learned.lookup("local send") == "LocalSend"


def test_lookup_ignores_spacing_and_case(store):
    learned.remember("Local Send", "LocalSend")
    assert learned.lookup("local-send") == "LocalSend"
    assert learned.lookup("LOCALSEND") == "LocalSend"


def test_an_unknown_phrase_returns_nothing(store):
    assert learned.lookup("something nobody taught") is None


def test_it_survives_a_restart(store):
    learned.remember("my editor", "Visual Studio Code")
    learned.reload()                # as if the process had restarted
    assert learned.lookup("my editor") == "Visual Studio Code"


def test_a_later_correction_wins(store):
    learned.remember("notes", "OneNote")
    learned.remember("notes", "Obsidian")
    assert learned.lookup("notes") == "Obsidian"


def test_nothing_is_learned_from_an_empty_phrase(store):
    learned.remember("", "Obsidian")
    learned.remember("   ", "Obsidian")
    assert learned.all_pairs() == {}


# -- it must never break app opening -----------------------------------------
def test_a_corrupt_store_behaves_as_an_empty_one(store):
    store.write_text("{ this is not json", encoding="utf-8")
    learned.reload()
    assert learned.lookup("anything") is None
    # and it must still be writable afterwards
    learned.remember("local send", "LocalSend")
    assert learned.lookup("local send") == "LocalSend"


def test_a_store_of_the_wrong_shape_is_ignored(store):
    store.write_text(json.dumps(["not", "a", "mapping"]), encoding="utf-8")
    learned.reload()
    assert learned.lookup("anything") is None


def test_an_unwritable_store_does_not_raise(monkeypatch, tmp_path):
    monkeypatch.setattr(learned, "STORE", tmp_path / "nope" / "deep" / "x.json")
    learned.reload()
    learned.remember("a phrase", "An App")      # must not raise


# -- recognising a correction ------------------------------------------------
@pytest.mark.parametrize("said,want", [
    ("no I meant local send", "local send"),
    ("no, I meant LocalSend", "LocalSend"),
    ("that's wrong, I meant cap cut", "cap cut"),
    ("wrong app, I meant obsidian", "obsidian"),
    ("I said local send", "local send"),
    ("no I said rust desk", "rust desk"),
    ("not that, I meant file zilla", "file zilla"),
])
def test_a_correction_is_recognised(said, want):
    assert parse.correction_request(said) == want


@pytest.mark.parametrize("said", [
    "open local send",
    "search for what I meant to say",
    "I meant to tell you about the meeting",
    "no",
    "",
    "that's wrong",             # a complaint with no replacement is not a correction
])
def test_ordinary_speech_is_not_a_correction(said):
    assert parse.correction_request(said) is None


def test_a_correction_needs_something_to_correct_to():
    assert parse.correction_request("no I meant") is None
