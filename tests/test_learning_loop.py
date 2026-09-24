"""The loop: it gets an app wrong, you correct it once, it stops being wrong."""
from __future__ import annotations

import pytest

from conftest import FakeJev
from jevflow import apps, learned
from jevflow.command import Commander


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(learned, "STORE", tmp_path / "learned.json")
    learned.reload()
    yield
    learned.reload()


@pytest.fixture
def two_apps(monkeypatch):
    a = apps.App(name="Send to OneNote", launch="a.lnk", target=r"C:\onenote.exe")
    b = apps.App(name="LocalSend", launch="b.lnk", target=r"C:\localsend.exe")
    monkeypatch.setattr(apps, "index", lambda: (a, b))
    return a, b


def test_a_learned_phrase_beats_the_scorer(store, two_apps):
    learned.remember("local send", "LocalSend")
    assert apps.shortlist("local send")[0].name == "LocalSend"


def test_a_learned_phrase_for_a_missing_app_is_ignored(store, two_apps):
    """Teaching it an app that is not installed must not empty the shortlist."""
    learned.remember("local send", "An App That Is Not Installed")
    hits = apps.shortlist("local send")
    assert hits, "a stale lesson wiped out the results entirely"


def test_correcting_it_teaches_it_and_opens_the_right_thing(store, two_apps):
    jev = FakeJev(intent="open_app", app="Send to OneNote", certain=0.9)
    c = Commander(jev=jev, dry_run=True)

    first = c.handle("open local send")
    assert first.ok and "OneNote" in first.detail      # the wrong one, as reported

    second = c.handle("no I meant LocalSend")
    assert second.ok, second.detail
    assert "LocalSend" in second.detail, "the correction did not open the right app"
    assert learned.lookup("local send") == "LocalSend", "it did not learn"


def test_the_lesson_applies_next_time(store, two_apps):
    jev = FakeJev(intent="open_app", app="Send to OneNote", certain=0.9)
    c = Commander(jev=jev, dry_run=True)
    c.handle("open local send")
    c.handle("no I meant LocalSend")

    # A fresh commander, as if the tool had been restarted.
    jev2 = FakeJev(intent="open_app", app="LocalSend", certain=0.9)
    c2 = Commander(jev=jev2, dry_run=True)
    c2.handle("open local send")
    assert "LocalSend" in jev2.last_candidates, "the lesson was not applied"
    assert jev2.last_candidates[0] == "LocalSend", "the lesson did not rank it first"


def test_a_correction_with_nothing_to_correct_does_not_learn(store, two_apps):
    jev = FakeJev(intent="unclear", certain=0.1)
    c = Commander(jev=jev, dry_run=True)
    out = c.handle("no I meant LocalSend")
    assert not out.ok
    assert learned.all_pairs() == {}, "it learned from a correction of nothing"


def test_ordinary_speech_never_teaches_it(store, two_apps):
    jev = FakeJev(intent="open_app", app="LocalSend", certain=0.9)
    c = Commander(jev=jev, dry_run=True)
    c.handle("open local send")
    c.handle("I meant to tell you about the meeting tomorrow")
    assert learned.all_pairs() == {}, "ordinary speech poisoned the store"


def test_you_can_teach_it_a_name_it_never_matched(store, two_apps):
    """The case that matters most. "open the sender" finds nothing at all, and
    "no, I meant LocalSend" is the obvious next thing to say. Recording only
    SUCCESSFUL opens left this - the teachable moment - with nothing to attach
    the lesson to."""
    jev = FakeJev(intent="open_app", app=None, certain=0.9)
    c = Commander(jev=jev, dry_run=True)

    miss = c.handle("open the sender")
    assert not miss.ok

    fixed = c.handle("no I meant LocalSend")
    assert fixed.ok, fixed.detail
    assert learned.lookup("the sender") == "LocalSend"
    assert apps.shortlist("the sender")[0].name == "LocalSend"
