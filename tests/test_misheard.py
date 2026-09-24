"""When the recogniser mangles a name.

From the log, 2026-09-22: "open Minecraft launcher" was heard as "open
Minecraft WATCHER".

    'open Minecraft watcher' -> ['Minecraft Launcher', 'Minecraft for Windows']
    intent=open_app app=none_of_these certain=0.69

Code shortlisted the right app. Jev refused it, correctly by its own lights -
"watcher" is not "Launcher" and it was told not to invent. The result surfaced
as "Step Unclear", which reads as the program not understanding English.

Neither side is wrong, so neither side should be overruled. A near-miss that
code can see and Jev will not vouch for is exactly the case the confirmation
gate exists for: ASK. And an answer of yes teaches it, so it only happens once.
"""
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
def minecraft(monkeypatch):
    made = (
        apps.App(name="Minecraft Launcher", launch="m.lnk", target=r"C:\mc.exe"),
        apps.App(name="Minecraft for Windows", launch="w.lnk", target=r"C:\mcw.exe"),
    )
    monkeypatch.setattr(apps, "index", lambda: made)
    monkeypatch.setattr(apps, "running", lambda c: [])
    return made


def test_a_near_miss_is_offered_rather_than_refused(store, minecraft):
    jev = FakeJev(intent="open_app", app="none_of_these", certain=0.69)
    c = Commander(never_ask=False, jev=jev, dry_run=True)
    out = c.handle("open minecraft watcher")
    assert out.action == "needs_confirm", f"it refused instead of asking: {out.detail}"
    assert "Minecraft Launcher" in out.detail


def test_saying_yes_opens_it_and_learns_it(store, minecraft):
    jev = FakeJev(intent="open_app", app="none_of_these", certain=0.69)
    c = Commander(never_ask=False, jev=jev, dry_run=True)
    c.handle("open minecraft watcher")
    out = c.handle("yes")
    assert out.ok, out.detail
    assert "Minecraft Launcher" in out.detail
    assert learned.lookup("minecraft watcher") == "Minecraft Launcher", "it did not learn"


def test_saying_no_teaches_it_nothing(store, minecraft):
    jev = FakeJev(intent="open_app", app="none_of_these", certain=0.69)
    c = Commander(never_ask=False, jev=jev, dry_run=True)
    c.handle("open minecraft watcher")
    c.handle("no")
    assert learned.all_pairs() == {}, "a refusal was learned as a lesson"


def test_something_genuinely_unrelated_is_still_refused(store, minecraft):
    """The guard. Offering the nearest app to ANY sound turns a refusal into a
    guess, and a guess that opens something is worse than a clean no."""
    jev = FakeJev(intent="open_app", app="none_of_these", certain=0.9)
    c = Commander(jev=jev, dry_run=True)
    out = c.handle("open photoshop")
    assert out.action != "needs_confirm", f"it offered {out.detail}"
    assert not out.ok


def test_a_confident_choice_is_not_second_guessed(store, minecraft):
    jev = FakeJev(intent="open_app", app="Minecraft Launcher", certain=0.95)
    c = Commander(jev=jev, dry_run=True)
    out = c.handle("open minecraft launcher")
    assert out.ok and out.action == "would_open"


def test_the_message_says_what_went_wrong_in_plain_words(store, minecraft):
    """"Step Unclear" reads as the program not understanding English."""
    jev = FakeJev(intent="open_app", app="none_of_these", certain=0.9)
    c = Commander(jev=jev, dry_run=True)
    out = c.handle("open photoshop")
    assert "photoshop" in out.detail.lower(), out.detail
