"""Typing, which did not work at all.

From the log, 2026-09-22, after "open Notepad and then type the word Apples":

    'type the word Apples' -> app_q 'type the word Apples' -> ['Word']
    intent=type_text app=none_of_these certain=0.46

Two separate failures in one line.

**`type_text` was refused.** It is Jev's verb for "type where the cursor
already is", which is exactly what "type the word Apples" means once Notepad is
open and focused - and the code answered "not an app request" and did nothing.
The capability existed; nothing was wired to it.

**"the WORD Apples" shortlisted Microsoft Word.** The typing verb was never
stripped out of the app query, so the words being dictated were searched for as
if they were an application name.
"""
from __future__ import annotations

import pytest

from conftest import FakeJev
from jevflow import apps, parse
from jevflow.command import Commander


@pytest.fixture
def some_apps(monkeypatch):
    made = (
        apps.App(name="Notepad", launch="n.lnk", target=r"C:\notepad.exe"),
        apps.App(name="Word", launch="w.lnk", target=r"C:\winword.exe"),
    )
    monkeypatch.setattr(apps, "index", lambda: made)
    monkeypatch.setattr(apps, "running", lambda c: [])
    return made


# -- the verb that did nothing -----------------------------------------------
def test_type_text_actually_types(some_apps):
    """Jev's verb for "type where the cursor is". It was refused."""
    jev = FakeJev(intent="type_text", app=None, certain=0.9)
    c = Commander(jev=jev, dry_run=True)
    out = c.handle("type the word apples", focused_window="Notepad", focused_hwnd=99)
    assert out.ok, f"typing was refused: {out.action} / {out.detail}"
    assert "apples" in out.detail.lower(), out.detail


def test_type_text_goes_to_whatever_is_focused(some_apps):
    jev = FakeJev(intent="type_text", app=None, certain=0.9)
    c = Commander(jev=jev, dry_run=True)
    out = c.handle("type hello world", focused_window="Notepad", focused_hwnd=99)
    assert out.ok and "hello world" in out.detail


def test_type_text_with_nothing_to_type_is_still_refused(some_apps):
    jev = FakeJev(intent="type_text", app=None, certain=0.9)
    c = Commander(jev=jev, dry_run=True)
    out = c.handle("type", focused_window="Notepad", focused_hwnd=99)
    assert not out.ok


# -- the words being dictated are not an app name ----------------------------
@pytest.mark.parametrize("said", [
    "type the word apples",
    "type the word notepad",
    "type out the word chrome",
    "write the word spotify",
])
def test_what_is_being_typed_is_not_searched_for_as_an_app(said):
    """"type the WORD apples" hunting for Microsoft Word is how it ended up
    offering to type into the wrong application entirely."""
    q = parse.app_query(said)
    assert "type" not in q.lower() and "write" not in q.lower(), q


def test_type_the_word_x_types_only_x():
    """"the word" is framing, not content. Typing it out is not what anyone
    means by "type the word apples"."""
    got = parse.type_request("type the word apples")
    assert got is not None
    assert got.text.lower() == "apples", got.text


@pytest.mark.parametrize("said,text", [
    ("type out hello world", "hello world"),
    ("type in hello world", "hello world"),
    ("type the phrase good morning", "good morning"),
    ("type hello world", "hello world"),
])
def test_framing_words_are_not_typed(said, text):
    got = parse.type_request(said)
    assert got is not None and got.text == text, got


def test_a_named_target_still_wins(some_apps):
    """"type hello into notepad" must still choose Notepad deliberately."""
    got = parse.type_request("type hello world into notepad")
    assert got.target.lower() == "notepad"


# -- a sequence from real use ------------------------------------------------
def test_a_real_sequence_plans_all_of_it(some_apps):
    said = ("open Google Chrome and search for Apples and then open Notepad "
            "and then type the word Apples")
    steps = parse.split_steps(said)
    assert len(steps) >= 3, steps
    assert any("type" in s.lower() for s in steps), steps
