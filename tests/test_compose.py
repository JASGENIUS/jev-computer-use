"""Asking it to WRITE something, rather than to type words you dictated.

"type hello world into notepad" is dictation - the words are right there.
"write a poem about apples" is not: the words do not exist yet, and something
has to compose them.

This tool cannot. Jev answers typed questions - a choice, a score, a certainty
- and by design cannot hand back free-form prose. So the honest answer is to
say so. Left alone, the typing path matched it and would have typed the literal
string "a poem about apples" into Notepad, which is a wrong answer delivered
confidently and with no sign anything went astray.
"""
from __future__ import annotations

import pytest

from conftest import FakeJev
from jevflow import apps, parse
from jevflow.command import Commander


@pytest.mark.parametrize("said,topic", [
    ("write a poem about apples", "a poem about apples"),
    ("write a story about a dog", "a story about a dog"),
    ("write an email to my landlord", "an email to my landlord"),
    ("write a poem about apples in notepad", "a poem about apples"),
    ("compose a haiku about winter", "a haiku about winter"),
    ("write me a paragraph about trading", "a paragraph about trading"),
])
def test_a_compose_request_is_recognised(said, topic):
    got = parse.compose_request(said)
    assert got is not None, f"{said!r} was not seen as composing"
    assert topic.split()[-1] in got


@pytest.mark.parametrize("said", [
    "type hello world into notepad",
    "write hello world in notepad",
    "type the meeting is at four",
    "open notepad",
    "search for poems about apples",
])
def test_dictation_is_not_composing(said):
    """The words are already there - that is typing, and it must keep working."""
    assert parse.compose_request(said) is None


def test_composing_is_refused_honestly_rather_than_typed_literally():
    fake = apps.App(name="Notepad", launch="n.lnk", target=r"C:\notepad.exe")
    jev = FakeJev(intent="type_into", app="Notepad", certain=0.95)
    c = Commander(jev=jev, dry_run=True)
    out = c.handle("write a poem about apples in notepad")
    assert not out.ok, "it accepted a job it cannot do"
    assert "poem" in out.detail.lower() or "write" in out.detail.lower(), out.detail


def test_a_sequence_containing_a_compose_step_refuses_the_whole_thing():
    """A real example. Better to refuse the sequence than to open three
    apps and type nonsense into the third."""
    fake = apps.App(name="Notepad", launch="n.lnk", target=r"C:\notepad.exe")
    jev = FakeJev(intent="open_app", app="Notepad", certain=0.9)
    c = Commander(jev=jev, dry_run=True)
    out = c.handle("open notepad and then write a poem about apples")
    assert not out.ok
    assert "nothing was done" in out.detail.lower(), out.detail


def test_ordinary_typing_still_works():
    fake = apps.App(name="Notepad", launch="n.lnk", target=r"C:\notepad.exe")
    jev = FakeJev(intent="type_into", app="Notepad", certain=0.95)
    c = Commander(jev=jev, dry_run=True)
    out = c.handle("type hello world into notepad")
    assert out.ok and "hello world" in out.detail
