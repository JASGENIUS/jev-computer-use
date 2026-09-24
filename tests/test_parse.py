"""Pulling the pieces out of a spoken sentence.

Every one of these is done in CODE, never by Jev. Jev answers typed questions -
it cannot hand back a free-form string - so the words to search for, the words
to type, and which media key to press are all extracted here by pattern. Jev
only judges what KIND of request it was.
"""
from __future__ import annotations

import pytest

from jevflow import parse


# -- what to search for ------------------------------------------------------
@pytest.mark.parametrize("said,want", [
    ("search for apples", "apples"),
    ("look up the weather in toronto", "the weather in toronto"),
    ("hey jev search up faster whisper", "faster whisper"),
    ("google the price of nvidia", "the price of nvidia"),
    # The browser's own name is not the instruction.
    ("open a google chrome tab and search for apples", "apples"),
    # Rightmost wins.
    ("open chrome and search for apples", "apples"),
    ("", ""),
    ("open notepad", ""),
])
def test_search_query(said, want):
    assert parse.search_query(said) == want


def test_google_chrome_is_not_a_search_for_chrome():
    assert parse.search_query("open google chrome") == ""


# -- which app ---------------------------------------------------------------
@pytest.mark.parametrize("said,want", [
    ("open notepad", "notepad"),
    ("hey jev, please open visual studio code", "visual studio code"),
    ("switch to obsidian", "obsidian"),
    ("pull up spotify for me", "spotify"),
])
def test_app_query(said, want):
    assert parse.app_query(said) == want


# -- what to type ------------------------------------------------------------
@pytest.mark.parametrize("said,text,target", [
    ("type hello world into notepad", "hello world", "notepad"),
    # "type INTO x" names a target; bare "type in x y" is framing, because
    # "type in hello world" must not read "hello" as an application.
    ("type into obsidian the meeting starts at four",
     "the meeting starts at four", "obsidian"),
    ("write good morning everyone in slack", "good morning everyone", "slack"),
    ("type hello world", "hello world", ""),
])
def test_type_request(said, text, target):
    got = parse.type_request(said)
    assert got is not None, f"{said!r} was not read as a typing request"
    assert got.text == text
    assert got.target == target


@pytest.mark.parametrize("said", [
    "open notepad",
    "search for apples",
    "close this window",
    "what type of file is that",      # 'type' as a noun, not the verb
])
def test_not_a_type_request(said):
    assert parse.type_request(said) is None


def test_typing_never_returns_an_empty_string():
    """An empty payload would send a confident no-op and report success."""
    assert parse.type_request("type") is None
    assert parse.type_request("type into notepad") is None


# -- media -------------------------------------------------------------------
@pytest.mark.parametrize("said,want", [
    ("play", "play_pause"),
    ("pause the music", "play_pause"),
    ("play pause", "play_pause"),
    ("next track", "next"),
    ("skip this song", "next"),
    ("go back a track", "previous"),
    ("previous song", "previous"),
    ("turn the volume up", "volume_up"),
    ("volume down", "volume_down"),
    ("mute", "mute"),
    ("unmute the sound", "mute"),
])
def test_media_action(said, want):
    assert parse.media_action(said) == want


@pytest.mark.parametrize("said", [
    "open notepad",
    "search for apples",
    "close chrome",
    "play the video on youtube",     # a site request, not a media key
])
def test_not_a_media_action(said):
    assert parse.media_action(said) is None


# -- yes and no --------------------------------------------------------------
@pytest.mark.parametrize("said", [
    "yes", "yeah", "yep", "do it", "go ahead", "confirm", "yes please",
    "sure", "okay do it", "that's right", "affirmative",
])
def test_affirmations(said):
    assert parse.is_affirmation(said) is True


@pytest.mark.parametrize("said", [
    "no", "nope", "cancel", "never mind", "nevermind", "stop", "don't",
    "no don't", "forget it",
])
def test_denials(said):
    assert parse.is_denial(said) is True


@pytest.mark.parametrize("said", [
    "yes I was telling Bob we should ship it on Friday",
    "no idea what the weather is doing, open the forecast",
    "open notepad",
    "sure thing but first search for apples",
    "",
])
def test_a_sentence_is_neither_yes_nor_no(said):
    """A confirmation must be the WHOLE utterance.

    Continuous mode hears the room. If any sentence containing 'yes' counted as
    a confirmation, a conversation happening nearby would authorise an action
    nobody asked for - which is exactly the failure the confirmation exists to
    prevent.
    """
    assert parse.is_affirmation(said) is False
    assert parse.is_denial(said) is False
