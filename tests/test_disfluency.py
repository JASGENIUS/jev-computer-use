"""Cleaning up how people actually talk.

Two different jobs live here and they fail in opposite directions:

- **Fillers** are safe to remove and the risk is removing too few.
- **Self-corrections** are dangerous to remove and the risk is removing too
  many. "what does that say, what does that mean" is one thought restated;
  "I want coffee, I want tea" is two thoughts. They look nearly identical to a
  similarity score, so the rule has to be tight enough to tell them apart and
  the default has to be to leave text alone when unsure.

The invariant that matters more than any individual rule: cleaning a non-empty
transcript must never produce an empty one.
"""
from __future__ import annotations

import pytest

from jevflow import disfluency as d


# -- fillers ------------------------------------------------------------------
@pytest.mark.parametrize("said,want", [
    ("um open notepad", "open notepad"),
    ("uh, open notepad", "open notepad"),
    ("open um notepad", "open notepad"),
    ("so um, uh, close this window", "close this window"),
    ("erm what time is it", "what time is it"),
    ("open notepad, you know", "open notepad"),
    ("mm hmm open notepad", "open notepad"),
])
def test_core_fillers_are_removed(said, want):
    assert d.clean(said) == want


@pytest.mark.parametrize("said", [
    "open notepad",
    "search for the weather in toronto",
    "close this window",
])
def test_clean_speech_is_left_alone(said):
    assert d.clean(said) == said


def test_like_is_not_stripped_by_default():
    """'like' is a real word far more often than it is a filler. Stripping it
    turns "search for something like this" into a different search."""
    assert d.clean("search for something like this") == "search for something like this"


def test_like_can_be_stripped_when_asked():
    assert "like" not in d.clean("open, like, notepad", aggressive=True)


def test_a_word_containing_a_filler_survives():
    """'um' inside 'umbrella' is not a filler, and 'a' inside 'ah' is not one."""
    assert d.clean("search for umbrella") == "search for umbrella"
    assert d.clean("open ahsoka") == "open ahsoka"


# -- stutters -----------------------------------------------------------------
@pytest.mark.parametrize("said,want", [
    ("the the meeting is at four", "the meeting is at four"),
    ("open open notepad", "open notepad"),
    ("I I think so", "I think so"),
])
def test_immediate_repetition_is_collapsed(said, want):
    assert d.clean(said) == want


def test_a_real_double_word_survives():
    """Some doubles are the sentence. 'had had' and 'that that' are grammar."""
    assert d.clean("I said that that was fine") == "I said that that was fine"


# -- self-correction ----------------------------------------------------------
def test_the_users_own_example():
    """'what does that say? what does that mean?' - the speaker restarts the question."""
    assert d.clean("what does that say, what does that mean") == "what does that mean"


@pytest.mark.parametrize("said,want", [
    ("open notepad, I mean open obsidian", "open obsidian"),
    ("close this, no wait, close chrome", "close chrome"),
    ("search for apples, sorry, search for oranges", "search for oranges"),
    ("play the next track, actually no, pause it", "pause it"),
    ("open spotify, scratch that, open steam", "open steam"),
])
def test_a_repair_marker_drops_what_came_before(said, want):
    assert d.clean(said) == want


def test_parallel_sentences_are_not_treated_as_a_correction():
    """The failure mode that matters. These share a prefix and are NOT a
    restatement - deleting the first one loses half of what was said."""
    assert d.clean("I want coffee, I want tea") == "I want coffee, I want tea"
    assert d.clean("open notepad, open chrome") == "open notepad, open chrome"


def test_an_abandoned_restart_needs_a_long_shared_prefix():
    said = "what is the, what is the weather today"
    assert d.clean(said) == "what is the weather today"


def test_a_repair_marker_at_the_start_is_just_dropped():
    assert d.clean("I mean, open notepad") == "open notepad"


def test_a_marker_word_opening_a_real_sentence_is_not_a_marker():
    """At position zero there is nothing behind it to correct, so "I mean" is
    only throat-clearing when a comma follows it. Otherwise it is the verb."""
    assert d.clean("I mean open notepad") == "I mean open notepad"


def test_a_repair_marker_with_nothing_after_it_keeps_what_came_before():
    """Otherwise 'open notepad, I mean-' erases the instruction entirely."""
    assert d.clean("open notepad, I mean") == "open notepad"


# -- the invariant ------------------------------------------------------------
@pytest.mark.parametrize("said", [
    "um", "uh", "um uh erm", "like", "you know", "I mean", "the the",
])
def test_cleaning_never_empties_a_non_empty_transcript(said):
    """An utterance that is ALL filler still has to come back as something.
    Returning "" would make the caller think it heard nothing, which is a
    different and much more confusing failure than hearing a filler."""
    out = d.clean(said)
    assert out, f"{said!r} cleaned away to nothing"


def test_empty_in_empty_out():
    assert d.clean("") == ""
    assert d.clean(None) == ""


def test_cleaning_is_idempotent():
    for said in ("um open notepad", "what does that say, what does that mean",
                 "the the meeting", "open notepad, I mean open obsidian"):
        once = d.clean(said)
        assert d.clean(once) == once, said


def test_punctuation_and_case_are_preserved_where_they_survive():
    assert d.clean("Um, open Notepad.") == "open Notepad."


# -- how people actually talk -------------------------------------------------
# Every case below came from running real messy sentences through a version
# that passed all the tidy tests above. Speech recognition frequently returns
# NO punctuation, which defeated the comma-delimited repair markers entirely,
# and restarts routinely straddle a clause boundary.
@pytest.mark.parametrize("said,want", [
    # A restart buried behind leading discourse markers.
    ("um so like what does that say, uh, what does that mean",
     "what does that mean"),
    # A restart split across a filler: "open, um, open notepad".
    ("hey jev open, um, open notepad", "hey jev open notepad"),
    ("I need you to, uh, search for the, the weather in toronto",
     "I need you to search for the weather in toronto"),
    # No commas at all - Whisper very often returns none.
    ("close this no wait close chrome", "close chrome"),
    ("can you open obsidian actually no open notepad instead",
     "open notepad instead"),
    # A repeated phrase, not a repeated word.
    ("the client the client wants it by Friday", "the client wants it by Friday"),
    ("yeah so um I was thinking we ship on Friday",
     "I was thinking we ship on Friday"),
])
def test_real_messy_speech(said, want):
    assert d.clean(said) == want


def test_a_bare_repeat_is_an_abandoned_start():
    """"open" then "open notepad" is one instruction, started twice."""
    assert d.clean("open, open notepad") == "open notepad"


def test_a_prefix_repeat_is_only_dropped_when_it_is_a_real_prefix():
    """The guard for the rule above: two commands that merely start the same
    way are two commands."""
    assert d.clean("open notepad, open chrome") == "open notepad, open chrome"


def test_sorry_in_the_middle_of_a_sentence_is_not_a_repair_marker():
    """'tell him I am sorry' must not be cut in half. Single ambiguous markers
    still require punctuation; only unmistakable multi-word ones float free."""
    assert d.clean("tell him I am sorry") == "tell him I am sorry"
    assert d.clean("I mean it when I say that") == "I mean it when I say that"


# -- reporting ----------------------------------------------------------------
def test_it_can_say_what_it_removed():
    """Dictation types the result into a real document. When it silently edits
    what someone said, they need to be able to find out what it changed."""
    res = d.clean_verbose("um open notepad, I mean open obsidian")
    assert res.text == "open obsidian"
    assert res.changed is True
    assert res.removed, "it edited the transcript but reported removing nothing"


def test_unchanged_text_reports_no_change():
    res = d.clean_verbose("open notepad")
    assert res.text == "open notepad"
    assert res.changed is False


def test_an_opener_made_entirely_of_markers_is_removed():
    """Two markers in a row used to survive where one did not.

    "okay so, like, stop media" splits into ("okay so", "like", "stop media").
    Stripping the head left nothing, a truthiness guard rejected the empty
    result, and the whole opener was kept untouched.
    """
    assert d.clean("okay so, like, stop media") == "stop media"
    assert d.clean("okay, so, yeah, run trading bot") == "run trading bot"
    assert d.clean("well anyway, open dashboard") == "open dashboard"


def test_an_utterance_that_is_only_markers_is_left_alone():
    """The invariant: anything that cleans away to nothing was not cleanable,
    and the original is a better answer than silence."""
    for said in ("okay so", "so", "um", "yeah okay so right"):
        assert d.clean(said) == said


def test_a_trailing_filler_is_dropped_even_from_an_all_marker_utterance():
    """"okay so, um" keeps its opener - the utterance is nothing but markers
    and the original beats silence - but the "um" still goes. Popping the head
    unconditionally would leave the whole thing untouched instead."""
    c = d
    assert c.clean("okay so, um") == "okay so"
    assert c.clean("okay so") == "okay so"
