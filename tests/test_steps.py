"""Breaking one sentence into the several things it actually asks for.

"open Chrome, search for apples, then open notepad" is three instructions, and
the whole sentence used to resolve to ONE - so everything after "search for"
got typed into the search box.

The hard part is the word "and". It joins instructions ("open notepad AND open
chrome") and it joins the contents of one instruction ("search for apples AND
oranges"). Split on it blindly and every search with two words in it becomes
two commands; never split on it and half of normal speech is one giant step.

The rule: "and" only separates when what follows it STARTS A COMMAND. "then",
"after that" and "also" always separate, because nothing else uses them.
"""
from __future__ import annotations

import pytest

from jevflow import parse


def steps(said):
    return parse.split_steps(said)


# -- single steps stay single -------------------------------------------------
@pytest.mark.parametrize("said", [
    "open notepad",
    "search for apples",
    "search for apples and oranges",
    "search for cats and dogs and birds",
    "type hello and goodbye into notepad",
    "close this window",
    "look up the weather in toronto and montreal",
    "open visual studio code",
])
def test_one_instruction_is_not_split(said):
    assert steps(said) == [said], f"{said!r} was broken up"


# -- sequences from real use -------------------------------------------------
def test_a_real_example():
    said = ("google search for apples and then open the notepad and then write "
            "a poem about apples and then open minecraft launcher")
    got = steps(said)
    assert len(got) == 4, got
    assert "apples" in got[0]
    assert "notepad" in got[1]
    assert "poem" in got[2]
    assert "minecraft" in got[3]


def test_his_parallel_example():
    got = steps("open notepad and also open google chrome and then open spotify")
    assert len(got) == 3, got
    assert "notepad" in got[0] and "chrome" in got[1] and "spotify" in got[2]


@pytest.mark.parametrize("said,count", [
    ("open notepad and then open chrome", 2),
    ("open notepad then open chrome", 2),
    ("open notepad, then open chrome", 2),
    ("search for apples then open notepad", 2),
    ("open notepad and open chrome", 2),
    ("open notepad and also open chrome", 2),
    ("open chrome, search for apples, and open notepad", 3),
    ("open notepad. open chrome. open spotify.", 3),
    ("open notepad after that open chrome", 2),
    ("switch to obsidian and then type hello into notepad", 2),
    ("pause the music and then open spotify", 2),
])
def test_sequences_split_into_the_right_number(said, count):
    assert len(steps(said)) == count, steps(said)


def test_the_and_that_joins_a_query_is_not_a_separator():
    """The case that breaks a naive splitter: 'apples and oranges' is one
    search, but 'apples and open notepad' is a search then a launch."""
    assert len(steps("search for apples and oranges")) == 1
    assert len(steps("search for apples and open notepad")) == 2


def test_each_step_can_stand_on_its_own():
    """A fragment that still says 'and then' is not a step, it is a mistake."""
    for step in steps("open chrome and then search for apples and then open notepad"):
        assert not step.lower().startswith(("then", "and", "also", "after that"))
        assert step.strip()


# -- guards -------------------------------------------------------------------
def test_empty_in_empty_out():
    assert steps("") == []
    assert steps(None) == []


def test_a_sequence_is_capped():
    """Twenty chained actions from one misheard sentence is a runaway, not a
    request. The cap is a safety rail, not a feature."""
    said = " and then ".join(["open notepad"] * 30)
    assert len(steps(said)) <= parse.MAX_STEPS


def test_trailing_conjunctions_do_not_make_an_empty_step():
    for said in ("open notepad and then", "open notepad and", "open notepad then"):
        got = steps(said)
        assert all(s.strip() for s in got), got
        assert len(got) == 1, got


def test_it_is_idempotent():
    said = "open notepad and then open chrome"
    once = steps(said)
    assert [s for one in once for s in steps(one)] == once


def test_a_single_step_reports_as_not_multi():
    assert parse.is_multi_step("open notepad") is False
    assert parse.is_multi_step("open notepad and then open chrome") is True


# -- conversation is not a sequence -------------------------------------------
@pytest.mark.parametrize("said", [
    "so then I told him it was fine",
    "and then he said he would call back",
    "I went to the shop and then came home",
    "well then that settles it",
    "first this and then that happened",
])
def test_ordinary_speech_with_then_is_not_a_sequence(said):
    """"then" and "and" are everywhere in normal speech. A sequence is only a
    sequence when at least two fragments actually BEGIN like commands - and
    "so then I told him it was fine" is not two instructions, nor one."""
    got = steps(said)
    assert len(got) == 1, f"conversation was split into {got}"
    assert parse.is_multi_step(said) is False


# -- the verb carries over ----------------------------------------------------
# From the log, 2026-09-22: "Open Chrome and Notepad." never split, because
# "Notepad" is a NAME and the rule wanted a verb. It became one step, and Jev
# picked one app out of a shortlist containing both - Chrome opened, Notepad
# did not, and nothing said so.
@pytest.mark.parametrize("said,count", [
    ("open chrome and notepad", 2),
    ("open notepad and chrome and spotify", 3),
    ("open notepad, chrome, and spotify", 3),
    ("open local send and obsidian", 2),
    ("open chrome and notepad and then search for apples", 3),
])
def test_a_repeated_verb_is_not_required(said, count):
    assert len(steps(said)) == count, steps(said)


def test_the_carried_verb_is_put_back():
    """Each step has to stand alone - "notepad" on its own is not an
    instruction, it is a word."""
    got = steps("open chrome and notepad")
    assert got[0].lower().startswith("open")
    assert got[1].lower().startswith("open"), got


@pytest.mark.parametrize("said", [
    "search for apples and oranges",
    "look up cats and dogs",
    "type hello and goodbye into notepad",
    "search for pots and pans",
])
def test_a_query_with_and_in_it_is_still_one_step(said):
    """The guard. Only OPEN carries its verb over - a search does not, because
    "apples and oranges" is one search and always was."""
    assert steps(said) == [said], steps(said)
