"""The word "like", which is two completely different words.

From real dictation, logged 2026-09-22:

    "maybe could you make like a bit of a dashboard that can open up like an
     app so I can like tweak the settings of JevFlow?"

Three fillers in one sentence, all of which survived, because "like" was in the
opt-in tier - and it was in the opt-in tier because stripping it blindly turns
"something like this" into "something this".

So it needs a rule rather than a switch. "like" is a COMPARISON after a small,
closed set of words - feels, sounds, looks, something, just, more - and a tic
almost everywhere else.
"""
from __future__ import annotations

import pytest

from jevflow import disfluency as d


# -- a real sentence ----------------------------------------------------------
def test_real_dictation():
    said = ("maybe could you make like a bit of a dashboard that can open up "
            "like an app so I can like tweak the settings of JevFlow")
    out = d.clean(said)
    assert "like" not in out.lower(), out
    # and the sentence has to survive
    for word in ("dashboard", "app", "tweak", "settings", "JevFlow"):
        assert word.lower() in out.lower(), f"{word} was lost: {out}"


@pytest.mark.parametrize("said,gone", [
    ("so I can like tweak the settings", True),
    ("like a bit of a dashboard", True),
    ("open up like an app", True),
    ("it was like really good", True),
])
def test_filler_like_is_removed(said, gone):
    assert ("like" not in d.clean(said).lower()) is gone, d.clean(said)


# -- the comparisons that must survive ---------------------------------------
@pytest.mark.parametrize("said", [
    "search for something like this",
    "I feel like we should ship it",
    "it looks like rain",
    "that sounds like a plan",
    "make it more like the other one",
    "nothing like the original",
    "it tastes like chicken",
    "a tool just like this one",
    "who else is like him",
])
def test_comparison_like_survives(said):
    """Stripping these changes what the sentence means. "something like this"
    becoming "something this" is worse than leaving a tic in."""
    assert "like" in d.clean(said).lower(), d.clean(said)


def test_the_sentence_is_never_left_mangled():
    """Whatever happens, real words must not disappear with the filler."""
    said = "so I can like tweak the settings of JevFlow"
    out = d.clean(said)
    for word in ("tweak", "settings", "JevFlow"):
        assert word.lower() in out.lower(), out


# -- self-correction, from the same dictation --------------------------------
def test_not_x_i_meant_y():
    """"Not JevFlow, I meant JCU" - corrected mid-sentence."""
    assert d.clean("Not JevFlow, I meant JCU") == "JCU"


def test_like_that_is_left_alone():
    """"like that" is comparative far more often than not - "make it like
    that" is an instruction. Leaving a tic is cheaper than deleting meaning."""
    assert "like that" in d.clean("make it work like that").lower()


# -- a correction must not eat the whole dictation ---------------------------
def test_a_correction_only_reaches_back_to_its_own_sentence():
    """Found by running real dictation through it: a paragraph ending in
    "Not JevFlow, I meant JCU." came out as "JCU." - one corrected word threw
    away everything said before it.

    A repair marker discards what it replaces, which is the CLAUSE it sits in.
    It has no business reaching across a full stop into a different sentence.
    """
    said = ("maybe could you make a dashboard so I can tweak the settings. "
            "Not JevFlow, I meant JCU.")
    out = d.clean(said)
    assert "dashboard" in out, f"the earlier sentence was destroyed: {out}"
    assert "JCU" in out, out
    assert "Not JevFlow" not in out, out


def test_a_correction_inside_one_sentence_still_replaces_it():
    assert d.clean("open notepad, I mean open obsidian") == "open obsidian"


def test_several_sentences_keep_their_own_corrections():
    said = "open notepad, I mean obsidian. search for cats, I mean dogs."
    out = d.clean(said)
    assert "obsidian" in out and "dogs" in out, out
    assert "notepad" not in out and "cats" not in out, out


def test_a_long_dictation_is_never_reduced_to_a_fragment():
    """The general guard. Whatever the rules do, an utterance of many
    sentences must not come back as two words."""
    said = ("First I want to talk about the schedule. Then we should review "
            "the budget. After that we can discuss hiring. Not hiring, I "
            "meant onboarding.")
    out = d.clean(said)
    assert len(out.split()) > 12, f"most of it vanished: {out}"
    assert "schedule" in out and "budget" in out, out
