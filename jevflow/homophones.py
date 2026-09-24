"""Resolving the words the recogniser was not sure about.

Whisper returns a per-word probability, and it drops exactly where you would
expect it to: homophones. "there", "their" and "they're" are the same sound, so
the model is guessing from whatever context it has - and it guesses wrong often
enough to be the most visible error in dictated text.

This is the same shape as the app matching, and deliberately so:

    **Code finds the uncertain word and builds the candidate set.
    Jev picks one.**

Jev answers a typed choice, so it cannot invent a third spelling - it can only
choose among words that are genuinely confusable with what was heard. That is
what makes this safe to run on every sentence.

Two rules keep it from making things worse:

**A replacement must be better supported than the original**, judged by the
CHOICE's own confidence in the word it picked. An earlier version gated on a
separate "the sentence makes it clear" judgment instead; swept over 30
sentences that cost five correct fixes and prevented nothing, because it
answers a question about the sentence and the decision here is about the word.

**Only the occurrence that was uncertain changes.** "there is a book over
there" has two of them, and being unsure about the second says nothing about
the first.

What actually keeps this safe is the closed option list, and it was measured
rather than hoped for: across 30 sentences - 20 with a planted error, 10 where
the recogniser was already right - no threshold between 0.40 and 0.80 ever
made a transcript worse. The gate is not there to stop damage. It is there to
stop a coin flip being presented as a correction.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Optional

log = logging.getLogger("jevflow.homophones")

# Below this, the recogniser was guessing. Whisper sits at 0.95+ on words it
# is sure of, so 0.80 is well clear of ordinary speech and still catches the
# genuine coin-flips.
DEFAULT_THRESHOLD = 0.80

# However unsure the recogniser was, the answer replacing its word has to be
# better than a coin flip in its own right. Swept over 30 sentences: nothing
# in the range 0.40-0.80 ever made a transcript worse, so this is set where it
# still lands almost every real fix.
MIN_CONFIDENCE = 0.50

# Sets of words that sound alike. Every member is a candidate for every other,
# which is why they are stored as groups rather than as pairs - an asymmetric
# table fixes an error in one direction only.
GROUPS: tuple[tuple[str, ...], ...] = (
    ("there", "their", "they're"),
    ("to", "too", "two"),
    ("your", "you're"),
    ("its", "it's"),
    ("hear", "here"),
    ("know", "no"),
    ("knew", "new"),
    ("right", "write", "rite"),
    ("by", "buy", "bye"),
    ("for", "four", "fore"),
    ("one", "won"),
    ("sea", "see"),
    ("week", "weak"),
    ("wear", "where", "were"),
    ("whose", "who's"),
    ("than", "then"),
    ("affect", "effect"),
    ("accept", "except"),
    ("lose", "loose"),
    ("hole", "whole"),
    ("principal", "principle"),
    ("peace", "piece"),
    ("plain", "plane"),
    ("brake", "break"),
    ("past", "passed"),
    ("allowed", "aloud"),
    ("bare", "bear"),
    ("board", "bored"),
    ("cell", "sell"),
    ("cent", "scent", "sent"),
    ("coarse", "course"),
    ("complement", "compliment"),
    ("desert", "dessert"),
    ("die", "dye"),
    ("fair", "fare"),
    ("flour", "flower"),
    ("grate", "great"),
    ("heal", "heel"),
    ("hour", "our"),
    ("mail", "male"),
    ("meat", "meet"),
    ("pair", "pear", "pare"),
    ("poor", "pour", "pore"),
    ("sight", "site", "cite"),
    ("some", "sum"),
    ("stationary", "stationery"),
    ("steal", "steel"),
    ("tail", "tale"),
    ("waist", "waste"),
    ("weather", "whether"),
    ("which", "witch"),
    ("wood", "would"),
    ("threw", "through"),
    ("aisle", "isle", "I'll"),
    ("role", "roll"),
    ("scene", "seen"),
    ("sole", "soul"),
    ("weight", "wait"),
    ("lead", "led"),
    ("read", "red"),
    ("advice", "advise"),
    ("elicit", "illicit"),
    ("ensure", "insure"),
    ("farther", "further"),
    ("moral", "morale"),
    ("personal", "personnel"),
    ("quiet", "quite"),
)

_BY_WORD: dict[str, tuple[str, ...]] = {}
for _group in GROUPS:
    for _w in _group:
        _BY_WORD[_w.lower()] = _group

ALL_WORDS = tuple(sorted(_BY_WORD))

_STRIP = re.compile(r"^[^\w']+|[^\w']+$")


def _bare(word: str) -> str:
    """The word itself, without surrounding space or punctuation."""
    return _STRIP.sub("", (word or "").strip()).lower()


def candidates(word: str) -> list[str]:
    """Spellings this word could have been, including itself.

    Empty when there is nothing to confuse it with - there is no point asking
    about "xylophone", and asking anyway would spend a round trip on every
    mumbled word.
    """
    group = _BY_WORD.get(_bare(word))
    return list(group) if group else []


def uncertain(words: list[Any], threshold: float = DEFAULT_THRESHOLD
              ) -> list[tuple[int, str, float]]:
    """Which words were both LOW CONFIDENCE and actually confusable.

    Returns (index, bare word, probability). The index is into the word list,
    so the right occurrence can be replaced later.
    """
    out: list[tuple[int, str, float]] = []
    for i, w in enumerate(words or []):
        try:
            prob = float(getattr(w, "probability", 1.0))
            text = _bare(getattr(w, "word", ""))
        except Exception:
            continue
        if not text or prob >= threshold:
            continue
        if not candidates(text):
            # Being unsure about a word with no homophone is not something a
            # spelling choice can fix.
            continue
        out.append((i, text, prob))
    return out


def _match_case(original: str, replacement: str) -> str:
    if original[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def apply_choice(text: str, heard: str, chosen: str, confidence: float,
                 original_probability: float, occurrence: int = 0) -> str:
    """Swap one occurrence, but only if the swap is better supported.

    `confidence` is the CHOICE's own confidence in the word it picked - not the
    separate "the sentence makes it clear" judgment. Gating on the latter was
    measured across 30 sentences and cost five correct fixes while preventing
    nothing: it answers a question about the sentence, and the decision here is
    about the word.

    `occurrence` is which appearance of the word to change, counted from zero.
    "there is a book over there" has two, and being unsure about the second
    says nothing about the first.
    """
    if not chosen or chosen.lower() == (heard or "").lower():
        return text
    floor = max(float(original_probability), MIN_CONFIDENCE)
    if confidence <= floor:
        # A replacement no better supported than what it replaces is a coin
        # flip dressed up as a correction. The floor is there because a word
        # the recogniser barely believed in must still be replaced by
        # something the answer actually believes in, not merely by something
        # it believed in slightly more.
        log.info("[homophones] not replacing %r with %r: %.2f does not clear %.2f",
                 heard, chosen, confidence, floor)
        return text

    pattern = re.compile(rf"\b{re.escape(heard)}\b", re.I)
    seen = -1

    def swap(m: re.Match) -> str:
        nonlocal seen
        seen += 1
        return _match_case(m.group(0), chosen) if seen == occurrence else m.group(0)

    out = pattern.sub(swap, text)
    if out != text:
        log.info("[homophones] %r -> %r (heard %.2f, chosen %.2f)",
                 heard, chosen, original_probability, confidence)
    return out


def blank_out(text: str, heard: str, occurrence: int = 0,
              blank: str = "____") -> str:
    """Replace one occurrence with a blank, so the context can be read alone.

    Handing the sentence over with the recogniser's guess still sitting in it
    anchors the answer to that guess. Measured, not assumed: "I left it over
    there" came back `there` at 1.00, and the same sentence transcribed "I left
    it over their" came back `their` at 0.50 with `there` at 0.49. The only
    thing that differed was the spelling we supplied.
    """
    pattern = re.compile(rf"\b{re.escape(heard)}\b", re.I)
    seen = -1

    def hide(m: re.Match) -> str:
        nonlocal seen
        seen += 1
        return blank if seen == occurrence else m.group(0)

    return pattern.sub(hide, text)


def resolve(text: str, words: list[Any], jev: Any,
            threshold: float = DEFAULT_THRESHOLD,
            max_questions: int = 4) -> tuple[str, list[tuple[str, str]]]:
    """Ask about each uncertain, confusable word and apply the answers.

    Returns (text, [(heard, chosen), ...]).

    Jev being unavailable must NEVER cost the transcript: dictation that fails
    because a spelling check could not reach the network is far worse than a
    "their" that should have been a "there". Every failure path returns the
    text unchanged.

    `max_questions` caps a pathological sentence. Each question is one round
    trip, and a mumbled paragraph could otherwise ask twenty.
    """
    from jevflow.jev import build_homophone_question, build_homophone_state

    if not text or jev is None:
        return text, []
    found = uncertain(words, threshold)
    if not found:
        return text, []

    changes: list[tuple[str, str]] = []
    out = text
    # Lowest confidence first: those are the ones most likely to be wrong, and
    # the cap should spend itself on them rather than on the near-misses.
    for _i, heard, prob in sorted(found, key=lambda t: t[2])[:max_questions]:
        options = candidates(heard)
        if len(options) < 2:
            continue
        # Which occurrence? The uncertain one is the nth appearance of that
        # word in the sentence, counted among the words list. It is needed
        # BEFORE the question, because that occurrence is what gets blanked.
        before = sum(1 for j, w in enumerate(words) if j < _i
                     and _bare(getattr(w, "word", "")) == heard)
        context = blank_out(out, heard, occurrence=before)
        try:
            res = jev.ask(build_homophone_state(heard, context, prob),
                          build_homophone_question(heard, options, context))
        except Exception as exc:
            log.warning("[homophones] could not ask about %r (%s); "
                        "keeping what was heard", heard, exc)
            return out, changes
        answers = res.get("answers") or {}
        word = answers.get("word") or {}
        chosen = word.get("choice")
        confidence = float(word.get("confidence") or 0.0)
        if not chosen:
            continue
        new = apply_choice(out, heard, chosen, confidence, prob, occurrence=before)
        if new != out:
            changes.append((heard, chosen))
            out = new
    return out, changes
