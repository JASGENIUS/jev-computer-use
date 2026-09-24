"""Deciding whether an ambiguous word is a tic or is doing a job.

Some words are a filler in one sentence and load-bearing in the next, and no
list of neighbours settles it. "like" is the worst of them:

    I would really LIKE to use it          the verb
    something LIKE this                    a comparison
    it's just LIKE, you know, pretty nice  a tic
    he was LIKE, no way                    a quotation

**Jev does not decide what to delete here, and that was measured rather than
assumed.** Over 28 real occurrences of "like" in actual dictation, the existing
two-list rule found 21 of the 23 genuine tics; Jev asked to make the same call
never matched it at any threshold - either it deleted more real words or it
left most of the tics in.

What Jev IS good at is the opposite question. The rule's worst error was
deleting the verb out of "I would really like to use it"; Jev named that one
`verb` at **0.99**. So the rule proposes and Jev holds a veto:

    **The rule finds the tic. Jev can say the word is doing a job.**

Measured on the same 28:

    rule alone            3 harmful deletions, 2 tics missed, 21 cut
    rule + veto (0.80)    2 harmful deletions, 2 tics missed, 21 cut

The veto never blocked a correct deletion at any strength from 0.60 to 0.95,
so it is a plateau rather than a tuned point. It does not fix everything: the
two it cannot save are ones Jev also believes are fillers, and no gate rescues
those.

The safety is structural, not numerical. Jev is asked which grammatical ROLE
the word is playing, from a closed list where exactly one option means "tic".
A veto only fires on a confident NON-filler answer, so a confused model can
only fail towards leaving the sentence alone - the direction a deletion has to
fail in. Swapping the wrong homophone is a typo the reader sees through;
deleting a word leaves nothing behind to notice.

Caveat worth keeping: n=28, one word, and the labels are a judgement call.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Optional

log = logging.getLogger("jevflow.filler")

# How sure Jev has to be that the word has a real job before the deletion is
# called off. Swept from 0.60 to 0.95 over 28 real cases with identical
# results and no correct deletion ever blocked, so this sits in the middle of
# a plateau rather than on a tuned edge.
VETO_ABOVE = 0.80

# One round trip per ambiguous word, so a rambling paragraph gets a ceiling.
MAX_QUESTIONS = 6

# The role that means "safe to remove". It is a single name on purpose: every
# other answer, right or wrong, keeps the word.
FILLER = "filler"

# What brackets the one occurrence being asked about.
MARKER = ">>>"

# Words that are a tic in some sentences and load-bearing in others. Anything
# NOT here is handled by the ordinary rules - "um" is never doing a job.
AMBIGUOUS: dict[str, dict[str, str]] = {
    "like": {
        FILLER: "a verbal tic - the sentence means exactly the same without it",
        "comparison": 'it means "similar to", as in "something like this"',
        "verb": 'to like or want something, as in "I would really like to"',
        "quotative": 'it introduces speech or thought, as in "he was like, no way"',
        "approximation": 'it means "roughly", as in "like twenty people"',
        "noun": "a like, of the kind a post receives",
    },
    "just": {
        FILLER: "a verbal tic - the sentence means exactly the same without it",
        "only": 'it means "only" or "merely", as in "just one of them"',
        "recently": 'it means "a moment ago", as in "I just finished"',
        "emphasis": 'it sharpens the word after it, as in "just stop"',
        "fairly": 'it means fair or deserved, as in "a just outcome"',
    },
    "right": {
        FILLER: "a verbal tic or a check for agreement that adds nothing",
        "direction": 'the direction, as in "turn right"',
        "correct": 'it means correct, as in "that is right"',
        "entitlement": 'a right that someone holds',
        "immediacy": 'it means "immediately", as in "right now"',
    },
    "so": {
        FILLER: "a verbal tic starting a sentence, adding nothing",
        "therefore": 'it means "therefore", joining a cause to its result',
        "degree": 'it means "to that extent", as in "so expensive"',
        "purpose": 'it introduces a purpose, as in "so I can tweak it"',
    },
    "actually": {
        FILLER: "a verbal tic - the sentence means exactly the same without it",
        "contrast": "it corrects an expectation, as in \"it actually works\"",
    },
    "basically": {
        FILLER: "a verbal tic - the sentence means exactly the same without it",
        "summarising": 'it means "in essence", introducing a summary',
    },
    "literally": {
        FILLER: "a verbal tic used for emphasis and meaning nothing",
        "exactly": "it means the thing described is true word for word",
    },
    "well": {
        FILLER: "a verbal tic starting a sentence, adding nothing",
        "manner": 'it describes doing something well',
        "health": 'it means healthy, as in "I am not well"',
        "noun": "a well that water comes from",
    },
}

_WORDS = re.compile(r"[A-Za-z']+")


def occurrences(text: str, words: Optional[list[str]] = None
                ) -> list[tuple[int, int, str]]:
    """Every ambiguous word in the text, as (start, end, word).

    Positions rather than counts, so the exact one that was asked about is the
    exact one removed - a sentence with four "like"s in it is the normal case,
    not the edge case.
    """
    found: list[tuple[int, int, str]] = []
    wanted = set(words or AMBIGUOUS)
    for m in _WORDS.finditer(text or ""):
        w = m.group(0).lower()
        if w in wanted and w in AMBIGUOUS:
            found.append((m.start(), m.end(), w))
    return found


def mark(text: str, start: int, end: int, marker: str = MARKER) -> str:
    """The sentence with ONE occurrence marked, so the question is unambiguous.

    Without this a sentence containing four "like"s asks four identical
    questions and gets four identical answers, which is not four decisions.
    """
    return f"{text[:start]}{marker}{text[start:end]}{marker}{text[end:]}"


def cut(text: str, start: int, end: int) -> str:
    """Remove that span and the space it leaves behind.

    Dropping the word alone welds its neighbours together; dropping the
    whitespace on both sides does too. One space back in, and a space before
    punctuation is not a space.
    """
    out = text[:start] + text[end:]
    out = re.sub(r"\s{2,}", " ", out)
    out = re.sub(r"\s+([,.;!?])", r"\1", out)
    out = re.sub(r"([,;])\s*\1+", r"\1", out)
    return out.strip()


class Veto:
    """Asks whether a word the rule wants to delete is actually doing a job.

    Answers are cached per (word, sentence) because dictation repeats itself
    and a round trip is not free.

    **Every failure returns False - allow the deletion.** That is deliberate
    and it is the one place this module fails towards losing a word: a veto
    that cannot reach the network must not silently change what the cleaner
    does, or the text would depend on the weather. The rule alone is what
    shipped before, and it is what you get back.
    """

    def __init__(self, jev: Any, sure_above: float = VETO_ABOVE,
                 max_questions: int = MAX_QUESTIONS) -> None:
        self.jev = jev
        self.sure_above = sure_above
        self.max_questions = max_questions
        self.asked = 0
        self._seen: dict[tuple[str, str], bool] = {}

    def __call__(self, text: str, start: int, end: int, word: str) -> bool:
        """True means DO NOT delete this word."""
        from jevflow.jev import build_filler_question, build_filler_state

        roles = AMBIGUOUS.get((word or "").lower())
        if self.jev is None or not roles:
            return False
        marked = mark(text, start, end)
        key = (word.lower(), marked)
        if key in self._seen:
            return self._seen[key]
        if self.asked >= self.max_questions:
            return False
        self.asked += 1
        try:
            res = self.jev.ask(build_filler_state(word, marked),
                               build_filler_question(word, roles))
        except Exception as exc:
            log.warning("[filler] could not ask about %r (%s); "
                        "leaving the rule to decide", word, exc)
            self._seen[key] = False
            return False
        answer = (res.get("answers") or {}).get("role") or {}
        role = answer.get("choice")
        confidence = float(answer.get("confidence") or 0.0)
        # A veto needs a confident NON-filler answer. No answer, a filler
        # answer, or an unsure one all let the rule have its way.
        held = bool(role) and role != FILLER and confidence > self.sure_above
        if held:
            log.info("[filler] keeping %r: it is a %s here (%.2f)",
                     word, role, confidence)
        self._seen[key] = held
        return held
