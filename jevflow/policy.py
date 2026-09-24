"""Who may act, and when a refusal should become a question instead.

This is the one part of the program a model never touches. Jev says what kind of
request it heard and how certain it is. What that certainty *earns* is decided
here, in code, where it can be read and argued with.

Two ideas:

**Bands, not a bar.** A single threshold only has two answers, and the second
one is "no" - so a slightly-unclear request that was obviously right got thrown
away, and saying it again more loudly was the only recourse. Each verb now has
two numbers: act above the top one, refuse below the bottom one, and in between
*ask*. The bars that were already measured in use (0.65 to open an app, 0.50 to
search) are unchanged, so nothing that used to run automatically has stopped.

**A verb can be marked always-confirm.** That is what makes a destructive verb
possible at all. Until now safety came entirely from absence - there was no
delete option to mis-select - which works until the first genuinely useful
destructive verb needs to exist. `always_confirm` means no certainty, however
high, lets it run unasked. A model cannot talk its way past it, because the
model is never asked this question.

A verb with no policy is refused. It does NOT fall through to a default: a new
intent that someone forgets to wire up must do nothing, not the most permissive
thing available.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

# What a certainty earns.
ACT = "act"
CONFIRM = "confirm"
REFUSE = "refuse"

# How a pending question was answered.
CONFIRMED = "confirmed"
DENIED = "denied"
EXPIRED = "expired"
SUPERSEDED = "superseded"

# An unanswered question goes stale. Without this, "yes" said a minute later
# about something else entirely would run an action nobody remembered offering.
CONFIRM_TTL_S = 20.0


@dataclass(frozen=True)
class Policy:
    verb: str
    act_at: float             # at or above this, run it without asking
    confirm_at: float         # at or above this, ask; below it, refuse
    always_confirm: bool = False
    why: str = ""


POLICIES: dict[str, Policy] = {
    # Measured in use and deliberately unchanged.
    "open_app": Policy("open_app", 0.65, 0.45,
                       why="a wrongly launched application is disruptive"),
    "web_search": Policy("web_search", 0.50, 0.30,
                         why="a stray browser tab is cheap to close"),
    # WM_CLOSE, so the app still gets to ask about unsaved work - but closing
    # the wrong window is annoying enough to deserve the same bar as opening.
    "close_window": Policy("close_window", 0.65, 0.45,
                           why="closing the wrong window interrupts real work"),
    # Focusing something already open changes nothing that cannot be undone by
    # clicking once, so the bar is lower.
    "switch_to": Policy("switch_to", 0.60, 0.40,
                        why="focus is trivially reversible"),
    # Text goes into a real document. Higher bar than anything else here,
    # because the keystrokes land in whatever is focused if focus moved.
    "type_into": Policy("type_into", 0.70, 0.50,
                        why="keystrokes land in a real document"),
    "media_control": Policy("media_control", 0.55, 0.40,
                            why="pressing play is harmless and obvious"),
    # The first genuinely destructive verb, and the reason `always_confirm`
    # exists at all. Force-killing a process used to be forbidden
    # outright. Decided 2026-09-21: keep the capability, and make it
    # unreachable without being asked out loud first. No certainty, however
    # high, runs this unasked - and the model is never asked THAT question, so
    # it cannot be talked past. A hung app keeps its escape hatch; a misheard
    # sentence can never silently destroy unsaved work.
    "force_quit": Policy("force_quit", 0.85, 0.60, always_confirm=True,
                         why="a forced kill has no undo and loses unsaved work"),
}


def for_verb(verb: str) -> Optional[Policy]:
    return POLICIES.get(verb or "")


def decide_with(p: Policy, certainty: float) -> str:
    if p.always_confirm:
        return CONFIRM if certainty >= p.confirm_at else REFUSE
    if certainty >= p.act_at:
        return ACT
    if certainty >= p.confirm_at:
        return CONFIRM
    return REFUSE


def decide(verb: str, certainty: float) -> str:
    """What this certainty earns for this verb. Unknown verbs earn nothing."""
    p = for_verb(verb)
    return REFUSE if p is None else decide_with(p, float(certainty))


@dataclass
class Pending:
    """A question that was asked and is waiting for a yes or a no.

    It holds the action that was *already decided*, so answering "yes" runs that
    one. It is deliberately not re-derived from the new utterance: re-asking
    would let a second, different answer run in place of the one offered, and
    the person said yes to what they were shown.
    """
    verb: str
    prompt: str
    payload: dict[str, Any] = field(default_factory=dict)
    created: float = field(default_factory=time.time)
    _spent: bool = False

    def expired(self, now: Optional[float] = None) -> bool:
        return (now or time.time()) - self.created > CONFIRM_TTL_S

    def resolve(self, transcript: str, now: Optional[float] = None) -> str:
        """Read an utterance as the answer to this question.

        Anything that is not plainly yes or no supersedes it - the person moved
        on, and a half-answered question must not linger where a later "yes"
        could land on it.
        """
        from jevflow import parse

        if self._spent or self.expired(now):
            return EXPIRED
        if parse.is_affirmation(transcript):
            self._spent = True
            return CONFIRMED
        if parse.is_denial(transcript):
            self._spent = True
            return DENIED
        return SUPERSEDED
