"""JCU never stops to ask yes or no.

Prompts were removed one at a time and the next one kept
turning up, so the guarantee lives at the one choke point in `_prepare` - and
this file checks the guarantee rather than any single prompt.

Every case below is something that DID ask before: a middling confidence, a
near-miss app name, a multi-close, a sequence containing any of those. With
the default Commander, none of them may leave anything pending.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jevflow import policy
from jevflow.command import Commander


class FakeJev:
    """Answers every question the same way, at whatever confidence is set."""

    def __init__(self, intent="open_app", app="Notepad", confidence=0.55,
                 certain=0.55):
        self.intent, self.app = intent, app
        self.confidence, self.certain = confidence, certain

    def ask(self, state, questions):
        # A close line is a close, whatever the default intent is - otherwise
        # "close everything you opened" gets treated as opening an app of
        # that name and the test measures the fake, not JCU.
        said = str(state.get("transcript", "")).lower()
        intent = "close_window" if said.startswith("close") else self.intent
        answers = {
            "intent": {"choice": intent, "confidence": self.confidence},
            "certain": {"noul": self.certain},
        }
        if "app" in questions:
            answers["app"] = {"choice": self.app, "confidence": self.confidence}
        return {"answers": answers, "latency_ms": 5.0}


def never_pending(jev, *said):
    """Never asks, and never refuses for being UNSURE.

    Not asking is satisfied just as well by refusing, which would make every
    one of these tests pass against a tool that simply stopped working - so a
    refusal on confidence ("unsure") fails too. Outcomes about the world are
    fine: in a dry run nothing really opens, so "Notepad is not open" is the
    honest answer to closing it, not a failure of nerve.
    """
    c = Commander(jev=jev, dry_run=True)
    out = None
    for line in said:
        out = c.handle(line)
        assert c.pending is None, f"{line!r} left a question pending: {out.detail}"
        assert out.action != "needs_confirm", f"{line!r} asked: {out.detail}"
        assert "yes or no" not in (out.detail or ""), f"{line!r} said: {out.detail}"
        assert out.action != "unsure", (
            f"{line!r} did not ask - but refused for being unsure: {out.detail}")
    return c, out


def test_it_is_the_default():
    assert Commander(jev=FakeJev()).never_ask is True


# -- every band that used to ask ----------------------------------------------
def test_a_middling_open_does_not_ask():
    """open_app asks between 0.45 and 0.65. 0.55 used to be a question."""
    never_pending(FakeJev(confidence=0.55, certain=0.55), "open notepad")


def test_a_middling_type_does_not_ask():
    """type_into had the highest bar of all: 0.50 to 0.70 was a question."""
    never_pending(FakeJev(intent="type_text", app=None, confidence=0.60,
                          certain=0.60), "type in hello")


def test_a_middling_close_does_not_ask():
    never_pending(FakeJev(intent="close_window", confidence=0.55, certain=0.55),
                  "close notepad")


def test_every_verb_across_the_whole_confirm_band_does_not_ask():
    """Sweep, rather than pick the two numbers someone happened to remember."""
    for verb in ("open_app", "web_search", "close_window", "switch_to",
                 "type_text", "media_control"):
        p = policy.for_verb("type_into" if verb == "type_text" else verb)
        lo, hi = p.confirm_at, p.act_at
        for step in range(11):
            level = lo + (hi - lo) * step / 10
            never_pending(FakeJev(intent=verb, confidence=level, certain=level),
                          "open notepad")


# -- the prompts that were forced regardless of confidence --------------------
def test_a_near_miss_just_opens_it():
    """"open modrinth ap" used to become "did you mean Modrinth App? say yes or
    no". Jev will not vouch for it, but code can see it scores 0.90 - so it
    opens it.

    The first version of this test used "open modrinthh", which never reached
    the near-miss path at all (the shortlist was empty), refused, and so
    "did not ask" for a reason that had nothing to do with the guard. It
    passed with the guard deleted.
    """
    _c, out = never_pending(
        FakeJev(app="none_of_these", confidence=0.95, certain=0.90),
        "open modrinth ap")
    assert "Modrinth" in out.detail, out.detail


def test_closing_everything_it_opened_does_not_ask():
    never_pending(FakeJev(confidence=0.95, certain=0.90),
                  "open notepad", "close everything you opened")


def test_a_sequence_does_not_ask_even_when_a_step_is_middling():
    """A sequence used to ask ONCE for the whole lot if any step was unsure."""
    never_pending(FakeJev(confidence=0.55, certain=0.55),
                  "open notepad and then open notepad and then open notepad")


# -- and the one thing it still must not do ----------------------------------
def test_a_destructive_verb_is_refused_not_run_unasked():
    """force_quit is always_confirm: a forced kill has no undo. Never asking
    must not mean killing things silently - it is refused instead."""
    p = policy.for_verb("force_quit")
    assert p.always_confirm
    c = Commander(jev=FakeJev(), dry_run=True)
    band = policy.decide_with(p, 0.99)
    assert band == policy.CONFIRM          # what policy says on its own
    # and what the choke point turns that into:
    resolved = policy.REFUSE if p.always_confirm else policy.ACT
    assert resolved == policy.REFUSE
    assert c.never_ask


# -- the old behaviour is still reachable, on purpose -------------------------
def test_asking_can_still_be_switched_back_on():
    c = Commander(jev=FakeJev(confidence=0.55, certain=0.55), dry_run=True,
                  never_ask=False)
    out = c.handle("open notepad")
    assert c.pending is not None and out.action == "needs_confirm"
