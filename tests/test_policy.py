"""Who may act, and when a refusal should become a question instead.

Policy is the one part of this program a model never touches. Jev says what kind
of request it heard and how certain it is; everything below decides what that
earns.
"""
from __future__ import annotations

import pytest

from jevflow import policy


def test_every_verb_has_a_policy():
    for verb in ("open_app", "web_search", "close_window", "switch_to",
                 "type_into", "media_control"):
        assert policy.for_verb(verb) is not None, verb


def test_an_unknown_verb_has_no_policy():
    """A verb with no policy must not fall through to a permissive default."""
    assert policy.for_verb("delete_everything") is None
    assert policy.for_verb("") is None


def test_the_thresholds_that_were_already_validated_did_not_move():
    """These two were measured in use. Changing them silently is a regression."""
    assert policy.for_verb("open_app").act_at == 0.65
    assert policy.for_verb("web_search").act_at == 0.50


def test_confirming_is_always_a_lower_bar_than_acting():
    for verb, p in policy.POLICIES.items():
        assert p.confirm_at < p.act_at, verb
        assert 0.0 < p.confirm_at, verb


@pytest.mark.parametrize("certainty,expect", [
    (0.95, policy.ACT),
    (0.65, policy.ACT),        # exactly at the bar acts
    (0.64, policy.CONFIRM),
    (0.45, policy.CONFIRM),
    (0.44, policy.REFUSE),
    (0.04, policy.REFUSE),
])
def test_open_app_bands(certainty, expect):
    assert policy.decide("open_app", certainty) == expect


def test_an_always_confirm_verb_never_acts_on_its_own():
    """The whole point of the gate: a destructive verb can exist, but not run
    unasked, no matter how certain the model claims to be."""
    p = policy.Policy("wipe_disk", act_at=0.65, confirm_at=0.45, always_confirm=True)
    assert policy.decide_with(p, 0.99) == policy.CONFIRM
    assert policy.decide_with(p, 0.44) == policy.REFUSE


def test_an_unknown_verb_is_refused_not_confirmed():
    assert policy.decide("delete_everything", 0.99) == policy.REFUSE


# -- the pending confirmation ------------------------------------------------
def test_a_pending_confirmation_is_answered_by_yes():
    pend = policy.Pending(verb="close_window", prompt="Close Notepad?",
                          payload={"hwnd": 7}, created=100.0)
    assert pend.resolve("yes", now=101.0) == policy.CONFIRMED


def test_a_pending_confirmation_is_dropped_by_no():
    pend = policy.Pending("close_window", "Close Notepad?", {}, created=100.0)
    assert pend.resolve("never mind", now=101.0) == policy.DENIED


def test_a_pending_confirmation_expires():
    """An unanswered question must not be answered by something said a minute
    later about something else entirely."""
    pend = policy.Pending("close_window", "Close Notepad?", {}, created=100.0)
    assert pend.resolve("yes", now=100.0 + policy.CONFIRM_TTL_S + 0.1) == policy.EXPIRED


def test_anything_else_abandons_the_confirmation():
    pend = policy.Pending("close_window", "Close Notepad?", {}, created=100.0)
    assert pend.resolve("open spotify", now=101.0) == policy.SUPERSEDED


def test_a_confirmation_cannot_be_answered_twice():
    """Once used, it is gone - so a second 'yes' does not close a second window."""
    pend = policy.Pending("close_window", "Close Notepad?", {}, created=100.0)
    assert pend.resolve("yes", now=101.0) == policy.CONFIRMED
    assert pend.resolve("yes", now=102.0) == policy.EXPIRED


# -- the destructive verb, and the gate that makes it possible ----------------
def test_force_quit_exists_and_always_asks():
    """Decided 2026-09-21: keep the ability to force-quit, but make it
    impossible to reach without being asked out loud first."""
    p = policy.for_verb("force_quit")
    assert p is not None, "the verb was dropped"
    assert p.always_confirm is True


@pytest.mark.parametrize("certainty", [1.0, 0.99, 0.9, 0.85, 0.7, 0.6])
def test_force_quit_never_acts_on_its_own_at_any_certainty(certainty):
    """The whole argument for keeping it. If any certainty could fire it, a
    misheard sentence destroys unsaved work with no undo and no prompt."""
    assert policy.decide("force_quit", certainty) == policy.CONFIRM


def test_force_quit_is_refused_outright_when_unclear():
    assert policy.decide("force_quit", 0.59) == policy.REFUSE


def test_no_other_verb_is_marked_destructive_by_accident():
    """A second always_confirm verb appearing silently would mean someone added
    a destructive capability without this test being updated to say so."""
    destructive = {v for v, p in policy.POLICIES.items() if p.always_confirm}
    assert destructive == {"force_quit"}, f"unexpected destructive verbs: {destructive}"
