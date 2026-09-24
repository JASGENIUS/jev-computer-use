"""The chain end to end, with Jev scripted and nothing actually launched.

Jev is faked here on purpose. What is being tested is the part Jev does not do:
which candidates code assembled, whether policy let it run, and whether the
result told the truth about what happened.
"""
from __future__ import annotations

import pytest

from conftest import FakeJev
from jevflow import policy
from jevflow.command import Commander


def cmd(jev, **kw):
    kw.setdefault("dry_run", True)
    return Commander(jev=jev, **kw)


# -- an address nobody curated ------------------------------------------------
def test_an_arbitrary_domain_can_be_opened():
    jev = FakeJev(intent="open_app", app="databento.com", certain=0.9)
    out = cmd(jev).handle("open databento.com")
    assert out.ok, out.detail
    assert "https://databento.com" in out.detail


def test_the_domain_reaches_jev_as_a_candidate():
    """If code never shortlists it, Jev cannot choose it however clearly it
    was said. The wide pass is code's job and its failure is silent."""
    jev = FakeJev(intent="open_app", app="databento.com")
    cmd(jev).handle("open databento.com")
    assert "databento.com" in jev.last_candidates


def test_a_curated_site_still_wins_over_the_bare_host():
    jev = FakeJev(intent="open_app", app="YouTube")
    out = cmd(jev).handle("open youtube.com")
    assert out.ok and "YouTube" in out.detail


def test_a_choice_outside_the_shortlist_is_refused():
    """Jev can only be offered what code found. If a name comes back that was
    never offered, something is wrong and nothing should open."""
    jev = FakeJev(intent="open_app", app="evil.com", certain=0.99)
    out = cmd(jev).handle("open databento.com")
    assert not out.ok


# -- honest verification ------------------------------------------------------
def test_opening_a_site_does_not_claim_verification_it_does_not_have(monkeypatch):
    """A browser window existing is not proof the address loaded.

    The old code returned verified=True unconditionally - the same lie as
    reporting success because a process was spawned.
    """
    from jevflow import actions

    monkeypatch.setattr(actions, "open_url",
                        lambda *a, **k: actions.Opened(True, "Brave", "brave.exe", ""))
    monkeypatch.setattr(actions, "wait_for_title", lambda *a, **k: None)
    jev = FakeJev(intent="open_app", app="databento.com")
    out = cmd(jev, dry_run=False).handle("open databento.com")
    assert out.ok
    assert out.verified is not True, "claimed verified without checking the title"


def test_a_browser_downgrade_is_reported(monkeypatch):
    """Chrome exits 0 with no window on this machine. Falling back to Brave is
    fine; doing it silently is not - the user has no other way to find out."""
    from jevflow import actions

    monkeypatch.setattr(actions, "open_url",
                        lambda *a, **k: actions.Opened(
                            True, "Brave", "brave.exe", "Google Chrome did not start"))
    monkeypatch.setattr(actions, "wait_for_title", lambda *a, **k: "Databento")
    jev = FakeJev(intent="open_app", app="databento.com")
    out = cmd(jev, dry_run=False).handle("open databento.com in chrome")
    assert "Chrome" in out.detail, out.detail


# -- policy bands -------------------------------------------------------------
def test_a_confident_request_acts():
    jev = FakeJev(intent="open_app", app="databento.com", certain=0.9)
    assert cmd(jev).handle("open databento.com").ok


def test_a_middling_request_asks_instead_of_refusing():
    jev = FakeJev(intent="open_app", app="databento.com", certain=0.55)
    out = cmd(jev, never_ask=False).handle("open databento.com")
    assert out.action == "needs_confirm"
    assert not out.ok


def test_a_vague_request_is_refused_outright():
    jev = FakeJev(intent="open_app", app="databento.com", certain=0.2)
    out = cmd(jev).handle("mumble mumble")
    assert out.action != "needs_confirm"
    assert not out.ok


def test_ambient_conversation_does_nothing():
    jev = FakeJev(intent="unclear", app=None, certain=0.08)
    out = cmd(jev).handle("so then I told him it was fine")
    assert not out.ok and out.action == "unclear"


# -- the confirmation path ----------------------------------------------------
def test_yes_runs_the_thing_that_was_offered():
    jev = FakeJev(intent="open_app", app="databento.com", certain=0.55)
    c = cmd(jev, never_ask=False)
    first = c.handle("open databento.com")
    assert first.action == "needs_confirm"
    second = c.handle("yes")
    assert second.ok, second.detail
    assert "databento.com" in second.detail


def test_no_drops_it():
    jev = FakeJev(intent="open_app", app="databento.com", certain=0.55)
    c = cmd(jev, never_ask=False)
    c.handle("open databento.com")
    out = c.handle("never mind")
    assert not out.ok and out.action == "cancelled"
    assert c.pending is None


def test_a_yes_with_nothing_pending_does_nothing():
    jev = FakeJev(intent="unclear", certain=0.1)
    out = cmd(jev).handle("yes")
    assert not out.ok


def test_answering_yes_does_not_ask_jev_again():
    """The decision was already made. Asking again would let a second, different
    answer run in place of the one that was actually offered."""
    jev = FakeJev(intent="open_app", app="databento.com", certain=0.55)
    c = cmd(jev, never_ask=False)
    c.handle("open databento.com")
    before = len(jev.asked)
    c.handle("yes")
    assert len(jev.asked) == before


def test_a_new_command_abandons_the_pending_one():
    jev = FakeJev(intent="open_app", app="databento.com", certain=0.9)
    c = cmd(jev, never_ask=False)
    c.jev = FakeJev(intent="open_app", app="databento.com", certain=0.55)
    c.handle("open databento.com")
    assert c.pending is not None
    c.jev = jev
    out = c.handle("open databento.com")
    assert out.ok
    assert c.pending is None, "the abandoned question is still waiting"


def test_a_stale_confirmation_is_not_honoured(monkeypatch):
    jev = FakeJev(intent="open_app", app="databento.com", certain=0.55)
    c = cmd(jev, never_ask=False)
    c.handle("open databento.com")
    c.pending = c.pending.__class__(c.pending.verb, c.pending.prompt,
                                    c.pending.payload,
                                    created=c.pending.created - policy.CONFIRM_TTL_S - 1)
    out = c.handle("yes")
    assert not out.ok


# -- the new verbs ------------------------------------------------------------
@pytest.fixture
def notepad(monkeypatch):
    """A shortlist that does not depend on what this machine has installed."""
    from jevflow import apps
    fake = apps.App(name="Notepad", launch=r"C:\notepad.lnk",
                    target=r"C:\Windows\notepad.exe")
    monkeypatch.setattr(apps, "shortlist", lambda q, limit=6: [fake])
    return fake


def test_switch_to_focuses_rather_than_launching(monkeypatch, notepad):
    from jevflow import actions

    focused: list[int] = []
    launched: list[str] = []
    monkeypatch.setattr(actions, "windows_for_exe", lambda exe: [(42, "Notepad")])
    monkeypatch.setattr(actions, "focus_window", lambda h, **k: focused.append(h) or True)
    monkeypatch.setattr("jevflow.apps.launch",
                        lambda a: launched.append(a.name) or (True, a.name))
    jev = FakeJev(intent="switch_to", app="Notepad", certain=0.9)
    out = Commander(jev=jev, dry_run=False).handle("switch to notepad")
    assert out.ok, out.detail
    assert focused == [42], "it did not focus the open window"
    assert not launched, "it launched a second copy instead of switching to the open one"


def test_switching_to_something_that_is_not_open_says_so(monkeypatch, notepad):
    from jevflow import actions
    monkeypatch.setattr(actions, "windows_for_exe", lambda exe: [])
    jev = FakeJev(intent="switch_to", app="Notepad", certain=0.9)
    out = Commander(jev=jev, dry_run=False).handle("switch to notepad")
    assert not out.ok and out.action == "not_open"


def test_a_media_key_needs_no_candidates():
    jev = FakeJev(intent="media_control", app=None, certain=0.8)
    out = cmd(jev).handle("next track")
    assert out.ok, out.detail
    assert "next" in out.detail.lower()


def test_a_media_key_is_never_claimed_as_verified():
    """Nothing on this machine can confirm that a song changed."""
    jev = FakeJev(intent="media_control", app=None, certain=0.8)
    out = cmd(jev).handle("pause")
    assert out.verified is None


def test_media_without_a_recognised_key_does_nothing():
    jev = FakeJev(intent="media_control", app=None, certain=0.9)
    out = cmd(jev).handle("do the thing with the music")
    assert not out.ok


def test_typing_into_an_app_carries_the_words():
    jev = FakeJev(intent="type_into", app="Notepad", certain=0.9)
    out = cmd(jev).handle("type hello world into notepad")
    assert out.ok, out.detail
    assert "hello world" in out.detail


def test_typing_with_nothing_to_type_is_refused():
    jev = FakeJev(intent="type_into", app="Notepad", certain=0.95)
    out = cmd(jev).handle("type into notepad")
    assert not out.ok


def test_an_intent_with_no_policy_is_refused():
    """Jev returning a verb code does not implement must not fall through to
    a default that runs it."""
    jev = FakeJev(intent="format_the_drive", app=None, certain=0.99)
    out = cmd(jev).handle("format the drive")
    assert not out.ok


# -- Jev being unreachable ----------------------------------------------------
def test_jev_being_down_is_reported_not_guessed():
    jev = FakeJev(raises=RuntimeError("connection refused"))
    out = cmd(jev).handle("open notepad")
    assert not out.ok and out.action == "error"


# -- a site and an app can share a name ----------------------------------------
def test_a_site_that_collides_with_an_app_is_named_by_its_host():
    """"Claude" is both a desktop app and claude.ai, so the option list had the
    same word twice. A choice between two identical labels is not a choice:
    Jev picked "Claude", the code resolved the first match, and the website won
    every time however clearly the app was asked for."""
    from jevflow.command import site_label
    from jevflow.sites import Site

    claude = Site("Claude", "https://claude.ai", None, ())
    assert site_label(claude, ["Claude"]) == "claude.ai"
    assert site_label(claude, ["Notepad", "Chrome"]) == "Claude"


def test_the_label_collision_is_case_insensitive():
    from jevflow.command import site_label
    from jevflow.sites import Site

    site = Site("Claude", "https://claude.ai", None, ())
    assert site_label(site, ["claude"]) == "claude.ai"
    assert site_label(site, ["CLAUDE"]) == "claude.ai"


def test_no_two_options_ever_carry_the_same_label():
    """The property that matters, checked against the real app index: whatever
    Jev is offered, it must be able to tell the options apart."""
    from jevflow import apps, parse, sites
    from jevflow.command import site_label

    for said in ("open claude", "open spotify", "open the claude app",
                 "open notion", "open discord", "open figma"):
        hits = sites.shortlist(said)
        query = parse.app_query(said)
        found = apps.shortlist(query, limit=4) if query else []
        app_names = [a.name for a in found]
        names = [site_label(s, app_names) for s in hits] + app_names
        assert len(names) == len(set(names)), f"{said!r} offered {names}"
