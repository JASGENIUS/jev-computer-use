"""Doing several things, in order, from one sentence.

The danger with a sequence is not failing - it is failing HALFWAY. Three of
four steps done and no word about the fourth is worse than refusing the whole
thing, because the person walks away believing it all happened.

So: every step is planned BEFORE any of them runs, one confirmation covers the
lot, execution stops at the first failure, and the report always says exactly
which steps ran and which did not.
"""
from __future__ import annotations

import pytest

from conftest import FakeJev
from jevflow import apps, policy
from jevflow.command import Commander


class ScriptedJev:
    """Answers each step differently, in the order they are asked."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.asked = []

    def ask(self, state, questions):
        self.asked.append((state, questions))
        intent, app, certain = self.answers[min(len(self.asked) - 1,
                                                len(self.answers) - 1)]
        ans = {"intent": {"choice": intent}, "certain": {"noul": certain}}
        if app is not None:
            ans["app"] = {"choice": app}
        return {"answers": ans, "latency_ms": 90.0}

    @property
    def last_candidates(self):
        return self.asked[-1][0]["installed_candidates"] if self.asked else []


@pytest.fixture
def three_apps(monkeypatch):
    made = [
        apps.App(name="Notepad", launch="n.lnk", target=r"C:\notepad.exe"),
        apps.App(name="Google Chrome", launch="c.lnk", target=r"C:\chrome.exe"),
        apps.App(name="Spotify", launch="s.lnk", target=r"C:\spotify.exe"),
    ]
    monkeypatch.setattr(apps, "index", lambda: tuple(made))
    monkeypatch.setattr(apps, "running", lambda c: [])
    return made


# -- the happy path -----------------------------------------------------------
def test_three_apps_open_in_order(three_apps):
    jev = ScriptedJev([("open_app", "Notepad", 0.9),
                       ("open_app", "Google Chrome", 0.9),
                       ("open_app", "Spotify", 0.9)])
    c = Commander(jev=jev, dry_run=True)
    out = c.handle("open notepad and also open google chrome and then open spotify")
    assert out.ok, out.detail
    assert len(jev.asked) == 3, "it did not plan each step separately"
    for name in ("Notepad", "Chrome", "Spotify"):
        assert name in out.detail, f"{name} missing from the report: {out.detail}"


def test_the_report_counts_the_steps(three_apps):
    jev = ScriptedJev([("open_app", "Notepad", 0.9)])
    c = Commander(jev=jev, dry_run=True)
    out = c.handle("open notepad and then open notepad and then open notepad")
    assert "3" in out.detail, out.detail


# -- failing halfway ----------------------------------------------------------
def test_a_failure_stops_the_sequence(three_apps, monkeypatch):
    """Carrying on after a failure means later steps act on a machine that is
    not in the state they were planned for."""
    jev = ScriptedJev([("open_app", "Notepad", 0.9),
                       ("open_app", "NOT_INSTALLED", 0.9),
                       ("open_app", "Spotify", 0.9)])
    c = Commander(jev=jev, dry_run=True)
    out = c.handle("open notepad and then open nonsense and then open spotify")
    assert not out.ok
    assert "Spotify" not in out.detail, "it carried on after a failure"


def test_a_step_that_cannot_be_PLANNED_stops_everything_before_it_runs(three_apps):
    """Planning ahead means this never half-runs: the report says so plainly."""
    jev = ScriptedJev([("open_app", "Notepad", 0.9),
                       ("open_app", "NOT_INSTALLED", 0.9)])
    c = Commander(jev=jev, dry_run=True)
    out = c.handle("open notepad and then open nonsense")
    assert not out.ok
    assert "nothing was done" in out.detail.lower(), out.detail


def test_a_step_that_fails_while_RUNNING_reports_what_did_happen(three_apps,
                                                                 monkeypatch):
    """Some failures can only happen at execution - an app that refuses to
    start. Then the report MUST say which steps already ran, or the person
    walks away believing the whole thing happened."""
    from jevflow import command as C

    jev = ScriptedJev([("open_app", "Notepad", 0.9),
                       ("open_app", "Google Chrome", 0.9)])
    c = Commander(jev=jev, dry_run=True)

    real = c._do_open
    calls = {"n": 0}

    def flaky(app, ms):
        calls["n"] += 1
        if calls["n"] == 1:
            return real(app, ms)
        return C.Outcome(False, "launch_failed", f"{app.name} refused to start", ms)

    monkeypatch.setattr(c, "_do_open", flaky)
    out = c.handle("open notepad and then open google chrome")
    low = out.detail.lower()
    assert not out.ok
    assert "notepad" in low, f"it did not say what HAD been done: {out.detail}"
    assert "stopped at step 2" in low, out.detail


def test_nothing_runs_if_a_step_cannot_be_planned(three_apps):
    """Planning happens for ALL steps before ANY of them runs, so a sequence
    that cannot complete never half-completes."""
    jev = ScriptedJev([("format_the_drive", None, 0.99)])
    c = Commander(jev=jev, dry_run=True)
    out = c.handle("open notepad and then format the drive")
    assert not out.ok
    assert "notepad" not in out.detail.lower() or "not" in out.detail.lower()


# -- confirmation covers the whole thing --------------------------------------
def test_one_confirmation_for_the_whole_sequence(three_apps):
    jev = ScriptedJev([("open_app", "Notepad", 0.55),
                       ("open_app", "Google Chrome", 0.55)])
    c = Commander(never_ask=False, jev=jev, dry_run=True)
    out = c.handle("open notepad and then open google chrome")
    assert out.action == "needs_confirm", out.action
    assert "2" in out.detail or "notepad" in out.detail.lower()

    done = c.handle("yes")
    assert done.ok, done.detail


def test_saying_no_runs_none_of_it(three_apps):
    jev = ScriptedJev([("open_app", "Notepad", 0.55),
                       ("open_app", "Google Chrome", 0.55)])
    c = Commander(never_ask=False, jev=jev, dry_run=True)
    c.handle("open notepad and then open google chrome")
    out = c.handle("never mind")
    assert not out.ok and out.action == "cancelled"


# -- a single step is untouched ----------------------------------------------
def test_a_single_instruction_behaves_exactly_as_before(three_apps):
    jev = FakeJev(intent="open_app", app="Notepad", certain=0.9)
    c = Commander(jev=jev, dry_run=True)
    out = c.handle("open notepad")
    assert out.ok and out.action == "would_open" and out.detail == "Notepad"


def test_progress_is_reported(three_apps):
    """Three apps opening behind a capsule that still says "Deciding" looks
    like it has hung.

    Reported once per GROUP. Sequentially - the default - every step is its own
    group, so three opens report three times and the capsule counts 1, 2, 3 in
    the order they were spoken.
    """
    jev = ScriptedJev([("open_app", "Notepad", 0.9)])
    seen = []
    c = Commander(jev=jev, dry_run=True,
                  on_progress=lambda i, n, label: seen.append((i, n, label)))
    c.handle("open notepad and then open notepad and then open notepad")
    assert len(seen) == 3, f"sequential steps reported {len(seen)} times: {seen}"
    assert [s[0] for s in seen] == [1, 2, 3]
    assert all(s[1] == 3 for s in seen)


def test_a_parallel_group_reports_once(three_apps=None):
    """With grouping ON, three opens happen at one instant, and calling that
    "step 1 of 3" three times would describe something that is not happening."""
    jev = ScriptedJev([("open_app", "Notepad", 0.9)])
    seen = []
    c = Commander(jev=jev, dry_run=True, parallel=True,
                  on_progress=lambda i, n, label: seen.append((i, n, label)))
    c.handle("open notepad and then open notepad and then open notepad")
    assert len(seen) == 1, f"a parallel group reported {len(seen)} times: {seen}"
    assert seen[0][1] == 3
    assert "+" in seen[0][2], seen[0][2]


def test_progress_reports_each_focus_step_separately(three_apps):
    """Steps that cannot overlap DO happen one after another, so each is its
    own event."""
    jev = ScriptedJev([("type_into", "Notepad", 0.95)])
    seen = []
    c = Commander(jev=jev, dry_run=True,
                  on_progress=lambda i, n, label: seen.append(i))
    c.handle("type hello into notepad and then type goodbye into notepad")
    assert seen == [1, 2], seen


def test_a_broken_progress_callback_does_not_stop_the_work(three_apps):
    def boom(*_a):
        raise RuntimeError("ui exploded")
    jev = ScriptedJev([("open_app", "Notepad", 0.9)])
    c = Commander(jev=jev, dry_run=True, on_progress=boom)
    out = c.handle("open notepad and then open notepad")
    assert out.ok, "a broken progress callback killed the sequence"
