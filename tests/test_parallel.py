"""Doing the independent parts at once, and the rest strictly in order.

**Grouping is OPT-IN.** By default every step runs strictly in the order it was
spoken: batching the opens was faster on paper and wrong in the room - three
windows appeared in a scramble and the search raced the browser it was meant to
run in. These tests keep the capability honest, so they ask for it explicitly
with `parallel=True`.

Opening three apps is three independent jobs, and running them one at a time
means waiting for each in turn - which matters because an app that never shows
a window costs a 12-second timeout, and those add up.

But only SOME steps are independent. Searching, typing and switching all take
the keyboard focus and then send keystrokes to whatever has it. Run two of
those at once and the keystrokes land in the wrong window - which is the exact
failure this whole layer exists to prevent. So focus-taking steps run alone,
in order, always.
"""
from __future__ import annotations

import threading
import time

import pytest

from jevflow import apps, command
from jevflow.command import Commander, Outcome, _Plan


def plan(verb, label, fn):
    return _Plan(verb, label, fn)


# -- which steps may share the machine ---------------------------------------
@pytest.mark.parametrize("verb,independent", [
    ("open_app", True),
    ("media_control", True),
    ("web_search", False),      # takes focus, then types
    ("type_into", False),       # types
    ("switch_to", False),       # takes focus
    ("close_window", False),    # changes what is focused
    ("force_quit", False),
    ("sequence", False),
])
def test_only_focus_free_steps_are_independent(verb, independent):
    assert command.is_independent(verb) is independent


def test_an_unknown_verb_is_never_treated_as_independent():
    """A verb nobody classified must not be assumed safe to run concurrently."""
    assert command.is_independent("something_new") is False


# -- grouping -----------------------------------------------------------------
def test_consecutive_opens_group_together():
    plans = [plan("open_app", "a", None), plan("open_app", "b", None),
             plan("open_app", "c", None)]
    groups = command.group_plans(plans, parallel=True)
    assert len(groups) == 1 and len(groups[0]) == 3


def test_a_focus_step_breaks_the_group():
    plans = [plan("open_app", "a", None), plan("web_search", "s", None),
             plan("open_app", "b", None)]
    groups = command.group_plans(plans, parallel=True)
    assert [len(g) for g in groups] == [1, 1, 1], groups


def test_order_is_preserved_across_groups():
    plans = [plan("open_app", "a", None), plan("open_app", "b", None),
             plan("type_into", "t", None), plan("open_app", "c", None)]
    groups = command.group_plans(plans, parallel=True)
    flat = [p.label for g in groups for p in g]
    assert flat == ["a", "b", "t", "c"]


# -- execution ----------------------------------------------------------------
def test_independent_steps_really_do_overlap():
    """The point of the exercise. Three half-second opens should take about
    half a second, not a second and a half."""
    def slow(label):
        def run():
            time.sleep(0.4)
            return Outcome(True, "opened", label, 1.0, verified=True)
        return run

    plans = [plan("open_app", n, slow(n)) for n in ("a", "b", "c")]
    c = Commander(parallel=True, jev=object(), dry_run=True)
    t0 = time.time()
    out = c._run_sequence(plans, 0.0)
    elapsed = time.time() - t0
    assert out.ok, out.detail
    assert elapsed < 0.9, f"they ran one at a time ({elapsed:.2f}s)"


def test_focus_taking_steps_never_overlap():
    """Two of these at once means keystrokes in the wrong window."""
    live = []
    peak = {"n": 0}
    lock = threading.Lock()

    def watched(label):
        def run():
            with lock:
                live.append(label)
                peak["n"] = max(peak["n"], len(live))
            time.sleep(0.1)
            with lock:
                live.remove(label)
            return Outcome(True, "typed", label, 1.0)
        return run

    plans = [plan("type_into", n, watched(n)) for n in ("a", "b", "c")]
    c = Commander(parallel=True, jev=object(), dry_run=True)
    c._run_sequence(plans, 0.0)
    assert peak["n"] == 1, f"{peak['n']} focus-taking steps ran at once"


def test_the_report_keeps_the_order_you_asked_for():
    def ok(label, delay):
        def run():
            time.sleep(delay)
            return Outcome(True, "opened", label, 1.0)
        return run

    # 'c' finishes first, but must still be reported last.
    plans = [plan("open_app", "a", ok("a", 0.3)),
             plan("open_app", "b", ok("b", 0.2)),
             plan("open_app", "c", ok("c", 0.05))]
    c = Commander(parallel=True, jev=object(), dry_run=True)
    out = c._run_sequence(plans, 0.0)
    assert out.detail.index("a") < out.detail.index("b") < out.detail.index("c"), out.detail


def test_a_failure_inside_a_parallel_group_is_still_reported():
    def ok(label):
        return lambda: Outcome(True, "opened", label, 1.0)

    def bad():
        return Outcome(False, "launch_failed", "nope", 1.0)

    plans = [plan("open_app", "a", ok("a")), plan("open_app", "b", bad),
             plan("open_app", "c", ok("c"))]
    c = Commander(parallel=True, jev=object(), dry_run=True)
    out = c._run_sequence(plans, 0.0)
    assert not out.ok
    assert "b" in out.detail


def test_nothing_after_a_failed_group_runs():
    ran = []

    def ok(label):
        def run():
            ran.append(label)
            return Outcome(True, "opened", label, 1.0)
        return run

    plans = [plan("open_app", "a", lambda: Outcome(False, "launch_failed", "no", 1.0)),
             plan("type_into", "later", ok("later"))]
    c = Commander(parallel=True, jev=object(), dry_run=True)
    c._run_sequence(plans, 0.0)
    assert "later" not in ran, "it carried on past a failure"


def test_a_step_that_raises_is_a_failure_not_a_crash():
    def boom():
        raise RuntimeError("the app exploded")

    plans = [plan("open_app", "a", boom)]
    c = Commander(parallel=True, jev=object(), dry_run=True)
    out = c._run_sequence(plans, 0.0)
    assert not out.ok
    assert "exploded" in out.detail or "failed" in out.detail.lower()
