"""The path that actually runs, driven end to end with no microphone.

This exists because a refactor changed `Overlay.show()`'s signature while
`app.py` still called the old one: both halves passed their own tests and
pressing the hotkey crashed every time.

The first version of this check (`smoke.py`) then caught the same disease. It
asserted only that *something* was rendered - and a crash renders an error
capsule, so it counted. It printed SMOKE OK for days while the command path
raised TypeError on every single utterance.

So the rules here are:

* every overlay call is bound against the REAL `Overlay` signature,
* the commander is bound against the REAL `Commander.handle` signature,
* and a rendered error is a FAILURE, not a passing call.
"""
from __future__ import annotations

import inspect
import threading
import types

import numpy as np
import pytest

from jevflow.app import VoiceApp
from jevflow.command import Commander, Outcome
from jevflow.overlay import Overlay
from jevflow.recorder import Clip


class RecordingOverlay:
    """Records calls, and validates each one against the real Overlay."""

    def __init__(self, *_a, **_k):
        self.calls: list[tuple[str, tuple, dict]] = []

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        real = getattr(Overlay, name, None)
        if real is None or not callable(real):
            raise AttributeError(name)

        def call(*a, **k):
            inspect.signature(real).bind(self, *a, **k)   # raises on drift
            self.calls.append((name, a, k))
        return call

    @property
    def statuses(self) -> list[str]:
        return [a[0] for n, a, _k in self.calls if n == "set_status" and a]

    @property
    def details(self) -> list[str]:
        return [a[1] for n, a, _k in self.calls if n == "set_status" and len(a) > 1]


class StubCommander:
    """A commander whose handle() must match the real one's signature."""

    def __init__(self, outcome: Outcome):
        self.outcome = outcome
        self.seen: list[tuple] = []

    def handle(self, *a, **k):
        inspect.signature(Commander.handle).bind(self, *a, **k)   # raises on drift
        self.seen.append((a, k))
        return self.outcome


@pytest.fixture
def app_factory(monkeypatch):
    """A real VoiceApp with the microphone, the model and the capsule stubbed.

    Built through the real constructor rather than __new__, so a field added to
    VoiceApp is picked up here instead of silently defaulting to nothing.
    """
    def build(transcript="open a chrome tab and search for apples", **kw):
        import jevflow.app as A
        import jevflow.stt as S

        monkeypatch.setattr(A, "Overlay", RecordingOverlay)
        monkeypatch.setattr(S, "transcribe",
                            lambda *a, **k: {"text": transcript, "seconds": 0.7,
                                             "audio_seconds": 2.0})
        monkeypatch.setattr(A, "Recorder", lambda **_k: types.SimpleNamespace(
            record=lambda: Clip(np.zeros(16000, "float32"), 2.0, 0.5, True),
            cancel=lambda: None))
        kw.setdefault("preload", False)
        kw.setdefault("title", "test")
        return VoiceApp(**kw)
    return build


def test_dictate_session_renders_and_never_errors(app_factory, monkeypatch):
    import jevflow.typing as T
    monkeypatch.setattr(T, "type_text", lambda *a, **k: (True, "Notepad"))
    monkeypatch.setattr(T, "foreground", lambda: (1234, "Notepad"))

    app = app_factory(dictate=True)
    app._session(command=False)

    assert app.overlay.calls, "nothing was rendered at all"
    assert "Error" not in app.overlay.statuses, (
        f"the dictation path raised; statuses were {app.overlay.statuses}")
    assert "Heard" in app.overlay.statuses


def test_command_session_renders_and_never_errors(app_factory, monkeypatch):
    import jevflow.typing as T
    monkeypatch.setattr(T, "foreground", lambda: (1234, "Notepad"))

    app = app_factory()
    app.commander = StubCommander(
        Outcome(True, "searched", "apples in Google Chrome", 120.0, verified=True))
    app._session(command=True)

    assert app.overlay.calls, "nothing was rendered at all"
    assert "Error" not in app.overlay.statuses, (
        f"the command path raised; statuses were {app.overlay.statuses}")
    assert app.commander.seen, "the commander was never reached"


def test_commander_signature_drift_is_caught(app_factory, monkeypatch):
    """Delete the fix and the test must fail: a stale commander is a failure.

    This is the exact bug smoke.py missed - handle() without focused_hwnd.
    """
    import jevflow.typing as T
    monkeypatch.setattr(T, "foreground", lambda: (1234, "Notepad"))

    class StaleCommander:
        def handle(self, text, focused_window=""):        # no focused_hwnd
            return Outcome(True, "searched", "apples", 1.0)

    app = app_factory()
    app.commander = StaleCommander()
    app._session(command=True)
    assert "Error" in app.overlay.statuses, (
        "a commander with the wrong signature was not reported as an error")
