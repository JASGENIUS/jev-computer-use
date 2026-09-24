"""Wake mode, driven through the real session loop.

The invariant under test is the one from the handoff's open questions: the
capsule must be visible the whole time the microphone is open, because it is the
only signal the user gets. A wake gate that went quiet by hiding the panel would
be trading noise for a listening microphone with no indicator - which is worse
than the noise it set out to fix.
"""
from __future__ import annotations

import types

import numpy as np
import pytest

from jevflow import wake
from jevflow.command import Outcome
from jevflow.recorder import Clip
from test_session import RecordingOverlay, StubCommander


@pytest.fixture
def wake_app(monkeypatch):
    def build(transcripts, **kw):
        import jevflow.app as A
        import jevflow.stt as S
        import jevflow.typing as T
        from jevflow.app import VoiceApp

        said = list(transcripts)
        monkeypatch.setattr(A, "Overlay", RecordingOverlay)
        monkeypatch.setattr(T, "foreground", lambda: (1234, "Notepad"))
        monkeypatch.setattr(S, "transcribe", lambda *a, **k: {
            "text": said.pop(0) if said else "", "seconds": 0.4, "audio_seconds": 1.0})
        monkeypatch.setattr(A, "Recorder", lambda **_k: types.SimpleNamespace(
            record=lambda: Clip(np.zeros(16000, "float32"), 1.0, 0.4, True),
            cancel=lambda: None))
        kw.setdefault("preload", False)
        app = VoiceApp(title="wake-test", wake=wake.TranscriptGate(word="jev"), **kw)
        app.commander = StubCommander(Outcome(True, "opened", "Notepad", 90.0,
                                              verified=True))
        return app
    return build


def test_an_unaddressed_sentence_never_reaches_the_commander(wake_app):
    app = wake_app(["so then I told him it was fine"])
    app._once(command=True, first=True)
    assert not app.commander.seen, "it acted on a conversation in the room"


def test_an_unaddressed_sentence_is_not_displayed(wake_app):
    """Not shown either. Putting the room's words on screen is the noise."""
    app = wake_app(["so then I told him it was fine"])
    app._once(command=True, first=True)
    assert "Heard" not in app.overlay.statuses
    assert "so then I told him it was fine" not in app.overlay.details


def test_the_capsule_stays_visible_while_it_waits(wake_app):
    """The microphone is open. Something must say so."""
    app = wake_app(["nothing to do with jev at all"])
    app._once(command=True, first=True)
    assert "Waiting" in app.overlay.statuses
    shown = [n for n, _a, _k in app.overlay.calls if n in ("show", "hide")]
    assert shown and shown[0] == "show"
    assert "hide" not in shown, "it hid the panel while still listening"


def test_being_addressed_runs_the_command_without_the_wake_word(wake_app):
    app = wake_app(["jev open notepad"])
    app._once(command=True, first=True)
    assert app.commander.seen, "it ignored a sentence addressed to it"
    text = app.commander.seen[0][0][0]
    assert text == "open notepad", f"the wake word was left in: {text!r}"


def test_a_bare_name_waits_for_the_next_sentence(wake_app):
    app = wake_app(["jev", "open notepad"])
    app._once(command=True, first=True)
    assert not app.commander.seen
    assert "Yes?" in app.overlay.statuses
    app._once(command=True, first=False)
    assert app.commander.seen
    assert app.commander.seen[0][0][0] == "open notepad"


def test_dictation_is_never_gated(wake_app, monkeypatch):
    """The dictation key is a deliberate press. Making it need a wake word too
    would mean saying the name before every sentence you wanted typed."""
    import jevflow.typing as T
    monkeypatch.setattr(T, "type_text", lambda *a, **k: (True, "Notepad"))
    app = wake_app(["the meeting is at four"], dictate=True)
    app._once(command=False, first=True)
    assert "Heard" in app.overlay.statuses


def test_no_gate_means_everything_is_heard(monkeypatch):
    import jevflow.app as A
    import jevflow.stt as S
    import jevflow.typing as T
    from jevflow.app import VoiceApp

    monkeypatch.setattr(A, "Overlay", RecordingOverlay)
    monkeypatch.setattr(T, "foreground", lambda: (1234, "Notepad"))
    monkeypatch.setattr(S, "transcribe", lambda *a, **k: {
        "text": "open notepad", "seconds": 0.4, "audio_seconds": 1.0})
    monkeypatch.setattr(A, "Recorder", lambda **_k: types.SimpleNamespace(
        record=lambda: Clip(np.zeros(16000, "float32"), 1.0, 0.4, True),
        cancel=lambda: None))
    app = VoiceApp(title="no-gate", preload=False, wake=None)
    app.commander = StubCommander(Outcome(True, "opened", "Notepad", 90.0))
    app._once(command=True, first=True)
    assert app.commander.seen, "the unGATED path stopped working"


def test_a_filler_before_the_wake_word_does_not_deafen_it(wake_app):
    """The ordering invariant. The gate is anchored to the START of the
    utterance, so "um, jev, open notepad" does not begin with the wake word
    until the "um" has gone. Clean first or the wake word stops working the
    moment someone hesitates - which is most of the time.
    """
    app = wake_app(["um, jev, open notepad"])
    app._once(command=True, first=True)
    assert app.commander.seen, "a hesitation before the wake word deafened it"
    assert app.commander.seen[0][0][0] == "open notepad"


def test_cleaning_can_be_turned_off(wake_app):
    app = wake_app(["jev open, um, notepad"], clean_speech=False)
    app._once(command=True, first=True)
    assert app.commander.seen
    assert "um" in app.commander.seen[0][0][0]
