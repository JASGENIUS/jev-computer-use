"""The streaming path, driven through the real session with no microphone.

Two things have to be true and they pull against each other:

* a step must actually run BEFORE the sentence ends, or the feature does
  nothing;
* nothing may run TWICE, because the final transcript still contains the words
  of everything that already fired.
"""
from __future__ import annotations

import threading
import time
import types

import numpy as np
import pytest

from jevflow.command import Outcome
from jevflow.recorder import Clip
from test_session import RecordingOverlay, StubCommander


@pytest.fixture
def streaming_app(monkeypatch):
    """A session whose microphone and model replay a sentence arriving in parts."""
    def build(parts, final, **kw):
        import jevflow.app as A
        import jevflow.stt as S
        import jevflow.typing as T
        from jevflow.app import VoiceApp

        said = list(parts)
        monkeypatch.setattr(A, "Overlay", RecordingOverlay)
        monkeypatch.setattr(T, "foreground", lambda: (1234, "Notepad"))

        def transcribe(audio, *a, **k):
            # Each partial returns the next growing fragment; the last call
            # (from the main path) returns the finished sentence.
            text = said.pop(0) if said else final
            return {"text": text, "seconds": 0.1, "audio_seconds": 1.0}
        monkeypatch.setattr(S, "transcribe", transcribe)

        class Mic:
            def __init__(self, **kw2):
                self.on_partial = kw2.get("on_partial")

            def record(self):
                for _ in range(len(parts)):
                    if self.on_partial:
                        self.on_partial(np.zeros(16000, "float32"))
                return Clip(np.zeros(16000, "float32"), 2.0, 0.4, True)

            def cancel(self):
                pass

            def finish(self):
                pass

        monkeypatch.setattr(A, "Recorder", Mic)
        app = VoiceApp(title="stream-test", preload=False, stream=True,
                       clean_speech=False, **kw)
        app.commander = StubCommander(Outcome(True, "opened", "done", 10.0))
        return app
    return build


def test_a_step_runs_before_the_sentence_ends(streaming_app):
    app = streaming_app(
        parts=["open chrome",
               "open chrome and then open notepad",
               "open chrome and then open notepad and then open spotify"],
        final="open chrome and then open notepad and then open spotify")
    app._once(command=True, first=True)
    ran = [c[0][0] for c in app.commander.seen]
    assert "open chrome" in ran, f"nothing streamed: {ran}"
    assert "open notepad" in ran, ran


def test_nothing_runs_twice(streaming_app):
    app = streaming_app(
        parts=["open chrome",
               "open chrome and then open notepad",
               "open chrome and then open notepad and then open spotify"],
        final="open chrome and then open notepad and then open spotify")
    app._once(command=True, first=True)
    ran = [c[0][0] for c in app.commander.seen]
    joined = " ".join(ran)
    assert joined.count("open chrome") == 1, ran
    assert joined.count("open notepad") == 1, ran
    assert joined.count("open spotify") == 1, ran


def test_the_last_step_still_runs(streaming_app):
    app = streaming_app(
        parts=["open chrome", "open chrome and then open spotify"],
        final="open chrome and then open spotify")
    app._once(command=True, first=True)
    ran = " ".join(c[0][0] for c in app.commander.seen)
    assert "spotify" in ran, "the final step was dropped"


def test_a_single_instruction_is_not_streamed(streaming_app):
    """One step has nothing behind it, so it must wait - and must still run."""
    app = streaming_app(parts=["open", "open notepad"], final="open notepad")
    app._once(command=True, first=True)
    ran = [c[0][0] for c in app.commander.seen]
    assert ran == ["open notepad"], ran


def test_streaming_off_behaves_exactly_as_before(streaming_app):
    app = streaming_app(parts=[], final="open chrome and then open notepad")
    app.stream = False
    app._once(command=True, first=True)
    ran = [c[0][0] for c in app.commander.seen]
    assert ran == ["open chrome and then open notepad"], ran
