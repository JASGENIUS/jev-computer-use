"""Pressing the key again should FINISH the clip, not throw it away.

Second press used to call cancel(), which discarded everything recorded and
showed "Cancelled". That is the opposite of what pressing the key again means
when you have just finished speaking: you are saying "I'm done, go", not
"forget it".

Esc still discards, because Esc means stop.
"""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from jevflow.recorder import Recorder


class Loud:
    """A microphone that never goes quiet, in real time."""

    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self, n):
        time.sleep(n / 16000.0)
        t = np.arange(n) / 16000.0
        return (0.3 * np.sin(2 * np.pi * 220 * t)).astype("float32").reshape(-1, 1), False


@pytest.fixture
def loud_mic(monkeypatch):
    import jevflow.recorder as R
    monkeypatch.setattr(R.sd, "InputStream", Loud)


def test_finish_keeps_the_audio(loud_mic):
    # Comfortably past the 0.35s calibration window, so there is real speech
    # to keep. A timer barely above it leaves 60ms of audio, which the
    # minimum-speech gate correctly rejects - that is not the case under test.
    rec = Recorder()
    threading.Timer(0.9, rec.finish).start()
    clip = rec.record()
    assert not clip.cancelled, "finishing threw the recording away"
    assert clip.spoke, "the audio it had already captured was lost"
    assert clip.seconds > 0.2


def test_cancel_still_discards(loud_mic):
    rec = Recorder()
    threading.Timer(0.4, rec.cancel).start()
    clip = rec.record()
    assert clip.cancelled, "cancel stopped discarding"


def test_finish_before_anything_was_said_is_not_speech(loud_mic, monkeypatch):
    """Tapping the key twice quickly should not produce a phantom transcript."""
    rec = Recorder()
    rec.finish()
    clip = rec.record()
    assert not clip.spoke


def test_a_longer_silence_window_is_honoured(monkeypatch):
    """The whole point of the revert: a pause while thinking must not end it."""
    import jevflow.recorder as R

    class PauseThenTalk(Loud):
        def __init__(self, *a, **k):
            self.n = 0

        def read(self, n):
            self.n += 1
            time.sleep(n / 16000.0)
            t = np.arange(n) / 16000.0
            # talk, then a 0.5s gap, then talk again
            talking = self.n < 20 or self.n > 35
            data = (0.3 * np.sin(2 * np.pi * 220 * t)) if talking else np.zeros(n)
            return data.astype("float32").reshape(-1, 1), False

    monkeypatch.setattr(R.sd, "InputStream", PauseThenTalk)
    rec = Recorder(silence_hold_s=1.5, max_clip_s=6.0)
    threading.Timer(2.0, rec.finish).start()
    clip = rec.record()
    # A 0.5s gap must NOT have ended it - the clip should run past the pause.
    assert clip.seconds > 1.2, "a pause mid-sentence ended the clip"


def test_the_default_silence_window_is_generous():
    """1.4s was cutting people off mid-thought; 4s left every clip hanging.
    2.5s is what was settled on after using both."""
    import jevflow.recorder as R
    assert 2.0 <= R.SILENCE_HOLD_S <= 3.0
