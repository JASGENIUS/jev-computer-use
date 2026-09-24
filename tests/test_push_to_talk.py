"""Hold the key and talk; let go when you are done.

Silence detection ends the clip when you stop making noise, which is wrong for
the way people actually dictate: a pause in the middle of forming a thought is
not the end of the sentence. It cut people off mid-thought, repeatedly.

Holding a key says "I am still talking" far more reliably than the absence of
silence does. It also makes a wake word redundant for that press - you cannot
hold a key by accident, so there is nothing to disambiguate.
"""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from jevflow import hotkeys
from jevflow.recorder import Recorder


# -- parsing a chord ----------------------------------------------------------
@pytest.mark.parametrize("chord,keys", [
    ("windows+alt", ["windows", "alt"]),
    ("ctrl+windows", ["ctrl", "windows"]),
    ("right ctrl", ["right ctrl"]),
    ("  Windows + Alt  ", ["windows", "alt"]),
])
def test_chord_splits_into_keys(chord, keys):
    assert hotkeys.keys_of(chord) == keys


def test_none_is_not_a_chord():
    assert hotkeys.keys_of("none") == []
    assert hotkeys.keys_of("") == []


def test_still_held_is_true_only_while_every_key_is_down(monkeypatch):
    down = {"windows": True, "alt": True}
    monkeypatch.setattr(hotkeys, "_is_pressed", lambda k: down.get(k, False))
    assert hotkeys.still_held("windows+alt") is True
    down["alt"] = False
    assert hotkeys.still_held("windows+alt") is False


def test_a_chord_nobody_can_read_is_treated_as_released(monkeypatch):
    """If the keyboard backend throws, recording must END, not run forever.
    A stuck recorder holds the microphone open with no way to stop it."""
    def boom(_k):
        raise OSError("no keyboard backend")
    monkeypatch.setattr(hotkeys, "_is_pressed", boom)
    assert hotkeys.still_held("windows+alt") is False


# -- the recorder ------------------------------------------------------------
class FakeStream:
    """A microphone that never goes quiet, so only should_stop can end it.

    It sleeps for the real duration of each block. Without that, blocks arrive
    in microseconds and a few milliseconds of wall time produce four minutes of
    audio - which made an earlier version of these tests fail against a
    perfectly correct `max_clip_s`, because that cap is wall-clock and the test
    was measuring generated samples.
    """

    def __init__(self, *a, **k):
        self.n = 0

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def _wait(self, n):
        time.sleep(n / 16000.0)

    def read(self, n):
        self.n += 1
        self._wait(n)
        # Loud, constant tone. Silence detection would never fire on this.
        t = np.arange(n) / 16000.0
        return (0.3 * np.sin(2 * np.pi * 220 * t)).astype("float32").reshape(-1, 1), False


def test_hold_mode_records_until_the_key_is_released(monkeypatch):
    import jevflow.recorder as R
    monkeypatch.setattr(R.sd, "InputStream", FakeStream)
    released = threading.Event()
    threading.Timer(0.35, released.set).start()

    clip = Recorder(should_stop=released.is_set).record()
    assert clip.spoke
    assert not clip.cancelled
    assert clip.seconds > 0.1, "stopped immediately instead of recording"


def test_hold_mode_ignores_silence(monkeypatch):
    """The whole point. A long pause mid-sentence must NOT end the clip."""
    import jevflow.recorder as R

    class Quiet(FakeStream):
        def read(self, n):
            self.n += 1
            self._wait(n)
            return np.zeros((n, 1), dtype="float32"), False

    monkeypatch.setattr(R.sd, "InputStream", Quiet)
    released = threading.Event()
    threading.Timer(0.6, released.set).start()
    t0 = time.time()
    Recorder(should_stop=released.is_set, silence_hold_s=0.05).record()
    assert time.time() - t0 > 0.4, "silence ended the clip while the key was held"


def test_without_should_stop_silence_still_ends_the_clip(monkeypatch):
    """Hold mode must not change the behaviour of the normal press-and-speak
    path, which many people prefer and which dictation has always used."""
    import jevflow.recorder as R

    class Quiet(FakeStream):
        def read(self, n):
            self.n += 1
            self._wait(n)
            # Loud for the first ~0.8s (past the 0.35s calibration), then quiet.
            loud = self.n < 25
            t = np.arange(n) / 16000.0
            data = (0.3 * np.sin(2 * np.pi * 220 * t)) if loud else np.zeros(n)
            return data.astype("float32").reshape(-1, 1), False

    monkeypatch.setattr(R.sd, "InputStream", Quiet)
    t0 = time.time()
    clip = Recorder(silence_hold_s=0.15).record()
    assert time.time() - t0 < 5.0
    assert clip.spoke


def test_hold_mode_still_respects_the_maximum_clip(monkeypatch):
    """A key that gets stuck down must not record forever."""
    import jevflow.recorder as R
    monkeypatch.setattr(R.sd, "InputStream", FakeStream)
    t0 = time.time()
    clip = Recorder(should_stop=lambda: False, max_clip_s=0.3).record()
    # The cap is wall-clock, which is what matters for a key stuck down.
    assert time.time() - t0 < 3.0, "a stuck key recorded forever"
    assert clip.seconds <= 3.0


# -- the noise floor, which had a real bug in it ------------------------------
def test_speaking_immediately_is_still_detected(monkeypatch):
    """The floor is measured from the room during the first fraction of a
    second. If someone starts talking straight away - which is the normal thing
    to do after pressing a key - that window contains their VOICE, and a floor
    set from it was three times louder than they are.

    Speech then never crossed it, `spoke` stayed False, the clip ran the full
    30-second cap, and the answer was "Heard nothing".
    """
    import jevflow.recorder as R

    class TalkingFromTheStart(FakeStream):
        def read(self, n):
            self.n += 1
            self._wait(n)
            loud = self.n < 30          # talking from block one
            t = np.arange(n) / 16000.0
            data = (0.25 * np.sin(2 * np.pi * 200 * t)) if loud else np.zeros(n)
            return data.astype("float32").reshape(-1, 1), False

    monkeypatch.setattr(R.sd, "InputStream", TalkingFromTheStart)
    t0 = time.time()
    clip = Recorder(silence_hold_s=0.2, max_clip_s=8.0).record()
    assert clip.spoke, "speech starting immediately was never detected"
    assert time.time() - t0 < 5.0, "ran to the cap instead of ending on silence"


def test_a_quiet_room_still_sets_a_usable_floor(monkeypatch):
    """The guard for the fix: a genuinely silent start must not produce a floor
    so low that the fan counts as speech."""
    import jevflow.recorder as R

    class QuietThenTalk(FakeStream):
        def read(self, n):
            self.n += 1
            self._wait(n)
            t = np.arange(n) / 16000.0
            if self.n < 15:
                data = 0.001 * np.sin(2 * np.pi * 60 * t)    # room hum only
            elif self.n < 40:
                data = 0.25 * np.sin(2 * np.pi * 200 * t)    # speech
            else:
                data = 0.001 * np.sin(2 * np.pi * 60 * t)
            return data.astype("float32").reshape(-1, 1), False

    monkeypatch.setattr(R.sd, "InputStream", QuietThenTalk)
    clip = Recorder(silence_hold_s=0.2, max_clip_s=8.0).record()
    assert clip.spoke
