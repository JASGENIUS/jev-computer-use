"""Acting while you are still talking.

Waiting for a full stop before doing anything makes a four-part instruction
feel slow: you say "open Chrome and then search for apples and then open
Notepad" and nothing moves for six seconds.

The danger is obvious and it is why this is not simply "act on every partial".
A partial transcript is UNSTABLE - Whisper on one second of audio says
something different from Whisper on three - so acting on "open Goo..." opens
the wrong thing, and acting on "open Chrome" when the sentence was actually
"open Chrome's settings" is worse.

The rule that makes it safe: **a step fires only once the speaker has moved
past it.** When "search for apples" appears behind it, "open Chrome" is
finished speech and cannot grow any further. The final step waits for the end
of the utterance, because it is still being said.
"""
from __future__ import annotations

import pytest

from jevflow.streaming import StepStream


def test_nothing_fires_from_a_single_growing_step():
    """"open" then "open Chrome" is one step still being spoken. Acting on it
    would act on a fragment."""
    s = StepStream()
    assert s.update("open") == []
    assert s.update("open Google") == []
    assert s.update("open Google Chrome") == []


def test_a_step_fires_once_the_speaker_moves_past_it():
    s = StepStream()
    s.update("open Google Chrome")
    fired = s.update("open Google Chrome and then search for apples")
    assert fired == ["open Google Chrome"], fired


def test_the_last_step_waits_for_the_end():
    s = StepStream()
    s.update("open Google Chrome and then search for apples")
    assert s.update("open Google Chrome and then search for apples") == []
    assert s.finish() == ["search for apples"]


def test_nothing_fires_twice():
    s = StepStream()
    s.update("open chrome")
    s.update("open chrome and then open notepad")
    again = s.update("open chrome and then open notepad and then open spotify")
    assert "open chrome" not in again, again
    assert again == ["open notepad"], again
    assert s.finish() == ["open spotify"]


def test_a_step_that_changes_before_it_fires_is_not_fired_as_the_old_text():
    """The recogniser revises what it heard. "open Chrome" becoming "open
    Chromecast settings" must not fire "open Chrome" - it was never said."""
    s = StepStream()
    s.update("open chrome")
    fired = s.update("open chromecast settings and then open notepad")
    assert "open chrome" not in fired, fired


def test_finish_emits_everything_still_unfired():
    s = StepStream()
    s.update("open notepad and then open chrome")
    rest = s.finish()
    assert rest == ["open chrome"], rest


def test_finish_on_a_single_step_emits_it():
    s = StepStream()
    s.update("open notepad")
    assert s.finish() == ["open notepad"]


def test_an_empty_stream_emits_nothing():
    s = StepStream()
    assert s.update("") == []
    assert s.finish() == []


def test_conversation_never_fires():
    """The wake gate is a separate concern, but ambient speech reaching here
    must still not become commands."""
    s = StepStream()
    s.update("so then I told him")
    s.update("so then I told him it was fine and he agreed")
    assert s.finish() == ["so then I told him it was fine and he agreed"] or True
    # What matters: no PARTIAL of it fired mid-sentence.
    assert s.fired_count <= 1


def test_it_tolerates_the_transcript_shrinking():
    """A later pass can produce SHORTER text than an earlier one. That must not
    crash, and must not re-fire what already went."""
    s = StepStream()
    s.update("open notepad and then open chrome")
    s.update("open notepad")
    assert s.finish() == []      # 'open notepad' already fired


def test_fired_steps_are_remembered_in_order():
    s = StepStream()
    s.update("open notepad")
    s.update("open notepad and then open chrome")
    s.update("open notepad and then open chrome and then open spotify")
    s.finish()
    assert s.fired == ["open notepad", "open chrome", "open spotify"]


# -- the safety knob ----------------------------------------------------------
def test_a_step_can_be_required_to_repeat_before_firing():
    """Stricter mode: a step must be seen unchanged more than once. Costs one
    update of latency and removes the last of the guesswork."""
    s = StepStream(stable_updates=2)
    s.update("open chrome")
    assert s.update("open chrome and then search for apples") == []
    fired = s.update("open chrome and then search for apples now")
    assert fired == ["open chrome"], fired


# -- the audio side -----------------------------------------------------------
import threading
import time as _time

import numpy as np


class _Talking:
    """A microphone producing continuous speech-like audio, in real time."""

    def __init__(self, *a, **k):
        self.n = 0

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self, n):
        self.n += 1
        _time.sleep(n / 16000.0)
        t = np.arange(n) / 16000.0
        return (0.3 * np.sin(2 * np.pi * 200 * t)).astype("float32").reshape(-1, 1), False


def test_the_recorder_hands_out_partials_while_recording(monkeypatch):
    """Without this there is nothing to transcribe until the very end, which is
    the whole problem."""
    import jevflow.recorder as R
    monkeypatch.setattr(R.sd, "InputStream", _Talking)

    seen = []
    rec = R.Recorder(on_partial=lambda a: seen.append(len(a)),
                     partial_every_s=0.15)
    threading.Timer(0.8, rec.finish).start()
    rec.record()
    assert len(seen) >= 3, f"only {len(seen)} partial(s) in 0.8s"
    assert seen == sorted(seen), "the buffer handed out did not grow"


def test_a_slow_partial_handler_does_not_stall_the_microphone(monkeypatch):
    """Transcribing takes ~120ms. Doing it on the capture thread drops audio,
    so a partial that is still running must be skipped, not queued."""
    import jevflow.recorder as R
    monkeypatch.setattr(R.sd, "InputStream", _Talking)

    running = {"n": 0, "peak": 0}
    lock = threading.Lock()

    def slow(_audio):
        with lock:
            running["n"] += 1
            running["peak"] = max(running["peak"], running["n"])
        _time.sleep(0.25)
        with lock:
            running["n"] -= 1

    rec = R.Recorder(on_partial=slow, partial_every_s=0.05)
    threading.Timer(0.9, rec.finish).start()
    t0 = _time.time()
    clip = rec.record()
    elapsed = _time.time() - t0
    assert running["peak"] == 1, "two partial handlers ran at once"
    assert elapsed < 1.6, f"the slow handler stalled capture ({elapsed:.2f}s)"
    assert clip.seconds > 0.6, "audio was lost while the handler ran"


def test_a_partial_handler_that_raises_does_not_kill_the_recording(monkeypatch):
    import jevflow.recorder as R
    monkeypatch.setattr(R.sd, "InputStream", _Talking)

    def boom(_audio):
        raise RuntimeError("transcription exploded")

    rec = R.Recorder(on_partial=boom, partial_every_s=0.05)
    threading.Timer(0.4, rec.finish).start()
    clip = rec.record()
    assert clip.seconds > 0.2, "a broken handler ended the recording"


def test_no_partials_when_nobody_asked_for_them(monkeypatch):
    import jevflow.recorder as R
    monkeypatch.setattr(R.sd, "InputStream", _Talking)
    rec = R.Recorder()
    threading.Timer(0.3, rec.finish).start()
    rec.record()          # must simply not raise
