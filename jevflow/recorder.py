"""Microphone capture that stops when you stop talking.

Push-to-talk means holding a key while thinking, which is worse than it sounds.
Instead: start on the hotkey, and end the clip once the level has been below the
noise floor for a moment. A fixed duration was tried first and was worse.

The noise floor is measured from the room at the start of every clip rather than
hard-coded, because a laptop fan, a desk mic and a headset sit at very different
levels and a fixed threshold silently fails on one of them.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import sounddevice as sd

log = logging.getLogger("jevflow.recorder")

SAMPLE_RATE = 16000          # what Whisper wants; resampling later only loses quality
BLOCK = 512                  # ~32ms
CALIBRATE_S = 0.35           # how long to listen to the room before arming
# Quiet for this long after speech -> stop. 1.4s was far too eager: a pause
# while working out the next words ended the sentence, over and over. 4s was
# then too patient - every clip sat there waiting. 2.5s was settled on after
# using both. Pressing the key again ends it immediately regardless.
SILENCE_HOLD_S = 2.5
MAX_CLIP_S = 30.0
MIN_SPEECH_S = 0.25          # below this it was a key noise, not a sentence
# Hold mode never establishes a measured noise floor, because it never needs
# one, so this is the absolute bar for "something was said". Deliberately low:
# holding a key down already states the intent, and the only thing worth
# rejecting is an accidental tap.
HOLD_SPEECH_PEAK = 0.004
# However loud the room is during calibration, the floor never rises above
# this. Speech sits well above it, so a floor that exceeds it can only mean the
# calibration window caught the speaker's own voice.
FLOOR_CEILING = 0.02


@dataclass
class Clip:
    audio: np.ndarray
    seconds: float
    peak: float
    spoke: bool
    cancelled: bool = False


@dataclass
class Recorder:
    sample_rate: int = SAMPLE_RATE
    silence_hold_s: float = SILENCE_HOLD_S
    max_clip_s: float = MAX_CLIP_S
    on_level: Optional[Callable[[float], None]] = None   # live meter for the UI
    # Push-to-talk. When set, the clip ends when this returns True, and
    # **silence is ignored entirely**. A pause while someone works out what to
    # say next is not the end of a sentence, and treating it as one cut people off
    # mid-thought over and over. Holding a key is a far more honest statement
    # of "I am still talking" than the absence of silence.
    should_stop: Optional[Callable[[], bool]] = None
    # Streaming. Called with the audio captured SO FAR, every
    # `partial_every_s`, so a command can be acted on while it is still being
    # said. Transcribing takes ~120ms, which is far too long to spend on the
    # capture thread, so it runs on a worker and a partial that is still
    # running is SKIPPED rather than queued - a backlog of stale partials is
    # worse than a missed one.
    on_partial: Optional[Callable[[np.ndarray], None]] = None
    partial_every_s: float = 0.6
    _cancel: threading.Event = field(default_factory=threading.Event)
    _finish: threading.Event = field(default_factory=threading.Event)

    def cancel(self) -> None:
        """Throw the recording away. Esc means stop."""
        self._cancel.set()

    def finish(self) -> None:
        """Stop recording and KEEP what was captured.

        Pressing the hotkey again means "I am done, go" - not "forget it".
        It used to call cancel(), which discarded a perfectly good sentence
        and showed "Cancelled".
        """
        self._finish.set()

    def _fire_partial(self, audio: np.ndarray) -> None:
        """Hand the buffer to the listener, on a worker thread.

        Wrapped because a handler that raises must not end the recording - the
        audio is still good and the next partial may well work.
        """
        try:
            self.on_partial(audio)
        except Exception as exc:
            log.warning("[rec] partial handler failed: %s", exc)

    def record(self) -> Clip:
        """Block until the speaker stops, the key is released, or cancel() fires."""
        self._cancel.clear()
        # `_finish` is deliberately NOT cleared. The hotkey fires on its own
        # thread and the session starts on another, so a fast double-tap can
        # call finish() before record() has begun. Clearing it here swallows
        # that press and records for the full thirty seconds instead.
        frames: list[np.ndarray] = []
        levels: list[float] = []
        started = time.time()
        floor = None
        spoke = False
        last_loud = None
        last_partial = 0.0
        partial_worker: Optional[threading.Thread] = None

        def rms(block: np.ndarray) -> float:
            return float(np.sqrt(np.mean(np.square(block))) + 1e-9)

        with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype="float32",
                            blocksize=BLOCK) as stream:
            while True:
                if self._cancel.is_set():
                    return Clip(np.concatenate(frames) if frames else np.zeros(0, "float32"),
                                time.time() - started, max(levels or [0.0]), spoke, cancelled=True)
                if self._finish.is_set():
                    break
                block, overflowed = stream.read(BLOCK)
                if overflowed:
                    log.debug("[rec] input overflow")
                block = block.reshape(-1)
                frames.append(block)
                lvl = rms(block)
                levels.append(lvl)
                if self.on_level:
                    self.on_level(lvl)

                elapsed = time.time() - started
                if self.on_partial is not None and                         elapsed - last_partial >= self.partial_every_s and                         (partial_worker is None or not partial_worker.is_alive()):
                    last_partial = elapsed
                    buf = np.concatenate(frames)
                    partial_worker = threading.Thread(
                        target=self._fire_partial, args=(buf,), daemon=True)
                    partial_worker.start()
                # Push-to-talk: the KEY decides when this ends, not the room.
                # Checked before the noise floor is even established, because
                # a short press should still produce the audio it captured.
                if self.should_stop is not None:
                    try:
                        if self.should_stop():
                            break
                    except Exception as exc:
                        # A stop-check that throws must END the clip, never
                        # extend it. A recorder that cannot be stopped holds
                        # the microphone open until the process is killed.
                        log.warning("[rec] stop check failed (%s); ending clip", exc)
                        break
                    if elapsed >= self.max_clip_s:
                        break          # a key stuck down must not record forever
                    continue

                if floor is None:
                    if elapsed < CALIBRATE_S:
                        continue
                    # Noise floor from the room itself, plus headroom. A speaking
                    # voice sits far above this; a fan does not.
                    #
                    # Measured from a LOW percentile, not a high one. People
                    # start talking the instant they press the key, so this
                    # window usually contains their voice - and a floor taken
                    # from the 75th percentile of their own speech came out
                    # three times louder than they are. Nothing ever crossed
                    # it, `spoke` stayed False, the clip ran the full 30-second
                    # cap, and the answer was "Heard nothing".
                    #
                    # The 20th percentile is the gaps between words, which is
                    # the room. The absolute cap is the backstop for someone
                    # who does not pause at all.
                    quiet = float(np.percentile(levels, 20))
                    floor = min(quiet * 3.0 + 0.004, FLOOR_CEILING)
                    log.debug("[rec] noise floor %.4f (quiet %.4f)", floor, quiet)
                    continue

                if lvl > floor:
                    spoke = True
                    last_loud = time.time()
                elif spoke and last_loud is not None:
                    if time.time() - last_loud >= self.silence_hold_s:
                        break
                if elapsed >= self.max_clip_s:
                    break

        audio = np.concatenate(frames) if frames else np.zeros(0, "float32")
        seconds = len(audio) / float(self.sample_rate)
        peak = max(levels or [0.0])
        if self.should_stop is not None:
            # Hold mode never establishes a noise floor, because it never
            # needed one - so `spoke` is decided here, from whether anything
            # audible was actually captured. The bar is deliberately low: the
            # person held a key down, which already says they meant to speak,
            # and the only thing worth rejecting is an accidental tap.
            spoke = peak > HOLD_SPEECH_PEAK and seconds >= MIN_SPEECH_S
            return Clip(audio, seconds, peak, spoke)
        speech = seconds - (CALIBRATE_S if spoke else 0)
        return Clip(audio, seconds, peak, spoke and speech >= MIN_SPEECH_S)


def list_devices() -> list[dict]:
    out = []
    for i, d in enumerate(sd.query_devices()):
        if d.get("max_input_channels", 0) > 0:
            out.append({"index": i, "name": d["name"], "channels": d["max_input_channels"],
                        "default": i == sd.default.device[0]})
    return out


def wait_for_wake(gate, on_level=None, cancel=None, block: int = 1280,
                  poll_s: float = 0.02) -> bool:
    """Hold the microphone open, feeding audio to a wake-word gate.

    Nothing is stored and nothing is transcribed here - blocks go to a small
    local model and are dropped. That is the whole point of the acoustic gate:
    the room is heard but never written down.

    Returns True when the phrase fires, False when cancelled. The level meter
    keeps running throughout, so the capsule still shows that the microphone is
    open while it waits.
    """
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                        blocksize=block) as stream:
        while True:
            if cancel is not None and cancel.is_set():
                return False
            audio, overflowed = stream.read(block)
            if overflowed:
                log.debug("[wake] input overflow")
            audio = audio.reshape(-1)
            if on_level is not None:
                on_level(float(np.sqrt(np.mean(np.square(audio))) + 1e-9))
            try:
                if gate.feed(audio):
                    return True
            except Exception as exc:
                # A detector that throws must not become a detector that never
                # fires: stop waiting and say so rather than listening forever.
                log.error("[wake] detector failed: %s", exc)
                return False
            time.sleep(poll_s)
