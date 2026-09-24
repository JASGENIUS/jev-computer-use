"""Staying on without acting on every conversation in the room.

Continuous mode hears everything. That was honest but noisy, and the fix is a
wake word - with one constraint that is not negotiable: **the capsule must stay
visible the whole time the microphone is open.** It is the only signal that
anything is listening, so a gate that hides the panel to look quiet would trade
noise for something much worse. In wake mode the capsule sits in a dim resting
state instead of disappearing.

There are two gates, and they are genuinely different things:

`TranscriptGate` (the default) still transcribes each utterance, then discards
it unless it began with the wake word. Nothing is displayed, nothing is sent to
Jev, and nothing runs. It works today, with the actual word "Jev", and costs one
local transcription per utterance - which never leaves the machine either way.

`AcousticGate` is a real wake-word detector: audio is matched against a small
ONNX model and nothing is transcribed at all until the phrase is heard. That is
the stronger version of the promise. The catch is that openWakeWord ships
pretrained models for "alexa", "hey jarvis", "hey mycroft" and "hey rhasspy" -
and not for "Jev", which would have to be trained. So the acoustic gate is real
and usable, just not yet with the right word. Asking it for a phrase it has no
model for raises rather than returning a detector that can never fire: a gate
that silently never opens turns the whole tool off with nothing to find.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

log = logging.getLogger("jevflow.wake")

# How long a bare "Jev" keeps the door open for the sentence that follows.
ARM_TTL_S = 12.0

# What the recogniser actually produces for "Jev". `stt.VOCABULARY` primes the
# model so the first spelling is overwhelmingly the common one, but a smaller
# model still slips, and a wake word that only works most of the time reads as
# a broken program rather than a mishearing.
HOMOPHONES = ("jev", "jeb", "jeff", "jev's", "gev", "chev", "jave")

_GREETING = r"(?:hey|ok(?:ay)?|hi|yo)"


def _wake_pattern(word: str) -> re.Pattern:
    forms = sorted({word.lower(), *HOMOPHONES} if word.lower() == "jev" else {word.lower()},
                   key=len, reverse=True)
    alt = "|".join(re.escape(f) for f in forms)
    # Anchored to the START of the utterance on purpose. "What did Jev say"
    # mentions the name; it does not address it.
    return re.compile(rf"^\s*(?:{_GREETING}\s+)?(?:{alt})\b[\s,.:!?-]*", re.I)


@dataclass(frozen=True)
class Heard:
    awake: bool
    command: str = ""          # the utterance with the wake word removed
    armed: bool = False        # a bare wake word: the next sentence is the command


@dataclass
class TranscriptGate:
    """Wake on the transcript. Cheap, exact, and available today."""

    word: str = "jev"
    arm_ttl_s: float = ARM_TTL_S
    _pattern: re.Pattern = field(init=False)
    _armed_until: float = 0.0

    def __post_init__(self) -> None:
        self._pattern = _wake_pattern(self.word)

    @property
    def label(self) -> str:
        return f"say '{self.word.title()}'"

    def is_armed(self, now: Optional[float] = None) -> bool:
        return (now or time.time()) < self._armed_until

    def disarm(self) -> None:
        self._armed_until = 0.0

    def check(self, transcript: str, now: Optional[float] = None) -> Heard:
        now = now if now is not None else time.time()
        text = (transcript or "").strip()
        if not text:
            return Heard(False)

        m = self._pattern.match(text)
        if m:
            rest = text[m.end():].strip(" ,.:!?-")
            if rest:
                self._armed_until = 0.0        # a whole instruction, act on it now
                return Heard(True, rest)
            # Just the name: hold the door open for the next sentence.
            self._armed_until = now + self.arm_ttl_s
            return Heard(True, "", armed=True)

        if self.is_armed(now):
            self._armed_until = 0.0            # one sentence per wake, not a session
            return Heard(True, text)
        return Heard(False)


@dataclass
class AcousticGate:
    """Wake on the audio itself, so nothing is transcribed until it fires."""

    phrase: str = "hey_jarvis"
    threshold: float = 0.5
    _model: object = None

    def __post_init__(self) -> None:
        import openwakeword
        from openwakeword.model import Model

        available = self.available()
        if self.phrase not in available:
            raise ValueError(
                f"no wake-word model for {self.phrase!r}. "
                f"openWakeWord ships: {', '.join(available)}. "
                "A model for 'Jev' has to be trained; until then use the "
                "transcript gate, or pick one of the above.")
        self._model = Model(wakeword_models=[self.phrase], inference_framework="onnx")
        log.info("[wake] acoustic gate on %r", self.phrase)

    @staticmethod
    def available() -> list[str]:
        """Phrases with a model on disk right now. Never a guess."""
        import glob
        import os

        import openwakeword
        base = os.path.join(os.path.dirname(openwakeword.__file__), "resources", "models")
        names = []
        for f in glob.glob(os.path.join(base, "*.onnx")):
            stem = os.path.basename(f).rsplit(".onnx", 1)[0]
            if stem in ("embedding_model", "melspectrogram", "silero_vad"):
                continue
            names.append(re.sub(r"_v\d+\.\d+$", "", stem))
        return sorted(set(names))

    @property
    def label(self) -> str:
        return f"say '{self.phrase.replace('_', ' ')}'"

    def is_armed(self, now: Optional[float] = None) -> bool:
        """An acoustic gate has no armed state: every command is preceded by
        the phrase, so there is never a sentence waiting on a previous one."""
        return False

    def feed(self, audio: np.ndarray) -> bool:
        """Push 16 kHz mono audio. True the moment the phrase is heard."""
        if self._model is None or audio is None or len(audio) == 0:
            return False
        pcm = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
        pcm = (pcm * 32767).astype(np.int16)
        scores = self._model.predict(pcm)
        best = max(scores.values()) if scores else 0.0
        if best >= self.threshold:
            log.info("[wake] heard %r (%.2f)", self.phrase, best)
            self._model.reset()
            return True
        return False


def build(engine: str = "transcript", word: str = "jev",
          threshold: float = 0.5) -> Optional[object]:
    """The gate named on the command line, or None for "listen to everything"."""
    engine = (engine or "none").lower()
    if engine in ("none", "off", ""):
        return None
    if engine == "transcript":
        return TranscriptGate(word=word)
    if engine == "acoustic":
        return AcousticGate(phrase=word, threshold=threshold)
    raise ValueError(f"unknown wake engine {engine!r}: use none, transcript or acoustic")
