"""Local speech-to-text. Nothing leaves the machine.

Two things in here were learned the hard way:

1. **The Windows CUDA path fix.** ctranslate2 loads `cublas64_12.dll` and
   `cudnn64_9.dll` with a plain `LoadLibrary`, which consults `PATH`.
   `os.add_dll_directory` alone is NOT enough. Without patching `PATH` before
   importing faster_whisper the model loads on CUDA and then dies at the first
   encode.
2. **Priming the vocabulary.** Whisper conditions on `initial_prompt`, so
   seeding the words you actually say is what stops "Jev" becoming "Jeff".
   Add your own names and jargon to VOCABULARY below.

`large-v3` and `large-v3-turbo` both transcribe a short clip in ~0.75 s on the
RTX 3070 Ti; turbo loads faster but mis-hears more, so large-v3 is the default.
"""
from __future__ import annotations

import contextlib
import glob
import logging
import os
import re
import site
import threading
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

log = logging.getLogger("jevflow.stt")

# Words the recogniser is primed with. Anything you say often belongs here -
# this is the cheapest accuracy win available.
VOCABULARY = [
    "Jev", "JevFlow", "TypeSafe", "Minecraft", "Claude",
    "Obsidian", "Telegram", "hotbar", "crosshair",
    # App names the recogniser mangles. "Modrinth" came through as "moderate"
    # and "Minecraft launcher" as "Minecraft watcher" - both cost a correction
    # each time, and seeding the word costs nothing.
    "Modrinth", "Minecraft Launcher", "Prism Launcher", "LocalSend", "CapCut",
    "FileZilla", "HandBrake", "RustDesk", "OneDrive", "Obsidian", "Spotify",
    "screenshot", "clipboard", "terminal", "browser", "window", "repo", "commit",
    # COMMAND OPENERS. Not names - the verbs themselves, because the recogniser
    # swaps them for near-homophones and a swapped verb changes the action.
    # Heard 2026-09-22: "find me the oldest YouTube video" came through as
    # "REMIND me the oldest YouTube video", twice, and a reminder is not a
    # search. Seeding the phrase is the cheapest fix there is.
    "find me", "search for", "look up", "show me", "open up", "close down",
    "switch to", "type in", "play", "pause",
    "YouTube",
]

_CUDA_READY = False
_LOCK = threading.Lock()
_MODEL: dict[str, Any] = {"obj": None, "key": None}


def ensure_cuda_libs() -> list[str]:
    """Put the pip-installed NVIDIA runtime DLLs on PATH for ctranslate2.

    Must run BEFORE faster_whisper is imported.
    """
    global _CUDA_READY
    if _CUDA_READY:
        return []
    roots = list(site.getsitepackages()) + [site.getusersitepackages()]
    dirs = [d for root in roots
            for d in glob.glob(os.path.join(root, "nvidia", "*", "bin"))
            if glob.glob(os.path.join(d, "*.dll"))]
    if dirs:
        os.environ["PATH"] = os.pathsep.join(dirs) + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            for d in dirs:
                with contextlib.suppress(OSError):
                    os.add_dll_directory(d)
    _CUDA_READY = True
    return dirs


def vocabulary_prompt(extra: Optional[list[str]] = None) -> str:
    words = list(VOCABULARY) + list(extra or [])
    return "Transcript mentioning " + ", ".join(words) + "."


# -- what this machine can actually run ---------------------------------------
# The code always fell back to CPU when CUDA was missing, but it fell back with
# `large-v3` still selected - a ~3GB model that takes many seconds per
# utterance on a CPU. A fallback that leaves you with something unusable is not
# a fallback, it is a slower failure.
#
# Requiring a particular graphics card in order to dictate a sentence is not a
# reasonable thing to ask of someone who just downloaded a tool, so "auto"
# looks at the machine and picks something that runs on it.

# Accuracy matters most and there is room for it.
CUDA_MODEL, CUDA_COMPUTE = "large-v3", "float16"
# Small enough to keep up on a CPU. int8 rather than float16 because float16 on
# CPU is slow and on many builds unsupported outright.
CPU_MODEL, CPU_COMPUTE = "base.en", "int8"

_CUDA_CHECKED: Optional[bool] = None


def _probe_cuda() -> bool:
    """Ask ctranslate2 directly. It is what actually runs the model."""
    ensure_cuda_libs()
    import ctranslate2
    return ctranslate2.get_cuda_device_count() > 0


def cuda_available() -> bool:
    """Is there a usable CUDA device? Never raises - a broken probe means no."""
    global _CUDA_CHECKED
    if _CUDA_CHECKED is None:
        try:
            _CUDA_CHECKED = bool(_probe_cuda())
        except Exception as exc:
            log.info("[stt] no usable CUDA device (%s); running on CPU", exc)
            _CUDA_CHECKED = False
    return _CUDA_CHECKED


def pick_runtime(device: str = "auto", model: str = "auto",
                 compute_type: str = "auto") -> tuple[str, str, str]:
    """Choose (device, model, compute_type) for this machine.

    An EXPLICIT choice is always respected. Someone who asks for large-v3 on a
    CPU gets large-v3 on a CPU - being helpful is not the same as overruling.
    The exception is asking for CUDA on a machine without it, which cannot be
    honoured at all and falls back rather than crashing.
    """
    want_cuda = device in ("auto", "cuda")
    have_cuda = cuda_available() if want_cuda else False
    chosen_device = "cuda" if (want_cuda and have_cuda) else "cpu"

    if chosen_device == "cuda":
        chosen_model = CUDA_MODEL if model == "auto" else model
        chosen_compute = CUDA_COMPUTE if compute_type in ("auto", "") else compute_type
    else:
        chosen_model = CPU_MODEL if model == "auto" else model
        # float16 is a GPU format. Asking for it on a CPU is almost always a
        # leftover default rather than an intention.
        chosen_compute = (CPU_COMPUTE if compute_type in ("auto", "", "float16")
                          else compute_type)
        if device == "cuda":
            log.warning("[stt] CUDA was requested but is not available; "
                        "using cpu/%s instead", chosen_model)

    log.info("[stt] running on %s with %s (%s)",
             chosen_device, chosen_model, chosen_compute)
    return chosen_device, chosen_model, chosen_compute


def load_model(model: str = "large-v3", device: str = "cuda", compute_type: str = "float16"):
    """Load once and keep it. Falls back to CPU rather than failing outright."""
    key = f"{model}/{device}/{compute_type}"
    with _LOCK:
        if _MODEL["obj"] is not None and _MODEL["key"] == key:
            return _MODEL["obj"]
        ensure_cuda_libs()
        from faster_whisper import WhisperModel

        t0 = time.time()
        try:
            obj = WhisperModel(model, device=device, compute_type=compute_type)
        except Exception as exc:
            log.warning("[stt] %s unavailable (%s); falling back to CPU", device, exc)
            obj = WhisperModel(model, device="cpu", compute_type="int8")
            key = f"{model}/cpu/int8"
        # One throwaway encode, so the first real utterance is not the one that
        # pays the kernel-compilation cost.
        with contextlib.suppress(Exception):
            segs, _ = obj.transcribe(np.zeros(16000, dtype="float32"), language="en",
                                     beam_size=1, vad_filter=False)
            list(segs)
        log.info("[stt] %s ready in %.1fs", key, time.time() - t0)
        _MODEL["obj"], _MODEL["key"] = obj, key
        return obj


_WORD = re.compile(r"[\w']+")


def _agreeing_prefix(text: str, words: list) -> list:
    """The leading run of `words` that matches `text`, word for word.

    Asking for word alignment can change what the model transcribes - see
    `transcribe` - so the two have to be reconciled rather than trusted. The
    text is authoritative; the words only say how sure the model was. Anything
    after the first disagreement is where the two passes diverged, and is
    dropped: a probability attached to a word that is not in the transcript
    cannot say anything useful about the transcript.
    """
    want = _WORD.findall(text.lower())
    out = []
    for i, w in enumerate(words):
        got = _WORD.findall(str(getattr(w, "word", "")).lower())
        if i >= len(want) or got != [want[i]]:
            break
        out.append(w)
    return out


def transcribe(audio: np.ndarray, sample_rate: int = 16000, *, model: str = "large-v3",
               device: str = "cuda", compute_type: str = "float16",
               extra_vocabulary: Optional[list[str]] = None,
               word_probabilities: bool = False) -> dict[str, Any]:
    """Transcribe mono float32 audio.

    Returns {text, seconds, audio_seconds, words}. `words` is empty unless
    `word_probabilities` is asked for.

    **The text never comes from the aligned pass.** Turning on word timestamps
    changes what the model transcribes, and on the small English models it
    changes it badly: measured over 16 sentences, `base.en` appended invented
    text to ALL SIXTEEN ("Thanks for watching", "Make sure to subscribe to our
    channel"), with and without the VAD. `large-v3` was unaffected, which is
    exactly why this cannot be left to chance - the damage only appears on the
    model people without a GPU are given.

    So the transcript is produced the way it always was, and a second aligned
    pass supplies the probabilities. The two are reconciled by
    `_agreeing_prefix`, so a divergent pass costs a spelling check rather than
    a sentence. It is a second decode, which is why it only happens when
    something downstream is actually going to read the numbers.
    """
    if audio is None or len(audio) == 0:
        return {"text": "", "seconds": 0.0, "audio_seconds": 0.0, "words": []}
    m = load_model(model, device, compute_type)
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    settings = dict(
        language="en", beam_size=5,
        vad_filter=True, vad_parameters={"min_silence_duration_ms": 400},
        initial_prompt=vocabulary_prompt(extra_vocabulary),
        condition_on_previous_text=False,
        temperature=[0.0, 0.2, 0.4], no_speech_threshold=0.6,
    )
    t0 = time.time()
    segments, info = m.transcribe(audio, **settings)
    text = re.sub(r"\s+", " ", " ".join(s.text.strip() for s in segments)).strip()

    words: list[Any] = []
    if word_probabilities and text:
        aligned, _ = m.transcribe(audio, **settings, word_timestamps=True)
        every: list[Any] = []
        for s in aligned:
            every.extend(getattr(s, "words", None) or [])
        words = _agreeing_prefix(text, every)
        if len(words) < len(every):
            log.info("[stt] aligned pass diverged after %d/%d words; "
                     "only the agreeing part is usable", len(words), len(every))

    return {
        "text": text,
        "seconds": round(time.time() - t0, 2),
        "audio_seconds": round(float(getattr(info, "duration", 0) or 0), 1),
        "words": words,
    }
