"""Minimal Jev client, and the one decision JevFlow asks it to make.

Jev answers typed questions - choice, score, noul - with calibrated
probabilities. It does not emit text, so "open Chrome" cannot be turned into a
free-form command string. It has to become a choice among options that already
exist, which is why `apps.py` does the wide pass first.

The boundary is kept deliberately: **Jev decides which app was meant. It does
not decide whether the app may be launched, and it does not launch anything.**
Both of those are in `command.py`, in code, where they can be read.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import requests

log = logging.getLogger("jevflow.jev")

DEFAULT_URL = "https://api.typesafe.ai/v1/systemone"
DEFAULT_MODEL = "jev-1.13.0"
NONE_OF_THESE = "none_of_these"
# What stands in for the word being asked about, so the recogniser's own
# guess does not sit in the sentence anchoring the answer.
BLANK = "____"

# Spelled out, because this file has twice been corrupted by a shell
# eating the backslash in an escaped newline.
NEWLINE = chr(10)


def load_key(explicit: Optional[str] = None) -> str:
    """Key from the JEV_API_KEY environment variable, or a .env in the repo root."""
    if explicit:
        return explicit
    if os.environ.get("JEV_API_KEY"):
        return os.environ["JEV_API_KEY"]
    for env in (Path(__file__).resolve().parents[1] / ".env",):
        if env.is_file():
            for line in env.read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.startswith("JEV_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


class JevError(RuntimeError):
    pass


@dataclass
class Jev:
    api_key: str = field(default_factory=load_key)
    url: str = DEFAULT_URL
    model: str = DEFAULT_MODEL
    timeout_s: float = 10.0

    def __post_init__(self) -> None:
        if not self.api_key:
            raise JevError(
                "No JEV_API_KEY found."
                + NEWLINE + NEWLINE +
                "JevFlow needs a Jev API key to decide which app you meant."
                + NEWLINE +
                "  1. Get a key at https://typesafe.ai"
                + NEWLINE +
                "  2. Then either set the environment variable:"
                + NEWLINE +
                '       setx JEV_API_KEY "your-key-here"'
                + NEWLINE +
                "     or copy .env.example to .env and fill it in."
                + NEWLINE + NEWLINE +
                "Dictation does not need a key - only computer use does.")
        self._s = requests.Session()
        self._s.headers.update({"Authorization": f"Bearer {self.api_key}",
                                "Content-Type": "application/json",
                                "User-Agent": "JevFlow/0.1"})

    def __repr__(self) -> str:
        return f"Jev(model={self.model!r})"

    def ask(self, state: dict[str, Any], questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        t0 = time.perf_counter()
        r = self._s.post(self.url, data=json.dumps(
            {"model": self.model, "state": state, "questions": questions}), timeout=self.timeout_s)
        if r.status_code in (401, 403):
            raise JevError(f"Jev rejected the key (HTTP {r.status_code})")
        if r.status_code >= 400:
            raise JevError(f"Jev HTTP {r.status_code}: {r.text[:200]}")
        data = r.json()
        answers = data.get("answers") or {}
        ms = (time.perf_counter() - t0) * 1000
        log.info("[jev] %d question(s) in %.0f ms", len(questions), ms)
        return {"answers": answers, "latency_ms": ms}


def build_questions(candidates: list[str]) -> dict[str, dict[str, Any]]:
    """Two judgments, asked together: what kind of request, and which candidate.

    The criteria below ARE the action space. Jev cannot answer with anything
    that is not listed, and `command.py` refuses any verb it has no policy for,
    so adding a line here is not enough to make something runnable - which is
    deliberate. Nothing destructive appears, and the one verb that touches real
    work (close_window) only ever sends WM_CLOSE.
    """
    qs: dict[str, dict[str, Any]] = {
        "intent": {
            "type": "choice",
            "instructions": "What is the speaker asking for?",
            "criteria": {
                "open_app": "start or open an application or a website, with nothing to look up",
                "web_search": "search the web or look something up, in a browser",
                "switch_to": "bring an application that is ALREADY open to the front",
                "close_window": "close, quit or shut something that is already open",
                "type_into": "type specific dictated words into a named application",
                "media_control": "play, pause, skip a track, or change the volume",
                "type_text": "dictate text to be typed where the cursor already is",
                "unclear": "the request is not one of the above, or is too vague to act on",
            },
        },
        "certain": {
            "type": "noul",
            "instructions": "The transcript is a clear, specific instruction that can be acted on now.",
        },
    }
    if candidates:
        qs["app"] = {
            "type": "choice",
            "instructions": "Which of these applications or websites did the speaker mean?",
            "criteria": {**{c: f"the application or website called {c}" for c in candidates},
                         NONE_OF_THESE: "none of these is what was asked for"},
        }
    return qs


def build_state(transcript: str, candidates: list[str], focused_window: str,
                search_query: str = "", already_open: Optional[list[str]] = None) -> dict[str, Any]:
    """Compact by construction: only what varies, and nothing repeated in the questions.

    `already_open` is measured from real windows before every question. It is
    the difference between "switch to Obsidian" and "open Obsidian", and a model
    cannot see a taskbar - so without being told, it guessed the common case
    every time and switch_to was effectively unreachable.
    """
    return {
        "transcript": transcript,
        "installed_candidates": candidates,
        "already_open_right_now": list(already_open or []),
        "focused_window": focused_window,
        "search_query_found_in_transcript": search_query or None,
        "notes": [
            "The transcript comes from speech recognition and may contain small errors.",
            "Only the listed applications and websites are reachable; anything else cannot be opened.",
            "A request for something not in the list should be answered with none_of_these.",
            "Asking to open a browser AND look something up is web_search, not open_app.",
            "'close this' or 'close that' with no name means the focused window.",
            "switch_to applies only to something in already_open_right_now; "
            "anything else, however clearly named, is open_app.",
            "A bare 'yes' or 'no' is an answer to a question, not an instruction: unclear.",
        ],
    }


def build_homophone_question(heard: str, options: list[str],
                             sentence: str) -> dict[str, dict[str, Any]]:
    """Which spelling fills the blank.

    This is what Jev is FOR. The recogniser heard a sound and had to guess a
    spelling with almost no context; Jev gets the whole sentence and a closed
    list of the spellings that sound like it, and returns a calibrated choice.

    It cannot invent a word. The criteria ARE the options, so the worst it can
    do is pick the wrong one of three - and `homophones.apply_choice` refuses
    any answer less certain than what the recogniser already had.
    """
    return {
        "word": {
            "type": "choice",
            "instructions": (f'The word at {BLANK} sounded like "{heard}". '
                             "Which of these spellings belongs there?"),
            "criteria": {o: f'the word "{o}"' for o in options},
        },
        "certain": {
            "type": "noul",
            "instructions": "The sentence makes it clear which spelling was meant.",
        },
    }


def build_homophone_state(heard: str, sentence: str, probability: float) -> dict[str, Any]:
    """The context, with the doubtful word REMOVED.

    Leaving the transcribed spelling in the sentence anchors the answer to it,
    and that was measured rather than assumed: "I left it over there" came back
    `there` at 1.00, while the same sentence transcribed "I left it over their"
    came back `their` at 0.50 with `there` at 0.49. The only thing that changed
    was the guess handed over, and it moved the answer half a point.

    So the sentence arrives with a blank where the word was. What it SOUNDED
    like is still given, separately - that is evidence. The spelling the
    recogniser happened to pick is not evidence about anything, and is withheld.
    """
    return {
        "sentence_with_one_word_blanked": sentence,
        "the_word_sounded_like": heard,
        "recogniser_confidence_in_its_own_guess": round(float(probability), 3),
        "notes": [
            "The sentence comes from speech recognition of a single speaker.",
            f"{BLANK} marks the word in question; everything else is as transcribed.",
            "These spellings are identical in sound; only the sentence distinguishes them.",
            "If the sentence does not make it clear, say so with a low certainty "
            "rather than guessing - the original spelling is kept when unsure.",
        ],
    }


def build_filler_question(word: str, roles: dict[str, str]) -> dict[str, dict[str, Any]]:
    """What job is the marked word doing?

    Asked as a ROLE, not as "should this be deleted". The difference matters:
    exactly one role means delete, so a confused answer can only fail towards
    keeping the word, which is the direction a deletion has to fail in. Asking
    the question the other way round makes every wrong answer destructive.
    """
    return {
        "role": {
            "type": "choice",
            "instructions": (f'The word marked with >>> is "{word}". '
                             "What is it doing in this sentence?"),
            "criteria": dict(roles),
        },
    }


def build_filler_state(word: str, marked_sentence: str) -> dict[str, Any]:
    """The sentence with the ONE occurrence under question marked.

    A sentence containing four "like"s otherwise asks four identical questions
    and gets four identical answers, which is one decision applied four times.
    """
    return {
        "sentence": marked_sentence,
        "word_in_question": word,
        "notes": [
            "This is dictated speech transcribed by a recogniser, so it is "
            "spoken English: unfinished sentences and repetition are normal.",
            "Only the occurrence marked with >>> is being asked about. The "
            "same word may appear elsewhere doing a different job.",
            "Speech has more verbal tics than writing does, but a word that "
            "is carrying meaning is not a tic however casual it sounds.",
        ],
    }
