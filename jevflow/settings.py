"""One place both tools read their settings from.

Everything lived in the .bat files, which meant changing anything meant editing
a batch script - so nothing ever got tuned, and the two launchers drifted apart
without anyone noticing. The dashboard needs somewhere to write and the
launchers need somewhere to read, and it has to be the same somewhere or the
same drift happens again.

Two rules:

**A broken file behaves as defaults.** A settings file that can stop the tool
starting is worse than no settings file, so an unreadable, corrupt or
hand-mangled store falls back silently and says so in the log.

**Unknown keys are dropped, known ones are kept.** A file from an older version
with one stale key must not lose the nine good ones alongside it.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger("jevflow.settings")

STORE = Path(__file__).resolve().parents[1] / "settings.json"

# The defaults ARE the documentation for what each tool does. Every one of
# these was chosen for a reason recorded elsewhere in the codebase.
DEFAULTS: dict[str, dict[str, Any]] = {
    "jevflow": {
        "hotkey": "windows+alt",
        "model": "auto",                # large-v3 on a GPU, base.en on CPU
        "silence_hold_s": 2.5,          # settled on after trying 1.4 and 4.0
        "clean_speech": True,
        "clean_aggressive": False,
        "hold_to_talk": False,
        # The only part of dictation that uses the network. Off by default:
        # "runs entirely on your machine" has to stay true unless asked, and
        # it is not free either - a second local decode, measured at +0.15s a
        # clip on a GPU and +3.4s on a CPU, where large-v3 is not the model
        # in use and there is almost never anything for it to fix.
        "fix_homophones": False,
        # Ask Jev before deleting an ambiguous filler. Same network caveat.
        # One round trip per candidate, capped at 6 a sentence.
        "fix_fillers": False,
        "accent": "#6cc4ff",
        "badge": "JevFlow",
    },
    "jcu": {
        # Right Ctrl ALONE was firing on left control too, so it now needs
        # both right-hand keys together. Neither is in Win+Alt, so the two
        # tools still cannot collide.
        "hotkey": "right ctrl+right shift",
        "model": "auto",                # small enough that both fit in 4GB of VRAM
        "silence_hold_s": 2.5,
        "clean_speech": True,
        "clean_aggressive": False,
        "stream": True,                 # act as you move past each instruction
        "partial_every_s": 0.35,        # transcribing costs ~120ms, so this fits
        "wake": "none",                 # back on when the acoustic model exists
        "wake_word": "jev",
        "dry_run": False,
        "continuous": True,
        "accent": "#f7bf54",
        "badge": "JCU",
    },
}

_CACHE: dict[str, dict[str, Any]] = {}


def _coerce(default: Any, value: Any) -> Any:
    """Keep a value only if it is the shape the default says it should be."""
    if isinstance(default, bool):
        return bool(value) if isinstance(value, bool) else default
    if isinstance(default, float):
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
    if isinstance(default, int):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default
    if isinstance(default, str):
        return value if isinstance(value, str) else default
    return value


def load() -> dict[str, dict[str, Any]]:
    """Settings, with defaults filled in for anything missing or wrong."""
    if _CACHE:
        return _CACHE
    raw: dict = {}
    try:
        raw = json.loads(STORE.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("not a mapping")
    except FileNotFoundError:
        raw = {}
    except Exception as exc:
        log.warning("[settings] %s is unusable (%s); using defaults", STORE, exc)
        raw = {}

    merged: dict[str, dict[str, Any]] = {}
    for tool, defaults in DEFAULTS.items():
        got = raw.get(tool) if isinstance(raw.get(tool), dict) else {}
        # Only keys we know about survive. A file from an older version with one
        # stale key must not take the good ones down with it.
        merged[tool] = {k: _coerce(v, got.get(k, v)) for k, v in defaults.items()}
    _CACHE.update(merged)
    return _CACHE


def save(data: dict[str, dict[str, Any]] | None = None) -> None:
    payload = data if data is not None else load()
    try:
        STORE.parent.mkdir(parents=True, exist_ok=True)
        STORE.write_text(json.dumps(payload, indent=2, sort_keys=True),
                         encoding="utf-8")
    except Exception as exc:
        # Never raise. Failing to remember a preference is small; failing to
        # start because a preference could not be written is not.
        log.warning("[settings] could not write %s: %s", STORE, exc)


def set_value(tool: str, key: str, value: Any) -> None:
    data = load()
    if tool not in data or key not in DEFAULTS.get(tool, {}):
        log.warning("[settings] ignoring unknown setting %s.%s", tool, key)
        return
    data[tool][key] = _coerce(DEFAULTS[tool][key], value)
    save(data)


def reload() -> dict[str, dict[str, Any]]:
    _CACHE.clear()
    return load()


def to_args(tool: str) -> list[str]:
    """The command line this tool should be started with.

    Built here rather than in the dashboard so there is exactly one place that
    knows how a setting becomes a flag - which is the thing the .bat files got
    wrong by having four copies of it.
    """
    s = load().get(tool, {})
    if tool == "jevflow":
        args = ["--model", str(s["model"]), "listen", "--dictate",
                "--hotkey", str(s["hotkey"]),
                "--silence", str(s["silence_hold_s"]),
                "--accent", str(s["accent"]), "--badge", str(s["badge"]),
                "--title", "JevFlow"]
        if not s["clean_speech"]:
            args.append("--raw")
        if s["clean_aggressive"]:
            args.append("--clean-hard")
        if s["hold_to_talk"]:
            args.append("--hold")
        if s.get("fix_homophones"):
            args.append("--fix-spelling")
        if s.get("fix_fillers"):
            args.append("--check-fillers")
        return args

    args = ["--model", str(s["model"]), "listen", "--command",
            "--command-hotkey", str(s["hotkey"]), "--hotkey", "none",
            "--silence", str(s["silence_hold_s"]),
            "--wake", str(s["wake"]), "--wake-word", str(s["wake_word"]),
            "--accent", str(s["accent"]), "--badge", str(s["badge"]),
            "--title", "Jev Computer Use"]
    if s["continuous"]:
        args.append("--continuous")
    if s["stream"]:
        args.append("--stream")
    if s["dry_run"]:
        args.append("--dry-run")
    if not s["clean_speech"]:
        args.append("--raw")
    if s["clean_aggressive"]:
        args.append("--clean-hard")
    return args
