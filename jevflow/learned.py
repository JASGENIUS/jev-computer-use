"""What it has been told, once, and should not get wrong again.

Matching will always have a gap. A nickname nobody could guess ("my editor"),
a name the recogniser mangles the same way every time, a company app with a
name that sounds like a different word. The cure for the long tail is not a
cleverer scorer - it is being able to say "no, I meant X" and have that stick.

Two rules shape this more than the storage does:

**Only an explicit correction teaches it.** Learning from ordinary speech would
fill the store with things nobody meant to teach, and a store full of noise
is worse than an empty one because it starts overriding matches that were
right.

**A broken store behaves as an empty one.** Everything here is wrapped so that
a corrupt file, an unwritable path or a store someone hand-edited into the
wrong shape costs nothing. A learning feature that can brick app-opening is
worse than no learning feature at all.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

log = logging.getLogger("jevflow.learned")

STORE = Path(__file__).resolve().parents[1] / "learned.json"

_CACHE: dict[str, str] = {}
_LOADED = False


def _key(phrase: str) -> str:
    """Spacing and punctuation carry no meaning here - "Local Send",
    "local-send" and "localsend" are one thing someone said."""
    return re.sub(r"[^a-z0-9]+", "", (phrase or "").lower())


def _load() -> dict[str, str]:
    global _LOADED
    if _CACHE or _LOADED:
        return _CACHE
    _LOADED = True
    try:
        data = json.loads(STORE.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return _CACHE
    except Exception as exc:
        log.warning("[learned] %s is unreadable (%s); starting empty", STORE, exc)
        return _CACHE
    if not isinstance(data, dict):
        log.warning("[learned] %s is not a mapping; ignoring it", STORE)
        return _CACHE
    for phrase, name in data.items():
        if isinstance(phrase, str) and isinstance(name, str) and phrase and name:
            _CACHE[_key(phrase)] = name
    return _CACHE


def _save() -> None:
    try:
        STORE.parent.mkdir(parents=True, exist_ok=True)
        STORE.write_text(json.dumps(_CACHE, indent=2, sort_keys=True), encoding="utf-8")
    except Exception as exc:
        # Never raise. Failing to remember is a small loss; failing to open the
        # app because remembering failed is not.
        log.warning("[learned] could not write %s: %s", STORE, exc)


def remember(phrase: str, app_name: str) -> None:
    """Teach it that this phrase means this app. A later correction wins."""
    key, name = _key(phrase), (app_name or "").strip()
    if not key or not name:
        return
    _load()
    _CACHE[key] = name
    _save()
    log.info("[learned] %r now means %r", phrase, name)


def lookup(phrase: str) -> Optional[str]:
    key = _key(phrase)
    return _load().get(key) if key else None


def reload() -> dict[str, str]:
    """Re-read the store from disk.

    Clearing the cache alone does NOT do this - an internal flag stops an empty
    store being re-read on every single lookup. Anything that wants a genuine
    reload has to say so, rather than relying on that subtlety.
    """
    global _LOADED
    _CACHE.clear()
    _LOADED = False
    return _load()


def all_pairs() -> dict[str, str]:
    return dict(_load())


def forget(phrase: str) -> bool:
    key = _key(phrase)
    _load()
    if key in _CACHE:
        del _CACHE[key]
        _save()
        return True
    return False


def forget_all() -> None:
    global _LOADED
    _CACHE.clear()
    _LOADED = True          # empty on purpose, not merely unread
    _save()
