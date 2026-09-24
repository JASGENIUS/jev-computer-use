"""Is the chord still being held?

Push-to-talk needs to know when to stop, and the honest signal is the key, not
the microphone. Silence detection ends a clip when the room goes quiet, which
is the wrong rule for dictation: a pause in the middle of forming a thought is
not the end of a sentence, and being cut off mid-thought is worse than a clip
that runs a second long.

The failure mode that matters here is the opposite one. If this cannot read the
keyboard it must report **released**, never held - a recorder that believes the
key is still down holds the microphone open with no way to stop it, and the
only cure is killing the process.
"""
from __future__ import annotations

import logging

log = logging.getLogger("jevflow.hotkeys")


def keys_of(chord: str) -> list[str]:
    """"windows+alt" -> ["windows", "alt"]. "none" is not a chord."""
    text = (chord or "").strip().lower()
    if not text or text == "none":
        return []
    return [k.strip() for k in text.split("+") if k.strip()]


def _is_pressed(key: str) -> bool:
    import keyboard
    return bool(keyboard.is_pressed(key))


def still_held(chord: str) -> bool:
    """True only while EVERY key of the chord is down."""
    keys = keys_of(chord)
    if not keys:
        return False
    try:
        return all(_is_pressed(k) for k in keys)
    except Exception as exc:
        # Fail RELEASED, deliberately. Believing a key is still held with no
        # way to check it leaves the microphone open forever.
        log.warning("[hotkeys] could not read the keyboard (%s); "
                    "treating the chord as released", exc)
        return False
