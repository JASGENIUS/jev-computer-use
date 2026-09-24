"""Stop two copies fighting over the same hotkey and the same microphone.

Autostart made this urgent. One copy now launches at login, and double-clicking
the .bat - which is the only way this has ever been started, so it is pure
muscle memory - used to add a second. Both register the same global hotkey, so
one keypress starts two recorders on one microphone, and the symptom is not
"it launched twice": it is dictation randomly dropping words, which looks like
a bug in the recogniser.

The lock is keyed on the **hotkeys**, not on the program. Two processes only
conflict if they grab the same keys, and JevFlow (Win+Alt) alongside Jev
Computer Use (Ctrl+Win) is a perfectly good arrangement that must keep working.

A Windows named mutex is used rather than a lock file because the kernel
releases it when the process dies, however it dies. A lock file left behind by
a crash - or by the `pythonw` process being killed - would lock the user out of
their own tool until they went looking for a stale file they do not know exists.
"""
from __future__ import annotations

import ctypes
import hashlib
import logging
from typing import Iterable, Optional

log = logging.getLogger("jevflow.single_instance")

ERROR_ALREADY_EXISTS = 183

# Held for the lifetime of the process. If this is garbage collected the mutex
# is released and a second copy can start, so the reference is deliberate.
_HANDLE: Optional[int] = None


def _name_for(hotkeys: Iterable[str]) -> str:
    keys = sorted({(k or "").strip().lower() for k in hotkeys
                   if k and k.strip().lower() not in ("", "none")})
    if not keys:
        return "JevFlow_nokeys"
    digest = hashlib.sha1("|".join(keys).encode("utf-8")).hexdigest()[:16]
    # "Local\\" keeps it per-session, which is what we want: another user
    # signed in on the same machine has their own keyboard focus.
    return f"Local\\JevFlow_hotkeys_{digest}"


def acquire(hotkeys: Iterable[str]) -> bool:
    """True if this process may own these hotkeys; False if someone already does."""
    global _HANDLE
    name = _name_for(hotkeys)
    try:
        k32 = ctypes.windll.kernel32
        k32.CreateMutexW.restype = ctypes.c_void_p
        handle = k32.CreateMutexW(None, ctypes.c_bool(True), ctypes.c_wchar_p(name))
        last = ctypes.get_last_error() if hasattr(ctypes, "get_last_error") else 0
        if not last:
            last = k32.GetLastError()
        if handle and last == ERROR_ALREADY_EXISTS:
            log.info("[single] another copy already owns %s", sorted(hotkeys))
            return False
        _HANDLE = handle
        return True
    except Exception as exc:
        # Never let the guard itself stop the tool from running. Failing open
        # is the right call: the worst case is the old behaviour.
        log.warning("[single] could not take the lock (%s); continuing", exc)
        return True


def tell_user_already_running(title: str, detail: str) -> None:
    """Say so somewhere the person can actually see it.

    The launchers use `pythonw`, which has no console, so printing a refusal
    sends it nowhere at all: a double-click did nothing, showed nothing, and
    logged nothing where anyone would look. That reads as the program being
    broken, which is precisely what happened.
    """
    print(f"{title} is already running ({detail}). Nothing to do.")
    try:
        import sys
        if sys.stdout is not None and sys.stdout.isatty():
            return          # a console run already showed it
    except Exception:
        pass
    try:
        body = "\n\n".join([
            f"{title} is already running.",
            f"Keys in use: {detail}",
            "Nothing was started. Use the running copy, or close it first.",
        ])
        ctypes.windll.user32.MessageBoxW(None, body, title, 0x40)  # MB_ICONINFORMATION
    except Exception:
        pass
