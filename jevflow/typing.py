"""Type a transcript into whatever window had focus.

This is the safe action: it is exactly what the user just said, it goes where
they were already looking, and it is trivially undone with ctrl+z. It needs no
judgement from anything, so nothing judges it.

Focus is the whole problem here. The overlay must never
take focus, so the target window still owns the caret when the transcript
arrives. We verify the foreground window is the SAME one that had focus when the
hotkey fired, and refuse to type if it changed - typing a sentence into whatever
happens to be in front is how automation destroys someone's afternoon.
"""
from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import logging
import time
from typing import Optional

log = logging.getLogger("jevflow.typing")

u32 = ctypes.windll.user32


def foreground() -> tuple[int, str]:
    hwnd = u32.GetForegroundWindow()
    n = u32.GetWindowTextLengthW(hwnd)
    buf = ctypes.create_unicode_buffer(n + 1)
    u32.GetWindowTextW(hwnd, buf, n + 1)
    return int(hwnd), buf.value


# --- SendInput unicode typing ------------------------------------------------
INPUT_KEYBOARD = 1
KEYEVENTF_UNICODE = 0x0004
KEYEVENTF_KEYUP = 0x0002


class _KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", wt.WORD), ("wScan", wt.WORD), ("dwFlags", wt.DWORD),
                ("time", wt.DWORD), ("dwExtraInfo", ctypes.POINTER(wt.ULONG))]


class _INPUTunion(ctypes.Union):
    _fields_ = [("ki", _KEYBDINPUT), ("padding", ctypes.c_byte * 32)]


class _INPUT(ctypes.Structure):
    _fields_ = [("type", wt.DWORD), ("u", _INPUTunion)]


def _unicode_event(ch: str, up: bool) -> _INPUT:
    flags = KEYEVENTF_UNICODE | (KEYEVENTF_KEYUP if up else 0)
    ki = _KEYBDINPUT(0, ord(ch), flags, 0, None)
    return _INPUT(INPUT_KEYBOARD, _INPUTunion(ki=ki))


def type_text(text: str, *, expect_hwnd: Optional[int] = None, chunk: int = 40,
              delay_s: float = 0.004) -> tuple[bool, str]:
    """Type `text` into the focused window. Returns (ok, reason).

    Sends real unicode key events rather than pasting, so it works in terminals
    and editors that ignore the clipboard, and it does not clobber whatever the
    user already had copied.
    """
    if not text:
        return False, "nothing to type"
    hwnd, title = foreground()
    if expect_hwnd is not None and hwnd != expect_hwnd:
        return False, f"focus moved to {title!r}; refusing to type"

    events: list[_INPUT] = []
    for ch in text:
        events.append(_unicode_event(ch, False))
        events.append(_unicode_event(ch, True))
    n = len(events)
    sent = 0
    i = 0
    while i < n:
        batch = events[i:i + chunk * 2]
        arr = (_INPUT * len(batch))(*batch)
        got = u32.SendInput(len(batch), ctypes.byref(arr), ctypes.sizeof(_INPUT))
        sent += int(got)
        if got != len(batch):
            return False, f"SendInput accepted {sent}/{n} events"
        i += len(batch)
        time.sleep(delay_s)
    log.info("[type] %d chars into %r", len(text), title)
    return True, title
