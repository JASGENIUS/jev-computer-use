"""Doing things to windows: focus one, open a tab, type into it, press a key.

Every lesson here was learned the hard way:

- `SetForegroundWindow` is refused for a background process unless the caller
  owns the foreground or an input event just happened, so focus is taken with
  the alt-tap trick AND then **verified** rather than assumed.
- Keystrokes sent to a window that is not actually focused land somewhere else.
  That is how automation types a sentence into the wrong application, so every
  send checks the foreground window first.
- Nothing here is chosen by a model. These are primitives that code calls once
  a decision has already been made.
"""
from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

log = logging.getLogger("jevflow.actions")

u32 = ctypes.windll.user32
k32 = ctypes.windll.kernel32

VK_MENU = 0x12
KEYEVENTF_KEYUP = 0x0002


def foreground() -> tuple[int, str]:
    hwnd = u32.GetForegroundWindow()
    n = u32.GetWindowTextLengthW(hwnd)
    b = ctypes.create_unicode_buffer(n + 1)
    u32.GetWindowTextW(hwnd, b, n + 1)
    return int(hwnd), b.value


def windows_for_exe(exe: str) -> list[tuple[int, str]]:
    """Visible top-level windows belonging to an executable.

    psutil is required, not optional. An earlier comment claimed a title-scan
    fallback that was never written, so on a machine without psutil every
    verification would have reported "unknown" forever and nothing would have
    looked wrong.
    """
    if not exe:
        return []
    import psutil
    found: list[tuple[int, str]] = []
    CB = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

    def cb(hwnd, _):
        if not u32.IsWindowVisible(hwnd):
            return True
        n = u32.GetWindowTextLengthW(hwnd)
        if n <= 0:
            return True
        pid = wt.DWORD()
        u32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        try:
            if psutil.Process(pid.value).name().lower() == exe.lower():
                b = ctypes.create_unicode_buffer(n + 1)
                u32.GetWindowTextW(hwnd, b, n + 1)
                found.append((int(hwnd), b.value))
        except Exception:
            pass
        return True

    u32.EnumWindows(CB(cb), None)
    return found


def focus_window(hwnd: int, timeout_s: float = 4.0) -> bool:
    """Bring a window to the front, and confirm it got there."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if u32.GetForegroundWindow() == hwnd:
            return True
        u32.keybd_event(VK_MENU, 0, 0, 0)          # satisfies the foreground lock
        u32.keybd_event(VK_MENU, 0, KEYEVENTF_KEYUP, 0)
        u32.ShowWindow(hwnd, 9)                     # SW_RESTORE
        cur = k32.GetCurrentThreadId()
        other = u32.GetWindowThreadProcessId(u32.GetForegroundWindow(), None)
        if other:
            u32.AttachThreadInput(cur, other, True)
        u32.BringWindowToTop(hwnd)
        u32.SetForegroundWindow(hwnd)
        if other:
            u32.AttachThreadInput(cur, other, False)
        time.sleep(0.25)
    return u32.GetForegroundWindow() == hwnd


def wait_for_window(exe: str, timeout_s: float = 20.0,
                    exclude: Optional[set[int]] = None) -> Optional[tuple[int, str]]:
    """Wait for a window of this exe to exist, ignoring ones already open."""
    exclude = exclude or set()
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        for hwnd, title in windows_for_exe(exe):
            if hwnd not in exclude:
                return hwnd, title
        time.sleep(0.3)
    return None


def visible_windows() -> list[tuple[int, str]]:
    """Every visible titled window, whichever process owns it."""
    found: list[tuple[int, str]] = []
    CB = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

    def cb(hwnd, _):
        if u32.IsWindowVisible(hwnd):
            n = u32.GetWindowTextLengthW(hwnd)
            if n > 0:
                b = ctypes.create_unicode_buffer(n + 1)
                u32.GetWindowTextW(hwnd, b, n + 1)
                found.append((int(hwnd), b.value))
        return True

    u32.EnumWindows(CB(cb), None)
    return found


def wait_for_new_titled(name: str, before: set[int],
                        timeout_s: float = 12.0) -> Optional[bool]:
    """Did a NEW window carrying this name appear?

    A Store app has no executable path, so windows cannot be counted for it and
    this is the only evidence available that it started. `explorer.exe` spawns
    happily whatever it is handed, which makes "it launched" worth nothing by
    itself - a Notepad that never opened was reported as opened exactly that way.
    """
    needle = (name or "").strip().lower()
    if not needle:
        return None
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            for hwnd, title in visible_windows():
                if hwnd not in before and needle in title.lower():
                    return True
        except Exception:
            return None
        time.sleep(0.4)
    return False


def wait_for_title(exe: str, needle: str, timeout_s: float = 8.0) -> Optional[str]:
    """Wait for a window title containing `needle`, and return it.

    This is the only evidence available that a page actually loaded. Returning
    None means it could not be confirmed - which is reported as unknown, never
    as success.
    """
    needle = (needle or "").strip().lower()
    if not needle:
        return None
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        for _h, title in windows_for_exe(exe):
            if needle in title.lower():
                return title
        time.sleep(0.4)
    return None


# --- keystrokes --------------------------------------------------------------
INPUT_KEYBOARD = 1
KEYEVENTF_UNICODE = 0x0004


class _KBD(ctypes.Structure):
    _fields_ = [("wVk", wt.WORD), ("wScan", wt.WORD), ("dwFlags", wt.DWORD),
                ("time", wt.DWORD), ("dwExtraInfo", ctypes.POINTER(wt.ULONG))]


class _U(ctypes.Union):
    _fields_ = [("ki", _KBD), ("pad", ctypes.c_byte * 32)]


class _IN(ctypes.Structure):
    _fields_ = [("type", wt.DWORD), ("u", _U)]


def _char_events(ch: str) -> list[_IN]:
    return [_IN(INPUT_KEYBOARD, _U(ki=_KBD(0, ord(ch), KEYEVENTF_UNICODE, 0, None))),
            _IN(INPUT_KEYBOARD, _U(ki=_KBD(0, ord(ch), KEYEVENTF_UNICODE | KEYEVENTF_KEYUP, 0, None)))]


def type_text(text: str, expect_hwnd: Optional[int] = None) -> tuple[bool, str]:
    """Type unicode into the focused window. Refuses if focus moved."""
    if not text:
        return False, "nothing to type"
    hwnd, title = foreground()
    if expect_hwnd is not None and hwnd != expect_hwnd:
        return False, f"focus moved to {title!r}; refusing to type"
    ev: list[_IN] = []
    for ch in text:
        ev += _char_events(ch)
    for i in range(0, len(ev), 80):
        batch = ev[i:i + 80]
        arr = (_IN * len(batch))(*batch)
        if u32.SendInput(len(batch), ctypes.byref(arr), ctypes.sizeof(_IN)) != len(batch):
            return False, "SendInput rejected part of the text"
        time.sleep(0.004)
    return True, title


def press(*vks: int, expect_hwnd: Optional[int] = None) -> bool:
    """Press a chord of virtual-key codes, e.g. press(0x11, 0x54) for Ctrl+T."""
    if expect_hwnd is not None and foreground()[0] != expect_hwnd:
        return False
    for vk in vks:
        u32.keybd_event(vk, 0, 0, 0)
        time.sleep(0.02)
    for vk in reversed(vks):
        u32.keybd_event(vk, 0, KEYEVENTF_KEYUP, 0)
        time.sleep(0.02)
    return True


VK_CONTROL, VK_RETURN, VK_T, VK_L = 0x11, 0x0D, 0x54, 0x4C

# Media keys go to whichever application owns media playback, so they need no
# focus and cannot be aimed at the wrong window - which is also why nothing here
# can confirm that anything happened.
MEDIA_KEYS: dict[str, int] = {
    "next": 0xB0,
    "previous": 0xB1,
    "play_pause": 0xB3,
    "mute": 0xAD,
    "volume_down": 0xAE,
    "volume_up": 0xAF,
}


def press_media(action: str) -> bool:
    """Press a media key. Returns whether the key EXISTS, not whether it worked."""
    vk = MEDIA_KEYS.get(action or "")
    if vk is None:
        return False
    return press(vk)


# --- browsers ----------------------------------------------------------------
# Browsers that launched but never showed a window. Chrome does exactly that on
# this machine: it exits 0 immediately, with no window and no error. Retrying it
# costs 14 seconds per command, so the failure is remembered - but on disk and
# with an expiry, because a browser that is broken today may be reinstalled
# tomorrow and a permanent in-memory blacklist would never find out.
STATE_FILE = Path(__file__).resolve().parents[1] / ".jevflow-state.json"
DEAD_BROWSER_TTL_S = 24 * 3600.0


def _load_state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_state(state: dict) -> None:
    try:
        STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")
    except Exception as exc:
        log.debug("[actions] could not write %s: %s", STATE_FILE, exc)


def dead_browsers(now: Optional[float] = None) -> dict[str, float]:
    """Browsers known not to start, and when that was last observed."""
    now = now if now is not None else time.time()
    dead = (_load_state().get("dead_browsers") or {})
    return {name: t for name, t in dead.items()
            if isinstance(t, (int, float)) and now - t < DEAD_BROWSER_TTL_S}


def mark_browser_dead(name: str, now: Optional[float] = None) -> None:
    state = _load_state()
    dead = state.setdefault("dead_browsers", {})
    dead[name] = now if now is not None else time.time()
    _save_state(state)


@dataclass(frozen=True)
class Opened:
    """What actually happened, including what did not.

    `note` carries the browsers that were tried and failed. A silent downgrade
    from Chrome to Brave is still a downgrade, and the person at the keyboard
    has no other way to learn that their default browser is broken.
    """
    ok: bool
    browser: str = ""
    exe: str = ""
    note: str = ""


def open_url(url: str, browsers: list, launch, prefer: str = "") -> Opened:
    """Open a URL, trying browsers until one actually shows a window.

    `browsers` are App records and `launch` is the fallback launcher. A browser
    is asked for the URL directly rather than going through the shell default,
    because the default here is Chrome and Chrome does not start.
    """
    import subprocess
    known_dead = dead_browsers()
    order = sorted(browsers, key=lambda b: (b.name in known_dead,
                                            0 if b.name == prefer else 1))
    failed: list[str] = []
    skipped = [b.name for b in browsers if b.name in known_dead]
    for b in order:
        if not b.target:
            continue
        try:
            subprocess.Popen([b.target, url])
        except Exception as exc:
            failed.append(f"{b.name} ({type(exc).__name__})")
            continue
        got = wait_for_window(b.exe, timeout_s=14.0)
        if got is not None:
            time.sleep(0.6)
            focus_window(got[0])
            note = ""
            if failed:
                note = ", ".join(failed) + " did not start"
            elif skipped and b.name not in skipped:
                note = ", ".join(skipped) + " skipped, known not to start"
            return Opened(True, b.name, b.exe, note)
        failed.append(b.name)
        mark_browser_dead(b.name)
        log.warning("[actions] %s launched but showed no window; "
                    "not trying it again for %.0f h", b.name, DEAD_BROWSER_TTL_S / 3600)
    return Opened(False, "", "", "no browser opened (tried " + ", ".join(failed) + ")")


WM_CLOSE = 0x0010


def close_window(hwnd: int, timeout_s: float = 6.0) -> tuple[bool, str]:
    """Ask a window to close, the same way clicking its X does.

    WM_CLOSE, never a forced kill: the application keeps its chance to say
    "save changes?" A voice command that force-kills an editor with unsaved work
    is not a feature, and there is no undo for it.
    """
    n = u32.GetWindowTextLengthW(hwnd)
    b = ctypes.create_unicode_buffer(n + 1)
    u32.GetWindowTextW(hwnd, b, n + 1)
    title = b.value
    if not u32.IsWindow(hwnd):
        return False, "that window is already gone"
    u32.PostMessageW(hwnd, WM_CLOSE, 0, 0)
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if not u32.IsWindow(hwnd) or not u32.IsWindowVisible(hwnd):
            return True, title
        time.sleep(0.25)
    # Still there: almost always a "save your work?" prompt waiting for a human.
    return False, f"{title!r} did not close - it may be asking you something"
