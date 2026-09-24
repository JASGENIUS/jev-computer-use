"""What is installed on this machine, and which of it you might have meant.

There are ~250 Start Menu shortcuts here. Handing all of them to a typed choice
would be the Banking77 mistake: options share a fixed token budget, so 250 of
them blur into noise and the model answers confidently at random.

So this follows the guide's rank-wide-read-narrow rule. **Code** does the wide
pass - cheap string matching over every installed app, filtering to things that
actually exist and can be launched - and hands a short list of real candidates
to Jev, which decides which one was meant. A model is never asked to choose an
option the system could not execute anyway.
"""
from __future__ import annotations

import functools
import json
import logging
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

log = logging.getLogger("jevflow.apps")

START_MENUS = [
    Path(os.environ.get("ProgramData", r"C:\ProgramData")) / "Microsoft/Windows/Start Menu/Programs",
    Path(os.environ.get("APPDATA", "")) / "Microsoft/Windows/Start Menu/Programs",
]

# Shortcuts that are never what someone means by "open X".
_NOISE = re.compile(
    r"(uninstall|readme|release notes|documentation|help|website|licen[cs]e|"
    r"changelog|repair|modify|troubleshoot|command prompt for|\.url)", re.I)

# What people say vs what the shortcut is called.
ALIASES = {
    "chrome": "google chrome", "vscode": "visual studio code", "vs code": "visual studio code",
    "code": "visual studio code", "explorer": "file explorer", "terminal": "windows terminal",
    "cmd": "command prompt", "powershell": "windows powershell", "word": "winword",
    "sheets": "excel", "docs": "word", "obsidian": "obsidian", "discord": "discord",
    "spotify": "spotify", "steam": "steam", "minecraft": "minecraft",
}


@dataclass(frozen=True)
class App:
    name: str
    launch: str          # the .lnk path - launching the shortcut keeps its args and working dir
    target: str          # the exe it points at, for window verification
    source: str = "start-menu"

    @property
    def exe(self) -> str:
        return Path(self.target).name.lower() if self.target else ""


def _shortcut_target(lnk: Path) -> str:
    """Resolve a .lnk without COM: read the target out of the shell link blob."""
    try:
        import struct
        data = lnk.read_bytes()
        if len(data) < 76 or data[:4] != b"L\x00\x00\x00":
            return ""
        flags = struct.unpack("<I", data[20:24])[0]
        off = 76
        if flags & 0x1:                                   # HasLinkTargetIDList
            off += struct.unpack("<H", data[off:off + 2])[0] + 2
        if not (flags & 0x2):                             # HasLinkInfo
            return ""
        info = data[off:]
        if len(info) < 20:
            return ""
        size, hdr = struct.unpack("<II", info[:8])
        lbp = struct.unpack("<I", info[16:20])[0]
        if not lbp or lbp >= len(info):
            return ""
        end = info.find(b"\x00", lbp)
        return info[lbp:end].decode("mbcs", "ignore")
    except Exception:
        return ""


def _apps_folder() -> list[App]:
    """Store apps, which have no Start Menu shortcut to find.

    Windows 11 ships Notepad, Calculator, Terminal, Photos and Settings as
    packaged apps. They are on the taskbar and in Start, and they have no .lnk
    anywhere - so a shortcut-only index answered "nothing installed matches" to
    "open notepad", which reads as the whole tool being broken rather than as a
    gap in one list.

    `Get-StartApps` lists them with the AppUserModelID that
    `shell:AppsFolder\\<id>` launches. There is no executable path in that
    listing, so these carry no `target` and cannot be verified by counting
    windows for an exe - which is reported as unknown, never as success.
    """
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command",
             "Get-StartApps | ConvertTo-Json -Compress"],
            capture_output=True, text=True, timeout=25,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if out.returncode != 0 or not out.stdout.strip():
            log.warning("[apps] Get-StartApps returned nothing; Store apps unavailable")
            return []
        rows = json.loads(out.stdout)
    except Exception as exc:
        log.warning("[apps] could not list Store apps (%s); shortcuts only", exc)
        return []
    if isinstance(rows, dict):
        rows = [rows]
    found: list[App] = []
    for row in rows:
        name = (row.get("Name") or "").strip()
        appid = (row.get("AppID") or "").strip()
        # A classic app's AppID is a path to its exe. Those are already covered
        # by the shortcut index, which also knows the args and working directory.
        if not name or not appid or "!" not in appid or _NOISE.search(name):
            continue
        found.append(App(name=name, launch=f"shell:AppsFolder\\{appid}",
                         target="", source="apps-folder"))
    log.info("[apps] %d Store apps", len(found))
    return found


# Where applications land when nobody makes a shortcut for them.
INSTALL_ROOTS = [
    Path(os.environ.get("LOCALAPPDATA", "")) / "Programs",
    Path(os.environ.get("LOCALAPPDATA", "")),
    Path(os.environ.get("ProgramFiles", r"C:\Program Files")),
    Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")),
]

# Executables that ship alongside an application and are not it. Without this
# a folder scan buries the shortlist in updaters, installers and helpers.
_NOT_AN_APP = re.compile(
    r"(unins|uninstall|setup|install|update[rs]?|upgrade|crashpad|"
    r"helper|service|daemon|launcher_helper|vcredist|dotnet|"
    r"report|diagnos|watchdog|elevate|squirrel|notification)", re.I)


def scan_install_dirs(roots=None, max_per_root: int = 400) -> list[App]:
    """Applications in the usual install folders, found without a shortcut.

    Modrinth is installed at `%LOCALAPPDATA%\\Modrinth App\\Modrinth App.exe`
    with no Start Menu entry and no `Get-StartApps` record, so both other
    sources missed it entirely - "open modrinth" could not work however clearly
    it was said, because the app was never a candidate.

    The rule that stops this becoming noise: **only an executable whose name
    matches its folder counts.** "Modrinth App\\Modrinth App.exe" is the
    application; "Modrinth App\\resources\\crashpad_handler.exe" is not.
    """
    found: list[App] = []
    for root in (roots if roots is not None else INSTALL_ROOTS):
        root = Path(root)
        if not root or not root.is_dir():
            continue
        n = 0
        try:
            children = sorted(root.iterdir())
        except Exception:
            continue
        for folder in children:
            if n >= max_per_root:
                break
            try:
                if not folder.is_dir() or _NOT_AN_APP.search(folder.name):
                    continue
                stem = _squash(folder.name)
                if not stem:
                    continue
                for exe in folder.glob("*.exe"):
                    if _NOT_AN_APP.search(exe.stem):
                        continue
                    # The executable has to BE the folder, not merely live in it.
                    if _squash(exe.stem) != stem:
                        continue
                    found.append(App(name=folder.name, launch=str(exe),
                                     target=str(exe), source="installed"))
                    n += 1
                    break
            except Exception:
                continue
    log.info("[apps] %d app(s) found by scanning install folders", len(found))
    return found


@functools.lru_cache(maxsize=1)
def index() -> tuple[App, ...]:
    """Every launchable app on this machine, de-duplicated by name.

    Shortcuts first: when both sources offer the same name, the .lnk wins
    because it carries the arguments and working directory the app expects.
    """
    seen: dict[str, App] = {}
    for root in START_MENUS:
        if not root or not root.is_dir():
            continue
        for lnk in root.rglob("*.lnk"):
            name = lnk.stem.strip()
            if not name or _NOISE.search(name) or _NOISE.search(str(lnk.parent)):
                continue
            key = name.lower()
            if key in seen:
                continue
            target = _shortcut_target(lnk)
            if target and not target.lower().endswith((".exe", ".com", ".bat", ".cmd")):
                continue
            seen[key] = App(name=name, launch=str(lnk), target=target)
    for app in _apps_folder():
        seen.setdefault(app.name.lower(), app)
    # Last, so a real shortcut always wins: a .lnk carries the arguments and
    # working directory an application expects, and a bare .exe does not.
    for app in scan_install_dirs():
        seen.setdefault(app.name.lower(), app)
    apps = tuple(sorted(seen.values(), key=lambda a: a.name.lower()))
    log.info("[apps] indexed %d launchable apps", len(apps))
    return apps


def running(candidates: list[App]) -> list[str]:
    """Which of these have a window open right now.

    Jev cannot see the taskbar. Without this, "switch to Obsidian" was always
    classified as open_app - the model had no way to know the difference between
    something running and something merely installed, so it guessed the common
    case every time.
    """
    open_now: list[str] = []
    for app in candidates:
        if not app.exe:
            continue          # a Store app: no exe to look for, so unknown
        try:
            from jevflow import actions
            if actions.windows_for_exe(app.exe):
                open_now.append(app.name)
        except Exception as exc:
            log.debug("[apps] could not check %s: %s", app.name, exc)
    return open_now


def _squash(text: str) -> str:
    """Letters and digits only. "Local Send", "local-send" and "LocalSend" are
    the same app, and only the person saying it out loud knows there is a space."""
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


# Below this, a fuzzy match is noise rather than a mishearing. Tuned so
# "handbreak" still finds HandBrake while nonsense finds nothing.
FUZZY_FLOOR = 0.84


def _score(query: str, app: App) -> float:
    """Cheap relevance. Deliberately generous - Jev makes the real call.

    Everything here compares SQUASHED forms - punctuation and spacing removed -
    because speech inserts spaces that the filename does not have. Comparing
    literally scored "local send" against LocalSend at **0.000** and returned
    "Send to OneNote", and the same held for CapCut, FileZilla, HandBrake,
    OneDrive, RustDesk and twenty more.
    """
    q, n = (query or "").lower().strip(), app.name.lower()
    if not q:
        return 0.0
    qs, ns = _squash(q), _squash(n)
    if not qs:
        return 0.0
    if q == n or qs == ns:
        return 1.0

    score = 0.0
    if ns.startswith(qs):
        score = max(score, 0.9)
    elif qs in ns:
        score = max(score, 0.75)
    exe = _squash(app.exe.removesuffix(".exe")) if app.exe else ""
    if exe and (qs == exe or qs in exe):
        score = max(score, 0.85)

    qt = set(re.findall(r"[a-z0-9]+", q))
    nt = set(re.findall(r"[a-z0-9]+", n))
    if qt and qt <= nt:
        score = max(score, 0.7)
    elif qt & nt:
        score = max(score, 0.35 + 0.3 * len(qt & nt) / max(1, len(qt)))

    # Last resort, for what the recogniser heard rather than what was said.
    # Only for queries long enough that a near-match means something - two
    # letters are similar to half the Start Menu.
    if score < 0.6 and len(qs) >= 5:
        import difflib
        ratio = difflib.SequenceMatcher(None, qs, ns).ratio()
        if ratio >= FUZZY_FLOOR:
            score = max(score, 0.55 + 0.25 * (ratio - FUZZY_FLOOR) / (1 - FUZZY_FLOOR))
    return score


def shortlist(query: str, limit: int = 6) -> list[App]:
    """Apps that plausibly match, best first. The wide pass, done in code."""
    from jevflow import learned

    raw = (query or "").lower().strip()
    q = ALIASES.get(raw) or ALIASES.get(_squash(raw)) or query

    # Something the user corrected by hand outranks anything the scorer works out.
    # It is the one signal here that came from a person rather than a guess.
    taught = learned.lookup(raw)
    pinned = [a for a in index() if a.name == taught] if taught else []
    if taught and not pinned:
        # The lesson names an app that is no longer installed. Drop it for this
        # query rather than returning nothing - a stale lesson must not be able
        # to make a working machine look empty.
        log.info("[apps] learned %r -> %r, which is not installed", raw, taught)

    scored = [(s, a) for a in index()
              if a not in pinned and (s := _score(q, a)) >= 0.34]
    # Ties broken by the SHORTER name. "Notepad" and "Notepad++" squash to the
    # same letters, and asking for one and getting the other is worse than
    # getting nothing, because it looks like it worked.
    scored.sort(key=lambda sa: (-sa[0], len(sa[1].name), sa[1].name.lower()))
    return (pinned + [a for _s, a in scored])[:limit]


def launch(app: App) -> tuple[bool, str]:
    """Start it. Launching the .lnk preserves its arguments and working directory.

    A Store app is NOT started by `os.startfile`. That call accepts a
    `shell:AppsFolder\\...` string, returns without raising, and opens nothing -
    so this function reported success for a Notepad that never appeared, which
    is the same lie as calling a spawned process a running application. Only
    `explorer.exe` resolves an AppUserModelID.
    """
    if app.source == "apps-folder":
        try:
            subprocess.Popen(["explorer.exe", app.launch], shell=False)
            return True, app.name
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"
    try:
        os.startfile(app.launch)  # noqa: S606 - a Start Menu shortcut the user already has
        return True, app.name
    except Exception as exc:
        try:
            subprocess.Popen(["cmd", "/c", "start", "", app.launch], shell=False)
            return True, app.name
        except Exception:
            return False, f"{type(exc).__name__}: {exc}"
