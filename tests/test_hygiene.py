"""Guards against the ways this codebase has actually been corrupted before."""
from __future__ import annotations

import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCES = [p for p in sorted(ROOT.rglob("*.py")) if "__pycache__" not in p.parts]


def test_there_are_sources_to_check():
    """A scan over nothing passes trivially, which is not a check at all."""
    assert len(SOURCES) >= 10


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_no_control_characters_in_source(path):
    """A patch script wrote a non-raw string, so `\\b` became a literal
    backspace (0x08) inside a regex. The file still looked correct in an editor
    and the pattern silently stopped matching. It happened twice."""
    raw = path.read_bytes()
    bad = [(i, b) for i, b in enumerate(raw) if b < 9 or 10 < b < 13 or 13 < b < 32]
    assert not bad, f"{path.name} contains control characters at {bad[:5]}"


@pytest.mark.parametrize("path", SOURCES, ids=lambda p: p.name)
def test_regexes_are_written_as_raw_strings(path):
    """re.compile("...") with a backslash in it is the same bug waiting."""
    src = path.read_text(encoding="utf-8")
    for m in re.finditer(r"re\.(?:compile|search|match|sub|findall)\(\s*(?!r[\"'])[\"']",
                         src):
        line = src[:m.start()].count("\n") + 1
        snippet = src[m.start():m.start() + 70].splitlines()[0]
        assert "\\" not in snippet, (
            f"{path.name}:{line} non-raw string in a regex: {snippet}")


def test_the_overlay_key_colour_is_never_drawn_with():
    """`-transparentcolor` punches exactly this colour out of the window, so
    anything painted in it becomes a hole rather than a pixel."""
    overlay = (ROOT / "jevflow" / "overlay.py").read_text(encoding="utf-8")
    key = re.search(r'KEY\s*=\s*"(#[0-9a-fA-F]{6})"', overlay)
    assert key, "the key colour is gone"
    body = overlay.split("def _draw", 1)[-1]
    assert key.group(1) not in body, "the capsule is drawn with the key colour"


def test_suppress_is_never_passed_to_a_hotkey():
    """suppress=True made the keyboard library buffer and replay every
    keystroke on the machine, and a crash mid-combo left Ctrl physically stuck
    down system-wide. It must never come back."""
    for path in SOURCES:
        src = path.read_text(encoding="utf-8")
        for m in re.finditer(r"add_hotkey\((.{0,200}?)\)", src, re.S):
            assert "suppress=True" not in m.group(1), f"{path.name}: {m.group(1)[:80]}"


def test_nothing_in_the_voice_path_terminates_a_process():
    """Closing by VOICE is WM_CLOSE only, so an app with unsaved work still
    gets to ask.

    Scoped to `jevflow/` on purpose. The dashboard stops JevFlow and JCU
    themselves, by an explicit button press, on processes it started - that is
    a person closing their own tool, not a misheard sentence killing an editor
    with unsaved work, and conflating the two would ban the wrong thing.
    """
    for path in SOURCES:
        # Relative to the repo root, so this keeps working when the repo
        # itself is called "jevflow" - which it is once published, and which
        # made this rule catch dashboard.py by accident.
        try:
            top = path.relative_to(ROOT).parts[0]
        except Exception:
            continue
        if path.name.startswith("test_") or top != "jevflow":
            continue
        src = path.read_text(encoding="utf-8")
        for banned in ("TerminateProcess", "taskkill", "SIGKILL"):
            assert banned not in src, f"{path.name} uses {banned}"
