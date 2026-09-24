"""The .bat files are the only way this is ever actually started.

They drifted: `Jev.bat` was still on ctrl+alt+1 and ctrl+windows long after the
handoff documented Win+Alt and Right Ctrl, and the two JevFlow launchers
disagreed with each other about which key does dictation. Nothing caught it,
because nothing reads the .bat files except a double-click.
"""
from __future__ import annotations

import pathlib
import shlex

import pytest

import main as entry

ROOT = pathlib.Path(__file__).resolve().parents[1]
def _invokes_main(path: pathlib.Path) -> bool:
    try:
        return "main.py" in path.read_text(encoding="ascii", errors="ignore")
    except Exception:
        return False


# Only the launchers that start a VOICE TOOL. Dashboard.bat opens the settings
# panel and has no hotkeys to collide, so holding it to these rules would be
# checking the wrong thing.
BATS = sorted(p for p in ROOT.glob("*.bat") if _invokes_main(p))


def command_line(bat: pathlib.Path) -> list[str]:
    for line in bat.read_text(encoding="ascii", errors="ignore").splitlines():
        if "main.py" in line:
            return shlex.split(line.split("main.py", 1)[1])
    return []


def test_there_are_launchers_to_check():
    assert BATS, "no .bat files found at all"


@pytest.mark.parametrize("bat", BATS, ids=lambda p: p.name)
def test_every_launcher_names_main(bat):
    assert command_line(bat), f"{bat.name} never invokes main.py"


@pytest.mark.parametrize("bat", BATS, ids=lambda p: p.name)
def test_every_launcher_parses(bat):
    """A typo in a flag is invisible until a double-click does nothing."""
    parser = entry.build_parser()
    parser.parse_args(command_line(bat))


def _keys(chord: str) -> frozenset:
    return frozenset(k.strip().lower() for k in (chord or "").split("+") if k.strip())


@pytest.mark.parametrize("bat", BATS, ids=lambda p: p.name)
def test_the_two_hotkeys_cannot_collide(bat):
    """The keyboard library fires a chord the moment its keys are down, so a
    chord whose keys are a SUBSET of the other's always wins the race.

    ctrl+alt and ctrl+alt+1 collided exactly that way. The rule is about key
    SETS, not string prefixes - an earlier version of this test compared the
    raw strings, and would have passed "windows" against "windows+alt" even
    though pressing the second always fires the first.
    """
    args = entry.build_parser().parse_args(command_line(bat))
    a = _keys(getattr(args, "hotkey", ""))
    b = _keys(getattr(args, "command_hotkey", ""))
    if not a or not b or "none" in (a | b) or not getattr(args, "command", False):
        return
    assert not (a <= b or b <= a), f"{set(a)} and {set(b)} collide"


def test_the_collision_rule_actually_rejects_a_collision():
    """Delete the fix to test the test. The pair that really did collide has to
    still be caught, and the pair now in use has to pass."""
    assert _keys("ctrl+alt") <= _keys("ctrl+alt+1")        # the historical bug
    assert _keys("windows") <= _keys("windows+alt")        # the near miss
    a, b = _keys("windows+alt"), _keys("ctrl+windows")     # what is in use now
    assert not (a <= b or b <= a)


@pytest.mark.parametrize("bat", BATS, ids=lambda p: p.name)
def test_no_launcher_suppresses_a_hotkey(bat):
    """suppress=True left Ctrl stuck down system-wide. There is no flag for it,
    and there must never be one."""
    assert "--suppress" not in bat.read_text(encoding="ascii", errors="ignore")


def test_the_console_and_windowless_pairs_agree():
    """A console build that behaves differently from the one you actually use
    is worse than no console build: it debugs a program you are not running."""
    for name in ("Jev", "JevFlow", "JevComputerUse"):
        quiet, loud = ROOT / f"{name}.bat", ROOT / f"{name}-console.bat"
        if not (quiet.exists() and loud.exists()):
            continue
        a = [x for x in command_line(quiet) if x != "pythonw"]
        b = [x for x in command_line(loud) if x != "python"]
        assert a == b, f"{name}.bat and {name}-console.bat disagree:\n  {a}\n  {b}"
