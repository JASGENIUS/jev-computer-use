"""One place both tools read their settings from.

Everything was in the .bat files, which means changing anything meant editing a
batch script - so nothing ever got tuned. The dashboard needs somewhere to
write to, and the launchers need somewhere to read from, and it has to be the
same somewhere or they drift apart the way the .bat files already did once.

A missing or broken file must behave as defaults. Settings that can stop the
tool starting are worse than no settings.
"""
from __future__ import annotations

import json

import pytest

from jevflow import settings


@pytest.fixture
def store(tmp_path, monkeypatch):
    path = tmp_path / "settings.json"
    monkeypatch.setattr(settings, "STORE", path)
    settings.reload()
    return path


def test_defaults_exist_for_both_tools(store):
    s = settings.load()
    assert "jevflow" in s and "jcu" in s


def test_defaults_match_what_the_tools_actually_use(store):
    s = settings.load()
    assert s["jevflow"]["hotkey"] == "windows+alt"
    assert s["jcu"]["hotkey"] == "right ctrl+right shift"
    # "auto" so nobody needs an NVIDIA GPU to dictate a sentence: large-v3 on
    # a GPU, base.en on CPU, decided at startup.
    assert s["jevflow"]["model"] == "auto"
    assert s["jcu"]["model"] == "auto"


def test_saving_and_loading_round_trips(store):
    settings.set_value("jcu", "silence_hold_s", 1.75)
    settings.reload()
    assert settings.load()["jcu"]["silence_hold_s"] == 1.75


def test_a_missing_file_gives_defaults(store):
    assert not store.exists()
    assert settings.load()["jevflow"]["hotkey"] == "windows+alt"


def test_a_corrupt_file_gives_defaults(store):
    store.write_text("{ not json at all", encoding="utf-8")
    settings.reload()
    assert settings.load()["jcu"]["model"] == "auto"


def test_an_unknown_key_is_ignored_rather_than_kept(store):
    store.write_text(json.dumps({"jcu": {"nonsense": 1, "model": "tiny.en"}}),
                     encoding="utf-8")
    settings.reload()
    loaded = settings.load()
    assert "nonsense" not in loaded["jcu"]
    assert loaded["jcu"]["model"] == "tiny.en", "the valid key was thrown away too"


def test_a_value_of_the_wrong_type_falls_back(store):
    store.write_text(json.dumps({"jcu": {"silence_hold_s": "not a number"}}),
                     encoding="utf-8")
    settings.reload()
    assert isinstance(settings.load()["jcu"]["silence_hold_s"], float)


def test_it_becomes_command_line_arguments(store):
    args = settings.to_args("jcu")
    assert "--command-hotkey" in args
    assert "right ctrl+right shift" in args
    assert "--model" in args


def test_the_arguments_it_produces_actually_parse(store):
    """The real check: a setting that cannot be turned into a working command
    line is a setting that silently stops the tool from starting."""
    import main as entry
    for tool in ("jevflow", "jcu"):
        entry.build_parser().parse_args(settings.to_args(tool))


def test_changing_a_setting_changes_the_arguments(store):
    before = settings.to_args("jcu")
    settings.set_value("jcu", "stream", False)
    after = settings.to_args("jcu")
    assert before != after
    assert "--stream" not in after


def test_an_unwritable_store_does_not_raise(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "STORE", tmp_path / "no" / "such" / "dir.json")
    settings.reload()
    settings.set_value("jcu", "model", "tiny.en")     # must not raise


def test_the_two_tools_can_never_share_a_key(store):
    """The rule that matters: a chord whose keys are a subset of the other's
    always wins the race, so neither may contain the other. Right Ctrl alone
    was firing on left control, which is why it became a two-key chord."""
    def keys(chord):
        return frozenset(k.strip().lower() for k in chord.split("+") if k.strip())
    a = keys(settings.load()["jevflow"]["hotkey"])
    b = keys(settings.load()["jcu"]["hotkey"])
    assert a and b
    assert not (a <= b or b <= a), f"{set(a)} and {set(b)} collide"
    assert not (a & b), f"they share {set(a & b)}"
