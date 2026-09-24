"""Staying on without acting on the room."""
from __future__ import annotations

import pytest

from jevflow import wake


@pytest.fixture
def gate():
    return wake.TranscriptGate(word="jev")


@pytest.mark.parametrize("said,command", [
    ("jev open notepad", "open notepad"),
    ("hey jev open notepad", "open notepad"),
    ("hey jev, open notepad", "open notepad"),
    ("okay jev search for apples", "search for apples"),
    ("Jev. Close this window", "Close this window"),
])
def test_the_wake_word_is_stripped_before_the_command(gate, said, command):
    heard = gate.check(said)
    assert heard.awake, f"{said!r} did not wake it"
    assert heard.command == command


@pytest.mark.parametrize("said", [
    "open notepad",
    "so then I told him it was fine",
    "what did jev say about that",      # mentions the name, does not address it
    "the jev flow project is going well",
    "",
])
def test_the_room_does_not_wake_it(gate, said):
    assert gate.check(said).awake is False


def test_a_misheard_wake_word_still_works(gate):
    """A smaller model slips on the name. A wake word that works most of the
    time reads as a broken program rather than as a mishearing."""
    for said in ("jeff open notepad", "hey jeb open notepad", "chev open notepad"):
        assert gate.check(said).awake, said


def test_a_bare_name_arms_it_for_the_next_sentence(gate):
    first = gate.check("jev", now=100.0)
    assert first.awake and first.armed and first.command == ""
    second = gate.check("open notepad", now=101.0)
    assert second.awake and second.command == "open notepad"


def test_being_armed_lasts_exactly_one_sentence(gate):
    gate.check("jev", now=100.0)
    gate.check("open notepad", now=101.0)
    assert gate.check("and now the weather", now=102.0).awake is False


def test_arming_expires(gate):
    gate.check("jev", now=100.0)
    late = gate.check("open notepad", now=100.0 + wake.ARM_TTL_S + 1)
    assert late.awake is False


def test_a_full_instruction_does_not_leave_it_armed(gate):
    gate.check("jev open notepad", now=100.0)
    assert gate.check("anyway as I was saying", now=101.0).awake is False


def test_a_custom_word_does_not_inherit_jevs_homophones():
    g = wake.TranscriptGate(word="computer")
    assert g.check("computer open notepad").awake
    assert g.check("jeff open notepad").awake is False


# -- choosing an engine -------------------------------------------------------
def test_no_gate_means_listen_to_everything():
    assert wake.build("none") is None
    assert wake.build("") is None


def test_the_transcript_gate_is_the_default():
    assert isinstance(wake.build("transcript"), wake.TranscriptGate)


def test_an_unknown_engine_is_an_error_not_a_silent_no_gate():
    with pytest.raises(ValueError):
        wake.build("magic")


def test_the_acoustic_gate_refuses_a_phrase_it_has_no_model_for():
    """A detector that can never fire turns the whole tool off, silently, with
    nothing anywhere to explain why nothing responds."""
    pytest.importorskip("openwakeword")
    with pytest.raises(ValueError) as exc:
        wake.AcousticGate(phrase="jev")
    assert "trained" in str(exc.value).lower()


def test_the_available_phrases_are_read_from_disk_not_assumed():
    pytest.importorskip("openwakeword")
    names = wake.AcousticGate.available()
    assert names, "no wake-word models found at all"
    # The helper models are not wake words and must never be offered as one.
    for helper in ("embedding_model", "melspectrogram", "silero_vad"):
        assert helper not in names


def test_the_acoustic_gate_loads_a_phrase_it_does_have():
    pytest.importorskip("openwakeword")
    phrase = wake.AcousticGate.available()[0]
    g = wake.AcousticGate(phrase=phrase)
    assert phrase.replace("_", " ") in g.label


def test_silence_does_not_trigger_the_acoustic_gate():
    pytest.importorskip("openwakeword")
    import numpy as np
    phrase = wake.AcousticGate.available()[0]
    g = wake.AcousticGate(phrase=phrase)
    assert g.feed(np.zeros(16000, dtype="float32")) is False
