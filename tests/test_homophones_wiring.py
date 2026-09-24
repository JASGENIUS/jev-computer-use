"""The spelling check as it is actually reached, not as a unit.

Everything here is about the seam. `test_homophones.py` proves the resolver is
right; this proves the resolver is CALLED, called at the right moment, and
cannot take dictation down with it when the network is not there.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jevflow import settings
from jevflow.app import VoiceApp


class W:
    def __init__(self, word, probability):
        self.word, self.probability = word, probability


def app(**kw):
    """A VoiceApp with no window, no model and no hotkey."""
    a = VoiceApp.__new__(VoiceApp)
    a.fix_homophones = kw.get("fix_homophones", True)
    a._jev = kw.get("jev", object())
    return a


# -- the seam -----------------------------------------------------------------
def test_no_words_means_no_question(monkeypatch):
    """Word probabilities are only requested when the feature is on, so the
    list is empty the rest of the time and must not be treated as a sentence
    with nothing uncertain in it."""
    called = []
    monkeypatch.setattr("jevflow.homophones.resolve",
                        lambda *a, **k: called.append(1) or ("x", []))
    assert app()._fix_homophones("over there", []) == "over there"
    assert called == []


def test_switched_off_means_no_question(monkeypatch):
    called = []
    monkeypatch.setattr("jevflow.homophones.resolve",
                        lambda *a, **k: called.append(1) or ("x", []))
    a = app(fix_homophones=False)
    assert a._fix_homophones("over there", [W(" there", 0.5)]) == "over there"
    assert called == []


def test_the_fixed_text_is_what_comes_back(monkeypatch):
    monkeypatch.setattr("jevflow.homophones.resolve",
                        lambda text, words, jev, **k: ("over their",
                                                       [("there", "their")]))
    assert app()._fix_homophones("over there", [W(" there", 0.5)]) == "over their"


# -- it must never cost the transcript ----------------------------------------
def test_a_missing_key_does_not_lose_the_sentence(monkeypatch):
    """The Jev constructor RAISES when no key is configured. Dictation that
    drops a sentence over a spelling check is far worse than a wrong 'their'."""
    from jevflow.jev import JevError

    def boom(*a, **k):
        raise JevError("No JEV_API_KEY found.")

    monkeypatch.setattr("jevflow.jev.Jev", boom)
    a = app(jev=None)
    assert a._fix_homophones("over there", [W(" there", 0.5)]) == "over there"


def test_a_dead_network_does_not_lose_the_sentence(monkeypatch):
    def boom(*a, **k):
        raise ConnectionError("no route to host")

    monkeypatch.setattr("jevflow.homophones.resolve", boom)
    assert app()._fix_homophones("over there", [W(" there", 0.5)]) == "over there"


def test_it_gives_up_rather_than_failing_on_every_utterance(monkeypatch):
    """A key that is not there will not be there next time either. Retrying
    once per sentence turns one missing key into a permanent tax."""
    calls = []

    def boom(*a, **k):
        calls.append(1)
        raise ConnectionError("no route to host")

    monkeypatch.setattr("jevflow.homophones.resolve", boom)
    a = app()
    for _ in range(5):
        a._fix_homophones("over there", [W(" there", 0.5)])
    assert len(calls) == 1
    assert a.fix_homophones is False


# -- ordering ------------------------------------------------------------------
def test_spelling_runs_before_cleaning():
    """Cleaning DELETES words, and the word list has to still line up with the
    text for the right occurrence to be replaced. Reading the source is the
    only way to check an ordering that nothing returns."""
    src = Path(__file__).resolve().parents[1] / "jevflow" / "app.py"
    body = src.read_text(encoding="utf-8")
    fix = body.index("self._fix_homophones(text,")
    clean = body.index("disfluency.clean_verbose(text")
    assert fix < clean, "cleaning would desynchronise the word list"


# -- the switch actually reaches the tool --------------------------------------
def test_the_setting_becomes_a_real_flag(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "STORE", tmp_path / "settings.json")
    settings.reload()
    settings.set_value("jevflow", "fix_homophones", True)
    assert "--fix-spelling" in settings.to_args("jevflow")
    settings.set_value("jevflow", "fix_homophones", False)
    assert "--fix-spelling" not in settings.to_args("jevflow")
    settings.reload()


def test_the_flag_is_one_the_parser_accepts(monkeypatch, tmp_path):
    """A setting that renders a flag the entry point rejects is a setting that
    stops the tool from starting at all."""
    import main as entry
    monkeypatch.setattr(settings, "STORE", tmp_path / "settings.json")
    settings.reload()
    settings.set_value("jevflow", "fix_homophones", True)
    args = entry.build_parser().parse_args(settings.to_args("jevflow"))
    assert args.fix_homophones is True
    settings.reload()


def test_it_is_off_by_default():
    """Dictation says it runs entirely on your machine. That has to stay true
    unless somebody asks for it not to be."""
    assert settings.DEFAULTS["jevflow"]["fix_homophones"] is False


# -- the recogniser has to be asked for the probabilities ----------------------
class FakeWord:
    def __init__(self, word, probability):
        self.word, self.probability = word, probability


class FakeSegment:
    def __init__(self, text, words):
        self.text, self.words = text, words


class FakeModel:
    """Answers like faster-whisper, including its worst habit.

    `hallucinates` reproduces what base.en really does when word timestamps
    are on: it appends invented text. Measured over 16 sentences, it did this
    to ALL SIXTEEN, while large-v3 did it to none.
    """

    def __init__(self, hallucinates=False):
        self.calls = []
        self.hallucinates = hallucinates

    def transcribe(self, audio, **kw):
        self.calls.append(kw)
        info = type("Info", (), {"duration": 1.0})()
        if kw.get("word_timestamps") and self.hallucinates:
            words = [FakeWord(" over", 0.99), FakeWord(" there", 0.42),
                     FakeWord(" Thanks", 0.31), FakeWord(" for", 0.22),
                     FakeWord(" watching", 0.19)]
            return iter([FakeSegment(" over there Thanks for watching", words)]), info
        if kw.get("word_timestamps"):
            words = [FakeWord(" over", 0.99), FakeWord(" there", 0.42)]
            return iter([FakeSegment(" over there", words)]), info
        return iter([FakeSegment(" over there", [])]), info


def _audio():
    import numpy as np
    return np.zeros(16000, dtype="float32")


def test_asking_for_probabilities_runs_a_second_aligned_pass(monkeypatch):
    """faster-whisper returns no per-word probability unless word_timestamps
    is set, and setting it on the first pass would change the transcript."""
    from jevflow import stt
    m = FakeModel()
    monkeypatch.setattr(stt, "load_model", lambda *a, **k: m)
    stt.transcribe(_audio(), word_probabilities=True)
    assert len(m.calls) == 2
    assert not m.calls[0].get("word_timestamps"), "the transcript pass was aligned"
    assert m.calls[1]["word_timestamps"] is True


def test_not_asking_costs_only_one_pass(monkeypatch):
    """A second decode is not free - on a CPU it was measured at roughly four
    times the cost of the first."""
    from jevflow import stt
    m = FakeModel()
    monkeypatch.setattr(stt, "load_model", lambda *a, **k: m)
    out = stt.transcribe(_audio())
    assert len(m.calls) == 1
    assert out["words"] == []


def test_the_words_come_back_with_their_probabilities(monkeypatch):
    """The whole feature rests on this number existing."""
    from jevflow import stt
    monkeypatch.setattr(stt, "load_model", lambda *a, **k: FakeModel())
    out = stt.transcribe(_audio(), word_probabilities=True)
    assert [w.probability for w in out["words"]] == [0.99, 0.42]
    assert out["text"] == "over there"


# -- the aligned pass must never reach the transcript --------------------------
def test_a_hallucinating_aligned_pass_does_not_change_the_text(monkeypatch):
    """base.en appends invented text when word timestamps are on. If the
    transcript came from that pass, every CPU user would get 'Thanks for
    watching' typed into their document."""
    from jevflow import stt
    monkeypatch.setattr(stt, "load_model", lambda *a, **k: FakeModel(hallucinates=True))
    out = stt.transcribe(_audio(), word_probabilities=True)
    assert out["text"] == "over there"
    assert "watching" not in out["text"]


def test_invented_words_are_dropped_from_the_probabilities(monkeypatch):
    """A probability attached to a word that is not in the transcript says
    nothing about the transcript, and would throw off which occurrence of a
    word gets replaced."""
    from jevflow import stt
    monkeypatch.setattr(stt, "load_model", lambda *a, **k: FakeModel(hallucinates=True))
    out = stt.transcribe(_audio(), word_probabilities=True)
    assert [w.word.strip() for w in out["words"]] == ["over", "there"]


def test_a_pass_that_diverges_immediately_yields_no_words(monkeypatch):
    """Better no spelling check than one working from a different sentence."""
    from jevflow import stt
    assert stt._agreeing_prefix("over there",
                                [FakeWord(" completely", 0.4),
                                 FakeWord(" different", 0.4)]) == []


def test_the_prefix_stops_at_the_first_disagreement():
    from jevflow import stt
    words = [FakeWord(" over", 0.9), FakeWord(" there", 0.4), FakeWord(" extra", 0.2)]
    assert len(stt._agreeing_prefix("over there", words)) == 2


def test_the_prefix_ignores_punctuation_and_case():
    """The aligned pass writes " There," where the transcript has "there"."""
    from jevflow import stt
    words = [FakeWord(" Over", 0.9), FakeWord(" there,", 0.4)]
    assert len(stt._agreeing_prefix("over there", words)) == 2


def test_no_alignment_pass_at_all_on_an_empty_transcript(monkeypatch):
    """Silence has no words to be unsure about, and the second decode is the
    expensive one."""
    from jevflow import stt

    class Silent(FakeModel):
        def transcribe(self, audio, **kw):
            self.calls.append(kw)
            return iter([]), type("Info", (), {"duration": 1.0})()

    m = Silent()
    monkeypatch.setattr(stt, "load_model", lambda *a, **k: m)
    out = stt.transcribe(_audio(), word_probabilities=True)
    assert len(m.calls) == 1
    assert out["words"] == []


def test_transcribe_always_returns_a_words_key():
    """Callers index it. An empty clip returning a dict without it is an
    AttributeError on the quietest possible input."""
    import numpy as np

    from jevflow import stt
    out = stt.transcribe(np.zeros(0, dtype="float32"))
    assert out["words"] == []
