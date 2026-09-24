"""The filler veto: Jev calling off a deletion the rule got wrong.

The measurements behind this live in `jevflow/filler.py`. What these check is
the property that makes it safe to run at all: **a veto can only ever keep a
word.** No answer, a wrong answer, a hesitant answer, a broken network - all of
them leave the cleaner exactly as it shipped before.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jevflow import disfluency as d
from jevflow import filler


class FakeJev:
    def __init__(self, role, confidence=0.99, raises=None):
        self.role, self.confidence, self.raises = role, confidence, raises
        self.asked = []

    def ask(self, state, questions):
        self.asked.append((state, questions))
        if self.raises:
            raise self.raises
        answer = {} if self.role is None else {"choice": self.role,
                                               "confidence": self.confidence}
        return {"answers": {"role": answer}, "latency_ms": 90.0}


# -- the real bug this exists to fix ------------------------------------------
def test_the_verb_survives():
    """The shipped rule turns this into "I would really to use it". It is the
    worst error the cleaner makes: a deletion leaves nothing to notice."""
    said = "I would really like to use it"
    assert d.clean_verbose(said).text == "I would really to use it"
    out = d.clean_verbose(said, veto=filler.Veto(FakeJev("verb", 0.99)))
    assert out.text == said


def test_a_real_tic_is_still_removed():
    """A veto that keeps everything is just switching the cleaner off."""
    out = d.clean_verbose("so like what do you want",
                          veto=filler.Veto(FakeJev("filler", 0.99)))
    assert "like" not in out.text


# -- a veto can only ever KEEP a word -----------------------------------------
def test_a_filler_answer_never_blocks_the_deletion():
    v = filler.Veto(FakeJev("filler", 0.99))
    assert v("so like what", 3, 7, "like") is False


def test_an_unsure_answer_never_blocks_the_deletion():
    """Deleting is what shipped. Overriding it needs more than a guess."""
    v = filler.Veto(FakeJev("verb", 0.50))
    assert v("I like to use it", 2, 6, "like") is False


def test_no_answer_at_all_never_blocks_the_deletion():
    v = filler.Veto(FakeJev(None))
    assert v("I like to use it", 2, 6, "like") is False


def test_a_dead_network_never_blocks_the_deletion():
    """The text must not depend on the weather: no network means the rule
    alone, which is exactly what shipped before."""
    v = filler.Veto(FakeJev("verb", raises=ConnectionError("no route")))
    assert v("I like to use it", 2, 6, "like") is False


def test_no_jev_at_all_is_fine():
    assert filler.Veto(None)("I like it", 2, 6, "like") is False


def test_a_broken_veto_does_not_stop_the_cleaner():
    """A crash in an optional accuracy feature must not cost the sentence."""
    def boom(*a, **k):
        raise RuntimeError("bang")
    out = d.clean_verbose("um so like what do you want", veto=boom)
    assert out.text and "um" not in out.text.lower()


# -- exactly one answer means delete ------------------------------------------
def test_only_one_role_in_the_list_means_tic():
    """The structural safety: a confused model can only fail towards keeping
    the word, because every other option keeps it."""
    for word, roles in filler.AMBIGUOUS.items():
        assert filler.FILLER in roles, word
        assert len(roles) >= 2, f"{word} has nothing to be mistaken for"


def test_an_unknown_word_is_never_asked_about():
    """"um" is never doing a job, and a round trip is not free."""
    jev = FakeJev("filler")
    assert filler.Veto(jev)("um hello", 0, 2, "um") is False
    assert jev.asked == []


# -- the question has to be about ONE occurrence -------------------------------
def test_the_occurrence_under_question_is_marked():
    """A sentence with four "like"s otherwise asks four identical questions
    and gets one decision applied four times."""
    jev = FakeJev("verb")
    text = "like I like it like this like"
    filler.Veto(jev)(text, 7, 11, "like")
    sentence = jev.asked[0][0]["sentence"]
    assert sentence.count(filler.MARKER) == 2, "not exactly one word marked"
    assert sentence == "like I " + filler.MARKER + "like" + filler.MARKER + " it like this like"


def test_marking_does_not_change_the_other_occurrences():
    out = filler.mark("like I like it", 7, 11)
    assert out == "like I " + filler.MARKER + "like" + filler.MARKER + " it"
    assert out.count(filler.MARKER) == 2


def test_two_occurrences_are_two_questions():
    jev = FakeJev("filler", 0.99)
    d.clean_verbose("so like what, like whatever", veto=filler.Veto(jev))
    assert len(jev.asked) == 2
    assert jev.asked[0][0]["sentence"] != jev.asked[1][0]["sentence"]


# -- budget --------------------------------------------------------------------
def test_the_number_of_questions_is_capped():
    """One round trip per candidate; a rambling paragraph needs a ceiling."""
    jev = FakeJev("verb")
    v = filler.Veto(jev, max_questions=2)
    text = "like like like like like like"
    for i in range(0, 25, 5):
        v(text, i, i + 4, "like")
    assert len(jev.asked) <= 2


def test_the_same_question_is_only_asked_once():
    jev = FakeJev("verb")
    v = filler.Veto(jev)
    for _ in range(4):
        v("I like to use it", 2, 6, "like")
    assert len(jev.asked) == 1


def test_running_out_of_budget_allows_the_deletion():
    """Not keeping the word - the rule is the fallback, in both directions."""
    v = filler.Veto(FakeJev("verb"), max_questions=0)
    assert v("I like to use it", 2, 6, "like") is False


# -- the span handed over is the word ------------------------------------------
def test_the_veto_is_handed_the_word_not_its_neighbours():
    """The rule's own regex swallows a neighbour on each side to judge the
    context, so its span is NOT the span of the word."""
    seen = []
    for text in ("I would really like to use it", "so like what do you want",
                 "like I don't have it", "he said, like,  two   spaces",
                 "well like, a comma after", "LIKE in capitals"):
        d.clean_verbose(text, veto=lambda t, s, e, w: seen.append(t[s:e]) or False)
    assert seen, "the veto was never consulted"
    assert all(x.lower() == "like" for x in seen), seen


# -- the switch reaches the tool -----------------------------------------------
def test_the_setting_becomes_a_real_flag(monkeypatch, tmp_path):
    from jevflow import settings
    monkeypatch.setattr(settings, "STORE", tmp_path / "settings.json")
    settings.reload()
    settings.set_value("jevflow", "fix_fillers", True)
    assert "--check-fillers" in settings.to_args("jevflow")
    settings.reload()


def test_the_flag_is_one_the_parser_accepts(monkeypatch, tmp_path):
    import main as entry

    from jevflow import settings
    monkeypatch.setattr(settings, "STORE", tmp_path / "settings.json")
    settings.reload()
    settings.set_value("jevflow", "fix_fillers", True)
    assert entry.build_parser().parse_args(settings.to_args("jevflow")).fix_fillers
    settings.reload()


def test_it_is_off_by_default():
    from jevflow import settings
    assert settings.DEFAULTS["jevflow"]["fix_fillers"] is False


def test_the_app_builds_no_veto_when_it_is_off(monkeypatch):
    """Off has to mean no Jev client, not a client that is never asked.

    The client is faked so this holds with or without a key. Built for real,
    a missing key made construction fail quietly, `_jev` stayed None, and the
    test passed even with the guard deleted - on every machine but one."""
    from jevflow.app import VoiceApp
    built = []
    monkeypatch.setattr("jevflow.jev.Jev", lambda *a, **k: built.append(1) or object())
    a = VoiceApp.__new__(VoiceApp)
    a.fix_fillers = False
    a._jev = None
    assert a._filler_veto() is None
    assert a._jev is None
    assert not built, "a Jev client was built with the feature off"


def test_a_missing_key_switches_it_off_instead_of_crashing(monkeypatch):
    from jevflow.app import VoiceApp
    from jevflow.jev import JevError

    def boom(*a, **k):
        raise JevError("No JEV_API_KEY found.")

    monkeypatch.setattr("jevflow.jev.Jev", boom)
    a = VoiceApp.__new__(VoiceApp)
    a.fix_fillers = True
    a._jev = None
    assert a._filler_veto() is None
    assert a.fix_fillers is False
