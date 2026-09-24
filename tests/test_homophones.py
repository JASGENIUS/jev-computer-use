"""Resolving the words the recogniser was not sure about.

Whisper returns a per-word probability, and it drops exactly where you would
expect: homophones. "there" and "their" sound identical, so the model is
guessing from context it does not have much of - and it guesses wrong often
enough to be the most visible error in dictation.

This is the same shape as the app matching. **Code finds the uncertain word and
builds the candidate set; Jev picks one.** Jev answers a typed choice, so it
cannot invent a third word - it can only choose among spellings that are
genuinely confusable with what was heard.

The rule that keeps it safe: a replacement must be better supported than the
original. Swapping a word the recogniser was 60% sure of for one Jev is 55%
sure of makes the transcript worse while looking like a feature.
"""
from __future__ import annotations

import pytest

from jevflow import homophones as h


# -- the candidate set --------------------------------------------------------
@pytest.mark.parametrize("word,expect", [
    ("there", {"their", "they're"}),
    ("their", {"there", "they're"}),
    ("to", {"too", "two"}),
    ("your", {"you're"}),
    ("its", {"it's"}),
    ("hear", {"here"}),
    ("then", {"than"}),
    ("weather", {"whether"}),
])
def test_confusable_words_have_candidates(word, expect):
    got = set(h.candidates(word))
    assert expect <= got, f"{word} -> {got}"


def test_the_word_itself_is_always_a_candidate():
    """Jev must be able to answer "it was right the first time"."""
    assert "there" in h.candidates("there")


@pytest.mark.parametrize("word", ["dashboard", "notepad", "transcription", "xylophone"])
def test_an_unconfusable_word_has_no_candidates(word):
    """No point asking about a word with nothing to confuse it with."""
    assert h.candidates(word) == []


def test_case_and_punctuation_do_not_hide_a_candidate():
    assert h.candidates("There,") and h.candidates("THEIR")


def test_the_table_is_symmetric():
    """If A can be mistaken for B, B can be mistaken for A. An asymmetric
    table fixes an error in one direction only."""
    for word in h.ALL_WORDS:
        for other in h.candidates(word):
            if other.lower() == word.lower():
                continue
            # Compared lower-cased: the table keeps display forms like "I'll",
            # and casing is restored at replacement time, not stored here.
            back = [c.lower() for c in h.candidates(other)]
            assert word.lower() in back, f"{word} -> {other} is one-way"


# -- finding the uncertain ones -----------------------------------------------
class W:
    def __init__(self, word, probability):
        self.word, self.probability = word, probability


def test_only_low_confidence_words_are_questioned():
    words = [W(" I", 0.99), W(" saw", 0.98), W(" there", 0.55), W(" dog", 0.97)]
    got = h.uncertain(words, threshold=0.80)
    assert [w for _i, w, _p in got] == ["there"]


def test_a_low_confidence_word_with_no_candidates_is_left_alone():
    """Being unsure about "xylophone" is not something a spelling choice can
    fix, and asking would waste a round trip on every mumble."""
    words = [W(" xylophone", 0.31), W(" there", 0.55)]
    got = h.uncertain(words, threshold=0.80)
    assert [w for _i, w, _p in got] == ["there"]


def test_a_confident_homophone_is_not_second_guessed():
    words = [W(" their", 0.97)]
    assert h.uncertain(words, threshold=0.80) == []


def test_nothing_to_do_on_an_empty_transcript():
    assert h.uncertain([], threshold=0.8) == []


# -- applying a decision ------------------------------------------------------
def test_a_confident_correction_is_applied():
    out = h.apply_choice("I left it over there", "there", "their", confidence=0.9,
                         original_probability=0.55)
    assert out == "I left it over their"


def test_a_correction_no_better_supported_than_the_original_is_refused():
    """The rule that stops this making things worse. 0.55 -> 0.50 is not a fix,
    it is a coin flip presented as an improvement."""
    out = h.apply_choice("I left it over there", "there", "their", confidence=0.50,
                         original_probability=0.55)
    assert out == "I left it over there"


def test_choosing_the_same_word_changes_nothing():
    out = h.apply_choice("over there", "there", "there", confidence=0.99,
                         original_probability=0.55)
    assert out == "over there"


def test_only_the_uncertain_occurrence_is_replaced():
    """"there" appearing twice must not both change because one was unsure."""
    text = "there is a book over there"
    out = h.apply_choice(text, "there", "their", confidence=0.95,
                         original_probability=0.5, occurrence=1)
    assert out == "there is a book over their", out


def test_capitalisation_is_preserved():
    out = h.apply_choice("There you go", "There", "Their", confidence=0.95,
                         original_probability=0.5)
    assert out.startswith("Their")


# -- asking Jev ---------------------------------------------------------------
class FakeJev:
    def __init__(self, choice, confidence=0.95, certain=0.95, raises=None):
        self.choice, self.confidence = choice, confidence
        self.certain, self.raises = certain, raises
        self.asked = []

    def ask(self, state, questions):
        self.asked.append((state, questions))
        if self.raises:
            raise self.raises
        return {"answers": {"word": {"choice": self.choice,
                                     "confidence": self.confidence},
                            "certain": {"noul": self.certain}},
                "latency_ms": 90.0}


def test_a_confident_answer_is_applied():
    words = [W(" over", 0.99), W(" there", 0.55)]
    out, changes = h.resolve("I left it over there", words, FakeJev("their"))
    assert out == "I left it over their"
    assert changes == [("there", "their")]


def test_jev_is_only_asked_about_confusable_words():
    words = [W(" xylophone", 0.20), W(" dashboard", 0.30)]
    jev = FakeJev("their")
    out, changes = h.resolve("xylophone dashboard", words, jev)
    assert jev.asked == [], "it spent a round trip on a word with no homophone"
    assert changes == []


def test_the_sentence_is_given_as_context():
    """Without it Jev is guessing from the same nothing the recogniser had."""
    words = [W(" there", 0.55)]
    jev = FakeJev("their")
    h.resolve("I left it over there", words, jev)
    state = jev.asked[0][0]
    assert "I left it over" in str(state)


def test_the_doubtful_word_is_blanked_out_of_the_context():
    """Leaving the guess in the sentence anchors the answer to it: the same
    sentence scored `there` 1.00 spelled one way and 0.50 spelled the other."""
    words = [W(" there", 0.55)]
    jev = FakeJev("their")
    h.resolve("I left it over there", words, jev)
    state = str(jev.asked[0][0])
    assert "____" in state
    assert "over there" not in state, "the recogniser guess was left in to anchor on"


def test_the_rest_of_the_sentence_survives_blanking():
    words = [W(" there", 0.55)]
    jev = FakeJev("their")
    h.resolve("I left it over there", words, jev)
    assert "I left it over" in str(jev.asked[0][0])


def test_blanking_hits_only_the_uncertain_occurrence():
    assert h.blank_out("there is a book over there", "there", occurrence=1) ==         "there is a book over ____"
    assert h.blank_out("there is a book over there", "there", occurrence=0) ==         "____ is a book over there"


def test_a_successful_replacement_does_not_crash_on_logging():
    """The log line named a variable the refactor removed, so every SUCCESSFUL
    swap raised NameError while every refused one passed."""
    import logging
    logging.getLogger("jevflow.homophones").setLevel(logging.INFO)
    out = h.apply_choice("over there", "there", "their", confidence=0.95,
                         original_probability=0.55)
    assert out == "over their"


def test_jev_being_down_never_costs_the_transcript():
    """Dictation failing because a spelling check could not reach the network
    is far worse than a wrong "their"."""
    words = [W(" there", 0.55)]
    out, changes = h.resolve("over there", words, FakeJev("their",
                                                          raises=RuntimeError("no net")))
    assert out == "over there"
    assert changes == []


def test_no_jev_at_all_is_fine():
    words = [W(" there", 0.55)]
    assert h.resolve("over there", words, None) == ("over there", [])


def test_a_hesitant_answer_is_not_applied():
    words = [W(" there", 0.75)]
    out, _ = h.resolve("over there", words, FakeJev("their", confidence=0.60))
    assert out == "over there", "it replaced on an answer weaker than the original"


def test_the_gate_reads_the_choice_confidence_not_the_sentence_noul():
    """They answer different questions. Gating on the noul was measured across
    30 sentences and cost five correct fixes while preventing nothing."""
    words = [W(" there", 0.55)]
    # Sure about the word, unsure whether the sentence settles it.
    out, changes = h.resolve("I left it over there", words,
                             FakeJev("their", confidence=0.93, certain=0.20))
    assert changes == [("there", "their")], "the sentence noul blocked a confident word"


def test_a_confident_answer_below_the_floor_is_refused():
    """A word the recogniser barely believed in still needs a real answer, not
    merely one that beats a very low bar."""
    words = [W(" there", 0.10)]
    out, _ = h.resolve("over there", words, FakeJev("their", confidence=0.45))
    assert out == "over there"


def test_the_number_of_questions_is_capped():
    """A mumbled paragraph could otherwise ask twenty times."""
    words = [W(" there", 0.3), W(" to", 0.3), W(" your", 0.3),
             W(" its", 0.3), W(" then", 0.3), W(" which", 0.3)]
    jev = FakeJev("their")
    h.resolve("there to your its then which", words, jev, max_questions=2)
    assert len(jev.asked) <= 2


def test_the_least_certain_words_are_asked_about_first():
    words = [W(" there", 0.78), W(" to", 0.20)]
    jev = FakeJev("too")
    h.resolve("there to", words, jev, max_questions=1)
    assert jev.asked[0][0]["the_word_sounded_like"] == "to"
