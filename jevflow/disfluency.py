"""Cleaning up how people actually talk, before anything acts on it.

Speech recognition is faithful, and faithful is not always what you want. A
transcript of real speech carries "um", false starts, and thoughts restated
halfway through - and all of it goes straight into a document in dictation
mode, or into a command shortlist in computer-use mode.

Two jobs live here and they fail in opposite directions:

- **Fillers** are safe to remove. The risk is removing too few.
- **Self-corrections** are dangerous to remove. The risk is removing too many.
  "what does that say, what does that mean" is one thought restated.
  "I want coffee, I want tea" is two thoughts. A similarity score cannot tell
  them apart, so the rule is a **shared prefix of at least three tokens**,
  which separates those two exactly, and the default everywhere else is to
  leave the words alone.

One invariant outranks every rule below: cleaning a non-empty transcript must
never produce an empty one. An utterance that is nothing but "um" still comes
back as "um", because reporting nothing heard is a different and much more
confusing failure than reporting a filler.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Sounds, not words. Removing these never changes what was meant.
CORE_FILLERS = (
    "um", "umm", "ummm", "uh", "uhh", "uhhh", "erm", "er", "ah", "ahh",
    "mm", "mmm", "hmm", "hmmm", "mhm", "uhhuh", "eh",
)

# Real words used as filler. Off by default: "like" is a preposition far more
# often than it is a tic, and stripping it turns "something like this" into a
# different search.
SOFT_FILLERS = ("like", "basically", "literally", "obviously", "seriously")

# Filler phrases, matched as a unit.
FILLER_PHRASES = ("you know", "i mean to say", "sort of", "kind of")

# A clause consisting of nothing but one of these is throat-clearing.
DISCOURSE_MARKERS = (
    "so", "well", "and", "but", "okay", "ok", "right", "anyway", "yeah",
    "like", "you know", "i guess", "or something",
)

# "I said X - no, Y." Everything before the marker was abandoned.
# `sorry` and `actually` are ordinary words too, so a marker only counts when
# it is delimited: at the start of the utterance, or after punctuation.
# Without that, "tell him I'm sorry" gets cut in half.
REPAIR_MARKERS = (
    # "i meant" is listed separately from "i mean": the word boundary after
    # the alternation fails on the trailing "t", so "Not JevFlow, I meant JCU"
    # matched nothing at all and the correction was typed out in full.
    "i mean", "i meant", "sorry", "rather", "or rather", "what i meant",
    "correction",
)

# Markers unmistakable enough to count ANYWHERE, with no punctuation in front.
# This matters more than it looks: speech recognition very often returns a
# sentence with no commas at all, and "close this no wait close chrome" is the
# normal shape of a spoken correction. These are multi-word and have no
# innocent reading, unlike "sorry" in "tell him I am sorry".
FLOATING_REPAIR_MARKERS = (
    "no wait", "wait no", "actually no", "no actually", "scratch that",
    "let me rephrase", "no i mean", "i meant to say",
)

# Doubles that are grammar, not stutter: "I said that that was fine".
GRAMMATICAL_DOUBLES = frozenset({"that", "had"})

# How many leading tokens two clauses must share before the first is read as an
# abandoned restart. Three is the number that keeps "I want coffee, I want tea"
# (shares two) while catching "what does that say, what does that mean"
# (shares three). Do not lower it without a test for the parallel case.
RESTART_PREFIX_TOKENS = 3

_CLAUSE_SPLIT = re.compile(r"\s*([,;]|[.!?](?=\s|$))\s*")
_WORD = re.compile(r"[A-Za-z']+")


@dataclass
class Cleaned:
    text: str
    changed: bool = False
    removed: list[str] = field(default_factory=list)


def _marker_pattern(markers: tuple[str, ...]) -> re.Pattern:
    alt = "|".join(re.escape(m) for m in sorted(markers, key=len, reverse=True))
    # After punctuation, or opening the utterance AND followed by a comma.
    # That second condition is load-bearing: "I mean, open notepad" is someone
    # correcting themselves, but "I mean it when I say that" is a sentence, and
    # at position zero there is nothing behind it to correct anyway.
    return re.compile(
        rf"(?:(?<=[,;.!?])\s*(?:{alt})\b[\s,;.!?-]*"
        rf"|^\s*(?:{alt})\s*[,;]\s*)", re.I)


_REPAIR = _marker_pattern(REPAIR_MARKERS)
_FLOATING = re.compile(
    r"\s*\b(?:" + "|".join(re.escape(m) for m in
                           sorted(FLOATING_REPAIR_MARKERS, key=len, reverse=True))
    + r")\b[\s,;.!?-]*", re.I)

_LEADING_NOISE = re.compile(
    r"^(?:\s*\b(?:" + "|".join(re.escape(m) for m in
                               sorted(DISCOURSE_MARKERS, key=len, reverse=True))
    + r")\b[\s,;-]*)+", re.I)


# -- "like", which is two completely different words --------------------------
# From real dictation: "make like a bit of a dashboard that can open up
# like an app so I can like tweak the settings". Three fillers in one sentence,
# all of which survived, because "like" sat in the opt-in tier - and it sat
# there because stripping it blindly turns "something like this" into
# "something this", which is worse than leaving the tic in.
#
# It needs a rule, not a switch. "like" is a COMPARISON after a small closed
# set of words and a tic almost everywhere else, so the set is what gets
# listed - anything outside it is filler.
_LIKE_IS_REAL_AFTER = (
    "feel", "feels", "felt", "look", "looks", "looked", "sound", "sounds",
    "sounded", "seem", "seems", "seemed", "taste", "tastes", "smell", "smells",
    "act", "acts", "acted", "behave", "behaves", "treat", "treats",
    "something", "anything", "nothing", "someone", "anyone", "everything",
    "just", "more", "less", "much", "exactly", "somewhat", "kind", "sort",
)
# Copulas are deliberately ABSENT above. "he was like, no way" is the quotative
# tic and by far the commoner use; the genuinely comparative "it is like THIS"
# is already caught by the list below, which looks at what FOLLOWS.
# ...and before these it is comparative regardless of what came first:
# "like this one", "like that", "like him".
_LIKE_IS_REAL_BEFORE = ("this", "that", "these", "those", "him", "her", "them",
                        "it", "us", "me", "you", "mine", "yours", "ours")

_LIKE = re.compile(r"(?:(?P<before>[A-Za-z']+)\s+)?\blike\b(?:\s+(?P<after>[A-Za-z']+))?",
                   re.I)
# The word on its own. `_LIKE` deliberately swallows a neighbour on each side
# to judge the context, so its span is not the span of the word.
_LIKE_WORD = re.compile(r"\blike\b", re.I)


def _strip_filler_like(text: str, removed: list, veto=None) -> str:
    """Remove "like" only where it is a tic, never where it is a comparison.

    `veto(text, start, end, word) -> True` calls off one removal. The lists
    below are right about 21 of 23 real tics but wrong in both directions at
    the edges, and the expensive direction is deleting a word that was doing a
    job - see `filler.py` for what that costs and what the veto recovers.
    """
    out, last = [], 0
    for m in _LIKE.finditer(text):
        before = (m.group("before") or "").lower()
        after = (m.group("after") or "").lower()
        if before in _LIKE_IS_REAL_AFTER or after in _LIKE_IS_REAL_BEFORE:
            continue                      # a genuine comparison; leave it alone
        if veto is not None:
            # The span of the WORD, not of the match - the match takes in a
            # neighbour on each side. Found by searching rather than by
            # arithmetic on the group offsets, which assumed exactly one space
            # and would have handed the veto the wrong characters after a
            # comma or a double space.
            here = _LIKE_WORD.search(text, m.start(), m.end())
            try:
                if here is not None and veto(text, here.start(), here.end(), "like"):
                    continue
            except Exception:
                pass                      # a broken veto must not stop cleaning
        start = m.start()
        if m.group("before"):
            start = m.start("before") + len(m.group("before"))
        end = m.end("after") if m.group("after") else m.end()
        if m.group("after"):
            end = m.start("after")        # keep the following word
        out.append(text[last:start])
        # A single space back in. The slice spans the whitespace on BOTH sides
        # of the word, so dropping it whole welded the neighbours together and
        # "so like what" came out as "sowhat".
        out.append(" ")
        removed.append("like")
        last = end
    out.append(text[last:])
    return re.sub(r"\s{2,}", " ", "".join(out))

def _strip_fillers(text: str, aggressive: bool, removed: list[str]) -> str:
    words = list(CORE_FILLERS) + (list(SOFT_FILLERS) if aggressive else [])
    for phrase in FILLER_PHRASES:
        pat = re.compile(rf"\b{re.escape(phrase)}\b", re.I)
        if pat.search(text):
            removed.append(phrase)
            text = pat.sub(" ", text)
    alt = "|".join(sorted(words, key=len, reverse=True))
    # \b on both sides, so "um" inside "umbrella" and "ah" inside "ahsoka"
    # are untouched - the word boundary is the whole guard here.
    pat = re.compile(rf"\b(?:{alt})\b", re.I)
    for m in pat.finditer(text):
        removed.append(m.group(0))
    return pat.sub(" ", text)


def _collapse_stutter(text: str, removed: list[str]) -> str:
    out_tokens: list[str] = []
    prev_word = None
    for token in re.split(r"(\s+)", text):
        if not token.strip():
            out_tokens.append(token)
            continue
        w = _WORD.search(token)
        key = w.group(0).lower() if w else None
        # Never collapse across a clause boundary. "open, open notepad" is a
        # restart of the whole clause, and deleting the second "open" here
        # turns it into "open, notepad" - which is worse than leaving it, and
        # hides it from the clause-level rule that knows how to fix it.
        crossed = bool(out_tokens) and bool(
            re.search(r"[,;.!?]\s*$", "".join(out_tokens[-2:])))
        if (key is not None and key == prev_word and not crossed
                and key not in GRAMMATICAL_DOUBLES):
            removed.append(token)
            # Drop this token AND the whitespace that preceded it.
            if out_tokens and not out_tokens[-1].strip():
                out_tokens.pop()
            continue
        prev_word = key
        out_tokens.append(token)
    return "".join(out_tokens)


def _collapse_repeated_phrase(text: str, removed: list[str]) -> str:
    """"the client the client wants it" - a whole phrase started twice.

    Single-word doubles are handled by `_collapse_stutter`; this catches the
    multi-word case, which is what someone does when they lose their place
    rather than trip over one syllable.
    """
    toks = list(re.finditer(r"\S+", text))
    words = [_WORD.search(t.group(0)) for t in toks]
    keys = [w.group(0).lower() if w else None for w in words]
    n = len(keys)
    for size in range(4, 1, -1):          # longest repeat first
        i = 0
        while i + 2 * size <= n:
            a, b = keys[i:i + size], keys[i + size:i + 2 * size]
            if None not in a and a == b:
                removed.append(" ".join(x for x in a if x))
                start, end = toks[i].start(), toks[i + size].start()
                return _collapse_repeated_phrase(text[:start] + text[end:], removed)
            i += 1
    return text


def _split_clauses(text: str) -> list[tuple[str, str]]:
    """[(clause, trailing separator)], so the text can be put back together."""
    parts = _CLAUSE_SPLIT.split(text)
    out: list[tuple[str, str]] = []
    i = 0
    while i < len(parts):
        clause = parts[i].strip()
        sep = parts[i + 1] if i + 1 < len(parts) else ""
        # Empty clauses are KEPT here on purpose. Stripping a filler leaves a
        # hole between two commas, and only the caller knows that the comma in
        # front of the hole was bracketing the filler and should go with it.
        out.append((clause, sep))
        i += 2
    return out


def _is_noise_clause(clause: str, aggressive: bool) -> bool:
    bare = re.sub(r"[^a-z' ]", " ", clause.lower())
    bare = re.sub(r"\s+", " ", bare).strip()
    if not bare:
        return True
    return bare in DISCOURSE_MARKERS or bare in CORE_FILLERS


def _tokens(clause: str) -> list[str]:
    return [m.group(0).lower() for m in _WORD.finditer(clause)]


def _drop_abandoned_restarts(clauses: list[tuple[str, str]],
                             removed: list[str]) -> list[tuple[str, str]]:
    """A clause restated immediately afterwards was a false start.

    Requires a long shared prefix. Two clauses that merely begin the same way -
    "I want coffee, I want tea" - are two things someone said, and deleting one
    loses half the sentence.
    """
    # A clause that ENDS on the word the next one BEGINS with is one phrase
    # interrupted and picked back up: "open, um, open notepad", "search for
    # the, the weather". Stitch them into one clause and drop the duplicate.
    merged: list[tuple[str, str]] = []
    for clause, sep in clauses:
        if merged:
            prev, prev_sep = merged[-1]
            pt, ct = _tokens(prev), _tokens(clause)
            if pt and ct and pt[-1] == ct[0]:
                cut = re.sub(rf"\s*\b{re.escape(pt[-1])}\b\s*[,;]?\s*$", "",
                             prev, flags=re.I).strip()
                removed.append(pt[-1])
                merged[-1] = ((cut + " " + clause).strip(), sep)
                continue
        merged.append((clause, sep))
    clauses = merged

    keep: list[tuple[str, str]] = []
    for i, (clause, sep) in enumerate(clauses):
        if i + 1 < len(clauses):
            a, b = _tokens(clause), _tokens(clauses[i + 1][0])
            shared = 0
            for x, y in zip(a, b):
                if x != y:
                    break
                shared += 1
            # A clause that is a STRICT PREFIX of the next one was abandoned
            # and started again - "open" then "open notepad", "the" then "the
            # weather". Length does not matter here, because a complete prefix
            # cannot be a second independent thought the way "open notepad,
            # open chrome" can.
            if a and shared == len(a) < len(b):
                removed.append(clause)
                continue
            # Otherwise require a long shared run with a short divergence:
            # enough to catch "what does that say / what does that mean"
            # without touching "I want coffee / I want tea".
            if shared >= RESTART_PREFIX_TOKENS and shared >= len(a) - 2:
                removed.append(clause)
                continue
        keep.append((clause, sep))
    return keep


_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def clean_verbose(text: str, aggressive: bool = False, veto=None) -> Cleaned:
    """Clean a transcript and say what was taken out.

    Sentences are cleaned INDEPENDENTLY. A repair marker discards what it
    replaces, and what it replaces is the clause it sits in - it has no
    business reaching across a full stop. Found the hard way on real
    dictation: a paragraph ending "Not JevFlow, I meant JCU." came back as
    "JCU.", one corrected word having thrown away everything before it.
    """
    original = (text or "").strip()
    if not original:
        return Cleaned("", False, [])

    sentences = [x for x in _SENTENCE.split(original) if x.strip()]
    if len(sentences) > 1:
        out_parts, removed_all = [], []
        for sentence in sentences:
            got = _clean_one(sentence, aggressive, veto)
            if got.text:
                out_parts.append(got.text)
            removed_all.extend(got.removed)
        joined = " ".join(out_parts).strip()
        if not joined:
            return Cleaned(original, False, [])
        return Cleaned(joined, joined != original, removed_all)
    return _clean_one(original, aggressive, veto)


def _clean_one(text: str, aggressive: bool = False, veto=None) -> Cleaned:
    """One sentence."""
    original = (text or "").strip()
    if not original:
        return Cleaned("", False, [])
    removed: list[str] = []

    # 1. Self-correction first: everything before the last repair marker was
    #    abandoned, so there is no point cleaning it. Floating markers are
    #    checked too, because a spoken correction usually arrives with no
    #    punctuation at all for the delimited pattern to anchor on.
    work = original
    floating = list(_FLOATING.finditer(work))
    if floating:
        last = floating[-1]
        tail = work[last.end():].strip()
        if tail:
            removed.append(work[:last.end()].strip())
            work = tail
        else:
            head = work[:last.start()].strip(" ,;.!?")
            work = head or work
    matches = list(_REPAIR.finditer(work))
    if matches:
        last = matches[-1]
        tail = work[last.end():].strip()
        if tail:
            removed.append(work[:last.end()].strip())
            work = tail
        elif last.start() == 0:
            work = tail or work
        else:
            # A marker with nothing after it - "open notepad, I mean". Keeping
            # the head is the only reading that does not erase the instruction.
            head = work[:last.start()].strip(" ,;.!?")
            work = head or work

    # 2. Fillers, then stutters, then whole repeated phrases.
    work = _strip_fillers(work, aggressive, removed)
    work = _strip_filler_like(work, removed, veto)
    work = _collapse_stutter(work, removed)
    work = _collapse_repeated_phrase(work, removed)

    # 3. Clause-level: drop throat-clearing and abandoned restarts.
    clauses = _split_clauses(work)
    kept: list[tuple[str, str]] = []
    for idx, (clause, sep) in enumerate(clauses):
        if _is_noise_clause(clause, aggressive):
            # The commas around a filler were only there to bracket the filler.
            # Leaving one behind turns "I need you to, uh, search for X" into
            # "I need you to, search for X", which reads as a typo in a
            # document someone is dictating. The trailing empty that every
            # split produces is exempt - taking ITS separator would eat the
            # full stop off the end of the sentence.
            more_follows = any(c.strip() for c, _s in clauses[idx + 1:])
            if more_follows and kept and kept[-1][1] in (",", ";"):
                kept[-1] = (kept[-1][0], "")
            continue
        kept.append((clause, sep))
    if kept:
        # Leading "so like what does..." - discourse markers that open the
        # utterance are throat-clearing even when they are not a whole clause.
        head, sep = kept[0]
        stripped = _LEADING_NOISE.sub("", head).strip()
        if stripped != head:
            if stripped:
                removed.append(head[:len(head) - len(stripped)].strip())
                kept[0] = (stripped, sep)
            elif any(c.strip() for c, _s in kept[1:]):
                # The WHOLE clause was throat-clearing: "okay so, like, stop
                # media". Stripping it left nothing, the truthiness guard
                # rejected that, and the opener survived untouched - so two
                # markers in a row got through where one did not. The check
                # for a surviving clause is what keeps the invariant: an
                # utterance that is nothing but markers stays as it was.
                removed.append(head.strip())
                kept.pop(0)
        kept = _drop_abandoned_restarts(kept, removed)

    rebuilt = ""
    for i, (clause, sep) in enumerate(kept):
        rebuilt += clause
        last = i == len(kept) - 1
        if sep and not last:
            rebuilt += sep + " "
        elif not last:
            # A separator that was deleted with a filler still has to leave a
            # space behind, or the clauses either side are welded together.
            rebuilt += " "
        elif sep in (".", "!", "?"):
            rebuilt += sep
    rebuilt = re.sub(r"\s+", " ", rebuilt).strip(" ,;")

    # The invariant. Anything that cleans away to nothing was not cleanable,
    # and the original is a better answer than silence.
    if not rebuilt:
        return Cleaned(original, False, [])
    return Cleaned(rebuilt, rebuilt != original, [r for r in removed if r.strip()])


def clean(text: str, aggressive: bool = False) -> str:
    return clean_verbose(text, aggressive).text
