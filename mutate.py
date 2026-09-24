"""Break each fix on purpose and confirm a test notices.

A green suite proves nothing on its own - it could be green because the checks
cannot fail. Each mutation below reintroduces a real bug that was fixed in this
session; every one of them must turn the suite red.
"""
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent

MUTATIONS = [
    ("jevflow/parse.py",
     'return bool(b) and bool(_AFFIRM.match(b))',
     'return "yes" in (transcript or "").lower()',
     "a sentence containing 'yes' authorises the pending action",
     "tests/test_parse.py"),

    ("jevflow/policy.py",
     'return REFUSE if p is None else decide_with(p, float(certainty))',
     'return decide_with(p or POLICIES["web_search"], float(certainty))',
     "an unknown verb falls through to acting",
     "tests/test_policy.py"),

    ("jevflow/sites.py",
     '|launch|load)"',
     '|launch|load|[a-z]+)"',
     "a spoken 'dot' after any word becomes a web address",
     "tests/test_sites.py"),

    ("jevflow/command.py",
     'if app is None:\n            # Jev named something code never offered.',
     'if False:\n            # Jev named something code never offered.',
     "a choice outside the shortlist gets launched",
     "tests/test_command.py"),

    ("jevflow/app.py",
     '            if not heard.awake:',
     '            if False:',
     "the wake gate stops gating and the room is acted on",
     "tests/test_session_wake.py"),

    ("jevflow/command.py",
     'verified=True if title else None',
     'verified=True',
     "opening a site claims verification it does not have",
     "tests/test_command.py"),

    ('jevflow/homophones.py',
     '    floor = max(float(original_probability), MIN_CONFIDENCE)',
     '    floor = -1.0',
     'any answer at all replaces the word, however unsure',
     'tests/test_homophones.py'),

    ('jevflow/homophones.py',
     '        return _match_case(m.group(0), chosen) if seen == occurrence else m.group(0)',
     '        return _match_case(m.group(0), chosen)',
     'every occurrence is replaced, not just the doubtful one',
     'tests/test_homophones.py'),

    ('jevflow/homophones.py',
     '        if not candidates(text):',
     '        if False:',
     'a round trip is spent on words with no homophone',
     'tests/test_homophones.py'),

    ('jevflow/homophones.py',
     '        context = blank_out(out, heard, occurrence=before)',
     '        context = out',
     'the recogniser guess is left in the sentence to anchor on',
     'tests/test_homophones.py'),

    ('jevflow/homophones.py',
     '    for _i, heard, prob in sorted(found, key=lambda t: t[2])[:max_questions]:',
     '    for _i, heard, prob in sorted(found, key=lambda t: t[2]):',
     'a mumbled paragraph asks unboundedly many questions',
     'tests/test_homophones.py'),

    ('jevflow/stt.py',
     '        aligned, _ = m.transcribe(audio, **settings, word_timestamps=True)',
     '        aligned, _ = m.transcribe(audio, **settings)',
     'word alignment never turns on, so there are no probabilities',
     'tests/test_homophones_wiring.py'),

    ('jevflow/stt.py',
     '        words = _agreeing_prefix(text, every)',
     '        words = every',
     'invented words from the aligned pass are trusted as real',
     'tests/test_homophones_wiring.py'),

    ('jevflow/stt.py',
     r'    text = re.sub(r"\s+", " ", " ".join(s.text.strip() for s in segments)).strip()',
     '    text = ""',
     'the transcript is thrown away',
     'tests/test_homophones_wiring.py'),

    ('jevflow/stt.py',
     '    if word_probabilities and text:',
     '    if word_probabilities:',
     'silence pays for a second decode it can never use',
     'tests/test_homophones_wiring.py'),

    ('jevflow/stt.py',
     '        return {"text": "", "seconds": 0.0, "audio_seconds": 0.0, "words": []}',
     '        return {"text": "", "seconds": 0.0, "audio_seconds": 0.0}',
     'an empty clip returns a dict callers index into and crash on',
     'tests/test_homophones_wiring.py'),

    ('jevflow/app.py',
     '        text = self._fix_homophones(text, result.get("words") or [])',
     '        pass',
     'the spelling check is never reached',
     'tests/test_homophones_wiring.py'),

    ('jevflow/app.py',
     '            self.fix_homophones = False\n            return text',
     '            return text',
     'a missing key is retried on every single utterance',
     'tests/test_homophones_wiring.py'),

    ('jevflow/settings.py',
     '        "fix_homophones": False,',
     '        "fix_homophones": True,',
     'dictation uses the network by default without being asked',
     'tests/test_homophones_wiring.py'),

    ('jevflow/settings.py',
     '        if s.get("fix_homophones"):\n            args.append("--fix-spelling")',
     '        pass',
     'the dashboard switch never reaches the tool',
     'tests/test_homophones_wiring.py'),

    ('jevflow/filler.py',
     '        held = bool(role) and role != FILLER and confidence > self.sure_above',
     '        held = role == FILLER',
     'the veto fires on a FILLER answer, deleting what it should keep',
     'tests/test_filler.py'),

    ('jevflow/filler.py',
     '        held = bool(role) and role != FILLER and confidence > self.sure_above',
     '        held = bool(role) and role != FILLER',
     'a hesitant answer is enough to override the rule',
     'tests/test_filler.py'),

    ('jevflow/filler.py',
     '            self._seen[key] = False\n            return False',
     '            raise',
     'a dead network takes the whole sentence with it',
     'tests/test_filler.py'),

    ('jevflow/filler.py',
     '        if self.asked >= self.max_questions:\n            return False',
     '        if False:\n            return False',
     'a rambling paragraph asks unboundedly many questions',
     'tests/test_filler.py'),

    ('jevflow/filler.py',
     '        if key in self._seen:\n            return self._seen[key]',
     '        if False:\n            return self._seen[key]',
     'the same question is paid for over and over',
     'tests/test_filler.py'),

    ('jevflow/filler.py',
     '        roles = AMBIGUOUS.get((word or "").lower())',
     '        roles = AMBIGUOUS.get((word or "").lower()) or AMBIGUOUS["like"]',
     'an unknown word falls back to a default role set and costs a round trip',
     'tests/test_filler.py'),

    ('jevflow/disfluency.py',
     '            here = _LIKE_WORD.search(text, m.start(), m.end())',
     '            here = m',
     'the veto is handed the neighbouring words, not the word',
     'tests/test_filler.py'),

    ('jevflow/disfluency.py',
     '                if here is not None and veto(text, here.start(), here.end(), "like"):\n                    continue',
     '                pass',
     'the veto is never consulted and the verb is deleted again',
     'tests/test_filler.py'),

    ('jevflow/settings.py',
     '        "fix_fillers": False,',
     '        "fix_fillers": True,',
     'the filler check uses the network by default without being asked',
     'tests/test_filler.py'),

    ('jevflow/settings.py',
     '        if s.get("fix_fillers"):\n            args.append("--check-fillers")',
     '        pass',
     'the filler switch never reaches the tool',
     'tests/test_filler.py'),

    ('jevflow/app.py',
     '        if not self.fix_fillers:\n            return None',
     '        if False:\n            return None',
     'a Jev client is built even when the feature is off',
     'tests/test_filler.py'),

    ('jevflow/command.py',
     '        if band == policy.CONFIRM and self.never_ask:',
     '        if False:',
     'JCU stops to ask yes or no again',
     'tests/test_never_asks.py'),
]


def run(target):
    p = subprocess.run([sys.executable, "-m", "pytest", target, "-q", "--no-header"],
                       cwd=str(ROOT), capture_output=True, text=True, timeout=300)
    return p.returncode


fails = 0
for rel, old, new, what, target in MUTATIONS:
    path = ROOT / rel
    original = path.read_text(encoding="utf-8")
    if old not in original:
        print(f"  ??  {rel}: anchor not found - CANNOT TEST: {what}")
        fails += 1
        continue
    path.write_text(original.replace(old, new, 1), encoding="utf-8")
    try:
        code = run(target)
    finally:
        path.write_text(original, encoding="utf-8")
    caught = code != 0
    print(f"  {'CAUGHT ' if caught else 'MISSED '} {what}")
    if not caught:
        fails += 1

print()
print("every mutation was caught" if not fails
      else f"{fails} mutation(s) went unnoticed - those checks cannot fail")
sys.exit(1 if fails else 0)
