"""Pulling the pieces out of a spoken sentence.

All of it happens in code, and that is not an implementation detail. Jev answers
typed questions - a choice, a score, a certainty - and cannot hand back a
free-form string. So "open Chrome and delete everything" can never become a
command string: the words to search for, the words to type and which media key
to press are extracted here by pattern, and Jev only judges what KIND of request
it was. The action space is bounded by what this file can produce.

Every regex here is a raw string. A patch script once wrote `\\b` through a
non-raw Python string, so it became a literal backspace and silently corrupted
a pattern that still looked correct in the file.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

# Words that introduce a command rather than name the thing being opened.
_LEAD = re.compile(
    r"^\s*(hey\s+jev|ok(ay)?\s+jev|jev)?[,.\s]*"
    r"(please\s+)?(can\s+you\s+)?(go\s+ahead\s+and\s+)?"
    r"(open|launch|start|run|bring\s+up|pull\s+up|switch\s+to|go\s+to|"
    r"type\s+(?:out\s+|in\s+)?(?:the\s+(?:word|phrase|text)\s+)?|"
    r"write\s+(?:out\s+)?(?:the\s+(?:word|phrase|text)\s+)?|enter\s+)\s*", re.I)
_TRAIL = re.compile(r"\s*(please|for\s+me|now|up)?\s*[.!?]*\s*$", re.I)

# Explicit search verbs, tried first and rightmost-first: in "open a Google
# Chrome tab and search for apples" the word "google" is part of the browser
# name, not the instruction, so matching it would capture the whole sentence.
_QUERY_STRONG = re.compile(
    r"(?:search\s+(?:for|up)|search|look\s+up|find\s+me)\s+(?P<q>.+?)"
    r"\s*(?:please|for\s+me|on\s+(?:google|the\s+web|the\s+internet))?\s*[.!?]*$", re.I)
# "google X" only counts when X is not itself the browser.
_QUERY_GOOGLE = re.compile(r"\bgoogle\s+(?!chrome\b)(?P<q>.+?)\s*[.!?]*$", re.I)

# Leading words of a close request, stripped so the app name is left.
CLOSE = re.compile(
    r"^\s*(hey\s+jev|jev)?[,.\s]*(please\s+)?(close|quit|exit|shut)\s+"
    r"(down\s+|out\s+of\s+)?(the\s+|this\s+|that\s+)?", re.I)

# "close this" names nothing: the target is whatever had focus.
_UNNAMED = ("this", "that", "it", "window", "this window", "that window", "")


def app_query(transcript: str) -> str:
    """Strip the command words so only the app name is matched against."""
    q = _LEAD.sub("", transcript or "")
    return _TRAIL.sub("", q).strip(" .,")


def search_query(transcript: str) -> str:
    """The thing to look up, if this sounds like a search.

    Rightmost match wins: "open Chrome and search for apples" should yield
    "apples", not everything after the first verb-ish word.
    """
    t = transcript or ""
    matches = list(_QUERY_STRONG.finditer(t))
    if matches:
        return matches[-1].group("q").strip(" .,")
    m = _QUERY_GOOGLE.search(t)
    return m.group("q").strip(" .,") if m else ""


def close_target(transcript: str) -> Optional[str]:
    """The app named in a close request, or None when it means "this window"."""
    if not CLOSE.search(transcript or ""):
        return None
    rest = CLOSE.sub("", transcript).strip(" .,")
    return None if rest.lower() in _UNNAMED else rest


# -- conversational framing ---------------------------------------------------
# Every verb pattern in this file is anchored at the start of the sentence,
# which is right - a "close" in the middle of a sentence is usually a word, not
# an instruction. But people do not open their mouths with a bare verb. They
# say "now let's close Notepad" and "can you open Claude", and the anchor then
# matches nothing at all.
#
# Measured on a real JCU session log, 2026-09-22: "Now let's close Modrinth and close
# Notepad" produced ONE step with no close target, so nothing happened and the
# capsule said "nothing named, and no window was focused". The same sentence
# without the two framing words split into two steps and resolved both.
#
# Stripped repeatedly, because they stack: "okay so now can you please open".
# Only ever from the FRONT - "search for how to close a window" keeps every
# word of its query.
_FRAMING = re.compile(
    r"^\s*(?:"
    r"okay|ok|alright|right|so|now|then|and|well|yeah|yes|um|uh|hey|please|"
    r"let\s*'?s|lets|"
    r"can\s+you|could\s+you|would\s+you|will\s+you|"
    r"i\s+want\s+you\s+to|i\s+need\s+you\s+to|i\s+want\s+to|"
    r"go\s+ahead\s+and|for\s+me"
    r")\b[,.\s]*",
    re.I)


def strip_framing(transcript: str) -> str:
    """Remove the polite run-up so the verb lands at the start.

    "Now let's close Notepad" -> "close Notepad". Leaves anything it cannot
    recognise exactly as it was, and never touches the middle of a sentence.
    """
    text = (transcript or "").strip()
    for _ in range(6):                      # they stack, but not forever
        stripped = _FRAMING.sub("", text, count=1)
        if stripped == text:
            break
        text = stripped
    # Everything was framing and nothing was asked. The original is a better
    # answer than an empty string, which would read as silence.
    return text.strip() or (transcript or "").strip()


# -- "close the ones you just opened" ----------------------------------------
# A close request can point BACKWARDS instead of naming anything: "close the
# three apps you just opened", "close those", "close them". Nothing is named,
# so the app shortlist comes back empty and the step dies with "nothing
# installed matches" - which reads as a broken tool rather than as a sentence
# it does not handle.
#
# The count is captured when it is there, because "close the three apps you
# opened" after opening five is a different request from "close everything".
_RECENT_CLOSE = re.compile(
    r"\b(?:the\s+)?(?P<count>\d+|one|two|three|four|five|six|seven|eight)?\s*"
    r"(?:apps?|windows?|ones?|things?|them|those|these|everything|all)\b"
    r"(?:\s+(?:that\s+)?(?:you|we|u)\s*(?:just\s+)?(?:opened|open|launched))?",
    re.I)
_OPENED_BY_YOU = re.compile(
    r"\b(?:you|we|u)\s*(?:just\s+)?(?:opened|open|launched)\b|\bjust\s+opened\b",
    re.I)
_WORD_NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4,
                 "five": 5, "six": 6, "seven": 7, "eight": 8}


def closes_recent(transcript: str) -> Optional[int]:
    """How many recently-opened apps this close request means, or None.

    `0` means "all of them" - "close everything you opened" names no number.
    None means this is not a backward-pointing close at all, and the ordinary
    named-app path should handle it.
    """
    text = (transcript or "").strip()
    if not CLOSE.search(text):
        return None
    rest = CLOSE.sub("", text).strip(" .,")
    if not rest:
        return None
    # It has to actually refer back. "close the apps" on its own is too vague
    # to act on, and guessing would close things nobody asked about.
    pronoun = rest.lower() in {"those", "them", "these", "it all", "everything", "all"}
    if not pronoun and not _OPENED_BY_YOU.search(rest):
        return None
    match = _RECENT_CLOSE.search(rest)
    if match is None:
        return 0 if pronoun else None
    raw = (match.group("count") or "").lower()
    if not raw:
        return 0
    return _WORD_NUMBERS.get(raw, int(raw) if raw.isdigit() else 0)


# -- typing into an application ----------------------------------------------
@dataclass(frozen=True)
class TypeRequest:
    text: str
    target: str = ""          # "" means the window that had focus


# "type X into Y" / "type in Y, X" / "write X in Y". The verb must be a verb:
# "what TYPE of file is that" is not a request to type anything, so the word has
# to lead the clause rather than merely appear in it.
_TYPE_INTO = re.compile(
    r"^\s*(?:hey\s+jev|jev)?[,.\s]*(?:please\s+)?(?:type|write|enter)\s+"
    r"(?:out\s+)?(?:the\s+(?:word|phrase|text|words)\s+)?"
    r"(?P<text>.+?)\s+(?:in|into|on)\s+(?:the\s+|my\s+)?(?P<target>[\w .+-]{2,40})"
    r"\s*[.!?]*$", re.I)
# "type INTO obsidian the meeting starts" names a target. Bare "type in hello
# world" does not - "in" there is framing, and reading "hello" as an
# application name turned "type in hello world" into typing just "world".
# "into" is the phrasing people actually use when they mean a target.
_TYPE_IN_FIRST = re.compile(
    r"^\s*(?:hey\s+jev|jev)?[,.\s]*(?:please\s+)?(?:type|write|enter)\s+"
    r"into\s+(?:the\s+|my\s+)?(?P<target>[\w.+-]{2,40})\s+(?P<text>.+?)"
    r"\s*[.!?]*$", re.I)
_TYPE_BARE = re.compile(
    r"^\s*(?:hey\s+jev|jev)?[,.\s]*(?:please\s+)?(?:type|write|enter)\s+"
    r"(?:out\s+|in\s+)?(?:the\s+(?:word|phrase|text|words)\s+)?"
    r"(?P<text>.+?)\s*[.!?]*$", re.I)


def type_request(transcript: str) -> Optional[TypeRequest]:
    """What to type and where, or None if this was not a typing request.

    Returns None rather than an empty payload. Typing nothing successfully is
    the worst possible outcome here: it reports success, changes nothing, and
    gives no reason to look.
    """
    t = (transcript or "").strip()
    for pattern in (_TYPE_IN_FIRST, _TYPE_INTO):
        m = pattern.match(t)
        if m:
            text = m.group("text").strip(" .,")
            target = m.group("target").strip(" .,")
            if text and target:
                return TypeRequest(text=text, target=target)
    m = _TYPE_BARE.match(t)
    if m:
        text = m.group("text").strip(" .,")
        # "type into notepad" has a target and no words - that is not a payload.
        if text and not re.match(r"^(in|into|on)\b", text, re.I):
            return TypeRequest(text=text, target="")
    return None


# -- media keys ---------------------------------------------------------------
# Order matters: "play pause" and "pause the music" must not be read as two
# different things, and "play the video on youtube" is a website, not a key.
_MEDIA: tuple[tuple[str, re.Pattern], ...] = (
    ("next", re.compile(r"\b(next|skip)\s+(track|song|one|this)\b|\bskip\s+(this|it)\b"
                        r"|^\s*(next|skip)\s*[.!?]*$", re.I)),
    ("previous", re.compile(r"\b(previous|last|go\s+back\s+a)\s+(track|song)\b"
                            r"|\bgo\s+back\s+a\s+track\b|^\s*previous\s*[.!?]*$", re.I)),
    ("mute", re.compile(r"^\s*(un)?mute\b|\b(un)?mute\s+(the\s+)?(sound|audio|volume)\b", re.I)),
    ("volume_up", re.compile(r"\bvolume\s+up\b|\bturn\s+(the\s+)?(volume|sound)\s+up\b"
                             r"|\bturn\s+it\s+up\b|\blouder\b", re.I)),
    ("volume_down", re.compile(r"\bvolume\s+down\b|\bturn\s+(the\s+)?(volume|sound)\s+down\b"
                               r"|\bturn\s+it\s+down\b|\bquieter\b", re.I)),
    ("play_pause", re.compile(r"^\s*(play|pause|resume)\s*(pause)?\s*[.!?]*$"
                              r"|\b(play|pause|resume)\s+(the\s+)?(music|song|track|video|audio)\s*[.!?]*$"
                              r"|^\s*(play|pause)\s+it\b", re.I)),
)


def media_action(transcript: str) -> Optional[str]:
    """Which media key this asks for, or None. Never a guess."""
    t = (transcript or "").strip()
    if not t:
        return None
    # A site named in the sentence means this is a navigation request that
    # happens to contain the word "play", not a keypress.
    if re.search(r"\b(on|in)\s+(youtube|netflix|spotify|twitch|tiktok)\b", t, re.I):
        return None
    for name, pattern in _MEDIA:
        if pattern.search(t):
            return name
    return None


# -- yes and no ---------------------------------------------------------------
# A confirmation must be the WHOLE utterance. Continuous mode hears the room,
# and if any sentence containing "yes" counted, a conversation happening nearby
# would authorise an action nobody asked for.
# "right" is NOT filler: stripping it turns "that's right" into "that's",
# which is no longer an affirmation at all.
_FILLER = re.compile(r"\b(please|jev|hey|thanks|thank\s+you|then|now|ok(ay)?)\b", re.I)
_AFFIRM = re.compile(
    r"^(yes|yeah|yep|yup|sure|affirmative|correct|confirm(ed)?|do\s+it|go\s+ahead|"
    r"go\s+for\s+it|that'?s\s+right|absolutely|definitely|carry\s+on|proceed)$", re.I)
_DENY = re.compile(
    r"^(no|nope|nah|cancel|stop|don'?t|do\s+not|never\s*mind|forget\s+it|"
    r"negative|abort|leave\s+it|no\s+don'?t)$", re.I)


def _bare(transcript: str) -> str:
    t = (transcript or "").strip().lower()
    t = re.sub(r"[.!?,]+", " ", t)
    t = _FILLER.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()


def is_affirmation(transcript: str) -> bool:
    b = _bare(transcript)
    return bool(b) and bool(_AFFIRM.match(b))


def is_denial(transcript: str) -> bool:
    b = _bare(transcript)
    return bool(b) and bool(_DENY.match(b))


# -- "no, I meant X" ----------------------------------------------------------
# A correction has to be explicit and has to name a replacement. "that's wrong"
# on its own is a complaint, not a correction - there is nothing to learn from
# it - and "I meant to tell you about the meeting" is a sentence.
_CORRECTION = re.compile(
    r"^\s*(?:no[,.\s]+|not\s+that[,.\s]+|that'?s?\s+(?:wrong|not\s+it)[,.\s]+|"
    r"wrong\s+app[,.\s]+)?"
    r"(?:i\s+(?:meant|said)|i\s+meant\s+to\s+say)\s+(?P<what>.+?)\s*[.!?]*$", re.I)


def correction_request(transcript: str) -> Optional[str]:
    """What they actually wanted, when they are correcting a wrong answer.

    Returns None far more often than not, deliberately. A false positive here
    teaches the wrong thing permanently, and an unlearned nickname only costs
    one more correction.
    """
    m = _CORRECTION.match((transcript or "").strip())
    if not m:
        return None
    what = m.group("what").strip(" .,")
    # "I meant to tell you about the meeting" is not a correction; a
    # replacement is a name, not a clause.
    if not what or len(what.split()) > 4:
        return None
    return what


# -- one sentence, several instructions ---------------------------------------
# "open Chrome, search for apples, then open notepad" is three things, and the
# whole sentence used to resolve to ONE - so everything after "search for" was
# typed into the search box.
#
# The hard part is "and". It joins instructions ("open notepad AND open
# chrome") and it joins the contents of one instruction ("search for apples AND
# oranges"). Split on it blindly and every two-word search becomes two
# commands; never split on it and half of ordinary speech is one giant step.
#
# So "and" only separates when what follows it STARTS A COMMAND. "then",
# "after that" and "also" always separate, because nothing else uses them.

# A runaway sequence from one misheard sentence is not a request. Twenty
# chained launches is a malfunction, and the cap is a safety rail.
MAX_STEPS = 8

# The words a new instruction can begin with. This list IS the rule - anything
# not here means "and" was joining a query, not starting a command.
_STEP_VERBS = (
    "open", "launch", "start", "run", "close", "quit", "exit", "shut",
    "search", "google", "look\\s+up", "find", "switch\\s+to", "go\\s+to",
    "type", "write", "enter", "play", "pause", "resume", "skip", "mute",
    "turn\\s+(?:the\\s+)?(?:volume|sound)", "next", "previous", "bring\\s+up",
    "pull\\s+up", "visit", "navigate\\s+to", "make", "create",
)
_VERB_ALT = "|".join(_STEP_VERBS)

# Always a separator - no instruction contains these in the middle of itself.
_HARD_SPLIT = re.compile(
    r"(?:[,;]\s*)?\b(?:and\s+then|then|after\s+that|and\s+also|also|"
    r"and\s+after\s+that|next\s+up)\b[,;\s]*", re.I)

# "and" followed by something that starts a command.
# A verb does not always sit right after the joining word. "close Chrome and
# let's open Claude" hid its second verb behind "let's", so the sentence never
# split, Jev saw the whole thing at once and answered for the LAST clause only
# - the close was silently dropped. Measured on a real session log, 2026-09-22.
_STEP_FRAMING = r"(?:let\s*'?s|lets|now|then|also|please|we\s+should|you\s+can|can\s+you|could\s+you)\s+"

_AND_SPLIT = re.compile(
    rf"(?:[,;]\s*)?\band\s+(?:{_STEP_FRAMING})*(?=(?:{_VERB_ALT})\b)", re.I)

# A comma directly before a command verb: "open chrome, search for apples".
_COMMA_SPLIT = re.compile(
    rf"[,;]\s*(?:{_STEP_FRAMING})*(?=(?:{_VERB_ALT})\b)", re.I)



# A full stop between sentences.
_STOP_SPLIT = re.compile(r"[.!?]+\s+")

_STARTS_COMMAND = re.compile(rf"^\s*(?:hey\s+jev|jev)?[,.\s]*(?:please\s+)?"
                             rf"(?:can\s+you\s+)?(?:{_VERB_ALT})\b", re.I)

_LEADING_JOIN = re.compile(
    rf"^\s*(?:and\s+then|and\s+also|and|then|also|after\s+that)\b[,;\s]*", re.I)


# "Open Chrome and Notepad." People say the verb once and let it carry. From
# the log, 2026-09-22: this never split, because "Notepad" is a NAME and the
# rule wanted a verb. It became one step, Jev picked one app out of a shortlist
# holding both, Chrome opened, Notepad did not, and nothing said so.
#
# Only OPEN carries over. A search must not: "apples and oranges" is one
# search, and always was. That asymmetry is the whole reason this is a separate
# pattern rather than a looser version of the main one.
_OPEN_VERB = r"(?:open|launch|start|run|bring\s+up|pull\s+up)"
_OPENS_SOMETHING = re.compile(rf"^\s*(?:hey\s+jev|jev)?[,.\s]*(?:please\s+)?"
                              rf"(?:can\s+you\s+)?{_OPEN_VERB}\b", re.I)
# The tail after "and"/"," that is a bare name rather than a new instruction.
_BARE_NAME = re.compile(r"^[\w][\w .+'&-]{0,38}$")


def _expand_carried_verb(step: str) -> list[str]:
    """"open chrome and notepad" -> ["open chrome", "open notepad"].

    Returns the step unchanged unless it is an OPEN whose tail is a list of
    plain names. Anything with its own verb has already been split by the
    patterns above, and anything that is not an open is left alone.
    """
    if not _OPENS_SOMETHING.match(step or ""):
        return [step]
    m = re.match(rf"^\s*((?:hey\s+jev|jev)?[,.\s]*(?:please\s+)?"
                 rf"(?:can\s+you\s+)?{_OPEN_VERB})\s+(?P<rest>.+)$", step, re.I)
    if not m:
        return [step]
    verb, rest = m.group(1).strip(), m.group("rest").strip()
    parts = [p.strip(" ,;") for p in re.split(r",|\band\b", rest, flags=re.I)]
    parts = [p for p in parts if p]
    if len(parts) < 2:
        return [step]
    # Every piece has to look like a name. One that does not means the "and"
    # was joining words inside a single title, not listing two apps.
    if not all(_BARE_NAME.match(p) for p in parts):
        return [step]
    return [f"{verb} {p}" for p in parts]

def split_steps(transcript: str) -> list[str]:
    """One sentence -> the instructions it actually contains, in order."""
    text = (transcript or "").strip()
    if not text:
        return []

    parts = [text]
    for pattern in (_STOP_SPLIT, _HARD_SPLIT, _AND_SPLIT, _COMMA_SPLIT):
        nxt: list[str] = []
        for chunk in parts:
            nxt.extend(pattern.split(chunk))
        parts = nxt

    steps: list[str] = []
    for chunk in parts:
        # A fragment that still opens with a joining word is a mistake, not a
        # step - it would be handed to the matcher as "then open notepad".
        cleaned = _LEADING_JOIN.sub("", (chunk or "")).strip(" ,;.!?")
        if cleaned:
            steps.append(cleaned)
    if not steps:
        return []

    # "open chrome and notepad" is two instructions with the verb said once.
    expanded: list[str] = []
    for one in steps:
        expanded.extend(_expand_carried_verb(one))
    steps = expanded

    # Ordinary speech uses "then" and "and" constantly. "so then I told him it
    # was fine" splits neatly into two fragments and is not two instructions -
    # it is not one instruction either. A sequence is only a sequence when at
    # least two fragments actually BEGIN like commands; otherwise the sentence
    # goes through whole and is judged as the conversation it is.
    if len(steps) > 1 and sum(bool(_STARTS_COMMAND.match(x)) for x in steps) < 2:
        return [text]
    return steps[:MAX_STEPS]


def is_multi_step(transcript: str) -> bool:
    return len(split_steps(transcript)) > 1


# -- "write a poem about apples" ----------------------------------------------
# Dictation and composition look alike and are not alike. "type hello world
# into notepad" carries its own words; "write a poem about apples" does not -
# they have to be invented, and nothing here can invent them. Jev answers typed
# questions and by design cannot return prose.
#
# Without this, the typing path matched it happily and would have typed the
# literal string "a poem about apples" into Notepad: a wrong answer delivered
# confidently, with nothing to show that anything had gone astray.
#
# Fixing it properly needs a language model that can write the text itself.
# Until then, saying "I cannot" is the only honest option.
_COMPOSE = re.compile(
    r"^\s*(?:hey\s+jev|jev)?[,.\s]*(?:please\s+)?(?:can\s+you\s+)?"
    r"(?:write|compose|draft|generate|make\s+up)\s+(?:me\s+)?"
    r"(?P<what>(?:a|an|the)\s+"
    r"(?:poem|story|song|haiku|essay|email|letter|note|paragraph|summary|"
    r"list|message|reply|draft|article|script|caption|tweet|post|joke)"
    r"\b.*?)"
    r"(?:\s+(?:in|into|on)\s+(?:the\s+|my\s+)?[\w .+-]{2,40})?\s*[.!?]*$", re.I)


def compose_request(transcript: str) -> Optional[str]:
    """What they asked to have WRITTEN, if the words do not exist yet.

    Returns None for dictation, which is the common case and must keep working:
    "write hello world in notepad" is typing, not composing.
    """
    m = _COMPOSE.match((transcript or "").strip())
    if not m:
        return None
    what = m.group("what").strip(" .,")
    return what or None
