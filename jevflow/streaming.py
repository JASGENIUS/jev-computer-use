"""Acting while you are still talking.

Waiting for a full stop before doing anything makes a four-part instruction
feel slow: "open Chrome and then search for apples and then open Notepad" sits
there doing nothing for six seconds, and only then does the machine wake up.

The obvious approach - act on every partial transcript - is unsafe, and the
reason matters. A partial is **unstable**: Whisper given one second of audio
says something different from Whisper given three, and it revises backwards as
it hears more. Acting on "open Goo..." opens the wrong thing. Acting on "open
Chrome" when the sentence turns out to be "open Chromecast settings" opens
something that was never asked for, and there is no undo for a launched
application.

So the rule is not "act early", it is:

    **A step fires only once the speaker has moved past it.**

When "search for apples" shows up behind it, "open Chrome" is finished speech -
it cannot grow any more, because the words after it already exist. The final
step always waits for the end of the utterance, because it is still being said.

That gives most of the speed with none of the guessing: in a four-part
instruction, three of the four fire while you are still talking.

This class is deliberately pure - transcripts in, steps out, no audio and no
model - because the interesting failures are all about text changing under you,
and those are far easier to get right when they can be tested directly.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from jevflow import parse

log = logging.getLogger("jevflow.streaming")


@dataclass
class StepStream:
    """Turns a growing transcript into steps that are safe to run now.

    `stable_updates` is the safety knob. At 1 a step fires as soon as the
    speaker moves past it. At 2 it must also have been seen unchanged an extra
    time, which costs one update of latency and removes the last of the
    guesswork for anyone who wants that trade.
    """

    stable_updates: int = 1
    fired: list[str] = field(default_factory=list)
    _seen: dict[str, int] = field(default_factory=dict)
    _last_steps: list[str] = field(default_factory=list)

    @property
    def fired_count(self) -> int:
        return len(self.fired)

    def _steps(self, transcript: str) -> list[str]:
        return parse.split_steps(transcript or "")

    def update(self, transcript: str) -> list[str]:
        """Feed the latest transcript. Returns steps that are ready to run NOW."""
        steps = self._steps(transcript)
        self._last_steps = steps
        if len(steps) < 2:
            # Nothing is behind anything yet, so nothing is finished speech.
            return []

        ready: list[str] = []
        # The final step is excluded on purpose: it is still being spoken.
        for step in steps[:-1]:
            if step in self.fired:
                continue
            seen = self._seen.get(step, 0) + 1
            self._seen[step] = seen
            if seen >= self.stable_updates:
                self.fired.append(step)
                ready.append(step)
                log.info("[stream] firing %r (speaker moved past it)", step)
        return ready

    def finish(self) -> list[str]:
        """The utterance ended. Whatever has not run yet, runs now."""
        rest = [s for s in self._last_steps if s not in self.fired]
        self.fired.extend(rest)
        if rest:
            log.info("[stream] end of utterance, running %d remaining", len(rest))
        return rest

    def reset(self) -> None:
        self.fired.clear()
        self._seen.clear()
        self._last_steps.clear()
