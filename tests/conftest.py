"""Shared fixtures. Nothing here touches a microphone, a GPU, or the network."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class FakeJev:
    """Stands in for the Jev API.

    Returns whatever the test scripted, and records what it was asked, so a test
    can assert on the candidate list code handed over - the wide pass is code's
    job and getting it wrong is invisible if you only check the final answer.
    """

    def __init__(self, intent="open_app", app=None, certain=0.9, latency_ms=120.0,
                 raises=None):
        self.intent, self.app, self.certain = intent, app, certain
        self.latency_ms, self.raises = latency_ms, raises
        self.asked: list[tuple[dict, dict]] = []

    def ask(self, state, questions):
        self.asked.append((state, questions))
        if self.raises is not None:
            raise self.raises
        answers = {
            "intent": {"choice": self.intent},
            "certain": {"noul": self.certain},
        }
        if self.app is not None:
            answers["app"] = {"choice": self.app}
        return {"answers": answers, "latency_ms": self.latency_ms}

    @property
    def last_candidates(self) -> list[str]:
        return self.asked[-1][0]["installed_candidates"] if self.asked else []


@pytest.fixture
def fake_jev():
    return FakeJev
