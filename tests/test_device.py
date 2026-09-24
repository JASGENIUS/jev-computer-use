"""Running without an NVIDIA GPU.

The code always fell back to CPU when CUDA was missing - but it fell back with
`large-v3` still selected, which is a ~3GB model that takes many seconds per
utterance on a CPU. A fallback that leaves you with something unusable is not
a fallback, it is a slower failure.

Requiring a specific graphics card to dictate a sentence is not a reasonable
ask of someone downloading a tool, so the default is now to look at the machine
and pick something that actually runs on it.
"""
from __future__ import annotations

import pytest

from jevflow import stt


def test_auto_picks_cuda_when_it_is_there(monkeypatch):
    monkeypatch.setattr(stt, "cuda_available", lambda: True)
    device, model, compute = stt.pick_runtime("auto", "auto")
    assert device == "cuda"
    assert compute == "float16"
    assert "large" in model


def test_auto_picks_something_usable_on_cpu(monkeypatch):
    """The point of the exercise: no GPU must still be a working tool."""
    monkeypatch.setattr(stt, "cuda_available", lambda: False)
    device, model, compute = stt.pick_runtime("auto", "auto")
    assert device == "cpu"
    assert compute == "int8", "float16 on CPU is slow and often unsupported"
    assert "large" not in model, f"{model} on a CPU is not a usable default"


def test_an_explicit_choice_is_always_respected(monkeypatch):
    """Someone who asks for large-v3 on CPU gets it. Being helpful is not the
    same as overruling."""
    monkeypatch.setattr(stt, "cuda_available", lambda: False)
    device, model, compute = stt.pick_runtime("cpu", "large-v3")
    assert model == "large-v3"


def test_asking_for_cuda_without_cuda_falls_back_rather_than_crashing(monkeypatch):
    monkeypatch.setattr(stt, "cuda_available", lambda: False)
    device, model, compute = stt.pick_runtime("cuda", "auto")
    assert device == "cpu"
    assert compute == "int8"


def test_cuda_detection_never_raises(monkeypatch):
    """Called at startup. A throw here would stop the tool running at all."""
    def boom(*a, **k):
        raise OSError("no driver")
    monkeypatch.setattr(stt, "_probe_cuda", boom)
    assert stt.cuda_available() in (True, False)


def test_the_choice_is_reported(monkeypatch, caplog):
    """A user on CPU should be able to find out why it is slower."""
    import logging
    monkeypatch.setattr(stt, "cuda_available", lambda: False)
    with caplog.at_level(logging.INFO):
        stt.pick_runtime("auto", "auto")
    assert any("cpu" in r.message.lower() for r in caplog.records)
