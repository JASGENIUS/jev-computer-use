"""Hotkey -> panel -> speak -> transcript.

Threading: Tk must own the main thread, the `keyboard` hook fires on its own
thread, and recording blocks. So the hotkey only enqueues; a single worker
thread does record-then-transcribe; the overlay is updated through its queue.
One session at a time - pressing the key again while listening cancels instead
of starting a second recorder on the same microphone.
"""
from __future__ import annotations

import atexit
import ctypes
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from jevflow import disfluency, hotkeys, stt
from jevflow.overlay import ACCENT, BAD, DIM as DIM_TONE, GOOD, WARN, Overlay
from jevflow.recorder import Recorder

# The computer-use half is OPTIONAL. Dictation needs a microphone, a model and
# somewhere to type; it does not need a command planner, a policy gate or a
# step splitter. Importing them at module level welded the two halves together
# so tightly that a dictation-only install could not start at all.
try:
    from jevflow import streaming
    from jevflow.policy import CONFIRM_TTL_S
    HAVE_COMMAND = True
except ImportError:                                   # dictation-only install
    streaming = None
    CONFIRM_TTL_S = 20.0
    HAVE_COMMAND = False

log = logging.getLogger("jevflow.app")

def release_modifiers() -> None:
    """Force a key-up for every modifier.

    A stuck modifier is the worst failure this program can have: Ctrl held down
    turns every keystroke into a shortcut and there is no obvious cause to find.
    Cheap to do, so do it on every exit path.
    """
    for vk in (0xA0, 0xA1, 0xA2, 0xA3, 0xA4, 0xA5, 0x5B, 0x5C):
        try:
            ctypes.windll.user32.keybd_event(vk, 0, 2, 0)   # KEYEVENTF_KEYUP
        except Exception:
            pass


DEFAULT_HOTKEY = "ctrl+windows"          # dictate: type what I said
DEFAULT_COMMAND_HOTKEY = "ctrl+alt+space"  # command: "hey Jev, open Chrome"


@dataclass
class VoiceApp:
    hotkey: str = DEFAULT_HOTKEY
    model: str = "large-v3"
    device: str = "cuda"
    compute_type: str = "float16"
    on_transcript: Optional[Callable[[str], None]] = None
    dictate: bool = False            # type the transcript into the window you were in
    command_hotkey: str = ""         # second key that runs commands instead of typing
    title: str = "JevFlow"
    continuous: bool = False         # keep listening until cancelled (command mode)
    commander: Optional[object] = None
    wake: Optional[object] = None    # a gate from wake.py, or None to hear everything
    clean_speech: bool = True        # strip fillers and abandoned restarts
    clean_aggressive: bool = False   # also strip "like", "basically", ...
    # Push-to-talk: record while the key is HELD, and ignore silence entirely.
    # Silence detection ends the clip when the room goes quiet, which is the
    # wrong rule for dictation - a pause while working out the next words is
    # not the end of the sentence, and being cut off mid-thought is worse than
    # a clip that runs a second long.
    hold_to_talk: bool = False
    # Act on each instruction as the speaker moves past it, instead of waiting
    # for the whole sentence. A step fires only once there is speech BEHIND it,
    # so it can no longer grow - see streaming.py for why acting on any partial
    # would be unsafe.
    stream: bool = False
    partial_every_s: float = 0.35   # how often to re-transcribe mid-sentence
    # How this tool's capsule looks. Two of them are on screen at once and an
    # identical panel tells you nothing about which one is listening.
    accent: str = ""
    badge: str = ""
    silence_hold_s: float = 0.0      # 0 = the recorder's own default
    preload: bool = True
    # Ask Jev which spelling was meant, for words the recogniser was unsure of
    # AND that have a homophone. Off unless a key is configured, because it is
    # the one part of dictation that leaves the machine - see homophones.py.
    fix_homophones: bool = False
    # Let Jev call off a filler deletion when the word is actually doing a job.
    # The rule deletes the verb out of "I would really like to use it"; this is
    # what catches that. Measured in filler.py.
    fix_fillers: bool = False

    _busy: threading.Event = field(default_factory=threading.Event)
    _stop: threading.Event = field(default_factory=threading.Event)
    _mode: bool = False              # which key started the running session
    _recorder: Optional[Recorder] = None
    _jev: Optional[object] = None    # built on first use, never at startup

    def __post_init__(self) -> None:
        self.overlay = Overlay(on_cancel=self.cancel, title=self.title,
                               accent=self.accent or ACCENT, badge=self.badge)

    # -- lifecycle --------------------------------------------------------
    def start(self) -> None:
        import keyboard

        if self.preload:
            threading.Thread(target=self._warm, daemon=True).start()
        # NEVER suppress=True. It makes the `keyboard` library buffer and replay
        # every keystroke through a low-level hook so it can decide whether a
        # combo completes. Because this hotkey starts with Ctrl, that meant
        # EVERY Ctrl press on the machine went through that path - and when the
        # process died mid-combo the Ctrl key-up was never delivered, leaving
        # Ctrl physically stuck down system-wide. Typing letters started firing
        # shortcuts. It is not worth it: Start only opens on a Win key-up with
        # no other key held, and Ctrl is held here, so nothing needed swallowing
        # in the first place.
        if self.hotkey and self.hotkey.lower() != "none":
            keyboard.add_hotkey(self.hotkey, self._triggered, suppress=False)
        # Deliberately NOT blocking the Windows key itself: Start only opens on a
        # Win keyup that had no other key held, and Ctrl is held here, so the
        # chord is already safe. Blocking it would cost the Start menu, Win+E and
        # every other Win shortcut for the sake of a problem that does not exist.
        if self.command_hotkey and self.commander is not None:
            keyboard.add_hotkey(self.command_hotkey,
                                lambda: self._triggered(command=True), suppress=False)
            log.info("[jevflow] command key %s", self.command_hotkey)
        atexit.register(release_modifiers)
        if self.hotkey and self.hotkey.lower() != "none":
            log.info("[jevflow] dictation key %s", self.hotkey)
        print(f"{self.title} ready.")
        if self.hotkey and self.hotkey.lower() != "none":
            print(f"  {self.hotkey:<16} speak -> types what you said")
        if self.command_hotkey and self.commander is not None:
            print(f"  {self.command_hotkey:<16} speak -> \"hey Jev, open Chrome\"")
        print("  Ctrl+C here to quit.")
        try:
            self.overlay.run()
        finally:
            # Whatever happens - crash, Ctrl+C, kill - never leave a modifier down.
            release_modifiers()

    def _warm(self) -> None:
        t0 = time.time()
        try:
            stt.load_model(self.model, self.device, self.compute_type)
            log.info("[jevflow] model warm in %.1fs", time.time() - t0)
        except Exception as exc:
            log.error("[jevflow] model failed to load: %s", exc)

    def _fix_homophones(self, text: str, words: list) -> str:
        """Ask Jev which spelling was meant, for the words Whisper doubted.

        Nothing here may cost the transcript. No key, no network, a bad
        response - every one of those returns the text exactly as heard,
        because dictation losing a sentence over a spelling check is far worse
        than a "their" that should have been a "there".
        """
        if not self.fix_homophones or not text or not words:
            return text
        try:
            from jevflow import homophones
            if self._jev is None:
                from jevflow.jev import Jev
                self._jev = Jev()
            fixed, changes = homophones.resolve(text, words, self._jev)
        except Exception as exc:
            # Includes JevError when no key is configured. Said once, at info,
            # rather than on every utterance.
            log.info("[jevflow] spelling check unavailable (%s)", exc)
            self.fix_homophones = False
            return text
        if changes:
            log.info("[jevflow] spelling: %s",
                     ", ".join(f"{a} -> {b}" for a, b in changes))
        return fixed

    def _filler_veto(self):
        """A fresh veto per utterance, or None when the feature is off.

        Fresh because its question budget and its cache are both per-sentence.
        Returns None on any failure, which leaves the cleaner exactly as it
        shipped before - the rule alone.
        """
        if not self.fix_fillers:
            return None
        try:
            from jevflow.filler import Veto
            if self._jev is None:
                from jevflow.jev import Jev
                self._jev = Jev()
            return Veto(self._jev)
        except Exception as exc:
            log.info("[jevflow] filler check unavailable (%s)", exc)
            self.fix_fillers = False
            return None

    def cancel(self) -> None:
        """Esc: stop listening and throw away what was recorded."""
        self._stop.set()
        if self._recorder is not None:
            self._recorder.cancel()

    def finish(self) -> None:
        """A second press of the same key: stop, and KEEP what was said.

        In continuous command mode this ends the session as well, because the
        press means "that's the last one" rather than "send this one and carry
        on listening".
        """
        self._stop.set()
        if self._recorder is not None:
            self._recorder.finish()

    def _triggered(self, command: bool = False) -> None:
        # In hold mode the chord can fire again WHILE it is being held - the
        # library re-triggers on key repeat inside a combo. Treating that as
        # "pressed again, so cancel" makes holding the key cancel itself the
        # instant you hold it, which looks exactly like the program being
        # broken. While held, a repeat means nothing: the release ends the clip.
        if self.hold_to_talk and self._busy.is_set():
            chord = self.command_hotkey if command else self.hotkey
            if hotkeys.still_held(chord):
                log.debug("[jevflow] key repeat while held; ignoring")
                return
        if self._busy.is_set():
            # Same key again = "I'm done, go". It used to CANCEL, which threw
            # away a perfectly good sentence and showed "Cancelled" - the exact
            # opposite of what pressing the key again means when you have just
            # finished speaking. Esc still discards.
            #
            # The OTHER key = switch to that mode, which is what pressing it
            # obviously means. There is one microphone, so a continuous command
            # session would otherwise silently swallow every press of the
            # dictation key.
            if command == self._mode:
                log.info("[jevflow] KEY AGAIN | finishing the clip")
                self.finish()
                return
            self.cancel()
            for _ in range(40):                 # wait for the session to unwind
                if not self._busy.is_set():
                    break
                time.sleep(0.05)
        log.info("[jevflow] KEY PRESSED | command=%s | hold=%s", command, self.hold_to_talk)
        self._mode = command
        threading.Thread(target=self._session, kwargs={"command": command},
                         daemon=True).start()

    # -- a listening session, one or many utterances ----------------------
    def _session(self, command: bool = False) -> None:
        """Stay listening until cancelled, when `continuous` is on.

        Commands come in runs - "open Brave", then "search for apples" - and
        reaching for the hotkey between each one defeats the point. Dictation
        stays one-shot, because there the pause IS the end of the sentence.
        """
        self._busy.set()
        self._stop.clear()
        keep_going = self.continuous and command
        try:
            first = True
            while True:
                if self._stop.is_set():
                    break
                if not self._once(command, first):
                    break
                first = False
                if not keep_going:
                    break
            if keep_going:
                self.overlay.set_status("Stopped", "", DIM_TONE)
                self.overlay.hide(700)
        except Exception as exc:
            log.exception("[jevflow] session failed")
            self.overlay.set_status("Error", str(exc)[:80], BAD)
            self.overlay.hide(2500)
        finally:
            self._recorder = None
            self._busy.clear()

    def _await_wake(self, gate) -> bool:
        """Hold until the wake phrase is heard. True means go on and record.

        Only an acoustic gate can answer before transcription; a transcript gate
        has nothing to wait for and decides afterwards instead.
        """
        if not hasattr(gate, "feed"):
            return True
        from jevflow.recorder import wait_for_wake
        return wait_for_wake(gate, on_level=self.overlay.set_level, cancel=self._stop)

    def _once(self, command: bool, first: bool) -> bool:
        """One utterance. Returns False when the session should end."""
        from jevflow.typing import foreground
        target_hwnd, target_title = foreground()
        # Holding a key down cannot happen by accident, so it already says
        # "this is for you" - which is the entire job of a wake word. Making
        # someone hold a key AND say a name is two intent signals for one
        # intent.
        gate = None if self.hold_to_talk else (self.wake if command else None)
        waiting = gate is not None and not gate.is_armed()

        # The capsule stays visible for as long as the microphone is open, in
        # wake mode too. It is the only signal that anything is listening, so a
        # gate that hid the panel to look quiet would trade noise for something
        # far worse than noise.
        self.overlay.show("Listening" if first else "Listening…")
        if waiting:
            self.overlay.set_status("Waiting", gate.label, DIM_TONE)
            if not self._await_wake(gate):
                return not self._stop.is_set()
            self.overlay.set_status("Listening", "", ACCENT)

        stop_when = None
        if self.hold_to_talk:
            chord = self.command_hotkey if command else self.hotkey
            # Give the chord a moment to actually register as down. The hotkey
            # fires on the first key of the combo, and polling immediately can
            # see it as already released before the second key lands.
            time.sleep(0.12)
            stop_when = lambda c=chord: not hotkeys.still_held(c)
        rec_kw = {"on_level": self.overlay.set_level, "should_stop": stop_when}
        if self.silence_hold_s:
            rec_kw["silence_hold_s"] = self.silence_hold_s

        step_stream = None
        if (self.stream and command and self.commander is not None
                and streaming is not None):
            step_stream = streaming.StepStream()
            rec_kw["on_partial"] = lambda audio: self._on_partial(
                audio, step_stream, target_title, target_hwnd)
            rec_kw["partial_every_s"] = self.partial_every_s

        self._recorder = Recorder(**rec_kw)
        clip = self._recorder.record()
        self._recorder = None

        if clip.cancelled or self._stop.is_set():
            if not self.continuous or not command:
                self.overlay.set_status("Cancelled", "", WARN)
                self.overlay.hide(600)
            return False
        if not clip.spoke:
            # Silence in continuous mode is just a pause, not an error.
            if self.continuous and command:
                return True
            self.overlay.set_status("Heard nothing", "try again", WARN)
            self.overlay.hide(900)
            return False

        self.overlay.set_status("Transcribing", f"{clip.seconds:.1f}s", ACCENT)
        result = stt.transcribe(clip.audio, model=self.model, device=self.device,
                                compute_type=self.compute_type,
                                word_probabilities=self.fix_homophones)
        text = result["text"]
        if not text:
            if self.continuous and command:
                return True
            self.overlay.set_status("Nothing recognised", "", WARN)
            self.overlay.hide(900)
            return False

        log.info("[jevflow] %.2fs -> %r", result["seconds"], text)

        # BEFORE cleaning, because cleaning deletes words and the word list has
        # to still line up with the text for the right occurrence to be
        # replaced. Fillers are never homophones, so nothing is missed by
        # running this first.
        text = self._fix_homophones(text, result.get("words") or [])

        # Cleaning happens BEFORE the wake gate, not after. The gate is
        # anchored to the start of the utterance, and "um, hey jev, open
        # notepad" does not start with the wake word until the "um" is gone.
        if self.clean_speech:
            cleaned = disfluency.clean_verbose(text, self.clean_aggressive,
                                               veto=self._filler_veto())
            if cleaned.changed:
                log.info("[jevflow] cleaned -> %r (removed %s)",
                         cleaned.text, ", ".join(cleaned.removed[:6]))
            text = cleaned.text

        # A transcript gate discards what it heard here: nothing is displayed,
        # nothing is sent to Jev, and nothing runs. The words never left the
        # machine to begin with, and now they do not leave this function either.
        if gate is not None and hasattr(gate, "check"):
            heard = gate.check(text)
            if not heard.awake:
                log.debug("[jevflow] not addressed to me; ignored")
                self.overlay.set_status("Waiting", gate.label, DIM_TONE)
                return True
            if heard.armed:
                self.overlay.set_status("Yes?", "", ACCENT)
                return True
            text = heard.command

        self.overlay.set_status("Heard", text, GOOD)

        if command:
            if step_stream is not None:
                remaining = self._finish_stream(step_stream, text)
                if remaining is not None:
                    text = remaining
                    if not text:
                        self.overlay.hide(1800)
                        return True
            self._run_command(text, target_title, target_hwnd)
            if self.continuous:
                time.sleep(1.6)          # let the result be read before listening again
            return True

        if self.dictate:
            from jevflow.typing import type_text
            ok, why = type_text(text, expect_hwnd=target_hwnd)
            if not ok:
                self.overlay.set_status("Not typed", why, WARN)
                self.overlay.hide(2500)
                return False
        if self.on_transcript:
            try:
                self.on_transcript(text)
            except Exception as exc:
                log.exception("on_transcript failed: %s", exc)
                self.overlay.set_status("Handler failed", str(exc)[:80], BAD)
                self.overlay.hide(2500)
                return False
        self.overlay.hide(2200)
        return True

    # -- acting while you are still talking -------------------------------
    def _on_partial(self, audio, step_stream, title: str, hwnd) -> None:
        """Transcribe what has been said so far and run anything now settled."""
        result = stt.transcribe(audio, model=self.model, device=self.device,
                                compute_type=self.compute_type)
        text = result.get("text") or ""
        if self.clean_speech and text:
            text = disfluency.clean(text, self.clean_aggressive)
        if self.wake is not None and hasattr(self.wake, "check"):
            heard = self.wake.check(text)
            if not heard.awake:
                return
            text = heard.command or text
        for step in step_stream.update(text):
            log.info("[jevflow] STREAMING step while still speaking: %r", step)
            self.overlay.set_status("Doing", step[:70], ACCENT)
            self.commander.handle(step, focused_window=title, focused_hwnd=hwnd)

    def _finish_stream(self, step_stream, final_text: str):
        """What is left of the sentence once the streamed steps are removed.

        Returns None when nothing streamed, so the normal path is untouched.
        Returns "" when the whole sentence already ran.
        """
        if not step_stream.fired:
            return None
        step_stream.update(final_text)
        remaining = step_stream.finish()
        log.info("[jevflow] streamed %d step(s) already; %d left",
                 len(step_stream.fired) - len(remaining), len(remaining))
        # Rejoined into one sentence so the ordinary command path handles them
        # - sequence logic, policy bands and the single confirmation all still
        # apply. Re-running the whole transcript would open everything twice.
        return " and then ".join(remaining)

    # -- "hey Jev, open Chrome" -------------------------------------------
    def _run_command(self, text: str, focused_window: str,
                     focused_hwnd: Optional[int] = None) -> None:
        """Route an utterance through Jev to a bounded action. Code executes."""
        if self.commander is None:
            self.overlay.set_status("No commander", "start with --command", WARN)
            self.overlay.hide(2000)
            return
        self.overlay.set_status("Deciding", text[:80], ACCENT)
        # Show a sequence advancing. Three apps opening behind a capsule that
        # still says "Deciding" looks like it has hung.
        if hasattr(self.commander, "on_progress"):
            self.commander.on_progress = (
                lambda i, n, label: self.overlay.set_status(
                    f"Step {i}/{n}", label[:70], ACCENT))
        out = self.commander.handle(text, focused_window=focused_window,
                                    focused_hwnd=focused_hwnd)
        # A question is not a failure. It stays up long enough to answer, and
        # the capsule is the ONLY place the question appears - if it vanished
        # after the usual two seconds there would be nothing left to say yes to.
        if out.action == "needs_confirm":
            self.overlay.set_status("Confirm", out.detail, WARN)
            self.overlay.hide(int(CONFIRM_TTL_S * 1000))
            return
        if out.ok:
            verb = {"opened": "Opened", "would_open": "Would open",
                    "searched": "Searched", "would_search": "Would search",
                    "opened_site": "Opened", "would_open_site": "Would open",
                    "closed": "Closed", "would_close": "Would close",
                    "switched": "Switched to", "would_switch": "Would switch to",
                    "typed": "Typed", "would_type": "Would type",
                    "pressed": "Pressed", "would_press": "Would press"}.get(out.action, "Done")
            mark = {True: "", False: "  (unconfirmed)", None: ""}[out.verified]
            self.overlay.set_status(verb, out.detail + mark,
                                    GOOD if out.verified is not False else WARN)
            self.overlay.hide(2600)
        else:
            tone = DIM_TONE if out.action == "cancelled" else WARN
            # Action names are code identifiers, and "Step Unclear" on a
            # capsule reads as the program not understanding English. The ones
            # you will actually see get words.
            title = {"couldnt_do_step": "Couldn't", "no_match": "Not found",
                     "sequence_failed": "Stopped", "cannot_compose": "Can't write",
                     "nothing_to_correct": "Nothing to fix",
                     "unsure": "Not sure", "no_policy": "Can't do that",
                     "not_open": "Not open", "no_query": "Search for what?",
                     "no_text": "Type what?", "unclear": "Didn't catch that",
                     }.get(out.action, out.action.replace("_", " ").title())
            self.overlay.set_status(title, out.detail, tone)
            self.overlay.hide(2600)
