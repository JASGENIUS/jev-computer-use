"""JevFlow - hold a hotkey, say what you want, Jev decides what to do.

    python main.py listen --dictate       # Ctrl+Win -> panel -> types what you said
    python main.py listen --hotkey ctrl+alt+space
    python main.py mic                    # list input devices
    python main.py say                    # record one clip from the terminal and transcribe
    python main.py warm                   # load the model and report timings
"""
from __future__ import annotations

import argparse
import logging
import pathlib
import sys
import time


LOG_DIR = pathlib.Path(__file__).resolve().parent / "logs"


def _log(level: str = "INFO", title: str = "jevflow") -> None:
    """Console AND a file.

    The launchers run under `pythonw`, which has no console at all, so every
    log line the running tool produced went nowhere. When something did not
    work there was literally nothing to look at - no error, no trace, no record
    that a key had even been pressed. Diagnosing that is guesswork, and
    guesswork is how you end up fixing things that were never broken.
    """
    handlers = [logging.StreamHandler()]
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        safe = "".join(c if c.isalnum() else "_" for c in title).strip("_") or "jevflow"
        handlers.append(logging.FileHandler(LOG_DIR / f"{safe}.log", encoding="utf-8"))
    except Exception:
        pass
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname).1s %(name)s %(message)s",
                        datefmt="%H:%M:%S", handlers=handlers, force=True)


def cmd_mic(_args) -> int:
    from jevflow.recorder import list_devices
    for d in list_devices():
        print(("  * " if d["default"] else "    ") + f"[{d['index']:>2}] {d['name']}")
    return 0


def cmd_warm(args) -> int:
    from jevflow import stt
    t0 = time.time()
    stt.load_model(args.model, args.device, args.compute_type)
    print(f"model ready in {time.time() - t0:.1f}s")
    return 0


def cmd_say(args) -> int:
    """One clip, from the terminal - the quickest way to check the mic and model."""
    from jevflow import stt
    from jevflow.recorder import Recorder

    stt.load_model(args.model, args.device, args.compute_type)
    print("speak now (it stops when you pause)...")
    clip = Recorder().record()
    if not clip.spoke:
        print("heard nothing")
        return 1
    r = stt.transcribe(clip.audio, model=args.model, device=args.device,
                       compute_type=args.compute_type)
    print(f"[{clip.seconds:.1f}s audio, {r['seconds']:.2f}s transcribe]")
    print(f"  {r['text']}")
    return 0


def cmd_listen(args) -> int:
    from jevflow.app import VoiceApp
    from jevflow import single_instance

    # Autostart means one copy is always running, and double-clicking the .bat
    # is muscle memory. Two copies grabbing the same hotkey start two recorders
    # on one microphone, and the symptom is dictation dropping words at random -
    # which reads as a broken recogniser, not as "it is running twice".
    # Hold-to-talk and continuous are contradictory: one utterance ends when
    # the key comes up, and continuous immediately starts another. Holding the
    # key IS the loop.
    if args.hold and args.continuous:
        args.continuous = False

    keys = [args.hotkey, args.command_hotkey if args.command else ""]
    if not single_instance.acquire(keys):
        live = ", ".join(k for k in keys if k and k.lower() != "none")
        single_instance.tell_user_already_running(args.title, live)
        return 0

    def handle(text: str) -> None:
        print(f"  heard: {text}")

    commander = None
    if args.command:
        from jevflow.command import Commander
        commander = Commander(dry_run=args.dry_run)

    gate = None
    if args.command:
        from jevflow import wake
        # A bad engine or a phrase with no model raises here, at startup, rather
        # than becoming a gate that silently never opens and a tool that has
        # quietly stopped responding with nothing anywhere to explain why.
        gate = wake.build(args.wake, word=args.wake_word)
        if gate is not None:
            print(f"  wake gate: {gate.label}")

    app = VoiceApp(hotkey=args.hotkey, model=args.model, device=args.device,
                   compute_type=args.compute_type, on_transcript=handle,
                   dictate=args.dictate, commander=commander, title=args.title,
                   continuous=args.continuous, wake=gate,
                   clean_speech=not args.raw,
                   clean_aggressive=args.clean_hard,
                   fix_homophones=getattr(args, "fix_homophones", False),
                   fix_fillers=getattr(args, "fix_fillers", False),
                   hold_to_talk=args.hold, stream=args.stream, partial_every_s=args.partial_every,
                   accent=args.accent, badge=args.badge,
                   silence_hold_s=args.silence or 0.0,
                   command_hotkey=args.command_hotkey if args.command else "")
    try:
        app.start()
    except KeyboardInterrupt:
        print("\nbye")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """The parser, separately, so the .bat launchers can be checked against it.

    They drifted: two of them were still on hotkeys the handoff had replaced,
    and nothing noticed, because nothing reads a .bat except a double-click.
    """
    p = argparse.ArgumentParser(prog="jevflow", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="auto",
                   help="auto picks a model your machine can run: large-v3 on "
                        "an NVIDIA GPU, base.en on CPU")
    p.add_argument("--device", default="auto",
                   help="auto, cuda or cpu. No GPU is required.")
    p.add_argument("--compute-type", dest="compute_type", default="auto")
    p.add_argument("--log", default="INFO")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("mic").set_defaults(fn=cmd_mic)
    sub.add_parser("warm").set_defaults(fn=cmd_warm)
    sub.add_parser("say").set_defaults(fn=cmd_say)
    lp = sub.add_parser("listen")
    lp.add_argument("--hotkey", default="ctrl+windows")
    lp.add_argument("--dictate", action="store_true",
                    help="type the transcript into the window you were in")
    lp.add_argument("--command", action="store_true",
                    help='enable the command key: "hey Jev, open Chrome"')
    lp.add_argument("--command-hotkey", dest="command_hotkey", default="ctrl+alt+space")
    lp.add_argument("--title", default="JevFlow")
    lp.add_argument("--continuous", action="store_true",
                    help="command mode keeps listening until you press the key again or Esc")
    lp.add_argument("--wake", default="none", choices=("none", "transcript", "acoustic"),
                    help="none hears every sentence; transcript acts only on ones that "
                         "start with the wake word; acoustic transcribes nothing at all "
                         "until a wake-word model fires")
    lp.add_argument("--wake-word", dest="wake_word", default="jev",
                    help="the wake word, or for --wake acoustic the name of an "
                         "openWakeWord model (there is no 'jev' model yet)")
    lp.add_argument("--dry-run", dest="dry_run", action="store_true",
                    help="decide and report, but do not actually do anything")
    lp.add_argument("--silence", type=float, default=None,
                    help="seconds of quiet that end a clip (default 4.0). "
                         "Pressing the key again ends it immediately.")
    lp.add_argument("--accent", default="",
                    help="capsule colour, so two tools on screen look different")
    lp.add_argument("--badge", default="",
                    help="short name shown in the capsule")
    lp.add_argument("--partial-every", dest="partial_every", type=float, default=0.35,
                    help="how often to re-transcribe while streaming (seconds). "
                         "Transcribing costs ~120ms, so this has room to be small.")
    lp.add_argument("--stream", action="store_true",
                    help="act on each instruction as you move past it, instead of "
                         "waiting for the whole sentence to finish")
    lp.add_argument("--hold", action="store_true",
                    help="push-to-talk: record while the key is HELD and ignore "
                         "silence, so a pause mid-thought does not end the clip")
    lp.add_argument("--raw", action="store_true",
                    help="type exactly what was said, fillers and false starts included")
    lp.add_argument("--clean-hard", dest="clean_hard", action="store_true",
                    help="also strip like, basically, literally")
    lp.add_argument("--fix-spelling", dest="fix_homophones", action="store_true",
                    help="ask Jev which spelling was meant for words the "
                         "recogniser was unsure of (there/their, to/too). "
                         "Needs a Jev key, and is the only part of dictation "
                         "that uses the network. Costs a second local decode: "
                         "measured at +0.15s per clip on a GPU and +3.4s on a "
                         "CPU, where it also almost never has anything to fix")
    lp.add_argument("--check-fillers", dest="fix_fillers", action="store_true",
                    help="ask Jev before deleting an ambiguous filler, so "
                         '"I would really like to use it" keeps its verb. '
                         "Needs a Jev key; one round trip per candidate word")
    lp.set_defaults(fn=cmd_listen)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    _log(args.log, getattr(args, 'title', 'jevflow'))
    # Resolve the machine BEFORE anything starts, so the choice is logged once
    # and every code path downstream sees a concrete device rather than "auto".
    from jevflow import stt
    args.device, args.model, args.compute_type = stt.pick_runtime(
        args.device, args.model, args.compute_type)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
