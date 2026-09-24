# Jev Computer Use

Control a Windows PC by talking to it. Press Right Ctrl + Right Shift and say
something like:

> open Chrome and search for pizza near me, then open Notepad and type hello,
> then close the apps you just opened

It does each step in the order you said it, one at a time, and it starts on the
first step before you've finished the sentence.

The repo also includes JevFlow, a dictation tool that types what you say into
whatever window you're in, with the "um"s, restarts and "no wait, I meant"
corrections taken out. Both share one speech model, so they can run in one
process.

## What it can do

```
open notepad                          launches it
open figma.com                        any address, not just ones it knows
search for pizza near me              uses your default browser
on youtube search up lofi             searches inside YouTube
switch to obsidian                    focuses it if it's already running
type hello world in notepad           types into a named app
close this                            closes the window you're in
close the three apps you just opened  it remembers what it opened
next track / pause                    media keys
```

Chain as many as you like with "and", "then" or commas. "Close the apps you
opened" only ever closes apps JCU opened itself in this session, and it closes
them the polite way (WM_CLOSE), so anything with unsaved work puts up its own
save prompt instead of losing it.

## What it won't do

It never stops to ask "did you mean X? yes or no". If it's sure enough, it acts.
If it isn't, it tells you it didn't catch that and you say it again. There's a
test that fails if a confirmation prompt ever comes back (`tests/test_never_asks.py`).

It won't write things for you. "Write a poem about apples" is refused rather
than typing the words "a poem about apples" into Notepad. Doing that properly
needs a language model that writes text, and this project doesn't have one.

It won't force-kill a process. The code for it exists, but JCU refuses it
outright, so a misheard sentence can't throw away someone's unsaved work.

## Setup

You need Windows 10 or 11, Python 3.10 or newer (developed on 3.13), and a
microphone. A GPU is optional.

```
git clone https://github.com/JASGENIUS/jev-computer-use
cd jev-computer-use
pip install -r requirements.txt
```

Computer use needs a Jev API key from [TypeSafe](https://typesafe.ai).
Dictation doesn't. Either set it once for your user:

```
setx JEV_API_KEY "your-key-here"
```

or copy `.env.example` to `.env` and put it there. `.env` is gitignored.

Then double-click one of the launchers:

| launcher | what it starts |
|---|---|
| `JevComputerUse.bat` | computer use on Right Ctrl + Right Shift |
| `JevFlow.bat` | dictation on Win + Alt |
| `Jev.bat` | both in one process, sharing one model |
| `Dashboard.bat` | a settings panel and the recent logs |

Each has a `-console` twin that keeps a terminal open so you can see what it
heard and what it decided. Use those the first time.

The first run downloads the speech model from Hugging Face. JCU uses `small.en`
(461 MB). Dictation picks `large-v3` (2.9 GB) on a GPU and `base.en` (138 MB) on
a CPU.

Press the hotkey, talk, and press it again when you're done, or just stop
talking: 2.5 seconds of quiet ends the clip on its own. Esc throws a clip away.

## What leaves your machine

Speech-to-text runs locally with [faster-whisper](https://github.com/SYSTRAN/faster-whisper).
Audio is never uploaded.

For computer use, the transcript of your command and a shortlist of candidate
apps or sites go to the Jev API, which answers one typed question: which of
these did you mean, and how sure are you? Dictation sends nothing by default.
Two optional dictation extras, `--fix-spelling` (there/their, to/too) and
`--check-fillers`, do send the sentence to Jev, and both are off unless you
turn them on.

Logs are written to `logs/` in the repo folder and contain your transcripts.
Corrections you teach it go to `learned.json`. Both stay on your machine and are
gitignored.

## How it decides

Code does the wide search and Jev only chooses between a few. A typical PC has a
couple of hundred installed apps, and handing all of them to one model call
gives confident noise, so
string matching cuts the list to about five first, and Jev picks one of those or
"none of these". It answers with a choice from a closed list and a confidence,
never free text, so a sentence like "open Chrome and delete everything" has no
way to turn into a command nobody wrote.

The search query, the text to type and the media key are pulled out of the
sentence by patterns in `jevflow/parse.py`. `jevflow/policy.py` decides what a
given confidence is allowed to do for each verb. A verb with no policy is
refused.

Nothing is assumed to have worked. A launch counts when a window appears, a
close when the window goes away. Things that can't be checked, like a media key,
are reported as unknown rather than as a success.

## Known limits

- Windows only. Focus handling, window lookup and app discovery all use Win32.
- English only. The step splitter, the command patterns and the filler cleaner
  are all written for English.
- No wake word yet. The code for one is in `jevflow/wake.py` (openWakeWord), but
  there's no trained model for "Jev", so the launchers go by hotkey alone.
  `--wake transcript` checks the start of each sentence for "Jev" instead.
- Store apps like Notepad and Calculator have no executable to track, so they
  are closed by window title, and a launch that can't be confirmed is reported
  as unknown.
- The numbers below are from one machine (RTX 3070 Ti). On a CPU it works but
  is a lot slower.

Measured on that machine: `large-v3` transcribes a short clip in about 0.75 s,
and Jev answered in 70 to 180 ms (median about 96 ms) across 14 kinds of
command.

## Layout

| file | does |
|---|---|
| `jevflow/stt.py` | faster-whisper, CUDA setup, vocabulary priming |
| `jevflow/recorder.py` | mic capture that stops when you stop talking |
| `jevflow/streaming.py` | runs each step once you've moved past it |
| `jevflow/disfluency.py` | strips fillers, restarts and self-corrections |
| `jevflow/apps.py` | every installed app: shortcuts, Store apps, bare installs |
| `jevflow/sites.py` | known sites, plus any address you say |
| `jevflow/parse.py` | splits steps, pulls out the query, text or key |
| `jevflow/jev.py` | the one typed question to Jev |
| `jevflow/policy.py` | what each confidence is allowed to do, per verb |
| `jevflow/command.py` | builds candidates, applies the policy, runs the step |
| `jevflow/actions.py` | focus, type, close, open a URL, press a media key |
| `jevflow/overlay.py` | the capsule at the top of the screen; never takes focus |
| `jevflow/app.py` | wires hotkeys, recorder, model and overlay together |

To make recognition better for your own words, add them to `VOCABULARY` in
`jevflow/stt.py`. Whisper is primed with that list, and it's the cheapest fix
for a name it keeps mishearing.

## Tests

```
python -m pytest     # 738 tests, no microphone or key needed
python smoke.py      # drives the real session path with a fake microphone
python mutate.py     # breaks each fix on purpose and checks a test notices
```

`mutate.py` exists because a test that can't fail is worse than no test. Every
fix in it was once "tested" by something that would have passed with the fix
deleted.

## License

MIT. See `LICENSE`.
