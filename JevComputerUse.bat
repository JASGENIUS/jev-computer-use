@echo off
REM Jev Computer Use - press Right Ctrl + Right Shift, say it, press again.
REM   "open notepad"                         -> launches the app
REM   "open figma.com"                       -> opens any address, not just known ones
REM   "search for pizza near me"             -> uses your browser, no need to name one
REM   "on youtube search up codex"           -> searches inside YouTube
REM   "switch to obsidian"                   -> focuses it if it is already running
REM   "type hello world in notepad"
REM   "close the three apps you just opened" -> it remembers what it opened
REM   "next track" / "pause"                 -> media keys
REM It never stops to ask yes or no. Sure enough, it acts; not sure enough,
REM it says so and you say it again. "Close the apps you opened" only
REM touches apps it opened itself this session, never ones you opened.
REM
REM The capsule is AMBER for this one and says JCU. JevFlow's is blue.
REM
REM --stream: each instruction runs as you move PAST it, so a four-part
REM sentence has three of them done before you stop talking. A step only
REM fires once there is speech behind it, so it can no longer change.
REM Steps run ONE AT A TIME, in the order you said them. They used to be
REM batched for speed and it looked like a scramble of windows.
REM
REM No wake word for now: it made the tool look dead, because a sentence it
REM decided was not addressed to it is ignored SILENTLY. It comes back once
REM the acoustic model is trained. Add --wake transcript to try it.
cd /d "%~dp0"
start "" pythonw main.py --model small.en listen --command --continuous --command-hotkey "right ctrl+right shift" --hotkey none --stream --accent #f7bf54 --badge JCU --title "Jev Computer Use"
