@echo off
REM JevFlow - dictation.  Press Win+Alt, speak, press again when you are done.
REM A pause to think does NOT end the clip: it waits 2.5 seconds of quiet.
REM Esc throws the clip away; pressing the key again keeps it.
REM
REM Fillers, false starts and self-corrections come out automatically.
REM Two extras are OFF by default because they are the only parts that use
REM the network, and dictation otherwise never leaves this machine:
REM   --fix-spelling   Jev picks there/their, to/too when Whisper was unsure.
REM                    +0.15s a clip on a GPU, +3.4s on a CPU.
REM   --check-fillers  Jev vetoes a filler deletion when the word has a job,
REM                    so "I would really like to use it" keeps its verb.
cd /d "%~dp0"
start "" pythonw main.py listen --dictate --hotkey windows+alt --accent #6cc4ff --badge JevFlow --title JevFlow
