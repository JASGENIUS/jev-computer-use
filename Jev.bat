@echo off
REM Both tools in ONE process - one Whisper model, so it fits in 4 GB of VRAM.
REM   Win+Alt      dictation: press, speak, press again when done
REM   RCtrl+RShift computer use: press, say the command, press again
REM
REM Neither key set contains the other, deliberately. ctrl+alt vs ctrl+alt+1
REM DID collide, because one was a subset of the other.
REM
REM A 2.5-second pause ends a clip on its own; pressing the key again ends it
REM straight away and KEEPS what you said. Esc throws it away.
REM
REM --stream is on, same as JevComputerUse.bat. Without it this launcher waited
REM for the whole sentence, so a four-part instruction did nothing until you
REM stopped talking - the one-process build behaved differently from the
REM two-process one for no reason anybody chose.
cd /d "%~dp0"
start "" pythonw main.py listen --dictate --hotkey windows+alt --accent #6cc4ff --badge JevFlow --command --continuous --command-hotkey "right ctrl+right shift" --stream --title Jev
