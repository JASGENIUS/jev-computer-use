@echo off
REM JevFlow dictation, with a console so you can see what it heard.
cd /d "%~dp0"
python main.py listen --dictate --hotkey windows+alt --accent #6cc4ff --badge JevFlow --title JevFlow
