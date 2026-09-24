@echo off
REM The settings panel for both tools. Everything used to live in six .bat
REM files, so nothing ever got tuned.
cd /d "%~dp0"
start "" pythonw dashboard.py
