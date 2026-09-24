@echo off

REM Same as Jev.bat, with a console so you can see what it heard and decided.

cd /d "%~dp0"

python main.py listen --dictate --hotkey windows+alt --accent #6cc4ff --badge JevFlow --command --continuous --command-hotkey "right ctrl+right shift" --stream --title Jev

