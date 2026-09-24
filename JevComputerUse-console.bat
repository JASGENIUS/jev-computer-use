@echo off
REM Same as JevComputerUse.bat, with a console so you can see what it decided.
cd /d "%~dp0"
python main.py --model small.en listen --command --continuous --command-hotkey "right ctrl+right shift" --hotkey none --stream --accent #f7bf54 --badge JCU --title "Jev Computer Use"
