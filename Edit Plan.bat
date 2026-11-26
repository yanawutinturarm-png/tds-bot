@echo off
rem Opens the build-plan editor: which towers the bot places, in what order, and how far it upgrades them.
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" call "%~dp0setup.bat" || exit /b 1
start "" ".venv\Scripts\pythonw.exe" plan_editor.py
