@echo off
rem Double-click to play: installs everything on the first run, starts the free Jev brain,
rem then the bot, which brings Roblox to the front and plays by itself.
rem   Q = pause / resume     ESC = stop the bot
rem Extra arguments go to the bot, e.g.  "Start TDS Bot.bat" --dry-run
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" goto setup
if not exist ".venv-onejev\Scripts\qev.exe" goto setup
goto run

:setup
call "%~dp0setup.bat"
if errorlevel 1 goto failed

:run
set PYTHONIOENCODING=utf-8
".venv\Scripts\python.exe" launch.py %*
echo.
pause
exit /b 0

:failed
echo Setup failed - see the messages above.
pause
exit /b 1
