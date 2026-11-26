@echo off
rem Free local Jev-compatible decision server (OneJev, Apache-2.0) for the bot's "jev" brain.
rem Usage:  start_onejev.bat                              (OneJev1.6-0.8B, ~0.16 s per decision)
rem         start_onejev.bat --model onejev1.6-4b ...     (any other qev serve options)
rem Listens on http://127.0.0.1:8000 only. Leave this window open while the bot runs.
setlocal
set "PY=%~dp0.venv-onejev\Scripts\python.exe"
if not exist "%PY%" (
  echo OneJev is not installed. See "Free local Jev" in README.md.
  exit /b 1
)
set PYTHONIOENCODING=utf-8
"%PY%" "%~dp0onejev_server.py" %*
