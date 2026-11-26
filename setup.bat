@echo off
rem One-time setup, run automatically by "Start TDS Bot.bat":
rem   .venv          the bot (screen capture, OpenCV, DirectInput)
rem   .venv-onejev   the free local Jev brain (PyTorch with CUDA, OneJev)
setlocal
cd /d "%~dp0"
where python >nul 2>nul
if errorlevel 1 (
  echo Python 3.10 or newer is not installed or not on PATH: https://www.python.org/downloads/
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo [1/2] Installing the bot...
  python -m venv .venv || exit /b 1
  ".venv\Scripts\python" -m pip install --disable-pip-version-check -q -r requirements.txt || exit /b 1
)

if not exist ".venv-onejev\Scripts\qev.exe" (
  echo [2/2] Installing the free Jev brain: PyTorch and OneJev, several GB, only this once...
  python -m venv .venv-onejev || exit /b 1
  ".venv-onejev\Scripts\python" -m pip install --disable-pip-version-check -q torch torchvision --index-url https://download.pytorch.org/whl/cu128 || exit /b 1
  ".venv-onejev\Scripts\python" -m pip install --disable-pip-version-check -q "qev[torch] @ git+https://github.com/OmniJev/OneJev.git@80d4e5f2f6dd5af55ca78e74d0dfa3536276106d" || exit /b 1
  rem Optional speed-up; the brain still works if this fails.
  ".venv-onejev\Scripts\python" -m pip install --disable-pip-version-check -q "triton-windows<3.8" flash-linear-attention
)

echo Setup complete.
exit /b 0
