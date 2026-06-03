@echo off
REM Cascadia launcher — double-click this, or run: run.bat [map|train|validate|serve|all]
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo [setup] Creating virtual environment...
  python -m venv .venv
  echo [setup] Installing dependencies ^(one time, ~1-2 min^)...
  ".venv\Scripts\python.exe" -m pip install --quiet --upgrade pip
  ".venv\Scripts\python.exe" -m pip install --quiet -r requirements.txt
)

set PYTHONIOENCODING=utf-8
".venv\Scripts\python.exe" run.py %*

echo.
pause
