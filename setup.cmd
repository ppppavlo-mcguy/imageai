@echo off
REM Windows: create the venv and install the CPU training stack.
cd /d "%~dp0"
py -3 -m venv .venv || python -m venv .venv
set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo could not create .venv - install Python 3.11+ and run this again
  exit /b 1
)
"%PY%" -m pip install --upgrade pip
"%PY%" -m pip install -r requirements.txt
echo.
echo Done. Now run:  train.cmd
