@echo off
REM Quantise checkpoints/latest.pt into web_model/ for the browser app.
REM Then refresh the picker:  node tools\make_models_json.mjs
cd /d "%~dp0"
set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo no .venv found - run setup.cmd first
  exit /b 1
)
"%PY%" src\export_web.py checkpoints\latest.pt -o web_model --verify %*
