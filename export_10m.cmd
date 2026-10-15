@echo off
REM Quantise checkpoints_10m\latest.pt into models\sgm-10m and refresh the model picker.
cd /d "%~dp0"
set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo no .venv found - run setup.cmd first
  exit /b 1
)
if not exist "checkpoints_10m\latest.pt" (
  echo checkpoints_10m\latest.pt is missing - run train_10m.cmd first
  exit /b 1
)
"%PY%" src\export_web.py checkpoints_10m\latest.pt --name sgm-10m --verify %*
if errorlevel 1 exit /b %errorlevel%
node tools\make_models_json.mjs
echo.
echo Now: serve.cmd, then pick SGM-10M in the model box.
