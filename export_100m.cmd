@echo off
REM Quantise checkpoints_100m\latest.pt into models\sgm-100m and refresh the model picker.
REM int8 keeps the download honest (~89 MB); add --bits 4 for roughly half that.
cd /d "%~dp0"
set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo no .venv found - run setup.cmd first
  exit /b 1
)
if not exist "checkpoints_100m\latest.pt" (
  echo checkpoints_100m\latest.pt is missing - run train_100m.cmd first
  exit /b 1
)
"%PY%" src\export_web.py checkpoints_100m\latest.pt --name sgm-100m --verify %*
if errorlevel 1 exit /b %errorlevel%
node tools\make_models_json.mjs
echo.
echo Now: serve.cmd, then pick SGM-100M in the model box.
