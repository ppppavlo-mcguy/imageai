@echo off
REM Quantise checkpoints_10m\latest.pt into models\sgm-10m and refresh the model picker.
call .venv\Scripts\activate
python src\export_web.py checkpoints_10m\latest.pt --name sgm-10m --verify %*
if errorlevel 1 exit /b %errorlevel%
node tools\make_models_json.mjs
echo.
echo Now: serve.cmd, then pick SGM-10M in the model box.
