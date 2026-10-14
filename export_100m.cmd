@echo off
REM Quantise checkpoints_100m\latest.pt into models\sgm-100m and refresh the model picker.
REM int8 keeps the download honest (~89 MB); add --bits 4 for roughly half that.
call .venv\Scripts\activate
python src\export_web.py checkpoints_100m\latest.pt --name sgm-100m --verify %*
if errorlevel 1 exit /b %errorlevel%
node tools\make_models_json.mjs
echo.
echo Now: serve.cmd, then pick SGM-100M in the model box.
