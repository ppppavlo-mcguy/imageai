@echo off
REM Quantise checkpoints/latest.pt into web_model/ for the browser app.
call .venv\Scripts\activate
python src/export_web.py checkpoints\latest.pt -o web_model --verify %*
