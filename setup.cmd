@echo off
REM Windows: create the venv and install the CPU training stack.
py -3 -m venv .venv || python -m venv .venv
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
echo.
echo Done. Now run:  train.cmd
