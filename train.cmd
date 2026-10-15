@echo off
REM Start or resume training (Ctrl+C is safe: it saves checkpoints/latest.pt first).
REM The venv interpreter is called directly, like train_code.cmd always did: `call
REM .venv\Scripts\activate` is unnecessary and on some machines prints
REM "The system cannot find the path specified." without telling you why.
cd /d "%~dp0"
set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo no .venv found - run setup.cmd first
  exit /b 1
)
"%PY%" src\train.py --workdir checkpoints --preset small --vocab 2048 --img-px 64 --patch 4 ^
  --synth-n 4000 --synth-code-n 128 --batch-tokens 2048 --epoch-minutes 120 --epochs 20 ^
  --ckpt-minutes 15 --val-every 200 --sample-every 500 %*
