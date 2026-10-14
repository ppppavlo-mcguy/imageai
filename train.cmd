@echo off
REM Start or resume training (Ctrl+C is safe: it saves checkpoints/latest.pt first).
call .venv\Scripts\activate
python src/train.py --workdir checkpoints --preset small --vocab 2048 --img-px 64 --patch 4 ^
  --synth-n 4000 --synth-code-n 128 --batch-tokens 2048 --epoch-minutes 120 --epochs 20 ^
  --ckpt-minutes 15 --val-every 200 --sample-every 500 %*
