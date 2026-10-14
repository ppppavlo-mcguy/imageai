@echo off
REM Pure-code training run: web code + Python + ascii->website. No image generation.
REM ~10x faster per token than a picture run because nothing emits image tokens.
.venv\Scripts\python.exe src\train.py --mix code ^
  --img-px 32 --no-detail --context 1024 --vocab 1400 ^
  --embed 160 --layers 6 --heads 4 --lr 6e-4 --warmup 60 ^
  --synth-code-n 256 --epoch-minutes 8 --ckpt-minutes 4 ^
  --workdir checkpoints_code
