@echo off
REM Multi-modal run: images, captions, ascii, code and all three combination tasks.
.venv\Scripts\python.exe src\train.py --mix multi ^
  --img-px 32 --no-detail --context 1024 --vocab 1400 ^
  --embed 160 --layers 6 --heads 4 --lr 6e-4 --warmup 60 ^
  --synth-n 4000 --synth-code-n 256 --epoch-minutes 15 --ckpt-minutes 5 ^
  --workdir checkpoints
