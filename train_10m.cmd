@echo off
REM Train the SGM-10M bundle (~9M parameters).  A real CPU run: see the sizing table in README.
REM Afterwards:  export_10m.cmd  (or: python src\export_web.py checkpoints_10m\latest.pt --name sgm-10m --verify)
call .venv\Scripts\activate
python src\train.py --workdir checkpoints_10m --preset 10m --mix multi ^
  --vocab 2048 --img-px 64 --patch 4 --synth-n 20000 --synth-code-n 256 ^
  --batch-tokens 4096 --max-batch-rows 32 --epochs 20 --epoch-minutes 240 ^
  --ckpt-minutes 20 --val-every 400 --sample-every 1000 %*
