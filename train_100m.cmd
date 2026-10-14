@echo off
REM Train the SGM-100M bundle (~89M parameters).  This is a multi-day CPU run on a small box:
REM start it on a machine you can leave alone (Ctrl+C always saves checkpoints_100m\latest.pt).
REM Afterwards:  export_100m.cmd
call .venv\Scripts\activate
python src\train.py --workdir checkpoints_100m --preset 100m --mix multi ^
  --vocab 4096 --img-px 64 --patch 4 --synth-n 40000 --synth-code-n 512 ^
  --batch-tokens 8192 --max-batch-rows 16 --loss-chunk 512 --epochs 40 --epoch-minutes 720 ^
  --ckpt-minutes 30 --val-every 500 --sample-every 2000 %*
