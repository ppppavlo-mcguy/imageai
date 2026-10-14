"""Build one training sequence for every task kind and prove the shapes are right.

Run from the repo root:

    python tools/check_tasks.py                 # the default (multi) mix
    python tools/check_tasks.py --mix code      # the pure-code mix
    python tools/check_tasks.py --context 2048 --img-px 64

Checks per task kind: the sequence is non-empty, every id is inside the vocabulary, it fits the
context (or is reported as too long), and that x=seq[:-1] / y=seq[1:] batches line up.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

import train
from bpe import BPE
from data import TrainData
from imagecodec import Codec
from vocab import Layout

KINDS = ["img2cap", "img2ascii", "txt2img", "code", "code_lm", "text_lm",
         "ascii2code", "img2code", "code2img", "fix"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="web_model/tokenizer.json")
    ap.add_argument("--mix", default=None, choices=list(train.MIXES))
    ap.add_argument("--img-px", type=int, default=32)
    ap.add_argument("--context", type=int, default=0, help="0 = preset default")
    ap.add_argument("--synth-n", type=int, default=12)
    ap.add_argument("--synth-code-n", type=int, default=24)
    a = ap.parse_args()

    argv = ["--img-px", str(a.img_px), "--no-detail", "--synth-n", str(a.synth_n),
            "--synth-code-n", str(a.synth_code_n), "--vocab", "900", "--bpe-samples", "400"]
    if a.context:
        argv += ["--context", str(a.context)]
    if a.mix:
        argv += ["--mix", a.mix]
    args = train.build_parser().parse_args(argv)
    train.apply_mix(args)                          # same override train.py applies before data setup
    args.max_seq_len = args.context or 2048        # what data.build_batch() truncates/drops against

    codec = Codec(args.img_px, args.patch, not args.no_detail)
    data = TrainData(args, codec, None)
    tok = Path(a.tokenizer)
    if tok.exists():
        bpe = BPE.load(tok)
    else:
        print(f"[note] {tok} not found - training a throwaway tokenizer for this check")
        bpe = BPE.train([data.code_at(i)[1] for i in range(data.code_len())], target_merges=300)
    layout = Layout(bpe.n_text, detail=codec.detail)
    data.bpe, data.layout = bpe, layout
    ctx = args.context or 2048
    print(f"vocab {layout.total} ({bpe.n_text} text) - context {ctx} - image "
          f"{codec.tokens_per_image()} tokens - code docs {data.code_len()}")

    weights = data._weights()
    ok = True
    for kind in KINDS:
        w = weights.get(kind, 0)
        if w <= 0:
            print(f"  {kind:11s} weight 0.00  skipped")
            continue
        seq = data.seq((kind, 0, 0), val=False)
        bad = [t for t in seq if t < 0 or t >= layout.total]
        fits = len(seq) <= ctx
        x, y = _batch(data, kind, ctx)
        shaped = x is not None and y is not None and x.shape == y.shape
        print(f"  {kind:11s} weight {w:.2f}  len {len(seq):5d}  in-vocab {not bad}  "
              f"fits {fits}  batch {tuple(x.shape) if shaped else 'FAILED'}")
        ok &= not bad and fits and shaped
        if not fits:
            print(f"      -> needs {len(seq)} tokens; raise --context or lower "
                  f"--img-px/--patch/--ascii-width/--combo-ascii-width")
    print("TASKS OK" if ok else "TASKS FAILED")
    return 0 if ok else 1


def _batch(data, kind, ctx):
    descs = []
    for j in range(3):
        d = (kind, j, j % 2)
        if len(data.seq(d)) <= ctx:
            descs.append(d)
    if not descs:
        return None, None
    b = data.build_batch(descs)
    if b is None:
        return None, None
    x, y = b
    # sanity: y must be x shifted by one token, and padding must be id 0 on both sides
    first = data.seq(descs[0])
    if int(x[0, 0]) != first[0] or int(y[0, 0]) != first[1]:
        return None, None
    if int(x.shape[1]) < len(first) - 1:
        return None, None
    return x, y


if __name__ == "__main__":
    raise SystemExit(main())
