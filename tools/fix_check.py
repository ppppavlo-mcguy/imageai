"""Does the "fix this, i got an error" task actually learn? Measured, not assumed.

Trains a tiny model on the procedural error cases, then reports
  (a) teacher-forced token accuracy over the fix region, and
  (b) greedy generation exact-match: prompt = snippet + error, target = the reference fix.

A broken sequence format, a wrong question template or an off-by-one in the answer region would
show up here as near-zero accuracy - that is the point of the check.

    python tools/fix_check.py                 # ~1 minute on a 2-core CPU
    python tools/fix_check.py --steps 200 --cases 6
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

import synth
import tasks
import train
from bpe import BPE
from data import TrainData
from generate import generate_ids
from imagecodec import Codec
from model import GPT, GPTConfig
from vocab import EOS, SEP, Layout

TF_MIN = 0.98      # teacher-forced character accuracy over the fix region
GREEDY_MIN = 0.75  # exact fixes reproduced from the snippet + error message alone


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=12)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--context", type=int, default=512)
    a = ap.parse_args()

    torch.manual_seed(0)
    random.seed(0)

    args = train.build_parser().parse_args([
        "--img-px", "32", "--no-detail", "--synth-n", "4", "--synth-code-n", "4",
        "--context", str(a.context), "--vocab", "300",
    ])
    for key in train.MIXES["multi"]:                       # keep only the fix task in this run
        setattr(args, key, 1.0 if key == "w_fix" else 0.0)
    args.max_seq_len = a.context

    codec = Codec(args.img_px, args.patch, not args.no_detail)
    data = TrainData(args, codec, None)
    corpus = [c for i in range(a.cases) for c in synth.error_case(i)[1:4:2]]
    bpe = BPE.train(corpus, target_merges=300, log_every=0, verbose=False)
    layout = Layout(bpe.n_text, detail=codec.detail)
    data.bpe, data.layout = bpe, layout

    case = lambda i: synth.error_case(i % len(synth.ERROR_CASES))
    descs = [("fix", i, 0) for i in range(a.cases)]
    seqs = [data.seq(d) for d in descs]
    longest = max(len(s) for s in seqs)
    cfg = GPTConfig(vocab_size=layout.total, embed_size=96, num_heads=4, num_layers=4,
                    context_length=max(a.context, longest + 8), loss_chunk=512)
    model = GPT(cfg)
    print(f"vocab {layout.total} ({bpe.n_text} text), {a.cases} error cases, "
          f"longest sequence {longest} tokens, params {model.n_params():,}")

    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, betas=(0.9, 0.95), weight_decay=0.01)
    model.train()
    t0 = time.time()
    last = 0.0
    for step in range(a.steps):
        batch = random.sample(descs, min(4, len(descs)))
        x, y = data.build_batch(batch)
        _, loss, _ = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        last = float(loss.detach())
        if (step + 1) % max(1, a.steps // 8) == 0:
            print(f"  step {step + 1:4d}/{a.steps}  loss {last:.4f}")
    model.eval()
    print(f"trained in {time.time() - t0:.0f}s, final loss {last:.4f}")

    # (a) teacher-forced accuracy over the fix region
    right = total = 0
    exact_forced = 0
    with torch.no_grad():
        for i, d in enumerate(descs):
            seq = data.seq(d)
            x = torch.tensor([seq[:-1]], dtype=torch.long)
            logits, _, _ = model(x)
            pred = logits[0].argmax(-1).tolist()
            answer_at = _answer_start(seq)
            got = [p for p, want in zip(pred[answer_at - 1:], seq[answer_at:])]
            want = seq[answer_at:]
            total += len(want)
            right += sum(1 for g, w in zip(got, want) if g == w)
            exact_forced += int(got == want)
    acc = right / max(1, total)
    print(f"teacher-forced: {right}/{total} tokens = {acc * 100:.1f}%  "
          f"({exact_forced}/{len(descs)} sequences exactly right)")

    # (b) greedy generation from the snippet + error message alone
    matches = []
    with torch.no_grad():
        for i, d in enumerate(descs):
            _lang, broken, err, fixed = case(i)
            prompt = tasks.ask_fix_seq(bpe, broken, tasks.FIX_QUESTIONS[0].format(e=err))
            out = generate_ids(model, prompt, max_new=len(bpe.encode(fixed)) + 8, temp=1.0,
                               top_k=1, top_p=1.0, stop_ids=(EOS,), seed=0, min_new=6)
            text = bpe.decode(out).strip()
            matches.append(text == fixed.strip())
    hit = sum(matches)
    rate = hit / len(matches)
    print(f"greedy fixes reproduced: {hit}/{len(matches)} = {rate * 100:.0f}%")

    ok = acc >= TF_MIN and rate >= GREEDY_MIN
    if not ok:
        print(f"FIX CHECK FAILED (need token accuracy >= {TF_MIN:.0%} and greedy >= {GREEDY_MIN:.0%})")
    else:
        print("FIX CHECK OK")
    return 0 if ok else 1


def _answer_start(seq) -> int:
    """Index of the first id after the second <sep> (the fix region)."""
    first = seq.index(SEP)
    second = seq.index(SEP, first + 1)
    return second + 1


if __name__ == "__main__":
    raise SystemExit(main())
