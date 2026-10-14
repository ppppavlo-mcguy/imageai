"""End-to-end check that "instruction -> image" really learns the pipeline's own image tokens.

Trains a tiny model on a handful of procedural scenes and then measures how many of the generated
level-1 image codes match the target scene exactly. If the code->id mapping, the sampler's image
region, the loss or the decoder were out of sync, this number would be near zero.

    python tools/overfit_test.py                 # ~1-3 minutes on a 2-core CPU
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

import synth
import tasks
from bpe import BPE
from generate import generate_ids, image_region
from imagecodec import Codec
from model import GPT, GPTConfig
from vocab import Layout

SCENES = 8
STEPS = 260
PX, PATCH = 32, 2


def main():
    torch.manual_seed(0)
    codec = Codec(PX, PATCH, detail=False)
    scenes = [synth.scene(i, seed=4242, px=PX) for i in range(SCENES)]
    captions = [c for _, c in scenes]
    bpe = BPE.train(captions, target_merges=200, log_every=0, verbose=False)
    layout = Layout(bpe.n_text, detail=codec.detail)
    cfg = GPTConfig(vocab_size=layout.total, embed_size=96, num_heads=4, num_layers=4,
                    context_length=768, loss_chunk=512)
    model = GPT(cfg)
    print(f"codec {codec.img_px}px patch {codec.patch} -> {codec.tokens_per_image()} image tokens, "
          f"vocab {layout.total}, params {model.n_params():,}")

    # pre-flight: incremental decoding (prefill + one token at a time) must equal a full forward.
    # If the KV cache were dropped, generation would lose its context after the first token and this
    # check would fail loudly instead of silently producing nonsense.
    probe = torch.randint(0, layout.total, (1, 24))
    with torch.no_grad():
        full, _, _ = model(probe)
        step_logits = []
        cache = None
        for t in range(24):
            lg, _, cache = model(probe[:, t:t + 1], cache=cache, pos=t, use_cache=True)
            step_logits.append(lg)
        cache_diff = float((torch.cat(step_logits, 1) - full).abs().max())
        prefill, _, cache2 = model(probe[:, :10], cache=None, pos=0, use_cache=True)
        lg2, _, _ = model(probe[:, 10:], cache=cache2, pos=10, use_cache=True)
        assert float((lg2 - full[:, 10:]).abs().max()) < 1e-4, "prefill + step disagrees with a full forward"
    print(f"kv-cache check: incremental vs full forward max |diff| = {cache_diff:.2e}")
    assert cache_diff < 1e-4, "KV cache is not accumulating - generation would lose its context"

    seqs = []
    targets = []
    for img, cap in scenes:
        l1, l2 = codec.encode(img)
        instr = tasks.IMAGE_INSTRUCTIONS[0].format(c=cap)
        seqs.append(tasks.text2img(bpe, layout, instr, l1, l2))
        targets.append(l1)
        assert all(layout.img1_base <= t < layout.img1_base + layout.l1
                   for t in tasks.img_prefix(layout, l1, l2)[1:]), "image ids outside their block"
    print(f"training set: {SCENES} scenes, sequence length {len(seqs[0])}, "
          f"image ids in [{layout.img1_base}, {layout.img1_base + layout.l1})")

    L = max(len(s) for s in seqs)
    x = torch.zeros((len(seqs), L - 1), dtype=torch.long)
    y = torch.zeros((len(seqs), L - 1), dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, :len(s) - 1] = torch.tensor(s[:-1])
        y[i, :len(s) - 1] = torch.tensor(s[1:])
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    model.train()
    t0 = time.time()
    for step in range(STEPS):
        _, loss, _ = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (step + 1) % 40 == 0:
            print(f"  step {step+1:4d}  loss {float(loss.detach()):.3f}  ({time.time()-t0:.0f}s)")

    model.eval()
    # (a) teacher-forced: feed the true prefix and check the argmax prediction for every image code.
    #     This isolates "is the mapping right" from "does free-running sampling drift".
    with torch.no_grad():
        lg, _, _ = model(x)
        pred = lg.argmax(-1)
        mask = y >= layout.img1_base
        tf_acc = float((pred[mask] == y[mask]).float().mean())
    # (b) greedy free-running generation from the caption alone.
    exact, pixel_err = [], []
    region = image_region(layout, codec)
    for (img, cap), tgt in zip(scenes, targets):
        instr = tasks.IMAGE_INSTRUCTIONS[0].format(c=cap)
        ids = generate_ids(model, tasks.ask_image_seq(bpe, instr), codec.tokens_per_image() + 2,
                           temp=1.0, top_k=1, top_p=1.0, region=region, stop_ids=(), seed=7)
        got = [t - layout.img1_base for t in ids if layout.img1_base <= t < layout.img1_base + layout.l1]
        got += [0] * (codec.n1 - len(got))
        hit = sum(1 for a, b in zip(got, tgt) if a == b) / len(tgt)
        rec = codec.decode(got[:codec.n1])
        pixel_err.append(float(np.abs(rec.astype(int) - img.astype(int)).mean()))
        exact.append(hit)
        print(f"  {cap[:46]:46s} exact level-1 codes {hit*100:5.1f}%  pixel MAE {pixel_err[-1]:5.1f}")
    mean_exact = float(np.mean(exact))
    print(f"\nteacher-forced image-code accuracy {tf_acc*100:.1f}%   (this must be ~100% on an "
          f"overfit set: it proves codes, ids, loss and decoder agree)")
    print(f"greedy generated image-code match {mean_exact*100:.1f}%, mean pixel MAE "
          f"{np.mean(pixel_err):.1f} (random would be ~{100.0/codec.n1:.1f}%)")
    ok = tf_acc > 0.95 and mean_exact > 0.9
    print("OVERFIT OK" if ok else "OVERFIT FAILED - pipeline mapping is suspect")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
