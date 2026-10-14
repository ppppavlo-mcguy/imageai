"""CPU trainer with a progress bar, time-based checkpoints and resume-from-latest.

    python src/train.py                          # procedural data, ~2 h epochs, checkpoints every 15 min
    python src/train.py --images C:/photos       # your own pictures (+ optional .txt captions)
    python src/train.py --hf-dataset flickr8k    # real photos through the `datasets` library
    python src/train.py --resume none            # start a fresh run in the same workdir

An epoch is time-boxed: it ends after --epoch-minutes (default 120) or, if you set a finite
--samples-per-epoch, after that many samples. Checkpoints land in the workdir every
--ckpt-minutes; the next run picks up checkpoints/latest.pt automatically.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import signal
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

import ascii_art
import generate as gen
from bpe import BPE
from data import TrainData
from imagecodec import Codec
from model import GPT, GPTConfig, PRESETS
from vocab import Layout

STOP = {"flag": False}


def handle_sigint(signum, frame):
    STOP["flag"] = True
    print("\n[train] Ctrl+C received - finishing step, then saving checkpoints/latest.pt ...")


# ----------------------------------------------------------------------- argparse
def build_parser():
    ap = argparse.ArgumentParser(description="train the image/text/code model on CPU")
    d = ap.add_argument_group("data")
    d.add_argument("--images", default=None, help="folder of your own images (+ optional .txt captions)")
    d.add_argument("--hf-dataset", default=None,
                   help="HuggingFace image dataset (flickr8k, cats_vs_dogs, coco, or any name)")
    d.add_argument("--hf-config", default=None)
    d.add_argument("--hf-split", default=None)
    d.add_argument("--hf-image-col", default=None)
    d.add_argument("--hf-caption-col", default=None)
    d.add_argument("--hf-caption-template", default=None)
    d.add_argument("--hf-samples", type=int, default=3000)
    d.add_argument("--synth-n", type=int, default=20000, help="procedural scene count (offline default)")
    d.add_argument("--synth-code-n", type=int, default=64)
    d.add_argument("--seed-code-dir", default="data/seed_code")
    d.add_argument("--code-dataset", default=None,
                   help="HuggingFace code dataset, e.g. codeparrot/codeparrot-clean")
    d.add_argument("--code-samples", type=int, default=1500)

    s = ap.add_argument_group("image / tokenizer shape")
    s.add_argument("--img-px", type=int, default=64, choices=[32, 64, 96, 128])
    s.add_argument("--patch", type=int, default=4, choices=[2, 4, 8])
    s.add_argument("--no-detail", action="store_true", help="level-1 only: 256 tokens/img, ~2x faster")
    s.add_argument("--ascii-width", type=int, default=40,
                   help="ASCII targets are this many characters wide (they share the context with image tokens)")
    s.add_argument("--combo-ascii-width", type=int, default=32,
                   help="width of the ascii art used by the ascii->code task (smaller = shorter sequence)")
    s.add_argument("--vocab", type=int, default=1536, help="BPE merges to learn")
    s.add_argument("--bpe-samples", type=int, default=4000)
    s.add_argument("--tokenizer", default=None, help="existing tokenizer.json to reuse")

    m = ap.add_argument_group("model")
    m.add_argument("--preset", default="small", choices=list(PRESETS))
    m.add_argument("--embed", type=int, default=None)
    m.add_argument("--layers", type=int, default=None)
    m.add_argument("--heads", type=int, default=None)
    m.add_argument("--context", type=int, default=None)
    m.add_argument("--expansion", type=int, default=None)
    m.add_argument("--dropout", type=float, default=0.0)

    o = ap.add_argument_group("optimisation")
    o.add_argument("--lr", type=float, default=3e-4)
    o.add_argument("--min-lr-ratio", type=float, default=0.1)
    o.add_argument("--warmup", type=int, default=200)
    o.add_argument("--lr-horizon", type=int, default=20000)
    o.add_argument("--wd", type=float, default=0.1)
    o.add_argument("--beta2", type=float, default=0.95)
    o.add_argument("--grad-clip", type=float, default=1.0)
    o.add_argument("--batch-tokens", type=int, default=3072)
    o.add_argument("--max-batch-rows", type=int, default=64)
    o.add_argument("--loss-chunk", type=int, default=1024,
                   help="tokens scored per loss chunk; lower it if you run out of RAM")
    o.add_argument("--samples-per-epoch", type=int, default=0,
                   help="0 = keep generating samples until the epoch time budget is used up")
    o.add_argument("--epochs", type=int, default=20)
    o.add_argument("--max-seq-len", type=int, default=None, help="defaults to the model context length")
    o.add_argument("--bucket-window", type=int, default=48)

    r = ap.add_argument_group("run")
    r.add_argument("--seed", type=int, default=1337)
    r.add_argument("--workdir", default="checkpoints")
    r.add_argument("--resume", default="latest", help="latest | none | path/to.pt")
    r.add_argument("--epoch-minutes", type=float, default=120.0)
    r.add_argument("--ckpt-minutes", type=float, default=15.0)
    r.add_argument("--keep-step-files", type=int, default=3)
    r.add_argument("--val-every", type=int, default=200)
    r.add_argument("--val-batches", type=int, default=4)
    r.add_argument("--val-samples", type=int, default=48)
    r.add_argument("--log-every", type=int, default=10)
    r.add_argument("--sample-every", type=int, default=500, help="0 disables sample previews")
    r.add_argument("--sample-n", type=int, default=2)
    r.add_argument("--max-hours", type=float, default=0.0, help="0 = run all epochs")
    r.add_argument("--threads", type=int, default=0)
    r.add_argument("--device", default="cpu")

    w = ap.add_argument_group("task mix")
    w.add_argument("--w-txt2img", type=float, default=0.22)
    w.add_argument("--w-img2txt", type=float, default=0.15)
    w.add_argument("--w-ascii", type=float, default=0.08)
    w.add_argument("--w-code", type=float, default=0.20)
    w.add_argument("--w-code-lm", type=float, default=0.10)
    w.add_argument("--w-text-lm", type=float, default=0.10)
    w.add_argument("--w-ascii2code", type=float, default=0.08,
                   help="ascii art -> the web page that displays it")
    w.add_argument("--w-img2code", type=float, default=0.04,
                   help="image -> the web page that describes it")
    w.add_argument("--w-code2img", type=float, default=0.03,
                   help="code -> a preview image of the page it builds")
    w.add_argument("--w-fix", type=float, default=0.09,
                   help="snippet + error message -> the fixed snippet (\"fix this i got an error\")")
    w.add_argument("--mix", default=None, choices=list(MIXES),
                   help="weight preset: multi (default) | code (text/code output only) | image | all")
    return ap


# Task-mix presets. Anything left out of a preset is switched off (weight 0).
MIXES = {
    # everything, balanced: image, text, code, the combination tasks and error fixing
    "multi": {"w_txt2img": 0.20, "w_img2txt": 0.14, "w_ascii": 0.07, "w_code": 0.18,
              "w_code_lm": 0.09, "w_text_lm": 0.09, "w_ascii2code": 0.07,
              "w_img2code": 0.04, "w_code2img": 0.03, "w_fix": 0.09},
    # "train purely for code": snippets, continuation, ascii->website and error fixing.
    # Nothing here emits image tokens, so it runs like a text model.
    "code": {"w_txt2img": 0.0, "w_img2txt": 0.0, "w_ascii": 0.05, "w_code": 0.35,
             "w_code_lm": 0.15, "w_text_lm": 0.05, "w_ascii2code": 0.20,
             "w_img2code": 0.0, "w_code2img": 0.0, "w_fix": 0.20},
    # picture-focused, like the shipped demo model
    "image": {"w_txt2img": 0.35, "w_img2txt": 0.25, "w_ascii": 0.12, "w_code": 0.08,
              "w_code_lm": 0.05, "w_text_lm": 0.15, "w_ascii2code": 0.0,
              "w_img2code": 0.0, "w_code2img": 0.0, "w_fix": 0.0},
    # one tenth each: maximum variety in a single model
    "all": {k: 1 / 10 for k in ("w_txt2img", "w_img2txt", "w_ascii", "w_code", "w_code_lm",
                                "w_text_lm", "w_ascii2code", "w_img2code", "w_code2img",
                                "w_fix")},
}


def apply_mix(args):
    """--mix overrides the individual --w-* flags (explicit flags win only when no --mix is given)."""
    if not args.mix:
        return
    for key, val in MIXES[args.mix].items():
        setattr(args, key, val)
    print(f"[mix ] {args.mix}: " + ", ".join(
        f"{k[2:]}={getattr(args, k):.2f}" for k in MIXES['multi'] if getattr(args, k) > 0))


def current_lr(step: int, a) -> float:
    base, floor = a.lr, a.lr * a.min_lr_ratio
    if step < a.warmup:
        return base * (step + 1) / max(1, a.warmup)
    t = min(1.0, (step - a.warmup) / max(1, a.lr_horizon))
    return floor + (base - floor) * 0.5 * (1.0 + np.cos(np.pi * t))


def human(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds/60:.0f}m"
    return f"{seconds/3600:.1f}h"


def bar(frac: float, width: int = 22) -> str:
    frac = 0.0 if frac != frac else min(1.0, max(0.0, frac))
    n = int(round(frac * width))
    return "[" + "=" * n + (">" if n < width else "") + " " * max(0, width - n - 1) + "]"


class Line:
    """Single-line progress bar that other messages can be printed over."""

    def __init__(self, width=110):
        self.width = width
        self.last = ""

    def set(self, text: str):
        pad = max(0, len(self.last) - len(text))
        sys.stdout.write("\r" + text + " " * pad)
        sys.stdout.flush()
        self.last = text

    def log(self, msg: str = ""):
        sys.stdout.write("\r" + " " * len(self.last) + "\r" + msg + "\n")
        sys.stdout.flush()
        self.last = ""


# ------------------------------------------------------------------------- setup
def resolve_resume(args, workdir: Path):
    r = (args.resume or "none").strip()
    if r == "none":
        return None
    p = workdir / "latest.pt" if r == "latest" else Path(r)
    return p if p.exists() else None


def make_config(args, vocab_size: int) -> GPTConfig:
    p = dict(PRESETS[args.preset])
    for key, val in (("embed_size", args.embed), ("num_layers", args.layers),
                     ("num_heads", args.heads), ("context_length", args.context),
                     ("expansion", args.expansion)):
        if val:
            p[key] = val
    cfg = GPTConfig(vocab_size=vocab_size, dropout=args.dropout, loss_chunk=args.loss_chunk, **p)
    if cfg.embed_size % cfg.num_heads:
        rd = max(1, cfg.embed_size // cfg.num_heads)
        print(f"[cfg] embed_size {cfg.embed_size} not divisible by heads {cfg.num_heads}; using {rd} heads")
        cfg.num_heads = rd
    return cfg


def build_bpe(args, data: TrainData, workdir: Path, resume_path) -> BPE:
    if resume_path is not None:
        ck = torch.load(resume_path, map_location="cpu", weights_only=False)
        if "bpe" in ck:
            print(f"[tok] reusing tokenizer from {resume_path.name}")
            return BPE([tuple(m) for m in ck["bpe"]["merges"]])
    if args.tokenizer:
        p = Path(args.tokenizer)
        if p.exists():
            print(f"[tok] loaded {p}")
            return BPE.load(p)
    corpus: list[str] = []
    ni = max(1, len(data.img))
    stride = max(1, ni // max(1, args.bpe_samples))
    for i in range(0, ni, stride):
        corpus.append(data.img.get(i)[1])
    nc = data.code_len()
    for i in range(min(nc, max(64, args.bpe_samples // 4))):
        corpus.append(data.code_at(i)[1])
    for i in range(0, min(ni, 96), max(1, ni // 32)):
        corpus.append(ascii_art.to_ascii(data.img.get(i)[0], width=args.ascii_width))
    size = sum(len(c) for c in corpus)
    print(f"[tok] training BPE on {len(corpus)} documents / {size/1e3:.0f} kB (target {args.vocab} merges)")
    t0 = time.time()
    bpe = BPE.train(corpus, target_merges=args.vocab, log_every=max(1, args.vocab // 4))
    print(f"[tok] {bpe.n_text} text tokens in {human(time.time()-t0)}")
    bpe.save(workdir / "tokenizer.json")
    return bpe


def save_ckpt(workdir: Path, model, opt, args, cfg, layout, codec, bpe, step, epoch, batch_in_epoch,
              tokens_seen, best_val, tag="latest"):
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": opt.state_dict() if opt is not None else None,
        "config": cfg.to_json(),
        "layout": layout.to_json(),
        "codec": codec.to_json(),
        "bpe": json.loads((workdir / "_bpe.json").read_text()) if (workdir / "_bpe.json").exists() else None,
        "step": int(step),
        "epoch": int(epoch),
        "batches_in_epoch": int(batch_in_epoch),
        "tokens_seen": int(tokens_seen),
        "best_val": None if best_val is None else float(best_val),
        "args": {k: v for k, v in vars(args).items() if isinstance(v, (int, float, str, bool, type(None)))},
    }
    p = workdir / f"{tag}.pt"
    tmp = workdir / f"{tag}.pt.tmp"
    torch.save(payload, tmp)
    os.replace(tmp, p)                      # atomic on Windows and POSIX
    return p


def write_bpe_sidecar(workdir: Path, bpe: BPE):
    """Token merges ride along with checkpoints; kept as a sidecar so saving stays cheap."""
    (workdir / "_bpe.json").write_text(json.dumps(
        {"base": 9, "specials": ["<pad>", "<bos>", "<eos>", "<sep>", "<img>", "<cap>", "<ascii>",
                                 "<code>", "<ask>"], "n_text": bpe.n_text,
         "merges": [[int(a), int(b)] for a, b in bpe.merges]}), encoding="utf-8")


# -------------------------------------------------------------------------- main
def main():
    args = build_parser().parse_args()
    apply_mix(args)
    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    samples = Path("samples")
    samples.mkdir(exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.threads:
        torch.set_num_threads(args.threads)
    signal.signal(signal.SIGINT, handle_sigint)

    resume_path = resolve_resume(args, workdir)
    codec = Codec(args.img_px, args.patch, not args.no_detail)
    data = TrainData(args, codec, None)

    if resume_path is not None:
        print(f"[run] resuming from {resume_path}")
        ck = torch.load(resume_path, map_location="cpu", weights_only=False)
        ckc, ckl = ck["codec"], ck["layout"]
        if (ckc["img_px"], ckc["patch"], ckc["detail"]) != (codec.img_px, codec.patch, codec.detail):
            raise SystemExit("checkpoint was trained with a different image codec shape "
                             f"({ckc['img_px']}px/patch{ckc['patch']}/detail={ckc['detail']}); "
                             "pass matching --img-px/--patch or move the workdir aside")
        bpe = build_bpe(args, data, workdir, resume_path)
        layout = Layout(bpe.n_text, l1=ckl["l1"], l2=ckl["l2"] or 512, detail=codec.detail)
        if layout.total != ckl["total"]:
            raise SystemExit("tokenizer in the checkpoint does not match this image shape")
        cfg = GPTConfig(**{k: v for k, v in ck["config"].items() if k in GPTConfig.__annotations__})
        start_epoch, start_batch = ck["epoch"], ck["batches_in_epoch"]
        model = GPT(cfg)
        model.load_state_dict(ck["model_state_dict"])
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, args.beta2), weight_decay=args.wd)
        if ck.get("optimizer_state_dict"):
            opt.load_state_dict(ck["optimizer_state_dict"])
        global_step, tokens_seen, best_val = ck["step"], ck.get("tokens_seen", 0), ck.get("best_val")
        print(f"[run] resumed: epoch {start_epoch}, step {global_step}, in-epoch batch {start_batch}")
    else:
        print("[run] fresh run (--resume none or no checkpoints/latest.pt)")
        bpe = build_bpe(args, data, workdir, None)
        layout = Layout(bpe.n_text, detail=codec.detail)
        cfg = make_config(args, layout.total)
        model = GPT(cfg)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, args.beta2), weight_decay=args.wd)
        start_epoch = start_batch = 0
        global_step = tokens_seen = 0
        best_val = None
    write_bpe_sidecar(workdir, bpe)

    ctx = cfg.context_length
    args.max_seq_len = min(args.max_seq_len or ctx, ctx)
    data.bpe = bpe
    data.layout = layout
    if codec.n1 + codec.n2 + 8 > args.max_seq_len:
        raise SystemExit(f"image sequences need {codec.n1 + codec.n2 + 8} tokens but max_seq_len is "
                         f"{args.max_seq_len}; raise --context or lower --img-px/--patch/--no-detail")
    n_img_tok = codec.tokens_per_image()
    print(f"[cfg] vocab {layout.total} ({bpe.n_text} text + {layout.l1} img1 + {layout.l2} img2), "
          f"context {ctx}, params {model.n_params():,}, image tokens {n_img_tok}")
    print(f"[cfg] data: images={data.img.kind}({len(data.img)}) code={data.code_len()} docs, "
          f"target {args.batch_tokens} tokens/batch")

    val_plan = data.plan(-1, args.val_batches * 6, val=True)
    val_batches = []
    for k, descs in enumerate(data.plan_batches(val_plan, args.batch_tokens, val=True)):
        if k >= args.val_batches:
            break
        b = data.build_batch(descs, val=True)
        if b is not None:
            val_batches.append(b)

    log_csv = workdir / "log.csv"
    if not log_csv.exists():
        log_csv.write_text("wall,epoch,step,step_in_epoch,tokens_seen,lr,loss,val_loss,tok_per_s\n",
                           encoding="utf-8")
    status = Line()
    warned_skip = False
    model.train()
    run_start = time.time()
    last_ckpt = time.time()
    step_files: list[Path] = sorted(workdir.glob("step_*.pt"))
    interrupted_at = None

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        deadline = epoch_start + args.epoch_minutes * 60.0
        batch_in_epoch = start_batch if epoch == start_epoch else 0
        batch_idx = 0
        loss_sum, loss_n, tok_since = 0.0, 0, 0
        t_last = time.time()
        data.bpe = bpe
        for chunk in range(10_000):
            if args.samples_per_epoch > 0 and chunk > 0:
                break
            plan = data.plan_chunk(epoch, chunk, args.samples_per_epoch or 2000)
            stop_epoch = False
            for descs in data.plan_batches(plan, args.batch_tokens):
                if batch_idx < batch_in_epoch:            # resume: skip already-trained batches
                    batch_idx += 1
                    continue
                if STOP["flag"] or time.time() > deadline:
                    stop_epoch = True
                    break
                nx = data.build_batch(descs)
                if nx is None:
                    batch_idx += 1
                    continue
                x, y = nx
                lr = current_lr(global_step, args)
                for g in opt.param_groups:
                    g["lr"] = lr
                _, loss, _ = model(x, y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if args.grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                opt.step()
                global_step += 1
                batch_idx += 1
                ntok = int((y != 0).sum())
                tokens_seen += ntok
                tok_since += ntok
                loss_sum += float(loss.detach())
                loss_n += 1
                now = time.time()
                tps = tok_since / max(1e-6, now - t_last)
                frac = min(1.0, (now - epoch_start) / max(1e-6, deadline - epoch_start))
                status.set(f"ep {epoch+1}/{args.epochs} {bar(frac)} {frac*100:3.0f}% "
                           f"| step {global_step} (e{epoch+1}:{batch_idx}) loss {float(loss.detach()):5.3f} "
                           f"avg {loss_sum/max(1,loss_n):5.3f} | lr {lr:.2e} | {tps/1000:.1f}k tok/s "
                           f"| {human(now-epoch_start)}/{human(deadline-epoch_start)} "
                           f"| ckpt in {human(args.ckpt_minutes*60-(now-last_ckpt))} ")
                if data.skipped and not warned_skip:
                    warned_skip = True
                    status.log(f"[warn] {data.skipped} {data.skipped_kind} sequence(s) dropped: they need "
                               f"{data.skipped_len} tokens but --max-seq-len/--context is {args.max_seq_len}. "
                               f"Raise --context (e.g. 2048) or lower --ascii-width/--img-px.")
                if args.log_every and global_step % args.log_every == 0:
                    with log_csv.open("a", encoding="utf-8") as f:
                        f.write(f"{int(now-run_start)},{epoch+1},{global_step},{batch_idx},{tokens_seen},"
                                f"{lr:.6g},{float(loss):.4f},, {tps:.0f}\n")
                # ---- timed checkpoint
                if (now - last_ckpt) / 60.0 >= args.ckpt_minutes:
                    p = save_ckpt(workdir, model, opt, args, cfg, layout, codec, bpe, global_step,
                                  epoch, batch_idx, tokens_seen, best_val, tag="latest")
                    sp = save_ckpt(workdir, model, opt, args, cfg, layout, codec, bpe, global_step,
                                   epoch, batch_idx, tokens_seen, best_val, tag=f"step_{global_step:07d}")
                    step_files.append(sp)
                    while len(step_files) > args.keep_step_files:
                        old = step_files.pop(0)
                        if old.exists():
                            old.unlink()
                    last_ckpt = now
                    status.log(f"[ckpt] {time.strftime('%H:%M:%S')} saved {p.name} + {sp.name} "
                               f"(loss {loss_sum/max(1,loss_n):.3f}, {human(now-run_start)} in)")
                # ---- validation
                if args.val_every and global_step % args.val_every == 0 and val_batches:
                    model.eval()
                    with torch.no_grad():
                        vl = float(np.mean([float(model(*b)[1].detach()) for b in val_batches]))
                    model.train()
                    best_val = vl if best_val is None else min(best_val, vl)
                    status.log(f"[val ] step {global_step} val_loss {vl:.3f} (best {best_val:.3f})")
                    with log_csv.open("a", encoding="utf-8") as f:
                        f.write(f"{int(time.time()-run_start)},{epoch+1},{global_step},{batch_idx},"
                                f"{tokens_seen},{lr:.6g},{float(loss):.4f},{vl:.4f},0\n")
                # ---- sample previews
                if args.sample_every and global_step % args.sample_every == 0:
                    try:
                        model.eval()
                        caps = []
                        if args.w_txt2img > 0:
                            for k in range(args.sample_n):
                                img, cap = data.val_img.get(k)
                                caps.append(cap)
                                gen.preview(model, bpe, layout, codec, cap,
                                            samples / f"e{epoch+1}_s{global_step}_{k}.png")
                        txt, _img, _ = gen.run_task(model, bpe, layout, codec, "code",
                                                    "write html for a blue card", max_new=160)
                        (samples / f"e{epoch+1}_s{global_step}_code.html").write_text(txt, encoding="utf-8")
                        if args.w_ascii2code > 0:               # the combination tasks get previews too
                            art = ascii_art.to_ascii(data.val_img.get(0)[0],
                                                     width=args.combo_ascii_width)
                            art_txt, _i2, _ = gen.run_task(model, bpe, layout, codec, "ascii2code",
                                                           "turn this ascii art into a website",
                                                           ascii_text=art, max_new=200)
                            (samples / f"e{epoch+1}_s{global_step}_ascii2code.html").write_text(
                                art_txt, encoding="utf-8")
                        if args.w_fix > 0:                       # "i got this error" preview
                            fix_txt, _i3, _ = gen.run_task(
                                model, bpe, layout, codec, "fix", "i got this error",
                                code_text="function add(a, b) {\n  return a + b;",
                                err_text="SyntaxError: Unexpected end of input", max_new=120)
                            (samples / f"e{epoch+1}_s{global_step}_fix.txt").write_text(
                                fix_txt, encoding="utf-8")
                        if args.w_code2img > 0:
                            _t, prev_img, _ = gen.run_task(
                                model, bpe, layout, codec, "code2img",
                                "turn this website into an image preview",
                                code_text="<div class=\"card\">blue page</div>", max_new=300)
                            if prev_img is not None:
                                from imagecodec import save_png
                                save_png(prev_img, samples / f"e{epoch+1}_s{global_step}_code2img.png",
                                         upscale=6)
                        status.log(f"[prev] step {global_step}: " + " | ".join(caps))
                        model.train()
                    except Exception as e:                     # previews must never kill training
                        status.log(f"[prev] skipped: {type(e).__name__}: {e}")
                if time.time() > deadline:
                    stop_epoch = True
                    break
            if stop_epoch or (args.samples_per_epoch > 0):
                break
        model.eval()
        vl = None
        if val_batches:
            with torch.no_grad():
                vl = float(np.mean([float(model(*b)[1].detach()) for b in val_batches]))
            best_val = vl if best_val is None else min(best_val, vl)
            with log_csv.open("a", encoding="utf-8") as f:
                f.write(f"{int(time.time()-run_start)},{epoch+1},{global_step},{batch_idx},"
                        f"{tokens_seen},{current_lr(global_step,args):.6g},{loss_sum/max(1,loss_n):.4f},"
                        f"{vl:.4f},0\n")
        model.train()
        save_ckpt(workdir, model, opt, args, cfg, layout, codec, bpe, global_step, epoch + 1, 0,
                  tokens_seen, best_val, tag="latest")
        status.log(f"[ep  ] {epoch+1} done: {batch_idx} batches, loss {loss_sum/max(1,loss_n):.3f}, "
                   f"val {vl if vl is None else round(vl,3)}, {human(time.time()-epoch_start)}, saved latest.pt")
        if STOP["flag"]:
            interrupted_at = epoch
            break
        if args.max_hours and (time.time() - run_start) / 3600.0 >= args.max_hours:
            status.log(f"[run ] --max-hours {args.max_hours} reached, stopping cleanly")
            break
    status.log(f"[run ] finished/interrupted. latest checkpoint is {workdir/'latest.pt'}")
    print(f"[run ] next: python src/export_web.py {workdir/'latest.pt'} -o web_model")


if __name__ == "__main__":
    main()
