"""Sampling + task glue. Used by train.py for previews and by the CLI for standalone inference.

Tasks:
  image       prompt text  -> image tokens -> PNG
  caption     image file   -> caption text
  ascii       image file   -> ascii text art
  code        instruction  -> html/css/js/python text
  text        continuation
  ascii2code  image file   -> the web page that displays its ascii art
  img2code    image file   -> the web page that describes the picture
  code2img    code snippet -> preview image of the page it builds
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

import ascii_art
import tasks
from bpe import BPE
from imagecodec import Codec, load_image, save_png
from model import GPT, GPTConfig
from vocab import ASCII, ASK, BOS, CAP, CODE, EOS, SEP, Layout

QUESTION = {"caption": "what is on this picture?", "ascii": "turn this into an ascii text art"}


# ------------------------------------------------------------------- utilities
def text_region(i):
    return None


def image_region(layout: Layout, codec: Codec):
    def f(i):
        if i < codec.n1:
            return (layout.img1_base, layout.img1_base + layout.l1)
        if i < codec.n1 + codec.n2:
            return (layout.img2_base, layout.img2_base + layout.l2)
        return EOS
    return f


def _top_k_top_p(logits: torch.Tensor, top_k: int, top_p: float):
    if top_k and top_k > 0:
        k = min(top_k, logits.numel())
        thresh = torch.topk(logits, k).values[-1]
        logits = logits.masked_fill(logits < thresh, float("-inf"))
    if top_p and top_p < 1.0:
        srt, idx = torch.sort(logits, descending=True)
        probs = torch.softmax(srt, dim=-1)
        cum = torch.cumsum(probs, dim=-1)
        cut = cum - probs > top_p
        srt = srt.masked_fill(cut, float("-inf"))
        out = torch.full_like(logits, float("-inf"))
        out.scatter_(0, idx, srt)
        logits = out
    return logits


def _penalize(logits: torch.Tensor, produced: list, penalty: float):
    if penalty and penalty != 1.0 and produced:
        for t in set(produced[-64:]):
            if logits[t] > 0:
                logits[t] /= penalty
            else:
                logits[t] *= penalty
    return logits


@torch.no_grad()
def generate_ids(model: GPT, prompt_ids, max_new: int, temp: float = 0.9, top_k: int = 64,
                 top_p: float = 0.95, rep_penalty: float = 1.1, region=None,
                 stop_ids=(EOS,), seed: int | None = None, min_new: int = 0) -> list:
    """Autoregressive sampling with a KV cache and optional allowed-id region per step."""
    if seed is not None:
        torch.manual_seed(seed)
    model.eval()
    produced: list[int] = []
    if not prompt_ids:
        prompt_ids = [BOS]
    logits, _, cache = model(torch.tensor([prompt_ids], dtype=torch.long), cache=None, pos=0,
                             logits_last_only=True, use_cache=True)
    logits = logits[0, -1].float().clone()
    pos = len(prompt_ids)
    for step in range(max_new):
        allowed = region(step) if region is not None else None
        if isinstance(allowed, int):                       # forced token (end of image)
            tok = allowed
            produced.append(tok)
            break
        lg = logits.clone()
        if allowed is not None:
            lo, hi = allowed
            mask = torch.full_like(lg, float("-inf"))
            mask[lo:hi] = lg[lo:hi]
            lg = mask
        if min_new and step < min_new:
            lg[EOS] = float("-inf")      # never stop before min_new tokens
        lg = _penalize(lg, produced, rep_penalty)
        if temp and temp > 0:
            lg = lg / temp
        lg = _top_k_top_p(lg, top_k, top_p)
        probs = torch.softmax(lg, dim=-1)
        tok = int(torch.multinomial(probs, 1).item())
        produced.append(tok)
        if tok in stop_ids:
            break
        logits, _, cache = model(torch.tensor([[tok]], dtype=torch.long), cache=cache, pos=pos,
                                 logits_last_only=True, use_cache=True)
        logits = logits[0, -1].float().clone()
        pos += 1
    return produced


def ids_to_image(layout: Layout, codec: Codec, ids) -> np.ndarray:
    l1, l2 = [], []
    for t in ids:
        if layout.img1_base <= t < layout.img1_base + layout.l1:
            l1.append(t - layout.img1_base)
        elif layout.detail and layout.img2_base <= t < layout.img2_base + layout.l2:
            l2.append(t - layout.img2_base)
    l1 = (l1 + [0] * codec.n1)[:codec.n1] if len(l1) < codec.n1 else l1[:codec.n1]
    if codec.detail:
        l2 = (l2 + [171] * codec.n2)[:codec.n2]        # 171 == zero delta in base-7 packing
    return codec.decode(l1, l2)


# ------------------------------------------------------------------- high level
def run_task(model, bpe, layout, codec, task: str, prompt: str = "", image_path: str = None,
             max_new: int = 420, temp: float = 0.9, top_k: int = 64, top_p: float = 0.95,
             seed: int | None = None, code_text: str = None, ascii_text: str = None,
             err_text: str = None):
    """Returns (text, image_array_or_None, produced_ids)."""
    min_new = {"caption": 2, "ascii": 24, "code": 8, "text": 4, "fix": 8,
               "ascii2code": 12, "img2code": 12}.get(task, 0)
    if task == "fix":
        snippet = code_text or "function add(a, b) {\n  return a + b;"
        err = err_text or "SyntaxError: Unexpected end of input"
        template = prompt if (prompt and "{e}" in prompt) else tasks.FIX_QUESTIONS[0]
        ids = tasks.ask_fix_seq(bpe, snippet, template.format(e=err))
        out = generate_ids(model, ids, max_new=max_new, temp=temp, top_k=top_k, top_p=top_p,
                           seed=seed, min_new=min_new)
        return bpe.decode(out), None, out
    if task in ("ascii2code", "img2code"):
        ids = None
        if task == "ascii2code":
            art = ascii_text
            if art is None:
                img = load_image(image_path, codec.img_px) if image_path else \
                    np.zeros((codec.img_px,) * 2 + (3,), np.uint8)
                art = ascii_art.to_ascii(img, width=32)
            ids = tasks.ask_ascii2code_seq(bpe, art, prompt)
        else:
            img = load_image(image_path, codec.img_px) if image_path else \
                np.zeros((codec.img_px,) * 2 + (3,), np.uint8)
            l1, l2 = codec.encode(img)
            ids = tasks.ask_img2code_seq(bpe, layout, l1, l2, prompt)
        out = generate_ids(model, ids, max_new=max_new, temp=temp, top_k=top_k, top_p=top_p,
                           seed=seed, min_new=min_new)
        return bpe.decode(out), None, out
    if task == "code2img":
        ids = tasks.ask_code2img_seq(bpe, code_text or prompt, prompt)
        out = generate_ids(model, ids, max_new=max_new, temp=temp, top_k=top_k, top_p=top_p,
                           region=image_region(layout, codec), stop_ids=(), seed=seed)
        return "", ids_to_image(layout, codec, out), out
    if task == "image":
        ids = tasks.ask_image_seq(bpe, prompt)
        out = generate_ids(model, ids, max_new=max_new, temp=temp, top_k=top_k, top_p=top_p,
                           region=image_region(layout, codec), stop_ids=(), seed=seed)
        return "", ids_to_image(layout, codec, out), out
    if task in ("caption", "ascii"):
        img = load_image(image_path, codec.img_px) if image_path else np.zeros((codec.img_px,) * 2 + (3,), np.uint8)
        l1, l2 = codec.encode(img)
        ids = tasks.ask_vision_seq(bpe, layout, l1, l2, QUESTION[task])
        out = generate_ids(model, ids, max_new=max_new, temp=temp, top_k=top_k, top_p=top_p,
                           seed=seed, min_new=min_new)
        return bpe.decode(out), img, out
    if task == "code":
        ids = [CODE] + bpe.encode(prompt) + [SEP]
        out = generate_ids(model, ids, max_new=max_new, temp=temp, top_k=top_k, top_p=top_p,
                           seed=seed, min_new=min_new)
        return bpe.decode(out), None, out
    ids = [BOS] + bpe.encode(prompt)
    out = generate_ids(model, ids, max_new=max_new, temp=temp, top_k=top_k, top_p=top_p,
                       seed=seed, min_new=min_new)
    return bpe.decode(out), None, out


def preview(model, bpe, layout, codec, caption: str, out_png: Path, temp: float = 0.7):
    _txt, img, _ = run_task(model, bpe, layout, codec, "image",
                            prompt=f"generate the image of {caption}", temp=temp, seed=0)
    save_png(img, out_png, upscale=5)
    return img


def load_bundle(ckpt_path, device: str = "cpu"):
    """Load model + tokenizer + codec from a self-contained checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    cfg = GPTConfig(**{k: v for k, v in ckpt["config"].items() if k in GPTConfig.__annotations__})
    model = GPT(cfg)
    model.load_state_dict(state)
    model.to(device).eval()
    bpe = BPE([tuple(m) for m in ckpt["bpe"]["merges"]])
    L = ckpt["layout"]
    layout = Layout(L["n_text"], l1=L.get("l1", 4096), l2=L.get("l2") or 512,
                    detail=bool(L.get("detail", True)))
    codec = Codec(ckpt["codec"]["img_px"], ckpt["codec"]["patch"], ckpt["codec"]["detail"])
    return model, bpe, layout, codec, ckpt


def main():
    ap = argparse.ArgumentParser(description="image/text/code sampling from a checkpoint")
    ap.add_argument("--ckpt", default="checkpoints/latest.pt")
    ap.add_argument("--task", default="image",
                    choices=["image", "caption", "ascii", "code", "text", "fix",
                             "ascii2code", "img2code", "code2img"])
    ap.add_argument("--prompt", default="generate the image of a white car")
    ap.add_argument("--image", default=None, help="image file for caption/ascii/img2code/ascii2code")
    ap.add_argument("--code", default=None, help="code snippet for --task code2img / fix")
    ap.add_argument("--error", default=None, help="the error message for --task fix")
    ap.add_argument("--code-file", default=None, help="read the snippet from a file (--task fix)")
    ap.add_argument("--out", default=None, help="png output for --task image")
    ap.add_argument("--max-new", type=int, default=420)
    ap.add_argument("--temp", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=64)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()
    if not Path(a.ckpt).exists():
        raise SystemExit(f"checkpoint not found: {a.ckpt}")
    model, bpe, layout, codec, _ = load_bundle(a.ckpt)
    snippet = a.code
    if a.code_file:
        snippet = Path(a.code_file).read_text(encoding="utf-8", errors="replace")
    text, img, _ids = run_task(model, bpe, layout, codec, a.task, a.prompt, a.image,
                               a.max_new, a.temp, a.top_k, a.top_p, a.seed,
                               code_text=snippet, ascii_text=None, err_text=a.error)
    if img is not None:
        out = Path(a.out or f"samples/{a.task}.png")
        out.parent.mkdir(parents=True, exist_ok=True)
        save_png(img, out, upscale=5)
        print(f"[{a.task}] wrote {out}")
    if text:
        print(text if a.task != "ascii" else text)
    print(f"params={model.n_params():,} tokens={len(_ids)}")


if __name__ == "__main__":
    main()
