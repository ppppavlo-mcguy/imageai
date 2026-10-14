"""Quantise a checkpoint to int8 and emit the browser bundle.

    python src/export_web.py checkpoints/latest.pt -o web_model

Writes  web_model/model.json  (architecture + layout + codec + int8 tensor index)
        web_model/weights.int8.bin
        web_model/tokenizer.json
        web_model/imagecodec.json
The bit format matches web/engine.js exactly; see README for the Render/HelioHost deploy step.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

from vocab import SPECIALS

DEFAULTS = dict(vocab_size=16000, embed_size=160, num_heads=16, num_layers=18,
                context_length=128, dropout=0.0, expansion=6)


def get_ckpt(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        state = ckpt["model"]
    elif isinstance(ckpt, dict):
        state = ckpt
    else:
        raise RuntimeError("Checkpoint is not a state dictionary.")
    cfg = dict(DEFAULTS)
    if isinstance(ckpt, dict) and isinstance(ckpt.get("config"), dict):
        cfg.update(ckpt["config"])
    return state, cfg, (ckpt if isinstance(ckpt, dict) else {})


def infer_cfg(state, cfg):
    for k, v in state.items():
        if getattr(v, "ndim", 0) == 2 and ("token_embedding" in k or "token_embeddings" in k):
            cfg["vocab_size"], cfg["embed_size"] = int(v.shape[0]), int(v.shape[1])
            break
    ids = set()
    for k in state:
        p = k.replace("\\", "/").split(".")
        for i, x in enumerate(p[:-1]):
            if x in ("blocks", "layers", "transformer_blocks") and p[i + 1].isdigit():
                ids.add(int(p[i + 1]))
    if ids:
        cfg["num_layers"] = max(ids) + 1
    for k in ("vocab_size", "embed_size", "num_heads", "num_layers", "context_length", "expansion"):
        cfg[k] = int(cfg[k])
    if cfg["embed_size"] % cfg["num_heads"]:
        raise RuntimeError(f"embed_size={cfg['embed_size']} is not divisible by num_heads={cfg['num_heads']}")
    return cfg


def _fp16_bytes(x) -> bytes:
    import numpy as np
    return np.asarray(x, dtype=np.float16).tobytes()


def quantize(t, bits: int = 8, group: int = 64, scale_dtype: str = "fp16"):
    """Quantise one tensor. Returns (payload_bytes, metadata).

    bits=8  : symmetric per-tensor int8 (1 byte/parameter).
    bits=4  : symmetric int4 with one fp16 scale per `group` values (~0.53 bytes/parameter).
    Integer tensors are stored raw. The metadata tells web/engine.js how to unpack it.
    """
    t = t.detach().cpu().contiguous()
    n = int(t.numel())
    if not torch.is_floating_point(t):
        raw = t.numpy().tobytes()
        return raw, [{"dtype": str(t.dtype).replace("torch.", ""), "quantized": False, "bits": 8,
                      "shape": list(t.shape), "length": len(raw)}]
    x = t.float().reshape(-1)
    if bits == 8:
        m = float(x.abs().max()) if n else 0.0
        scale = 1.0 if m == 0 else m / 127.0
        q = (torch.zeros_like(x, dtype=torch.int8) if m == 0
             else torch.clamp(torch.round(x / scale), -127, 127).to(torch.int8))
        raw = q.numpy().tobytes()
        return raw, [{"dtype": "int8", "quantized": True, "bits": 8, "scale": scale,
                      "shape": list(t.shape), "length": len(raw)}]
    if bits != 4:
        raise SystemExit("only --bits 4 or --bits 8 are supported")
    pad = (-n) % group
    xp = torch.cat([x, torch.zeros(pad)]) if pad else x
    g = xp.reshape(-1, group)
    scales = g.abs().amax(dim=1) / 7.0
    scales = torch.where(scales == 0, torch.ones_like(scales), scales)
    q = torch.clamp(torch.round(g / scales[:, None]), -8, 7).to(torch.int8) + 8      # 0..15
    q = q.reshape(-1).to(torch.uint8)
    if q.numel() % 2:
        q = torch.cat([q, torch.zeros(1, dtype=torch.uint8)])
    packed = (q[0::2] | (q[1::2] << 4)).to(torch.uint8).numpy().tobytes()
    if scale_dtype == "fp16":
        sc = _fp16_bytes(scales.numpy())
    else:
        sc = scales.numpy().astype("float32").tobytes()
    return packed + sc, [{"dtype": "int4", "quantized": True, "bits": 4, "group_size": group,
                          "scale_dtype": scale_dtype, "groups": int(scales.numel()),
                          "padded": int(pad), "shape": list(t.shape),
                          "length": len(packed), "scale_length": len(sc)}]


def dequant_np(buf, m, np):
    """Reference dequantisation (mirrors dequant() in web/engine.js)."""
    if not m["quantized"]:
        return np.frombuffer(buf, dtype=np.float32, count=np.prod(m["shape"]),
                             offset=m["offset"]).astype(np.float32)
    if m.get("bits", 8) == 8:
        q = np.frombuffer(buf, dtype=np.int8, count=m["length"], offset=m["offset"]).astype(np.float32)
        return q * m["scale"]
    packed = np.frombuffer(buf, dtype=np.uint8, count=m["data_length"], offset=m["offset"])
    lo = (packed & 0x0F).astype(np.int16)
    hi = ((packed >> 4) & 0x0F).astype(np.int16)
    q = np.empty(packed.size * 2, dtype=np.int16)
    q[0::2], q[1::2] = lo, hi
    q = (q - 8).astype(np.float32)
    dt = np.float16 if m["scale_dtype"] == "fp16" else np.float32
    sc = np.frombuffer(buf, dtype=dt, count=m["groups"], offset=m["scale_offset"]).astype(np.float32)
    out = q * np.repeat(sc, m["group_size"])[:q.size]
    n = int(np.prod(m["shape"]))
    return out[:n]


def main():
    ap = argparse.ArgumentParser(description="export a checkpoint as an int8 browser bundle")
    ap.add_argument("checkpoint", nargs="?", default="checkpoints/latest.pt")
    ap.add_argument("-o", "--output", default="web_model")
    ap.add_argument("--tokenizer", default="checkpoints/tokenizer.json")
    ap.add_argument("--bits", type=int, default=8, choices=[8, 4],
                    help="8 = per-tensor int8 (1.0 B/param), 4 = group-scaled int4 (~0.53 B/param)")
    ap.add_argument("--group", type=int, default=64, help="values per int4 scale (32/64/128)")
    ap.add_argument("--scale-dtype", default="fp16", choices=["fp16", "fp32"])
    ap.add_argument("--int8-keep", default="token_embedding,norm",
                    help="comma-separated name fragments that stay int8 when --bits 4 "
                         "(the tied embedding doubles as the output head, so it is the most "
                         "quantisation-sensitive tensor)")
    ap.add_argument("--no-gzip", action="store_true",
                    help="skip writing <weights>.gz (the runtime downloads it and gunzips in "
                         "the browser - ~12%% smaller with bit-identical weights)")
    ap.add_argument("--verify", action="store_true", help="reload the export and check the maths")
    a = ap.parse_args()
    src = Path(a.checkpoint)
    out = Path(a.output)
    if not src.exists():
        raise SystemExit(f"Checkpoint not found: {src}")
    out.mkdir(parents=True, exist_ok=True)
    state, cfg, raw_ckpt = get_ckpt(src)
    cfg = infer_cfg(state, cfg)
    keep = [k.strip() for k in (a.int8_keep or "").split(",")]
    fname = "weights.int4.bin" if a.bits == 4 else "weights.int8.bin"
    tensors, offset, params = {}, 0, 0
    with (out / fname).open("wb") as f:
        for name, t in state.items():
            if not hasattr(t, "shape") or not hasattr(t, "dtype"):
                continue
            bits = 8 if any(k and k in name for k in keep) else a.bits
            raw, metas = quantize(t, bits, a.group, a.scale_dtype)
            f.write(raw)
            part = metas[0]
            part["offset"] = offset
            if part["quantized"] and part["bits"] == 4:
                part["data_length"] = part["length"]
                part["scale_offset"] = offset + part["length"]
            tensors[name.replace("\\", "/")] = part
            offset += len(raw)
            params += t.numel()
    for stale in (out / ("weights.int8.bin" if a.bits == 4 else "weights.int4.bin"),):
        if stale.exists():
            stale.unlink()
    tied = not any(k.endswith("head.weight") for k in tensors)
    layout = raw_ckpt.get("layout") or {
        "specials": SPECIALS, "n_text": cfg["vocab_size"] - 4096 - 512 - len(SPECIALS),
        "text_base": len(SPECIALS), "l1": 4096, "img1_base": len(SPECIALS) + cfg["vocab_size"] - 4608,
        "l2": 512, "img2_base": len(SPECIALS) + cfg["vocab_size"] - 512, "detail": True,
        "total": cfg["vocab_size"],
    }
    codec = raw_ckpt.get("codec") or {"img_px": 64, "patch": 4, "sub": 2, "grid": 16, "detail": True,
                                      "n1": 256, "n2": 1024, "l1_levels": 15, "l1_bits": 4,
                                      "l2_min": -3, "l2_max": 3, "det_step": 15.3}
    mixed = a.bits == 4 and any(k and k in name for name in state for k in keep)
    quant_name = ("symmetric-per-tensor-int8" if a.bits == 8
                  else (f"mixed-int4-group{a.group}({'+'.join(k for k in keep if k)}@int8)-"
                        f"{a.scale_dtype}scale" if mixed
                        else f"symmetric-group{a.group}-int4-{a.scale_dtype}scale"))
    # Which tasks this checkpoint was actually trained for, so the UI can say so (and so a model
    # trained with --mix code is not asked to draw pictures it has never seen).
    weight_tasks = {"w_txt2img": "image", "w_img2txt": "caption", "w_ascii": "ascii",
                    "w_code": "code", "w_code_lm": "code_lm", "w_text_lm": "text",
                    "w_ascii2code": "ascii2code", "w_img2code": "img2code",
                    "w_code2img": "code2img"}
    ck_args = raw_ckpt.get("args") or {}
    trained_tasks = [name for key, name in weight_tasks.items() if float(ck_args.get(key, 0) or 0) > 0]
    meta = {"format": "local-gpt-int8-v1" if a.bits == 8 else "local-gpt-int4-v1", "version": 1,
            "tasks": trained_tasks,
            "quantization": quant_name, "bits": a.bits, "group_size": a.group,
            "scale_dtype": a.scale_dtype,
            "architecture": cfg, "tied_embeddings": bool(tied), "layout": layout, "codec": codec,
            "parameter_count": params, "weight_file": fname, "weight_bytes": offset,
            "checkpoint_step": raw_ckpt.get("step"), "tensors": tensors}
    if not a.no_gzip:
        import gzip as _gzip
        raw_bytes = (out / fname).read_bytes()
        with _gzip.open(out / f"{fname}.gz", "wb", compresslevel=9) as g:
            g.write(raw_bytes)
        gz_path = out / f"{fname}.gz"
        meta["weight_file_gz"] = gz_path.name
        meta["weight_bytes_gz"] = gz_path.stat().st_size
    (out / "model.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out / "imagecodec.json").write_text(json.dumps(codec, indent=2), encoding="utf-8")

    tok_src = Path(a.tokenizer)
    if raw_ckpt.get("bpe"):
        (out / "tokenizer.json").write_text(json.dumps(raw_ckpt["bpe"]), encoding="utf-8")
        tok_note = "from checkpoint"
    elif tok_src.exists():
        shutil.copy2(tok_src, out / "tokenizer.json")
        tok_note = str(tok_src)
    else:
        tok_note = None
        print(f"WARNING: tokenizer not found: {tok_src}")

    print(json.dumps({k: cfg[k] for k in ("vocab_size", "embed_size", "num_heads", "num_layers",
                                          "context_length", "expansion")}, indent=2))
    print(f"tokenizer: {tok_note}")
    print(f"layout: text {layout['n_text']}, img1 base {layout['img1_base']} x{layout['l1']}, "
          f"img2 base {layout['img2_base']} x{layout['l2']}, detail={layout['detail']}, tied={tied}")
    print(f"codec: {codec['img_px']}px, patch {codec['patch']}, {codec['n1']}+{codec['n2']} tokens/image")
    print(f"Output: {out.resolve()}")
    print(f"weights: {fname} ({quant_name}) = {offset/1048576:.2f} MiB "
          f"({offset/max(1, params):.3f} bytes/parameter)")
    if meta.get("weight_bytes_gz"):
        gz = meta["weight_bytes_gz"]
        print(f"gzip: {meta['weight_file_gz']} = {gz/1048576:.2f} MiB "
              f"({100 * (1 - gz / max(1, offset)):.0f}% smaller, decompressed in the browser)")
    print(f"Parameters: {params:,}")
    print(f"Run: node web/server.js   (then open http://localhost:3000 and pick the {out} folder)")

    if a.verify:
        import numpy as np
        buf = (out / fname).read_bytes()
        worst = 0.0
        for name, m in tensors.items():
            q = dequant_np(buf, m, np)
            ref = state[name].float().numpy().reshape(-1)
            den = max(float(np.abs(ref).max()), 1e-9)
            worst = max(worst, float(np.abs(q - ref).max()) / den)
        print(f"verify: max relative error {worst*100:.3f}% over {len(tensors)} tensors (vs fp32)")


if __name__ == "__main__":
    main()
