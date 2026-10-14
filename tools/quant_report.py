"""Size vs quality report for the exported int8/int4 bundles.

For a checkpoint it exports each candidate setting, reloads the dequantised weights and compares
them with the fp32 model on the three things that actually matter here:

  * logits            max |difference| on a real training batch
  * text generation   greedy continuation of a prompt (longest common prefix vs fp32)
  * image generation  how many generated image codes match the fp32 model's own generation

    python tools/quant_report.py checkpoints/latest.pt
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import tasks  # noqa: E402
from generate import generate_ids, image_region, load_bundle, text_region  # noqa: E402
from vocab import BOS, EOS, IMG, SEP  # noqa: E402

TEXT_PROMPT = "write css for a dark page"
IMG_PROMPT = "generate the image of a big red car in front of a pink background"


def greedy(model, prompt, n, **kw):
    return generate_ids(model, prompt, n, temp=1.0, top_k=1, top_p=1.0, rep_penalty=1.0, seed=1, **kw)


def common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/latest.pt"
    variants = [(4, 64, ""), (3, 64, ""), (3, 64, "token_embedding,norm"),
                (2, 64, "token_embedding,norm"), (2, 64, "token_embedding,norm,blocks.0,blocks.1")]
    model, bpe, layout, codec, _ = load_bundle(ckpt)
    model.eval()
    print(f"checkpoint {ckpt}: {model.n_params():,} params, vocab {layout.total}, "
          f"{codec.tokens_per_image()} image tokens/image")

    # fp32 reference: logits on a real batch plus greedy generations
    n_img = min(96, codec.tokens_per_image())
    with torch.no_grad():
        img, cap = __import__("synth").scene(3, seed=31, px=codec.img_px)
        l1, l2 = codec.encode(img)
        ref_seq = tasks.text2img(bpe, layout, tasks.IMAGE_INSTRUCTIONS[0].format(c=cap), l1, l2)
        x = torch.tensor([ref_seq[:-1]])
        ref_logits = model(x)[0][0, -1].clone()
        code_seq = tasks.code_seq(bpe, "write css for a dark page",
                                  "body { color: red; margin: 0 auto; }")
        xc = torch.tensor([code_seq[:-1]])
        ref_code_argmax = model(xc)[0][0].argmax(-1)
        ref_text = greedy(model, [BOS] + bpe.encode(TEXT_PROMPT), 24)
        ref_img = greedy(model, [IMG] + bpe.encode(IMG_PROMPT) + [SEP], n_img,
                         region=image_region(layout, codec), stop_ids=())

    rows = []
    for bits, group, keep in variants:
        tmp = Path(tempfile.mkdtemp(prefix=f"q{bits}g{group}_"))
        pybin = ROOT / ".venv" / "bin" / "python"
        cmd = [str(pybin) if pybin.exists() else sys.executable,
               str(ROOT / "src" / "export_web.py"), ckpt, "-o", str(tmp),
               "--bits", str(bits), "--group", str(group)] + (["--int8-keep", keep] if keep else [])
        out = subprocess.run(cmd, capture_output=True, text=True)
        if out.returncode != 0:
            print(out.stdout, out.stderr)
            raise SystemExit("export failed")
        meta = json.loads((tmp / "model.json").read_text())
        size = sum(f.stat().st_size for f in tmp.iterdir() if f.suffix in (".bin", ".json"))
        blob_mb = (tmp / meta["weight_file"]).stat().st_size / 1048576
        sys.path.insert(0, str(ROOT / "tools"))
        from check_parity import load_export
        qmodel, *_ = load_export(str(tmp))
        qmodel.eval()
        with torch.no_grad():
            ql = qmodel(x)[0][0, -1]
            dlogit = float((ql - ref_logits).abs().max())
        with torch.no_grad():
            ref_argmax = model(x)[0][0].argmax(-1)
            q_argmax = qmodel(x)[0][0].argmax(-1)
            q_code_argmax = qmodel(xc)[0][0].argmax(-1)
            img_argmax = sum(1 for a, b in zip(ref_argmax.tolist(), q_argmax.tolist()) if a == b)
            code_argmax = sum(1 for a, b in zip(ref_code_argmax.tolist(), q_code_argmax.tolist()) if a == b)
            text_agree = 0
            for i, (a, b) in enumerate(zip(ref_code_argmax.tolist(), q_code_argmax.tolist())):
                text_agree += (a == b)
        # code task end to end: greedy generation through the text region, exact + token level
        code_prompt = [BOS] + bpe.encode("write css for a dark page") + [SEP]
        with torch.no_grad():
            ref_code = greedy(model, code_prompt, 40, region=text_region(layout), stop_ids=(EOS,))
            q_code = greedy(qmodel, code_prompt, 40, region=text_region(layout), stop_ids=(EOS,))
        code_tok = sum(1 for a, b in zip(ref_code, q_code) if a == b)
        q_text = greedy(qmodel, [BOS] + bpe.encode(TEXT_PROMPT), 24)
        q_img = greedy(qmodel, [IMG] + bpe.encode(IMG_PROMPT) + [SEP], n_img,
                       region=image_region(layout, codec), stop_ids=())
        rows.append(dict(
            setting=(f"int{bits} g{group}" if group else f"int{bits} per-tensor")
                    + (f" +8:{keep.split(',')[0]}" if keep else ""),
            mb=blob_mb, total_mb=size / 1048576, bytes_per_param=blob_mb * 1048576 / model.n_params(),
            dlogit=dlogit,
            text_prefix=common_prefix(ref_text, q_text), text_len=len(ref_text),
            img_argmax=img_argmax, img_total=len(ref_argmax), code_agree=code_argmax,
            code_total=len(ref_code_argmax),
            img_match=sum(1 for a, b in zip(ref_img, q_img) if a == b), img_len=len(ref_img),
            code_tok=code_tok, code_len=len(ref_code),
            code_exact=int(ref_code == q_code),
        ))
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{'setting':22s} {'weights':>8s} {'B/par':>6s} {'max|dlogit|':>11s} "
          f"{'img argmax':>12s} {'code gen':>10s} {'greedy pfx':>10s} {'img codes':>10s}")
    for r in rows:
        print(f"{r['setting'][:22]:22s} {r['mb']:7.2f}M {r['bytes_per_param']:6.3f} {r['dlogit']:11.4f} "
              f"{r['img_argmax']:>5d}/{r['img_total']:<6d} {r['code_tok']}/{r['code_len']}e{r['code_exact']:<4d} "
              f"{r['text_prefix']:>4d}/{r['text_len']:<5d} {r['img_match']:>5d}/{r['img_len']:<4d}")
    best = [r for r in rows
            if r["img_argmax"] >= 0.98 * r["img_total"] and r["img_match"] >= 0.98 * r["img_len"]
            and r["code_tok"] >= 0.95 * r["code_len"] and r["text_prefix"] == r["text_len"]]
    if best:
        b = min(best, key=lambda r: r["mb"])
        print(f"\nsmallest setting with >=98% image argmax/codes, >=95% code tokens and identical text: "
              f"{b['setting']} ({b['mb']:.2f} MiB weights)")
    else:
        print("\nno candidate met the >95% image-code bar; raise bits or lower the group size")


if __name__ == "__main__":
    main()
