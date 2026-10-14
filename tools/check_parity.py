"""Verify that the JS runtime and the Python trainer agree bit-for-bit on the two shared formats:
image tokens (encode + decode) and BPE ids. Run from the repo root:

    python tools/check_parity.py checkpoints/tokenizer.json
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

import synth
from bpe import BPE
from imagecodec import Codec, save_png


def main():
    tok_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/tokenizer.json"
    if not Path(tok_path).exists():
        raise SystemExit(f"tokenizer not found: {tok_path} (train first, or pass a path)")
    tmp = Path(tempfile.mkdtemp(prefix="parity_"))
    codec = Codec(64, 4, True)
    img, cap = synth.scene(3)
    l1, l2 = codec.encode(img)
    rec = codec.decode(l1, l2)
    save_png(img, tmp / "parity_src.png", upscale=1)
    (tmp / "parity_py.json").write_text(json.dumps(
        {"l1": l1, "l2": l2, "decode": rec.reshape(-1).tolist(), "codec": codec.to_json()}))
    bpe = BPE.load(tok_path)
    text = "generate the image of a big white car on the left on a green background"
    (tmp / "parity_tok.json").write_text(json.dumps({"ids": bpe.encode(text), "text": text}))
    (tmp / "tokenizer.json").write_text(Path(tok_path).read_text())
    # task-format parity: the JS prompt builders must produce byte-identical sequences
    import tasks as T
    from imagecodec import Codec as C2
    from vocab import Layout
    c2 = C2(64, 4, True)
    lay = Layout(bpe.n_text, l1=4096, l2=512, detail=True)
    img2, cap2 = synth.scene(3)
    a1, a2 = c2.encode(img2)
    code_instr, code_txt = synth.synth_code(3)[0], synth.synth_code(3)[2]
    text_in = "a white car on a green background"
    py_tasks = {
        "img2cap": T.img2text(bpe, lay, a1, a2, T.CAPTION_QUESTIONS[0], cap2, "cap"),
        "img2ascii": T.img2text(bpe, lay, a1, a2, T.ASCII_QUESTIONS[0],
                                __import__("ascii_art").to_ascii(img2, width=40), "ascii"),
        "txt2img": T.text2img(bpe, lay, T.IMAGE_INSTRUCTIONS[0].format(c=cap2), a1, a2),
        "code": T.code_seq(bpe, code_instr, code_txt),
        "text": T.lm_seq(bpe, text_in),
    }
    # --- combination tasks (ascii->code, image->code, code->image) + the ascii converter itself
    import ascii_art
    art32 = ascii_art.to_ascii(img2, width=32)
    page_art = synth.art2page(3, art32)
    page_scene = synth.scene2page(3, cap2, img2)
    q1, q2 = c2.encode(synth.page2scene(3, code_txt, 64))
    py_tasks["ascii2code"] = T.ascii2code_seq(bpe, art32, T.ASCII2CODE_INSTRUCTIONS[0], page_art)
    py_tasks["img2code"] = T.img2code_seq(bpe, lay, a1, a2, T.IMG2CODE_INSTRUCTIONS[0], page_scene)
    py_tasks["code2img"] = T.code2img_seq(bpe, lay, code_txt, T.CODE2IMG_INSTRUCTIONS[0], q1, q2)
    _lang, broken, err_msg, fixed = synth.error_case(0)
    fix_question = T.FIX_QUESTIONS[0].format(e=err_msg)
    py_tasks["fix"] = T.fix_seq(bpe, broken, fix_question, fixed)
    (tmp / "parity_tasks.json").write_text(json.dumps({
        "l1": a1, "l2": a2, "layout": lay.to_json(), "caption_question": T.CAPTION_QUESTIONS[0],
        "ascii_question": T.ASCII_QUESTIONS[0], "answer_cap": cap2,
        "answer_ascii": __import__("ascii_art").to_ascii(img2, width=40),
        "image_instruction": T.IMAGE_INSTRUCTIONS[0].format(c=cap2),
        "code_instruction": code_instr, "code": code_txt, "text": text_in,
        "combo_ascii": art32, "combo_code": page_art, "combo_scene_page": page_scene,
        "ascii2code_instruction": T.ASCII2CODE_INSTRUCTIONS[0],
        "img2code_instruction": T.IMG2CODE_INSTRUCTIONS[0],
        "code2img_instruction": T.CODE2IMG_INSTRUCTIONS[0],
        "code2img_l1": q1, "code2img_l2": q2,
        "fix_code": broken, "fix_question": fix_question, "fix_answer": fixed,
        "py": py_tasks}))
    root = Path(__file__).resolve().parents[1]
    model_dir = sys.argv[2] if len(sys.argv) > 2 else None
    if model_dir:
        forward_check(model_dir, tmp, root)
        greedy_fixture(model_dir, tmp)
    cmd = ["node", str(root / "tools" / "parity_js.mjs"), str(tmp)] + ([model_dir] if model_dir else [])
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=root)
    if out.returncode != 0:
        print(out.stdout)
        print(out.stderr)
        raise SystemExit("js parity script failed")
    res = json.loads(out.stdout)
    print(f"reference caption: {cap}")
    ok = True
    for key in ("codecL1Match", "codecL2Match", "tokenizerMatch", "tokenizerRoundTrip"):
        print(f"  {key:20s} {res[key]}")
        ok &= bool(res[key])
    print(f"  decode max pixel diff  {res['codecDecodeMaxDiff']}")
    ok &= res["codecDecodeMaxDiff"] == 0
    print(f"  js l1[:8] {res['jsL1']}  py l1[:8] {res['pyL1']}")
    print(f"  js ids[:12] {res['jsIds']}")
    print(f"  py ids[:12] {res['pyIds']}")
    tf = res.get("taskFormats", {})
    print(f"  task formats          {len(tf.get('checked', []))} checked, mismatched: {tf.get('mismatched')}")
    ok &= not tf.get("mismatched")
    print(f"  ascii converter       js == py: {res['asciiMatch']} ({res['asciiDiffChars']} chars differ)")
    ok &= bool(res["asciiMatch"])
    g = res.get("greedy")
    if g:
        print(f"  greedy text ids       match={g['textMatch']}  js={g['jsText'][:8]} py={g['pyText'][:8]}")
        print(f"  greedy image ids      match={g['imageMatch']} (first {g['imageLen']} tokens of the image region)")
        ok &= bool(g["textMatch"]) and bool(g["imageMatch"])
    if res.get("forward"):
        f = res["forward"]
        print(f"  forward max |dlogit|  {f['maxAbsDiff']:.6f}")
        print(f"  forward argmax        js={f['jsArgmax']} py={f['pyArgmax']} match={f['match']}")
        ok &= f["maxAbsDiff"] < 0.05 and f["match"]
    print("PARITY OK" if ok else "PARITY MISMATCH")
    raise SystemExit(0 if ok else 1)




def dequant_py(buf, m):
    """Mirror of dequant() in web/engine.js and dequant_np in src/export_web.py."""
    if not m["quantized"]:
        return np.frombuffer(buf, dtype=np.float32, count=int(np.prod(m["shape"])),
                             offset=m["offset"]).astype(np.float32)
    if m.get("bits", 8) == 8:
        q = np.frombuffer(buf, dtype=np.int8, count=m["length"], offset=m["offset"]).astype(np.float32)
        return q * m["scale"]
    packed = np.frombuffer(buf, dtype=np.uint8, count=m["data_length"], offset=m["offset"])
    lo = (packed & 0x0F).astype(np.int16)
    hi = ((packed >> 4) & 0x0F).astype(np.int16)
    q = np.empty(packed.size * 2, dtype=np.int16)
    q[0::2], q[1::2] = lo, hi
    dt = np.float16 if m["scale_dtype"] == "fp16" else np.float32
    sc = np.frombuffer(buf, dtype=dt, count=m["groups"], offset=m["scale_offset"]).astype(np.float32)
    out = (q - 8).astype(np.float32) * np.repeat(sc, m["group_size"])[:q.size]
    return out[:int(np.prod(m["shape"]))]


def load_export(model_dir: str):
    """Rebuild the int8-dequantised PyTorch model, tokenizer and codec from an export bundle."""
    import torch
    from bpe import BPE
    from imagecodec import Codec
    from model import GPT, GPTConfig
    from vocab import Layout
    meta = json.loads((Path(model_dir) / "model.json").read_text())
    buf = (Path(model_dir) / meta.get("weight_file", "weights.int8.bin")).read_bytes()
    state = {}
    for name, m in meta["tensors"].items():
        arr = dequant_py(buf, m)
        state[name] = torch.from_numpy(arr.reshape(m["shape"]).copy())
    cfg = GPTConfig(**{k: v for k, v in meta["architecture"].items() if k in GPTConfig.__annotations__})
    model = GPT(cfg).eval()
    model.load_state_dict(state)
    tok = json.loads((Path(model_dir) / "tokenizer.json").read_text())
    bpe = BPE([tuple(m) for m in tok["merges"]])
    L = meta["layout"]
    layout = Layout(L["n_text"], l1=L["l1"], l2=L["l2"] or 512, detail=bool(L["detail"]))
    codec = Codec(meta["codec"]["img_px"], meta["codec"]["patch"], meta["codec"]["detail"])
    return model, bpe, layout, codec, meta


def greedy_fixture(model_dir: str, tmp: Path):
    """Greedy generation must pick the same tokens in Python and in the JS runtime."""
    from generate import generate_ids, image_region
    from vocab import BOS, IMG, SEP
    model, bpe, layout, codec, _ = load_export(model_dir)
    text_prompt = [BOS] + bpe.encode("a white car on a green background")
    text_ids = generate_ids(model, text_prompt, 16, temp=1.0, top_k=1, top_p=1.0, rep_penalty=1.0, seed=1)
    image_prompt = [IMG] + bpe.encode("generate the image of a white car") + [SEP]
    n_img = min(64, codec.n1 + codec.n2)
    image_ids = generate_ids(model, image_prompt, n_img, temp=1.0, top_k=1, top_p=1.0, rep_penalty=1.0,
                             region=image_region(layout, codec), stop_ids=(), seed=1)
    (tmp / "parity_greedy.json").write_text(json.dumps({
        "text_prompt": text_prompt, "text_len": 16, "text_ids": text_ids,
        "image_prompt": image_prompt, "image_len": n_img, "image_ids": image_ids}))
    return text_ids, image_ids


def forward_check(model_dir: str, tmp: Path, root: Path):
    """Compare JS transformer logits against the int8-dequantised PyTorch model."""
    import torch
    model, bpe, layout, codec, meta = load_export(model_dir)
    layout = meta["layout"]
    ids = [layout["specials"].index("<bos>")] + list(range(20, 30)) + [
        layout["img1_base"] + 1234, layout["img1_base"] + 4000,
        layout["text_base"] + 5, layout["specials"].index("<sep>")]
    if layout.get("l2"):
        ids.append(layout["img2_base"] + 77)
    with torch.no_grad():
        logits, _, _ = model(torch.tensor([ids], dtype=torch.long))
    (tmp / "parity_fwd.json").write_text(json.dumps({"ids": ids, "logits": logits[0, -1].tolist()}))
    return ids


if __name__ == "__main__":
    main()
