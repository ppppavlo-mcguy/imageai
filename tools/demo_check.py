"""Does the shipped demo really turn a caption into the right picture?

Not a vibes check: for every scene in the demo's training set it generates the image codes from the
caption alone (greedy, exactly like the website) and compares them with the codes the trainer used
as the target. Also reports the caption/ascii round trip on the same pictures.

    python tools/demo_check.py checkpoints_demo2/latest.pt [--synth-n 8] [--seed 1337]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import synth  # noqa: E402
import tasks  # noqa: E402
from generate import generate_ids, image_region, load_bundle, text_region  # noqa: E402
from vocab import BOS, EOS, IMG, SEP  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", nargs="?", default="checkpoints_demo2/latest.pt")
    ap.add_argument("--synth-n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--png-dir", default=None,
                    help="also write target vs generated PNGs here (samples/demo by default)")
    a = ap.parse_args()

    model, bpe, layout, codec, _ = load_bundle(a.checkpoint)
    model.eval()
    print(f"{a.checkpoint}: {model.n_params():,} params, {codec.tokens_per_image()} codes/image, "
          f"{a.synth_n} scenes, temp {a.temp}")

    png_dir = Path(a.png_dir) if a.png_dir else (ROOT / "samples" / "demo")
    png_dir.mkdir(parents=True, exist_ok=True)
    img_ok = img_all = text_ok = 0
    for i in range(a.synth_n):
        img, cap = synth.scene(i, a.seed, codec.img_px)
        l1, l2 = codec.encode(img)
        prompt = [IMG] + bpe.encode(f"generate the image of {cap}") + [SEP]
        got = generate_ids(model, prompt, codec.n1 + codec.n2, temp=a.temp, top_k=1, top_p=1.0,
                           rep_penalty=1.0, seed=i, region=image_region(layout, codec), stop_ids=())
        # generate_ids yields *token ids*; the codec yields raw codes, so add the block base
        want = [layout.img1_base + int(c) for c in l1]
        if layout.l2:
            want += [layout.img2_base + int(c) for c in l2]
        # rebuild a picture from the generated tokens exactly like web/tasks.js does
        codes = [t - (layout.img2_base if layout.l2 and t >= layout.img2_base else layout.img1_base)
                 for t in got]
        gen_l1 = codes[:codec.n1]
        gen_l2 = codes[codec.n1:] or [0] * codec.n2
        gen_img = codec.decode(gen_l1, gen_l2)
        from imagecodec import save_png  # noqa: PLC0415
        save_png(img, png_dir / f"scene{i}_target.png")
        save_png(gen_img, png_dir / f"scene{i}_generated.png")
        match = sum(1 for x, y in zip(got, want) if x == y)
        img_ok += match
        img_all += len(want)
        # caption task: show the image back to the model, ask what it is (greedy, 24 tokens)
        vision = tasks.ask_vision_seq(bpe, layout, l1, l2, "what is on this picture?")
        answer = bpe.decode(generate_ids(
            model, vision, 24, temp=0.0 if a.temp == 0 else a.temp, top_k=1, top_p=1.0,
            rep_penalty=1.05, seed=i, region=text_region(layout), min_new=2, stop_ids=(EOS,)))
        good = sum(1 for w in cap.split() if w in answer.split())
        text_ok += good
        print(f"  [{i}] image codes {match}/{len(want)} ({100 * match / len(want):5.1f}%)   "
              f"caption '{cap}' -> {answer.strip()!r} ({good}/{len(cap.split())} words)")
    print(f"\ncaption -> image: {100 * img_ok / max(1, img_all):.1f}% of image codes are exactly right")
    print(f"image -> caption: {text_ok} of the caption words reproduced")


if __name__ == "__main__":
    main()
