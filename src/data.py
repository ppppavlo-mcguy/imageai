"""Corpora + epoch plans.

The `datasets` library is used for the big real-image sets and for scraped web code; a procedural
corpus (src/synth.py) plus data/seed_code are the offline default so training always works.
An epoch plan is a deterministic list of sample descriptors, so resuming = skipping N batches.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np

import tasks
import synth

# How many distinct (broken, error, fixed) tuples the procedural error corpus cycles through.
FIX_CASES = 96

IMAGE_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif")
# .py is included so `--code-dataset` and data/seed_code can teach Python alongside web code.
CODE_EXT = (".html", ".htm", ".css", ".js", ".mjs", ".py")


# --------------------------------------------------------------------- sources
class SynthImages:
    kind = "synth"

    def __init__(self, n: int, seed: int, px: int, offset: int = 0):
        self.n, self.seed, self.px, self.offset = n, seed, px, offset

    def __len__(self):
        return self.n

    def get(self, i: int):
        return synth.scene(self.offset + i, self.seed, self.px)


class FolderImages:
    kind = "folder"

    def __init__(self, folder: str, px: int):
        from imagecodec import load_image
        self._load = load_image
        self.px = px
        root = Path(folder)
        if not root.exists():
            raise SystemExit(f"image folder not found: {root}")
        self.files = [p for p in sorted(root.rglob("*")) if p.suffix.lower() in IMAGE_EXT]
        if not self.files:
            raise SystemExit(f"no images ({', '.join(IMAGE_EXT)}) in {root}")

    def __len__(self):
        return len(self.files)

    def get(self, i: int):
        p = self.files[i % len(self.files)]
        cap = ""
        txt = p.with_suffix(".txt")
        if txt.exists():
            cap = txt.read_text(encoding="utf-8", errors="replace").strip().splitlines()[0]
        if not cap:
            cap = "a photo of a " + p.stem.replace("_", " ").replace("-", " ").lower()
        return self._load(p, self.px), cap


HF_PRESETS = {
    # name: (dataset, config, split, image col, caption col, caption template)
    # `mnist` is the guaranteed-tiny one (verified: parquet conversion, ~11 MB, real images).
    # Any other Hub dataset works through the explicit --hf-* flags; the identifier has to resolve
    # for your installed `datasets` version.
    "mnist": ("ylecun/mnist", None, "train", "image", "label", "a handwritten digit"),
    "flickr8k": ("nlphuji/flickr8k", None, "train", "image", "caption_0", "{c}"),
    "cats_vs_dogs": ("huggingface/cats_vs_dogs", None, "train", "image", "labels",
                     "a photo of a {label}"),
    "imagenette": ("frgfm/imagenette", "320px", "train", "image", "label",
                   "a photo of a {label}"),
}


class HFDatasetImages:
    kind = "hf"

    def __init__(self, name: str, px: int, n: int, config=None, split=None,
                 image_col=None, caption_col=None, caption_template=None):
        import datasets
        from PIL import Image
        preset = HF_PRESETS.get(name, (name, None, "train", "image", "caption_0", "{c}"))
        ds_name, cfg, sp, icol, ccol, ctpl = preset
        cfg = config or cfg
        split = split or sp
        image_col = image_col or icol
        caption_col = caption_col or ccol
        caption_template = caption_template or ctpl
        kwargs = {"split": split, "streaming": True}
        if cfg:
            kwargs["name"] = cfg
        try:
            ds = datasets.load_dataset(ds_name, **kwargs)
        except Exception as e:
            raise SystemExit(f"could not load {ds_name}: {e}\n"
                             "Tip: any image dataset works via --hf-dataset NAME --hf-split SPLIT "
                             "--hf-image-col IMG --hf-caption-col LABEL (add --hf-caption-template "
                             "'a photo of a {c}'); the offline synth corpus needs no download.")
        label_names = None
        for key in (caption_col, "label", "labels"):
            try:
                names = getattr(ds.features.get(key), "names", None)
            except Exception:
                names = None
            if names:
                label_names = names
                break
        self.images, self.captions = [], []
        for row in ds:
            try:
                im = row[image_col]
                if not hasattr(im, "convert"):
                    im = Image.fromarray(np.asarray(im))
                im = im.convert("RGB").resize((px, px), Image.BICUBIC)
                cap = ""
                if caption_col in row and row[caption_col] is not None:
                    v = row[caption_col]
                    if isinstance(v, (list, tuple)):
                        v = v[0]
                    if isinstance(v, dict):
                        v = v.get("raw", v.get("sentences", ""))
                    cap = str(v).strip()
                if caption_col in ("labels", "label") or not cap:
                    raw = row.get("label", row.get("labels", "object"))
                    if isinstance(raw, int) and label_names:
                        cap = "a photo of a " + str(label_names[raw]).replace("_", " ")
                    else:
                        cap = "a photo of a " + str(raw).lower()
                cap = caption_template.format(c=cap, label=str(row.get("label", "object")).lower())
                cap = " ".join(cap.split())[:160]
            except Exception:
                continue
            self.images.append(np.asarray(im, dtype=np.uint8))
            self.captions.append(cap)
            if len(self.images) >= n:
                break
        if not self.images:
            raise SystemExit(f"no usable rows pulled from {ds_name}")
        self.px = px

    def __len__(self):
        return len(self.images)

    def get(self, i: int):
        i %= len(self.images)
        return self.images[i], self.captions[i]


class SeedCode:
    kind = "seed_code"

    def __init__(self, folder: str = "data/seed_code"):
        root = Path(folder)
        self.items = []
        if root.exists():
            for p in sorted(root.rglob("*")):
                if p.suffix.lower() not in CODE_EXT:
                    continue
                text = p.read_text(encoding="utf-8", errors="replace").strip()
                if not text:
                    continue
                instr = None
                first = text.splitlines()[0].strip()
                bare = first
                for marker in ("<!--", "//", "#", "/*"):   # one instruction header, four comment styles
                    if bare.lower().startswith(marker):
                        bare = bare[len(marker):].strip()
                        break
                if bare.lower().startswith("instruction:"):
                    instr = bare.split(":", 1)[1].strip().rstrip("->").rstrip("*/").strip()
                    text = "\n".join(text.splitlines()[1:]).strip()
                if instr is None:
                    instr = f"write {p.suffix[1:]} for a small web page"
                self.items.append((instr, text))
        if not self.items:
            self.items = [(i, c) for i, _, c in (synth.synth_code(0), synth.synth_code(1), synth.synth_code(2))]

    def __len__(self):
        return len(self.items)

    def get(self, i: int):
        return self.items[i % len(self.items)]


class SynthCode:
    kind = "synth_code"

    def __init__(self, n: int, seed: int):
        self.n, self.seed = n, seed

    def __len__(self):
        return self.n

    def get(self, i: int):
        instr, _fname, code = synth.synth_code(self.seed + i * 13)
        return instr, code


class HFCode:
    kind = "hf_code"

    def __init__(self, dataset: str, n: int, exts=CODE_EXT, split="train"):
        import datasets
        ds = datasets.load_dataset(dataset, split=split, streaming=True)
        self.items = []
        for row in ds:
            path = str(row.get("path", ""))
            if not path.endswith(exts):
                continue
            code = row.get("content") or row.get("code") or ""
            code = code.strip()
            if len(code) < 80 or len(code) > 1200 or "\x00" in code:
                continue
            lang = path.rsplit(".", 1)[-1]
            self.items.append((f"write a small {lang} snippet like the one in {path}", code))
            if len(self.items) >= n:
                break
        if not self.items:
            raise SystemExit(f"no usable code rows pulled from {dataset}")

    def __len__(self):
        return len(self.items)

    def get(self, i: int):
        return self.items[i % len(self.items)]


class Subset:
    """Held-out view over another image source (validation set)."""

    def __init__(self, src, indices, tag="val"):
        self.src, self.indices = src, list(indices)
        self.kind = f"{src.kind}:{tag}"

    def __len__(self):
        return len(self.indices)

    def get(self, i: int):
        return self.src.get(self.indices[i % len(self.indices)])


# ------------------------------------------------------------------ plan + batch
class TrainData:
    """Corpora, deterministic epoch plans and batches.

    Training and validation always come from the same distribution: for your own folders and for
    HuggingFace datasets the last ~5% of the images (and code files) are held out, and for the
    procedural corpus validation uses a disjoint index range (offset 1e6).
    """

    VAL_FRACTION = 0.05

    def __init__(self, args, codec, bpe, layout=None):
        self.args, self.codec, self.bpe = args, codec, bpe
        self.layout = layout
        self.skipped = 0
        self.skipped_kind = None
        self.skipped_len = 0
        self.px = codec.img_px
        self.img_tokens = codec.tokens_per_image()

        full = self._image_source(args)
        self.img_full = full
        n = len(full)
        holdout = int(n * self.VAL_FRACTION) if full.kind not in ("synth",) and n >= 40 else 0
        if holdout:
            self.img = Subset(full, range(n - holdout))
            self.val_img = Subset(full, range(n - holdout, n))
        else:
            self.img = full
            self.val_img = SynthImages(max(8, args.val_samples), args.seed, codec.img_px, offset=10 ** 6)

        items = self._code_items(args)
        code_hold = max(2, int(len(items) * self.VAL_FRACTION)) if len(items) >= 40 else 0
        self.code_items = items[: len(items) - code_hold] if code_hold else items
        self.val_code = items[len(items) - code_hold:] if code_hold else items[: max(1, min(8, len(items)))]

    # -------------------------------------------------------------- sources
    @staticmethod
    def _image_source(args):
        if args.images:
            return FolderImages(args.images, args.img_px)
        if args.hf_dataset:
            print(f"[data] pulling up to {args.hf_samples} images from {args.hf_dataset} ...")
            src = HFDatasetImages(args.hf_dataset, args.img_px, args.hf_samples, args.hf_config,
                                  args.hf_split, args.hf_image_col, args.hf_caption_col,
                                  args.hf_caption_template)
            print(f"[data] held {len(src)} real images at {args.img_px}x{args.img_px}")
            return src
        return SynthImages(args.synth_n, args.seed, args.img_px)

    @staticmethod
    def _code_items(args):
        items = [p.get(i) for p in [SynthCode(args.synth_code_n, args.seed)] for i in range(args.synth_code_n)]
        seed = SeedCode(args.seed_code_dir)
        items += [seed.get(i) for i in range(len(seed))]
        if args.code_dataset:
            print(f"[data] pulling up to {args.code_samples} code files from {args.code_dataset} ...")
            hf = HFCode(args.code_dataset, args.code_samples)
            items += [hf.get(i) for i in range(len(hf))]
        return items

    def code_at(self, i: int):
        return self.code_items[i % len(self.code_items)]

    def code_len(self):
        return len(self.code_items)

    # ------------------------------------------------------------------ plan
    def _weights(self) -> dict:
        a = self.args
        return {
            "img2cap": a.w_img2txt, "img2ascii": a.w_ascii, "txt2img": a.w_txt2img,
            "code": a.w_code, "code_lm": a.w_code_lm, "text_lm": a.w_text_lm,
            "ascii2code": a.w_ascii2code, "img2code": a.w_img2code, "code2img": a.w_code2img,
            "fix": a.w_fix,
        }

    def _sample_descs(self, rng, n: int, ni: int, nc: int) -> list:
        """Sample n (kind, index, variant) descriptors for the configured task mix."""
        w = {k: v for k, v in self._weights().items() if v > 0}
        kinds, weights = list(w), list(w.values())
        out = []
        for _ in range(n):
            k = rng.choices(kinds, weights)[0]
            if k in ("img2cap", "img2ascii"):
                qs = tasks.ASCII_QUESTIONS if k == "img2ascii" else tasks.CAPTION_QUESTIONS
                out.append((k, rng.randrange(ni), rng.randrange(len(qs))))
            elif k in ("txt2img", "ascii2code", "img2code"):
                pools = {"txt2img": tasks.IMAGE_INSTRUCTIONS, "ascii2code": tasks.ASCII2CODE_INSTRUCTIONS,
                         "img2code": tasks.IMG2CODE_INSTRUCTIONS}[k]
                out.append((k, rng.randrange(ni), rng.randrange(len(pools))))
            elif k == "code":
                out.append((k, rng.randrange(nc), rng.randrange(len(tasks.CODE_INSTRUCTIONS))))
            elif k == "code2img":
                out.append((k, rng.randrange(nc), rng.randrange(len(tasks.CODE2IMG_INSTRUCTIONS))))
            elif k == "code_lm":
                out.append((k, rng.randrange(nc), 0))
            elif k == "fix":
                out.append((k, rng.randrange(FIX_CASES), rng.randrange(len(tasks.FIX_QUESTIONS))))
            else:
                out.append((k, rng.randrange(ni), 0))
        return out

    def _bucket(self, descs: list, rng, window: int) -> list:
        """Length bucketing: sort by rough size, shuffle inside small groups -> less padding."""
        est = [self._estimate(d) for d in descs]
        order = sorted(range(len(descs)), key=lambda i: est[i])
        win = max(2, window)
        for s in range(0, len(order), win):
            part = order[s:s + win]
            rng.shuffle(part)
            order[s:s + win] = part
        return [descs[i] for i in order]

    def plan_chunk(self, epoch: int, chunk: int, n: int) -> list:
        """One deterministic chunk of an epoch plan; chunk index lets a time-boxed epoch keep
        pulling fresh samples without repeating or breaking resume determinism."""
        a = self.args
        rng = random.Random(a.seed * 7919 + epoch * 1009 + chunk)
        descs = self._sample_descs(rng, n, len(self.img), self.code_len())
        return self._bucket(descs, rng, a.bucket_window)

    def plan(self, epoch: int, n: int, val: bool = False) -> list:
        a = self.args
        rng = random.Random(a.seed * 7919 + epoch)
        ni = len(self.val_img if val else self.img)
        nc = len(self.val_code) if val else self.code_len()
        return self._bucket(self._sample_descs(rng, n, ni, nc), rng, a.bucket_window)

    def _estimate(self, desc) -> int:
        k = desc[0]
        if k == "img2ascii":
            return self.img_tokens + 700
        if k == "img2cap":
            return self.img_tokens + 40
        if k == "ascii2code":
            return 420 + 420                       # ascii art + the page written from it
        if k == "img2code":
            return self.img_tokens + 420
        if k == "code2img":
            return 420 + self.img_tokens
        if k == "fix":
            return 320
        if k == "txt2img":
            return self.img_tokens + 30
        if k == "code":
            return 160
        if k == "code_lm":
            return 200
        return 24

    # -------------------------------------------------------------- sequences
    def seq(self, desc, val: bool = False):
        import ascii_art
        k, i, j = desc
        codec, bpe = self.codec, self.bpe
        img_src = self.val_img if val else self.img
        if k in ("img2cap", "img2ascii", "txt2img", "text_lm", "ascii2code", "img2code"):
            img, cap = img_src.get(i)
        else:
            img = cap = None
        if k == "img2cap":
            l1, l2 = codec.encode(img)
            return tasks.img2text(bpe, self.layout, l1, l2, tasks.CAPTION_QUESTIONS[j], cap, "cap")
        if k == "img2ascii":
            l1, l2 = codec.encode(img)
            art = ascii_art.to_ascii(img, width=self.args.ascii_width)
            return tasks.img2text(bpe, self.layout, l1, l2, tasks.ASCII_QUESTIONS[j], art, "ascii")
        if k == "txt2img":
            l1, l2 = codec.encode(img)
            instr = tasks.IMAGE_INSTRUCTIONS[j].format(c=cap)
            return tasks.text2img(bpe, self.layout, instr, l1, l2)
        if k in ("code", "code2img"):
            instr, code = (self.val_code[i % len(self.val_code)] if val else self.code_at(i))
            if k == "code":
                return tasks.code_seq(bpe, tasks.CODE_INSTRUCTIONS[j].format(i=instr), code)
            page = synth.page2scene(i, code, codec.img_px)        # -> "what this page looks like"
            l1, l2 = codec.encode(page)
            return tasks.code2img_seq(bpe, self.layout, code, tasks.CODE2IMG_INSTRUCTIONS[j], l1, l2)
        if k == "code_lm":
            _, code = (self.val_code[i % len(self.val_code)] if val else self.code_at(i))
            return tasks.lm_seq(bpe, code, tag=None)
        if k == "ascii2code":
            art = ascii_art.to_ascii(img, width=self.args.combo_ascii_width)
            return tasks.ascii2code_seq(bpe, art, tasks.ASCII2CODE_INSTRUCTIONS[j],
                                        synth.art2page(i, art))
        if k == "img2code":
            l1, l2 = codec.encode(img)
            return tasks.img2code_seq(bpe, self.layout, l1, l2, tasks.IMG2CODE_INSTRUCTIONS[j],
                                      synth.scene2page(i, cap, img))
        if k == "fix":
            _lang, broken, err, fixed = synth.error_case(i)
            return tasks.fix_seq(bpe, broken, tasks.FIX_QUESTIONS[j].format(e=err), fixed)
        return tasks.lm_seq(bpe, cap)

    def build_batch(self, descs, val: bool = False):
        """Returns (x, y) int64 tensors, x=seq[:-1], y=seq[1:], padded with <pad>=0."""
        import torch
        ctx = self.args.max_seq_len
        rows = []
        for d in descs:
            s = self.seq(d, val)
            if len(s) > ctx:
                if d[0] in ("img2cap", "img2ascii", "txt2img", "img2code", "code2img", "fix"):
                    # An image sequence cannot be truncated without destroying it, so it is dropped.
                    # train.py warns about this and tells you which flag to change.
                    self.skipped += 1
                    self.skipped_kind = d[0]
                    self.skipped_len = len(s)
                    continue
                s = s[:ctx - 1] + [s[-1]]
            rows.append(s)
        if not rows:
            return None
        L = max(len(r) for r in rows)
        x = np.zeros((len(rows), L - 1), dtype=np.int64)
        y = np.zeros((len(rows), L - 1), dtype=np.int64)
        for i, r in enumerate(rows):
            x[i, :len(r) - 1] = r[:-1]
            y[i, :len(r) - 1] = r[1:]
        return torch.from_numpy(x), torch.from_numpy(y)

    def plan_batches(self, plan, batch_tokens: int, val: bool = False):
        """Groups descriptors into batches by estimated length (no encoding, so it is cheap
        enough to use for resume skipping as well)."""
        i = 0
        max_rows = max(1, int(getattr(self.args, "max_batch_rows", 64)))
        while i < len(plan):
            chunk, ntok = [], 0
            while i < len(plan):
                d = plan[i]
                est = self._estimate(d)
                if chunk and (ntok + est > batch_tokens or len(chunk) >= max_rows):
                    break
                chunk.append(d)
                ntok += est
                i += 1
            yield chunk

    def batches(self, plan, batch_tokens: int, val: bool = False):
        for chunk in self.plan_batches(plan, batch_tokens, val):
            b = self.build_batch(chunk, val)
            if b is not None:
                yield b
