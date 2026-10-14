"""Token-space layout shared by training (Python) and inference (JS).

Every id belongs to exactly one region:

    0 .. 8                 specials (<pad> <bos> <eos> <sep> <img> <cap> <ascii> <code> <ask>)
    text_base .. +n_text   byte-level BPE text/code tokens
    img1_base .. +4096     image level-1 codes (4 bits per RGB channel, 16 levels each)
    img2_base .. +512      image level-2 codes (per-block signed 3-bit RGB deltas)

Keeping one shared vocabulary is what lets a single small transformer do
text, code and image tokens with the same weights.
"""
from __future__ import annotations

SPECIALS = ["<pad>", "<bos>", "<eos>", "<sep>", "<img>", "<cap>", "<ascii>", "<code>", "<ask>"]
PAD, BOS, EOS, SEP, IMG, CAP, ASCII, CODE, ASK = range(len(SPECIALS))

L1_CODES = 4096          # 12-bit mean colour
L2_CODES = 512           # 9-bit signed residual, 343 used


class Layout:
    def __init__(self, n_text: int, l1: int = L1_CODES, l2: int = L2_CODES, detail: bool = True):
        self.n_text = int(n_text)
        self.text_base = len(SPECIALS)
        self.img1_base = self.text_base + self.n_text
        self.img2_base = self.img1_base + int(l1)
        self.l1 = int(l1)
        self.l2 = int(l2) if detail else 0
        self.detail = bool(detail)
        self.total = self.img2_base + self.l2

    def to_json(self) -> dict:
        return {
            "specials": SPECIALS,
            "n_text": self.n_text,
            "text_base": self.text_base,
            "l1": self.l1,
            "img1_base": self.img1_base,
            "l2": self.l2,
            "img2_base": self.img2_base,
            "detail": self.detail,
            "total": self.total,
        }


def id_of(name: str) -> int:
    return SPECIALS.index(name)


def is_image_token(layout: Layout, tid: int) -> bool:
    if tid < layout.img1_base:
        return False
    return tid < layout.img1_base + layout.l1 or (layout.detail and tid < layout.total)
