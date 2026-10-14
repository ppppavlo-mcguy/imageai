"""Deterministic image <-> token codec. Identical maths in src/imagecodec.py and web/engine.js.

Level 1: mean colour of every PATCH x PATCH block -> 12-bit code (4 bits/channel, 16 levels).
Level 2: every 2x2 block inside a patch stores a signed 3-bit/channel delta from the patch mean.
So a 64x64 image with patch=4 becomes 256 level-1 tokens (+1024 level-2 tokens), and decoding
gives real per-2x2 detail with full colour resolution instead of a flat 32x32 icon.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

L1_LEVELS = 15           # 0..15 -> 4 bits
L1_BITS = 4
L2_MIN, L2_MAX = -3, 3   # 7 signed levels -> 3 bits
DET_STEP = 255.0 / 15.0 * 0.9   # ~15.3 grey levels per residual step


class Codec:
    def __init__(self, img_px: int = 64, patch: int = 4, detail: bool = True):
        if img_px % patch or patch % 2:
            raise ValueError("img_px must be divisible by patch, patch must be even")
        self.img_px = int(img_px)
        self.patch = int(patch)
        self.sub = int(patch) // 2          # 2x2 sub-blocks per patch
        self.grid = self.img_px // self.patch
        self.detail = bool(detail)
        self.n1 = self.grid * self.grid
        self.n2 = self.n1 * self.sub * self.sub if self.detail else 0

    # ------------------------------------------------------------------ encode
    def encode(self, img: np.ndarray):
        """img: uint8 HxWx3 (RGB). Returns (l1_codes:list[int], l2_codes:list[int])."""
        if img.shape[0] != self.img_px or img.shape[1] != self.img_px:
            raise ValueError(f"expected {self.img_px}x{self.img_px}, got {img.shape[:2]}")
        a = img.astype(np.float32)
        p, g = self.patch, self.grid
        patches = a.reshape(g, p, g, p, 3).mean(axis=(1, 3))          # (g,g,3)
        q1 = np.clip(np.rint(patches * (L1_LEVELS / 255.0)), 0, L1_LEVELS).astype(np.int32)
        l1 = (q1[..., 0] * 256 + q1[..., 1] * 16 + q1[..., 2]).reshape(-1).tolist()

        l2 = []
        if self.detail:
            s = self.sub
            blocks = a.reshape(g, s, 2, g, s, 2, 3).mean(axis=(2, 5))  # (g,s,g,s,3)
            blocks = blocks.transpose(0, 2, 1, 3, 4)                  # (g,g,s,s,3)
            delta = blocks - patches[:, :, None, None, :]             # (g,g,s,s,3)
            q2 = np.clip(np.rint(delta / DET_STEP), L2_MIN, L2_MAX).astype(np.int32) - L2_MIN
            l2 = (q2[..., 0] * 49 + q2[..., 1] * 7 + q2[..., 2]).reshape(-1).tolist()
        return l1, l2

    # ------------------------------------------------------------------ decode
    def decode(self, l1, l2=None) -> np.ndarray:
        p, g = self.patch, self.grid
        q1 = np.asarray(l1, dtype=np.int32).reshape(g, g)
        rgb4 = np.stack([(q1 >> 8) & 15, (q1 >> 4) & 15, q1 & 15], axis=-1)
        base = rgb4 * (255.0 / L1_LEVELS)
        img = base[:, None, :, None, :].repeat(p, axis=1).repeat(p, axis=3)  # (g,p,g,p,3)
        if self.detail and l2 is not None and len(l2):
            s = self.sub
            q2 = np.asarray(l2, dtype=np.int32).reshape(g, g, s, s)
            tri = np.stack([q2 // 49, (q2 // 7) % 7, q2 % 7], axis=-1) + L2_MIN
            delta = (tri * DET_STEP).transpose(0, 2, 1, 3, 4)                 # (g,s,g,s,3)
            delta = np.repeat(np.repeat(delta, 2, axis=1), 2, axis=3)         # (g,p,g,p,3)
            img = img + delta
        # (grid, patch, grid, patch, 3) is row-major (gy, py, gx, px) -> straight reshape
        img = img.reshape(self.img_px, self.img_px, 3)
        return np.clip(np.rint(img), 0, 255).astype(np.uint8)

    # -------------------------------------------------------------------- meta
    def to_json(self) -> dict:
        return {
            "img_px": self.img_px,
            "patch": self.patch,
            "sub": self.sub,
            "grid": self.grid,
            "detail": self.detail,
            "n1": self.n1,
            "n2": self.n2,
            "l1_levels": L1_LEVELS,
            "l1_bits": L1_BITS,
            "l2_min": L2_MIN,
            "l2_max": L2_MAX,
            "det_step": DET_STEP,
        }

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_json(), indent=2), encoding="utf-8")

    @staticmethod
    def load(path) -> "Codec":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        c = Codec(d["img_px"], d["patch"], d.get("detail", True))
        return c

    def tokens_per_image(self) -> int:
        return self.n1 + self.n2


def load_image(path, size: int) -> np.ndarray:
    from PIL import Image
    im = Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC)
    return np.asarray(im, dtype=np.uint8)


def save_png(arr: np.ndarray, path, upscale: int = 6):
    from PIL import Image
    im = Image.fromarray(arr, "RGB")
    if upscale > 1:
        im = im.resize((arr.shape[1] * upscale, arr.shape[0] * upscale), Image.BICUBIC)
    im.save(path)
    return path
