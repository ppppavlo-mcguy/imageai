"""Deterministic image -> ASCII/text target generator.

The model needs supervised targets for "turn this into an ascii text art", so during data
preparation we render every training image to ASCII with this exact converter. At inference the
same task is asked as a question (`<img> ... <sep> <ask> turn this into an ascii art <sep> ...`).
"""
from __future__ import annotations

import numpy as np

RAMP = " .:-=+*#%@"


def to_ascii(img: np.ndarray, width: int = 48, aspect: float = 0.5, ramp: str = RAMP) -> str:
    """img: HxWx3 uint8 -> ascii art text (newline separated)."""
    h, w = img.shape[:2]
    out_w = max(8, int(width))
    out_h = max(2, int(round(out_w * (h / w) * aspect)))
    # area-average downsample
    ys = np.linspace(0, h, out_h + 1).astype(int)
    xs = np.linspace(0, w, out_w + 1).astype(int)
    lum = (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]).astype(np.float32)
    rows = []
    for i in range(out_h):
        y0, y1 = ys[i], max(ys[i] + 1, ys[i + 1])
        line = []
        for j in range(out_w):
            x0, x1 = xs[j], max(xs[j] + 1, xs[j + 1])
            v = float(lum[y0:y1, x0:x1].mean())
            line.append(ramp[min(len(ramp) - 1, int(v / 256.0 * len(ramp)))])
        rows.append("".join(line).rstrip())
    return "\n".join(rows)
