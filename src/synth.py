"""Procedural captioned images + web code (the offline half of the training data).

Why: a ~5M-parameter CPU model will never reproduce photographs, but it can learn a deterministic
visual grammar. Every scene is a pure function of its index, so the dataset is infinite, perfectly
captioned ("a big white car on the left on a green background") and reproducible on resume.
"""
from __future__ import annotations

import math

import numpy as np

PX = 64

COLORS = {
    "white": (245, 245, 245), "black": (34, 34, 38), "gray": (132, 134, 140),
    "red": (206, 52, 48), "orange": (235, 128, 40), "yellow": (242, 208, 58),
    "green": (58, 156, 84), "blue": (48, 96, 208), "purple": (140, 84, 200),
    "pink": (232, 118, 168), "brown": (126, 88, 54),
}
BGS = {"green": (120, 176, 120), "blue": (128, 172, 224), "gray": (176, 178, 182),
       "yellow": (238, 226, 158), "pink": (238, 186, 208), "black": (44, 46, 52)}
SHAPES = ["car", "house", "tree", "face", "ball", "mug", "star", "fish", "flower"]

TEMPLATES = [
    "a {sz}{col} {shape}{pos} on a {bg} background",
    "on a {bg} background there is a {sz}{col} {shape}{pos}",
    "a picture of a {sz}{col} {shape}{pos} against a {bg} background",
    "a {sz}{col} {shape}{pos} in front of a {bg} background",
    "there is a {sz}{col} {shape}{pos} on a {bg} background",
]
POS_WORDS = {-1: " on the left", 0: "", 1: " on the right"}


class Raster:
    def __init__(self, px: int = PX):
        self.px, self.img = int(px), np.zeros((int(px), int(px), 3), np.float32)

    def fill(self, color):
        self.img[:] = color
        return self

    def poly(self, pts, color, inflate=0.0):
        pts = np.asarray(pts, np.float32)
        if inflate:
            c, d = pts.mean(0), pts - pts.mean(0)
            pts = pts + d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-6) * inflate
        yy, xx = np.mgrid[0:self.px, 0:self.px].astype(np.float32)
        inside = np.zeros((self.px, self.px), bool)
        for i in range(len(pts)):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % len(pts)]
            inside ^= ((y1 > yy) != (y2 > yy)) & (xx < (x2 - x1) * (yy - y1) / (y2 - y1 + 1e-9) + x1)
        return self._paint(inside, color)

    def circle(self, cx, cy, r, color, inflate=0.0):
        yy, xx = np.mgrid[0:self.px, 0:self.px].astype(np.float32)
        return self._paint((xx - cx) ** 2 + (yy - cy) ** 2 <= (r + inflate) ** 2, color)

    def ellipse(self, cx, cy, rx, ry, color, inflate=0.0):
        yy, xx = np.mgrid[0:self.px, 0:self.px].astype(np.float32)
        return self._paint(((xx - cx) / (rx + inflate)) ** 2 + ((yy - cy) / (ry + inflate)) ** 2 <= 1, color)

    def rect(self, x0, y0, x1, y1, color, inflate=0.0):
        yy, xx = np.mgrid[0:self.px, 0:self.px].astype(np.float32)
        return self._paint((xx >= x0 - inflate) & (xx <= x1 + inflate) &
                           (yy >= y0 - inflate) & (yy <= y1 + inflate), color)

    def rrect(self, x0, y0, x1, y1, r, color, inflate=0.0):
        self.rect(x0 + r, y0, x1 - r, y1, color, inflate)
        self.rect(x0, y0 + r, x1, y1 - r, color, inflate)
        for cx, cy in ((x0 + r, y0 + r), (x1 - r, y0 + r), (x0 + r, y1 - r), (x1 - r, y1 - r)):
            self.circle(cx, cy, r, color, inflate)
        return self

    def _paint(self, mask, color):
        if mask.any():
            self.img[mask] = color
        return self


def shape_spec(shape: str):
    """Primitive list. Last element of each tuple is the colour:
    0 -> shape colour (outlined), -1 -> punch back to background, else a fixed RGB triple."""
    C = COLORS
    if shape == "car":
        return [("poly", [(9, 32), (16, 21), (40, 21), (55, 32), (55, 46), (9, 46)], 0),
                ("circle", 19, 46, 5.5, C["black"]), ("circle", 45, 46, 5.5, C["black"]),
                ("poly", [(19, 30), (23, 23), (29, 23), (29, 30)], (150, 196, 226)),
                ("poly", [(33, 23), (39, 23), (43, 30), (33, 30)], (150, 196, 226)),
                ("rect", 52, 34, 54.5, 38, C["yellow"])]
    if shape == "house":
        return [("poly", [(13, 35), (32, 14), (51, 35)], 0),
                ("rect", 18, 33, 46, 55, 0),
                ("rect", 29, 43, 36, 55, C["brown"]),
                ("rect", 21, 37, 27, 43, (150, 196, 226))]
    if shape == "tree":
        return [("rect", 29, 40, 35, 57, C["brown"]),
                ("circle", 32, 30, 15, 0), ("circle", 23, 35, 9, 0), ("circle", 41, 35, 9, 0)]
    if shape == "face":
        return [("circle", 32, 32, 19, 0),
                ("circle", 25, 27, 2.6, C["black"]), ("circle", 39, 27, 2.6, C["black"]),
                ("rrect", 25, 40, 39, 43, 1.5, C["black"])]
    if shape == "ball":
        return [("circle", 32, 33, 18, 0), ("circle", 25, 26, 4.5, (250, 250, 250))]
    if shape == "mug":
        return [("rrect", 17, 24, 40, 51, 4, 0),
                ("circle", 42, 37, 8.5, 0), ("circle", 42, 37, 4.5, -1),
                ("ellipse", 28.5, 25, 10, 3, (250, 250, 250))]
    if shape == "star":
        pts = []
        for i in range(10):
            ang = -math.pi / 2 + i * math.pi / 5
            r = 21 if i % 2 == 0 else 9.5
            pts.append((32 + r * math.cos(ang), 33 + r * math.sin(ang)))
        return [("poly", pts, 0)]
    if shape == "fish":
        return [("ellipse", 29, 32, 16, 11, 0),
                ("poly", [(43, 32), (57, 21), (57, 43)], 0),
                ("circle", 21, 29, 2.6, C["black"])]
    if shape == "flower":
        prims = [("rect", 31, 34, 34, 57, (86, 152, 84))]
        for i in range(6):
            a = i * math.pi / 3
            prims.append(("circle", 32 + 9 * math.cos(a), 26 + 9 * math.sin(a), 6.5, 0))
        prims.append(("circle", 32, 26, 4.5, C["yellow"]))
        return prims
    raise KeyError(shape)


def _scale(pts, s, ox, oy, px):
    c = px / 2.0
    return [(c + (x - c) * s + ox, c + (y - c) * s + oy) for x, y in pts]


def _draw(ras: Raster, prims, s, ox, oy, body, inflate=0.0, override=None):
    px = ras.px

    def X(v):
        return px / 2 + (v - px / 2) * s + ox

    def Y(v):
        return px / 2 + (v - px / 2) * s + oy

    for pr in prims:
        kind, *rest = pr
        col = rest[-1]
        if col == 0:
            col = override if override is not None else body
        elif col == -1:
            continue
        if kind == "poly":
            ras.poly(_scale(rest[0], s, ox, oy, px), col, inflate)
        elif kind == "circle":
            ras.circle(X(rest[0]), Y(rest[1]), rest[2] * s, col, inflate)
        elif kind == "ellipse":
            ras.ellipse(X(rest[0]), Y(rest[1]), rest[2] * s, rest[3] * s, col, inflate)
        elif kind == "rect":
            ras.rect(X(rest[0]), Y(rest[1]), X(rest[2]), Y(rest[3]), col, inflate)
        elif kind == "rrect":
            ras.rrect(X(rest[0]), Y(rest[1]), X(rest[2]), Y(rest[3]), rest[4] * s, col, inflate)
    return ras


def far_enough(cname: str, bgname: str) -> bool:
    a, b = np.array(COLORS[cname], float), np.array(BGS[bgname], float)
    return float(np.abs(a - b).mean()) > 55


def scene(i: int, seed: int = 1234, px: int = PX):
    """Deterministic (image uint8 px*px*3, caption str) for sample index i."""
    rng = np.random.default_rng((int(seed) * 1000003 + int(i) * 7919) % (2 ** 63))
    shape = SHAPES[int(rng.integers(len(SHAPES)))]
    cname = list(COLORS)[int(rng.integers(len(COLORS)))]
    bgname = list(BGS)[int(rng.integers(len(BGS)))]
    for _ in range(12):
        if far_enough(cname, bgname):
            break
        bgname = list(BGS)[int(rng.integers(len(BGS)))]
    s = [0.72, 1.0, 1.14][int(rng.integers(3))]
    szword = {0.72: "small ", 1.0: "", 1.14: "big "}[s]
    pos = [-1, 0, 0, 1][int(rng.integers(4))]
    ox = pos * 6.0
    body = COLORS[cname]
    dark = tuple(int(c * 0.30) for c in body) if sum(body) > 400 else (250, 250, 250)
    prims = shape_spec(shape)
    ras = Raster(px).fill(BGS[bgname])
    _draw(ras, [p for p in prims if p[-1] == 0], s, ox, 0, body, inflate=1.8, override=dark)
    _draw(ras, [p for p in prims if p[-1] != -1], s, ox, 0, body)
    _draw(ras, [p for p in prims if p[-1] == -1], s, ox, 0, body, override=BGS[bgname])
    img = np.clip(ras.img + rng.normal(0, 1.5, ras.img.shape), 0, 255).astype(np.uint8)
    caption = TEMPLATES[int(rng.integers(len(TEMPLATES)))].format(
        sz=szword, col=cname, shape=shape, pos=POS_WORDS[pos], bg=bgname)
    return img, caption


# ------------------------------------------------------------------ web code
THINGS = ["gallery", "shop", "blog", "todo list", "weather app", "quiz", "dashboard", "portfolio"]
CODE_COLORS = ["red", "blue", "green", "purple", "orange", "black", "teal", "crimson"]


def fill(template: str, **kw) -> str:
    """@@NAME@@ placeholders (braces belong to CSS/JS, so str.format is unusable here)."""
    out = template
    for k, v in kw.items():
        out = out.replace(f"@@{k}@@", str(v))
    return out


def synth_code(i: int, seed: int = 1234):
    """Deterministic (instruction, filename, code) for web-code sample index i."""
    rng = np.random.default_rng((int(seed) * 7717 + int(i) * 104729) % (2 ** 63))
    thing = THINGS[int(rng.integers(len(THINGS)))]
    color = CODE_COLORS[int(rng.integers(len(CODE_COLORS)))]
    variant = int(rng.integers(13))
    title = {"gallery": "Photo Gallery", "shop": "My Shop", "blog": "My Blog",
             "todo list": "Todo List", "weather app": "Weather", "quiz": "Quiz",
             "dashboard": "Dashboard", "portfolio": "Portfolio"}[thing]
    if variant == 0:
        code = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>@@T@@</title>
  <link rel="stylesheet" href="style.css">
</head>
<body>
  <header class="top">
    <h1>@@T@@</h1>
  </header>
  <main id="app"></main>
  <script src="app.js"></script>
</body>
</html>"""
        return f"write html for a {color} {thing} page", "index.html", fill(code, T=title)
    if variant == 1:
        code = """body {
  margin: 0;
  font-family: system-ui, sans-serif;
  background: #f7f7f9;
  color: #222;
}

.top {
  padding: 24px;
  background: @@C@@;
  color: white;
}

h1 {
  margin: 0;
  font-size: 2rem;
}

main {
  padding: 24px;
}

.card {
  border: 2px solid @@C@@;
  border-radius: 12px;
  padding: 16px;
  background: white;
}"""
        return f"write css for a {color} {thing} page", "style.css", fill(code, C=color)
    if variant == 2:
        code = """const app = document.getElementById('app');
const button = document.createElement('button');
let count = 0;
button.textContent = 'Clicked 0 times';
button.addEventListener('click', () => {
  count += 1;
  button.textContent = `Clicked ${count} times`;
});
app.appendChild(button);"""
        return f"write javascript for a {thing} counter button", "app.js", code
    if variant == 3:
        code = """<ul class="list">
  <li>First item</li>
  <li>Second item</li>
  <li>Third item</li>
</ul>"""
        return f"write html for a {thing} list with three items", "list.html", code
    if variant == 4:
        code = """.card {
  padding: 16px;
  border-radius: 10px;
  background: #fff;
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.15);
  transition: transform 0.2s;
}

.card:hover {
  transform: translateY(-4px);
  color: @@C@@;
}"""
        return f"write css for a {thing} card with a hover effect", "card.css", fill(code, C=color)
    if variant == 5:
        code = """async function load() {
  const res = await fetch('/api/@@K@@');
  const data = await res.json();
  document.querySelector('#app').textContent = JSON.stringify(data);
}

load().catch(err => console.error(err));"""
        return (f"write javascript that fetches the {thing} data and renders it", "fetch.js",
                fill(code, K=thing.replace(" ", "-")))
    if variant == 6:
        return f"write html for a {color} button", "button.html", '<button class="btn" type="button">Save</button>'
    if variant == 7:
        code = """<!DOCTYPE html>
<html>
<head>
<style>
  body { background: #101014; color: #eee; }
  .box { border: 2px solid @@C@@; padding: 20px; border-radius: 8px; }
</style>
</head>
<body>
  <div class="box">
    <h1>@@T@@</h1>
    <p>Built with plain HTML and CSS.</p>
  </div>
</body>
</html>"""
        return f"write a complete {color} page for a {thing} with inline css", "page.html", fill(code, C=color, T=title)
    if variant == 9:
        code = """const canvas = document.getElementById('game');
const ctx = canvas.getContext('2d');
const size = 16;
let snake = [{ x: 8, y: 8 }];
let dir = { x: 1, y: 0 };
let food = { x: 5, y: 5 };
let score = 0;

function step() {
  const head = { x: snake[0].x + dir.x, y: snake[0].y + dir.y };
  if (head.x < 0 || head.y < 0 || head.x >= size || head.y >= size) return reset();
  snake.unshift(head);
  if (head.x === food.x && head.y === food.y) {
    score += 1;
    food = { x: Math.floor(Math.random() * size), y: Math.floor(Math.random() * size) };
  } else {
    snake.pop();
  }
}

document.addEventListener('keydown', event => {
  if (event.key === 'ArrowUp') dir = { x: 0, y: -1 };
  if (event.key === 'ArrowDown') dir = { x: 0, y: 1 };
  if (event.key === 'ArrowLeft') dir = { x: -1, y: 0 };
  if (event.key === 'ArrowRight') dir = { x: 1, y: 0 };
});

setInterval(() => { step(); draw(); }, 120);"""
        return f"make a {color} snake game in javascript", "snake.js", fill(code, C=color)
    if variant == 10:
        code = """const display = document.querySelector('#display');
let current = '';

function press(value) {
  if (value === 'C') {
    current = '';
  } else if (value === '=') {
    current = String(eval(current));
  } else {
    current += value;
  }
  display.textContent = current || '0';
}

for (const button of document.querySelectorAll('button')) {
  button.addEventListener('click', () => press(button.textContent));
}"""
        return f"write javascript for a {color} calculator", "calculator.js", code
    if variant == 11:
        code = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Image preview</title>
  <style>
    body { margin: 0; background: #0f1115; color: #e9edf3; font-family: system-ui, sans-serif; }
    .card { border: 2px solid @@C@@; border-radius: 12px; padding: 20px; }
  </style>
</head>
<body>
  <div class="card">
    <h1>Image preview</h1>
    <img id="preview" alt="preview" width="240">
  </div>
</body>
</html>"""
        return (f"write html that shows an image preview of a {color} {thing}",
                "preview.html", fill(code, C=color))
    if variant == 12:
        code = """def @@FN@@(items: list) -> dict:
    # count how often each name appears
    counts = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    return counts


if __name__ == "__main__":
    print(@@FN@@(['a', 'b', 'a']))"""
        fn = "count_items" if thing != "quiz" else "score_quiz"
        return f"write python for a {thing} helper", "helper.py", fill(code, FN=fn)
    code = """const root = document.documentElement;
let dark = false;
function toggle() {
  dark = !dark;
  root.style.background = dark ? '#101014' : '#f7f7f9';
  root.style.color = dark ? '#eeeeee' : '#222222';
}
document.querySelector('button').addEventListener('click', toggle);"""
    return f"write javascript that toggles a {color} theme", "theme.js", code


# ------------------------------------------------------------------ combinator tasks
# These three build paired data for the combination tasks, so the model learns real
# transforms rather than a random (input, output) pairing:
#
#   ascii -> code   take the ascii art of an image and write the page that displays it
#   image -> code   take an image + caption and write the page that describes it
#   code  -> image  take a code snippet and render the preview image of the page it makes
PAGE_COLORS = ["red", "blue", "green", "purple", "orange", "teal", "gray", "gold"]
COLOR_HEX = {
    "red": "#c0392b", "blue": "#2d6cdf", "green": "#27864f", "purple": "#7d4fc4",
    "orange": "#e0812a", "teal": "#1f8a8a", "gray": "#5c6470", "black": "#181a1f",
    "gold": "#c9a227", "pink": "#d1589b", "white": "#f2f3f5", "crimson": "#b02535",
    "brown": "#7b5636", "yellow": "#d8b427",
}


def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _page(title: str, bg: str, body_lines: list, extra_css: str = "") -> str:
    """A compact page (fits in a small context) used as the training target for the
    ascii->code and image->code tasks."""
    body = "\n".join("    " + line for line in body_lines)
    return ("<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
            f"<title>{title}</title>\n<style>\n"
            f"  body {{ margin: 0; background: {bg}; color: #f5f6f8;"
            " font-family: system-ui, sans-serif; }\n"
            "  .wrap { padding: 24px; max-width: 720px; }\n"
            "  pre { background: #10131a; color: #d8e2f0; padding: 16px; border-radius: 10px;"
            " overflow: auto; line-height: 1.15; }\n"
            "  h1 { margin: 0 0 12px; font-size: 1.6rem; }\n"
            f"{extra_css}</style>\n</head>\n<body>\n  <div class=\"wrap\">\n"
            f"    <h1>{title}</h1>\n{body}\n  </div>\n</body>\n</html>")


def art2page(i: int, art: str) -> str:
    """ascii art -> the small web page that shows it (target for the ascii->code task)."""
    rng = np.random.default_rng(9001 + int(i))
    color = PAGE_COLORS[int(rng.integers(len(PAGE_COLORS)))]
    title = f"{color} ascii art"
    return _page(title, COLOR_HEX[color],
                 ["<p>Rendered from an ascii text art on a " + color + " background.</p>",
                  f"<pre>{_esc(art)}</pre>"])


def scene2page(i: int, caption: str, img: np.ndarray) -> str:
    """image + caption -> the page that describes it (target for the image->code task)."""
    cap = " ".join(str(caption).split())[:60]
    rng = np.random.default_rng(5003 + int(i))
    flat = img.reshape(-1, 3).mean(0)
    bg = "#%02x%02x%02x" % tuple(int(max(18, min(210, c * 0.55))) for c in flat)
    accent = PAGE_COLORS[int(rng.integers(len(PAGE_COLORS)))]
    return _page(cap or "a made up page", bg,
                 [f"<p>{_esc(cap)}</p>",
                  "<div class=\"card\">This page was written from the picture.</div>"],
                 extra_css=(f"  .card {{ border: 2px solid {COLOR_HEX[accent]}; border-radius: 12px;"
                            " padding: 16px; background: rgba(255,255,255,.06); }\n"))


def page2scene(i: int, code: str, px: int = 64) -> np.ndarray:
    """code -> the preview image of the page it builds (target for the code->image task).
    The colours are read back out of the code, so the task is a real function of the input."""
    low = str(code).lower()
    color = next((c for c in PAGE_COLORS if c in low), None)
    if color is None:
        color = PAGE_COLORS[int(np.random.default_rng(2003 + int(i)).integers(len(PAGE_COLORS)))]
    accent = np.array([int(COLOR_HEX[color][1:3], 16), int(COLOR_HEX[color][3:5], 16),
                       int(COLOR_HEX[color][5:7], 16)], np.float32)
    px = int(px)
    img = np.zeros((px, px, 3), np.float32)
    img[:] = accent * 0.45 + 30.0                       # page background
    m = max(3, px // 10)
    img[m:px - m, m:px - m] = (246, 247, 249)           # the white content card
    img[m:px - m, m:px - m][:max(3, px // 10)] = accent  # header bar in the page colour
    return np.clip(img, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- error fixing
# Paired (broken code, error message, fixed code) examples for the "fix this, i got an error"
# task. Each case is a real, reproducible failure with a one-step fix, so the task is a genuine
# transform of the input rather than a random pairing.
ERROR_CASES = [
    ("javascript",
     "function add(a, b) {\n  return a + b;",
     "SyntaxError: Unexpected end of input",
     "function add(a, b) {\n  return a + b;\n}"),
    ("python",
     "def greet(name)\n    print(name)",
     "SyntaxError: expected ':'",
     "def greet(name):\n    print(name)"),
    ("python",
     "for item in items:\nprint(item)",
     "IndentationError: expected an indented block",
     "for item in items:\n    print(item)"),
    ("javascript",
     "let count = 0;\nif (count = 5) {\n  count += 1;\n}",
     "the condition is always true: '=' assigns instead of comparing",
     "let count = 0;\nif (count === 5) {\n  count += 1;\n}"),
    ("html",
     '<div class="card">\n  <p>hello</p>',
     "the page layout collapses: a tag is not closed",
     '<div class="card">\n  <p>hello</p>\n</div>'),
    ("javascript",
     "console.log(total);",
     "ReferenceError: total is not defined",
     "const total = 0;\nconsole.log(total);"),
    ("python",
     "print(nmae)",
     "NameError: name 'nmae' is not defined",
     "print(name)"),
    ("javascript",
     "document.querySelector('#app').textContent = 'ready';",
     "TypeError: Cannot set properties of null (setting 'textContent')",
     "const app = document.querySelector('#app');\nif (app) app.textContent = 'ready';"),
    ("python",
     "average = total / count",
     "ZeroDivisionError: division by zero",
     "average = total / count if count else 0"),
    ("css",
     "body {\n  colour: red;\n}",
     "the text colour does not change",
     "body {\n  color: red;\n}"),
    ("python",
     "print('n = ' + 5)",
     "TypeError: can only concatenate str (not \"int\") to str",
     "print(f'n = {5}')"),
    ("javascript",
     "const data = await response.json();\nconsole.log(data);",
     "SyntaxError: await is only valid in async functions",
     "async function load() {\n  const data = await response.json();\n  console.log(data);\n}"),
]

def error_case(i: int):
    """Deterministic (language, broken code, error message, fixed code) for case index i."""
    lang, broken, err, fixed = ERROR_CASES[int(i) % len(ERROR_CASES)]
    return lang, broken, err, fixed


def synth_text(i: int, seed: int = 1234):
    """Short natural-language lines (captions) used for plain language modelling."""
    _, caption = scene(i, seed)
    return caption


if __name__ == "__main__":
    import sys
    from PIL import Image
    from imagecodec import Codec, save_png
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    codec = Codec(64, 4, True)
    tiles = []
    for k in range(n):
        img, cap = scene(k)
        l1, l2 = codec.encode(img)
        rec = codec.decode(l1, l2)
        tiles.append(np.concatenate([img, rec], axis=1))
        print(f"{k:3d} {cap}")
    sheet = np.concatenate([np.concatenate(tiles[i:i + 4], axis=1) for i in range(0, n, 4)], axis=0)
    save_png(sheet, "samples/synth_sheet.png", upscale=2 if sheet.shape[0] < 300 else 1)
    print("wrote samples/synth_sheet.png (left = original, right = token round-trip)")
    print(synth_code(1))
