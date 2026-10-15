# SGM — Small Generative Model (CPU-trained, JS-served)

A small multimodal model you can train on a Windows PC with no GPU, then run **entirely in the
browser** as plain HTML + CSS + JS (or from Node). One transformer, one vocabulary, five jobs:

The UI is a **chat app** (`web/`): a tool rail and the **Gallery** on the left, the conversation on
the right, and the answer for each turn as an image, a code card with *Copy* and *Preview as page*,
or ASCII art. The **model selector sits in the composer on the right** (`◈ SGM-3M-code ▾`) and opens
the model list, where a bundle is chosen or downloaded. Dark and light themes, uploads with **＋**,
and everything — chats, uploads, model — stays in the browser tab (no external requests).

* **Gallery** (`Gallery` button): every file a chat created — PNGs, code, ASCII art — in one grid,
  each downloadable on its own or all together as a `.zip` (zipped in the tab by `web/zip.js`).
* **Model picker, in the composer**: the pill next to the send button shows which bundle will
  answer; clicking it opens **Models & downloads** on the *models* tab, where `SGM-1M`,
  `SGM-3M-code`, `SGM-1M-int4` and (planned) `SGM-10M` / `SGM-100M` are listed with their size and
  tasks, each with a *use* button and its `.zip` download in the same row. The list comes from
  `web/models.json` plus what the bundled server finds on disk, so a freshly exported bundle shows
  up by itself. A bundle is only loaded **when you send the first message** (or press *Load now*),
  and the rail greys out the tools a model was never trained for instead of producing noise.
* **Streaming answers** (`stream on` next to the composer, remembered): tokens appear in the bubble
  while the model writes them, instead of after the whole answer is finished. Turn it off to get
  the finished answer in one go.
* **A token budget you can see**: *max new tokens* in the **sampling** panel (default **512**, the
  code task used to stop at a hard 240 and cut pages off mid-attribute). If an answer does run into
  the cap, the message says so instead of looking like a syntax error, and the budget is fitted to
  the model's context window automatically.
* **Downloads** (`⬇` in the header, `/api/downloads`): every exported model as a versioned `.zip`
  (`sgm-1m-v2.zip`) and the desktop build (`sgm-desktop-v2.zip`, a Windows `.exe`).
* **Desktop app** (`desktop/`): the same UI in an Electron window, packageable to a real
  `SGM.exe` — see section 10.

Chats are named after the first thing you asked, not "New chat".

| job | example prompt | output |
|---|---|---|
| generate an image | `a big white car on a blue background` | 64×64 image tokens → PNG/canvas |
| understand an instruction | `draw a big red car on the left` | image |
| recognise an image | attach a picture → *what is on this picture?* | short caption |
| recognise an image | attach a picture → *turn this into an ascii text art* | ASCII art |
| write code | `make a snake game`, `write css for a dark page`, `python to reverse a string` | HTML / CSS / JS / Python |
| ASCII → website | paste ASCII art → *turn this ascii art into a website* | the page that displays the art |
| image → website | attach a picture → *build a web page that looks like this* | a page describing the picture |
| website → image | paste code → *turn this website into an image preview* | a preview image of the page |
| fix an error | attach a file and say *fix this, i got an error: SyntaxError…* | the corrected snippet |

Everything is trained from scratch by PyTorch on your CPU (`datasets` supplies the image/code
corpora) and exported to a single int8 blob that the JS runtime (`web/engine.js`) executes.

---

## Read this first: what a CPU-trained model can and cannot do

Be realistic about the scale, or you will be disappointed at hour 20.

* **It can learn simple visual grammars**: coloured shapes, cars/houses/trees/faces on coloured
  backgrounds, sprites, diagrams, icons, your own small objects — *provided* the image is small
  and the caption fully determines it. That is what `src/synth.py` produces, and it is why the
  demo prompt "a white car" works at all after a couple of hours.
* **It cannot do photorealistic people.** "Will Smith" at 64×64 from a ~6M-parameter model trained
  on a CPU for a few hours is not achievable. Faces need a large dataset, a GPU and a real
  image model (diffusion + VAE). A 64×64 photo of a person here would come out as an unreadable
  blob, so don't spend hours on Flickr data expecting portraits.
* **Code is where quality-per-hour is best.** Web snippets are short, structured and learnable;
  after a few epochs the model produces plausible HTML/CSS/JS skeletons. Keep snippets short
  (5–20 lines) and the format identical between training and inference.
* Rule of thumb: image tokens are ~1280 per picture, so one pass over 5 000 pictures is ~6.4M
  tokens. On a 2-core laptop at ~3k tokens/s (measured, see below) that is ~35 min per epoch for
  a 1M-parameter model, and several hours for a 6M-parameter one. See *Sizing a real run*.

---

## 1. Requirements

* Windows 10/11, Linux or macOS. Python 3.10–3.12.
* Node 18+ only if you want the Node CLI / the local web server.
* Intel integrated graphics are fine: everything runs on the CPU (nothing else is installed).

```
python -m venv .venv
.venv\Scripts\activate            # Windows   (source .venv/bin/activate on Linux/macOS)
pip install -r requirements.txt   # torch CPU wheel + numpy + pillow + datasets
```

On Windows the default PyPI `torch` wheel is the CPU build, so no extra index URL is needed.
Windows helpers are included: `setup.cmd`, `train.cmd`, `export.cmd`, `serve.cmd`. They call the
interpreter as `.venv\Scripts\python.exe` directly (like `train_code.cmd` always did) — `call
.venv\Scripts\activate` is not needed and on some machines prints *"The system cannot find the path
specified."* without saying what is wrong.

## 2. Smoke test (2 minutes) — proves the whole pipe works

```
python src/train.py --workdir checkpoints_smoke --img-px 32 --no-detail --preset xs ^
    --vocab 400 --epoch-minutes 0.5 --epochs 1 --sample-every 0
python src/export_web.py checkpoints_smoke/latest.pt -o web_model --verify
node web/cli.mjs --model web_model --task image --prompt "a white car" --out samples/smoke.png
```

Then `node web/server.js` and open <http://localhost:3000>. Expect colourful blobs — it trained for
30 seconds. The point is that training, checkpointing, int8 export and JS inference all line up.

## 3. A real run

```
python src/train.py --workdir checkpoints --preset small --vocab 2048 ^
    --img-px 64 --patch 4 ^
    --synth-n 4000 --synth-code-n 128 ^
    --batch-tokens 2048 --lr 3e-4 --epoch-minutes 120 --epochs 20 ^
    --ckpt-minutes 15 --val-every 200 --sample-every 500
```

* `--epoch-minutes 120` gives you the 2-hour epochs you asked for: the epoch stops at the time
  budget (or when `--samples-per-epoch` samples are consumed, if you set it), and every
  `--ckpt-minutes` it writes `checkpoints/latest.pt` plus a numbered `step_*.pt`.
* Re-running the same command **resumes from `checkpoints/latest.pt`** automatically, mid-epoch
  included (`--resume none` starts fresh, `--resume path/to.pt` picks a specific file).
  Ctrl+C is safe: the current step finishes and `latest.pt` is written; run again to continue.
* Each checkpoint stores the model, the optimiser, the tokenizer, the image codec and the exact
  batch position, so all of them stay in sync.
* `samples/` gets a generated image + a generated HTML file every `--sample-every` steps, so you
  can watch quality grow without exporting. `checkpoints/log.csv` has loss/val/lr history.

### Sizing a real run (measured on this 2 vCPU / 4 GB build box, PyTorch CPU, no GPU)

| preset | params | image size | image tokens/img | measured speed | 5 000 images (image tokens only) |
|---|---|---|---|---|---|
| `xs` 96d/4L | 1.07M | 64px, patch 8 | 64 | ~3.8k tok/s | ~1.5 min (very blocky) |
| `xs` 96d/4L | 1.07M | 32px, patch 2 | 256 | ~3.3k tok/s | ~6.5 min |
| `small` 192d/6L | 4.6M | 64px, patch 4 + detail | 1280 | ~1.0k tok/s | ~1.7 h |
| `base` 256d/8L | ~9.8M | 64px, patch 4 + detail | 1280 | ~0.5k tok/s (estimate) | ~3.5 h |

The numbers in the last two columns are a worst case in which every token is an image token; the
default task mix spends ~40% of tokens on much cheaper text/code, so wall-clock is better. Your
CPU will differ - read the `tok/s` field in the progress bar on the first 100 steps and pick
`--epoch-minutes` from that.

Tokens per image is the lever that matters most: `--no-detail` (1280 → 256) makes a `small` model
as fast as `xs` above, `--patch 8` is cheaper again, and a smaller `--vocab` speeds up the output
projection that every token pays for. The parameters printed at startup are the truth for your
config, so trust `[cfg] ... params ...` over this table.

For the 2-hour epochs you asked for, run `--epoch-minutes 120` with a dataset big enough to fill
it (the procedural corpus is effectively infinite; `--samples-per-epoch 0` keeps pulling fresh
samples until the clock runs out).

## 4. Data

Three sources, plus nine task kinds and four ready-made mixes (see *Task mixes* below).
`--w-txt2img`, `--w-img2txt`, `--w-ascii`, `--w-code`, `--w-code-lm`, `--w-text-lm`,
`--w-ascii2code`, `--w-img2code`, `--w-code2img` set how often each task appears:

1. **`src/synth.py` — procedural, offline, always available.** Deterministic scenes with captions
   (`a big white car on the left on a green background`) plus web-code templates. It is the only
   data source that reliably teaches "generate a white car", because the caption fully determines
   the picture. `--synth-n 20000` is a *virtual* dataset: scenes are generated on demand from the
   sample index, so it costs no disk and resumes deterministically.
2. **Your own pictures** — `--images C:\photos`. Every `foo.jpg` may have a sibling `foo.txt`
   holding the caption (first line is used); otherwise the filename becomes the caption.
3. **HuggingFace `datasets`** — `--hf-dataset mnist` is the tiny verified preset (11 MB, real
   images); `flickr8k`, `cats_vs_dogs` and `imagenette` are wired up too, and anything else works
   with the explicit flags:
   `--hf-dataset NAME --hf-split SPLIT --hf-image-col IMG --hf-caption-col LABEL`.
   Check a dataset resolves before a long run:
   `python -c "import datasets; ds=datasets.load_dataset('NAME',split='train',streaming=True); print(next(iter(ds)).keys())"`.
   Classification datasets get captions from their label names automatically. Rows are pulled once
   at startup (`--hf-samples`) and kept in RAM. For code, `--code-dataset NAME --code-samples N`
   streams a dataset and keeps small `.html/.css/.js` rows (`codeparrot/codeparrot-clean` is the
   obvious candidate) - resolve the name first as shown above, since Hub datasets that your
   `datasets` version cannot load will stop the run with a clear error. `data/seed_code/` plus the
   built-in templates already cover the offline case. Real photos mostly teach *those* photos; long
   Flickr captions plus a small model is seasoning, not the whole diet.

ASCII-art targets are generated deterministically from each training image
(`src/ascii_art.py`), so the "turn this into ascii art" task is supervised for free.

### Task mixes

`--mix` overrides the individual weights in one word:

| mix | what it trains | weights |
|---|---|---|
| `multi` (default) | everything: pictures, captions, ascii, code, the three combinations and error fixing | txt2img .20, img2txt .14, ascii .07, code .18, code_lm .09, text_lm .09, ascii2code .07, img2code .04, code2img .03, fix .09 |
| `code` | **pure code**: web + Python, continuation, ascii→website, error fixing. Nothing here emits image tokens, so it runs like a text model | code .35, code_lm .15, ascii2code .20, fix .20, text_lm .05, ascii .05 |
| `image` | picture-focused, like the shipped demo | txt2img .35, img2txt .25, ascii .12, text_lm .15, code .08, code_lm .05 |
| `all` | one tenth each — maximum variety of combinations | 1/10 per task |

```bat
train_code.cmd                     :: == src\train.py --mix code  (web + python + ascii->website)
train_ui.cmd                       :: == src\train.py --mix multi (everything)
python src/train.py --mix code --code-dataset codeparrot/github-code   :: real Python/JS corpus
```

A `code` run is the best quality-per-hour on a CPU: nothing emits the ~64–1280 image tokens, so
tokens/s is high and 20+ minute epochs already produce plausible snippets. `checkpoints_code/` in
this repo is such a run; export it with
`python src/export_web.py checkpoints_code/latest.pt -o web_model_code` and point the sidebar's
*model folder* box at `web_model_code`.

`img2code` / `code2img` move image tokens in or out, so they only fit a context that already fits a
picture: use `--no-detail` (64 image tokens at `--img-px 32`, 256 at 64) and `--context 1024`+.
`tools/check_tasks.py` tells you the exact length of every task before you burn an hour on a run:

```
python tools/check_tasks.py --mix code --context 1024
```

### Which snippets teach which instruction

`data/seed_code/` uses one comment header per file, in any of the four comment styles:

```
// instruction: make a snake game in javascript     (JS)
# instruction: python that counts words in a file   (Python)
<!-- instruction: ... -->                          (HTML)
```

The header is stripped from the target and becomes the instruction the model is trained to answer,
so adding `my_app.py` with `# instruction: python that renames files` in your data folder is all it
takes to teach a new prompt. `--seed-code-dir` points at a different folder if you keep your own.

## 5. How the one vocabulary works

```
ids 0..8            <pad> <bos> <eos> <sep> <img> <cap> <ascii> <code> <ask>
ids 9..9+n_text-1   byte-level BPE tokens (trained on your captions + code, --vocab merges)
next 4096 ids       image level-1: mean colour of a 4x4 patch, 12-bit (4 bits per channel)
next  512 ids       image level-2: signed 3-bit/channel delta per 2x2 block inside the patch
```

Training sequences (all just next-token prediction, see `src/tasks.py`):

```
caption/ascii   <img> img1… img2… <sep> <ask> what is on this picture? <sep> <cap|ascii> … <eos>
draw an image   <img> INSTRUCTION <sep> img1… img2… <eos>
write code      <code> INSTRUCTION <sep> …html/css/js/python… <eos>
ascii -> code   <ascii> …art… <sep> <code> INSTRUCTION <sep> …page… <eos>
image -> code   <img> img1… img2… <sep> <code> INSTRUCTION <sep> …page… <eos>
code -> image   <code> …snippet… <sep> <img> INSTRUCTION <sep> img1… img2… <eos>
fix an error    <code> …broken… <sep> <ask> i got this error: MESSAGE <sep> …fixed… <eos>
plain text      <bos> … <eos>
```

The error-fixing pairs come from `synth.error_case()` — twelve reproducible failures (missing brace,
missing colon, `=` vs `===`, unclosed tag, `NameError`, `ZeroDivisionError`, `await` outside `async`, …)
each with the exact message and the one-step fix, so "attach the file that failed and ask" is a real
transform rather than a guess.

The combination sequences are real transforms, not random pairings: `src/synth.py` renders the
ascii art of an image into a page (`art2page`), writes a page that describes a picture
(`scene2page`), and draws the preview image of a code snippet by reading its colours back out
(`page2scene`). Every sequence format is mirrored by hand in `web/tasks.js` and checked token by
token by `tools/check_parity.py`.

The **mode tag comes first** (`<img>` to draw or to look at a picture, `<code>` to write web code),
so the model commits to a task before it reads the instruction. An image is either a run of image
tokens right after `<img>` (something to look at) or text after `<img>` ending in `<sep>`
(something to draw) - one token tells the two apart, and the same three-line rule covers every
task, which is why one small model can do all of them.

At sampling time the image region is enforced (after `<sep>` the sampler is restricted to image
codes, then to the level-2 codes, then forced to `<eos>`), and a minimum length stops the model
from ending a caption or snippet before it started.

Why two levels of image tokens: 64×64 with 4×4 patches alone is a flat 16×16 mosaic, which is what
makes tiny image models look like 1990s sprites. Level-2 stores the residual inside each patch, so
decoding restores genuine per-2×2 detail with full colour resolution — the 64×64 test image in
`samples/synth_sheet.png` (left: original, right: token round-trip) shows the difference. If you
want *more* detail, raise `--img-px` or lower `--patch`; token count (and time) grows quadratically.

## 6. Export to the web bundle

```
python src/export_web.py checkpoints/latest.pt -o web_model --verify
```

Writes `web_model/{model.json, weights.int8.bin, weights.int8.bin.gz, tokenizer.json, imagecodec.json}` —
symmetric per-tensor int8, ~1 byte per parameter, plus everything the JS runtime needs (architecture,
token layout, codec constants, vocab merges). The `.gz` is the same bytes compressed; the browser
fetches that one and inflates it in memory, so the served model is smaller without changing a single
number the model computes.

### How small is the served model? (measured on the shipped 1.07M-param demo)

| bundle | banner | on the wire (`.gz`) | quality vs int8 |
|---|---|---|---|
| `--bits 8` (default) | `weights.int8.bin` 1.02 MiB · 1.000 B/param | **0.87 MiB** | reference |
| `--bits 4 --group 64` | `weights.int4.bin` 0.54 MiB · 0.531 B/param | 0.35 MiB | **51.8% mean agreement**, 4 of 8 prompts identical |

int8 is the smallest setting that keeps the model's behaviour exactly: over 8 prompts sampled greedy
and with temperature, `tools/quant_compare.mjs` reports byte-identical continuations and identical
pixels. Below 8 bits a 1M parameter model has no slack — the logits move enough to change the greedy
choice (measured on this demo: `max |Δlogit|` 0.55, mean continuation agreement 51.8%, one image
completely different), so int4 is off by default. On a bigger model (the 4.6M `small` preset or a
real run) int4 usually holds up; measure it before shipping:

```
python src/export_web.py checkpoints/latest.pt -o web_model_int4 --bits 4 --group 64
node tools/quant_compare.mjs web_model web_model_int4    # identical outputs? exit code 1 if not
```

`--no-gzip` skips the compressed copy (for hosts that already compress, or if you want the folder to
hold only raw files). The gzipped download is only ~11% smaller on this model because int8 weights
are close to random; the bigger the model, the more gzip helps (the same ratio held for the 4.6M one).

Total download for the shipped demo: **0.87 MiB weights + 10 KB model.json + 10 KB tokenizer**, and
the raw 1.02 MiB is materialised in the visitor's RAM, not on your host's bandwidth cap.

Verify that the JS runtime and the trainer agree exactly:

```
python tools/check_parity.py checkpoints/tokenizer.json web_model
```

That compares the image codec encode/decode, the BPE ids, all five training sequence formats, greedy
generation ids, and transformer logits against the int8-dequantised PyTorch model - see section 8.

## 7. Inference

**Browser (the app you deploy).** `node web/server.js` → <http://localhost:3000>. The page is a chat
UI: the left rail holds the tool list (*Auto*, image, code, **fix an error**, ASCII → website,
describe, …) and the **Gallery** button that opens every file the chat created, while the
conversation is on the right with the composer at the bottom. The composer carries the **model
selector on the right** (`◈ SGM-3M-code ▾` — it opens the model list; the bundle loads on your first
message), the **stream on/off** toggle and the send button; the sampling panel on the left has the
temperature/top-k/top-p/seed sliders plus **max new tokens** (default 512). **＋** or drag-and-drop
attaches files — up to **12 per chat**, **4 MB each**, **32 MB per
chat** including message text; when a limit is hit the **oldest file is deleted first**. A chat with
**24 h of inactivity is cleared** (on load, and every minute while the page is open), and `☰` lists
the recent chats. `◐` switches **dark / light** (remembered). *Auto* routes by the prompt —
“make a snake game” → code, **naming a language** ("python to reverse a string") → code even without
a verb, an error message plus an attached file → **fix**, “turn this ASCII art into a website” →
ascii→code, “turn this website into an image preview” → code→image, otherwise an image (that table
is in `web/code.js` and unit-tested, because getting it wrong is invisible: the wrong tool still
answers). Messages keep their tools: *Copy*, *Preview as page* (sandboxed iframe), *Download PNG*.

Two footnotes can appear under an answer, and both are there to stop the model's limits looking like
bugs: the answer hit the token budget (raise it and ask again), or you asked for another language
and it answered in web markup anyway (see *Honest caveats* in `RESULTS.md` — the 1M/3M bundles are
trained mostly on html/css/js, and no prompt makes them write Python).

Nothing leaves the tab: no CDN, no web fonts, no analytics, no API, no WebSocket. `web/engine.js`
only fetches model files from the same folder the page was served from (a model download is a plain
`GET` to your own server), so Render's free tier needs nothing but static hosting. Chats and uploads
live in IndexedDB (`web/store.js`), not on the server. The model picker switches bundles, and
`model.json` records which `tasks` a checkpoint was trained for: it shows as a badge next to the
title, and tools the model never learned are greyed out. The page fetches
`model.json` + `weights.int8.bin` + `tokenizer.json`, dequantises once (~40 ms for 1M params) and
runs the transformer in a Web Worker with a KV cache. Nothing is uploaded; the model runs on the
visitor's machine, which is why a free host only has to serve static files. The UI lets you
generate images, caption/ASCII-ify an uploaded picture, generate HTML/CSS/JS and preview generated
HTML in a sandboxed iframe.

`web_model/` in this repo ships a **demo run of ~18 minutes on a free 2-core CPU** — 2,437 steps / 4.6M
tokens, **1,069,920 parameters**, 32px with 64 image tokens per picture, 96-dim / 4-layer / 512 context,
final loss 0.087, served as int8 (0.87 MiB on the wire). It is trained on a deliberately tiny set of
**8 procedural scenes** (`--synth-n 8 --vocab 1200`) so that it can genuinely memorise them, and it does:
`python tools/demo_check.py` scores **100% of image codes exactly right** for all eight captions
(picture reproduced code-for-code) and **87 of 89 caption words** on the image → caption task. The
example buttons in the UI are those eight captions. Anything outside the set gives shapes from the
same visual grammar, not the picture you asked for — a real run wants `--synth-n 4000` or more, or your
own images (`python src/train.py --preset small --synth-n 4000`). Overwrite the bundle with your own
`python src/export_web.py checkpoints/latest.pt -o web_model`.

Code generation is in the same mixed diet but only gets ~15% of a 1M-parameter budget, so the demo
emits HTML/CSS-shaped text (`<!DOCTYPE html> ... </head>`) rather than a correct page. Code is where
a CPU run pays off best, so train it on purpose: `train_code.cmd` (or
`python src/train.py --mix code --preset small --code-dataset codeparrot/github-code`) gives the
code task the whole model instead of 15% of it, and the combination tasks (`--mix multi`) then let
the same model go ascii → website, image → website and website → image preview. Note the demo
`web_model/` in this repo predates the combination tasks, so its badge lists the five original
tasks; a `multi` run gets all nine.

**Node CLI** (any machine, handy before deploying):

```
node web/cli.mjs --model web_model --task image --prompt "a white car" --out car.png
node web/cli.mjs --model web_model --task ascii --image photo.png
node web/cli.mjs --model web_model --task code  --prompt "write css for a dark page"
```

**Deploy free — Render.** Push this repo to GitHub, then *New → Web Service → Build command*
`npm install` (there are no dependencies) *→ Start command* `node web/server.js`, instance type
*Free*. Render injects `PORT`, which `web/server.js` already honours, and binds `0.0.0.0`. A
*Static Site* also works (no server at all): publish the repo and use `/web/index.html` — the
model files are fetched relatively, so nothing else changes. Free instances sleep when idle, so
the first request after a while is slow; the model download then comes from the same origin.

**Deploy free — HelioHost.** Create a Node.js app pointing at `web/server.js` (their app manager
asks for a startup file), or simply upload the whole folder into `public_html` and use it as
static files — no Node process is required for the browser app.

Keep the deployment to one model: `web_model/` holds one 1–15 MiB blob, well inside free-tier
limits. Preview and production both serve the same files, so a redeploy is just a new push.

## 8. Verifying the build (no browser needed)

`npm test` runs the JavaScript ones; the list below is the whole set, and all of them pass on this
codebase:

```
node tools/check_ui.mjs            # markup/JS wiring, task buttons, "no external requests" promises
node tools/check_code.mjs          # tool routing, the trained code phrasing, the language notes
node tools/check_http.mjs          # the page, /api/models, /api/downloads, a model zip that loads
node tools/check_web_tasks.mjs     # all 11 task paths x every bundle, through web/tasks.js
node tools/check_gallery.mjs       # transcript -> created files (names, extensions, byte counts)
node tools/check_zip.mjs           # zip writer round-trip (also read back by Python's zipfile)
node tools/check_models.mjs        # bundle naming + manifest/discovery merge
node tools/check_desktop.mjs       # desktop bootstrap, staged app, the built .exe
python tools/check_tasks.py --mix multi --context 1024                   # every task format builds, and fits
node tools/check_store.mjs                                              # 32 MB / 12 file / FIFO / 24 h rules + chat titles
python tools/fix_check.py                                              # is "fix this i got an error" learnable?
python tools/demo_check.py checkpoints_demo2/latest.pt                  # does the shipped demo really work?
python tools/overfit_test.py                                            # is "caption -> image" actually right?
python tools/check_parity.py checkpoints/tokenizer.json web_model       # Python trainer vs JS runtime
python tools/quant_report.py checkpoints/latest.pt                      # int8 vs int4 size/quality sweep
node tools/quant_compare.mjs web_model web_model_int4                   # do two exports behave the same?
node tools/check_http_load.mjs http://127.0.0.1:3000/web_model web_model  # gzip transit path
node web/cli.mjs --model web_model --task image --prompt "a white car" --out samples/car.png
```

* `tools/overfit_test.py` memorises eight procedural scenes and then generates them again from the
  caption alone. It reports teacher-forced image-code accuracy (must be ~100%) and the exact match
  of the generated image codes (must be ~100% on an overfit set). It also asserts that stepping the
  model token by token through the KV cache equals one full forward pass - the two classic silent
  failures in a from-scratch implementation, both caught here.
* `tools/check_store.mjs` exercises the storage policy directly (pure functions): the 13th file
  evicts the oldest, a full chat frees space oldest-first, an upload larger than the whole budget is
  refused without evicting anything, message text counts toward the budget, and the 24 h boundary is
  exact at 23 h vs 25 h.
* `tools/check_tasks.py` builds one training sequence per task kind and checks the ids are inside
  the vocabulary, the sequence fits the context and `x=seq[:-1] / y=seq[1:]` batches line up — the
  cheap way to catch a broken format before an eight-hour run.
* `tools/check_parity.py` proves the JS runtime and the Python trainer agree bit for bit: image
  codec encode/decode, BPE ids, all eight training sequence formats (including the three
  combination tasks), the JS image→ascii converter against `src/ascii_art.py`, greedy generation
  ids, and transformer logits against the int8-dequantised PyTorch model.
* `tools/quant_compare.mjs` runs two exported bundles through the *real* inference path
  (`web/tasks.js`) on image / caption / ascii / code / text prompts and prints the mean continuation
  agreement, so "int4 still works" is a measurement rather than a hope. `tools/check_http_load.mjs`
  loads a bundle over HTTP with the gzip transit and asserts the browser path is bit-identical to
  reading the files from disk.

* `tools/check_web_tasks.mjs` is the end-to-end one: it runs every task the UI can route to
  (including `fix` with and without an error message, and `ascii2code` from pasted art) through the
  real `web/tasks.js` with each shipped bundle, checks the answer shape, and checks that the tasks
  which need an image - or a snippet - refuse to invent one.
* `tools/check_http.mjs` starts the server on a free port and walks the routes the browser uses
  (`/`, `/web/*`, `/api/models`, `/api/downloads`, `/api/download/model/<id>`), then unzips the
  downloaded model in memory, generates with it and asserts the tokens match the bundle on disk -
  a download that cannot reproduce the local model is a broken download.
* `tools/check_gallery.mjs` / `check_zip.mjs` / `check_models.mjs` pin the three new pieces of pure
  logic: gallery file names and counts, the zip format (round-tripped plus a Python `zipfile`
  cross-check), and bundle naming/merging (1.07M params -> `SGM-1M`, a planned bundle that appears
  on disk becomes selectable).
* `tools/check_ui.mjs` and `tools/check_desktop.mjs` cover the parts a browser would otherwise have
  to prove: every `$('id')` in `app.js` exists, every tool button has a task entry, the model
  selector is inside the composer (not the rail) and opens the model list, the stream toggle and the
  token slider are wired, your own prompt is rendered when it is sent, the page loads
  nothing from a CDN, only same-origin fetches are used, the desktop bootstrap resolves its root in
  both layouts, and `SGM.exe` is a real x86-64 PE executable when a build exists.
* `tools/check_code.mjs` is the routing table as data: 15 prompts -> the tool they must reach (the
  ones that used to go wrong — "python to reverse a string" landing on the image task, "turn this
  website into an image preview" being swallowed by the code rule), plus the phrasing rewrite and
  the language notes. `tools/shellcmd.mjs` is why the Windows build no longer dies with
  `spawnSync npm.cmd EINVAL`: the command lines (`.cmd` shims, quoting, a checkout path with a
  space) are built by a pure function, and `tools/check_desktop.mjs` checks them on Linux.

If you change `src/tasks.py` or `src/imagecodec.py`, run both - the JS side mirrors them by hand in
`web/tasks.js` and `web/engine.js`.

## 9. Troubleshooting

* **Out of memory while training** → lower `--batch-tokens` (2048 → 1024) and/or `--loss-chunk`.
  The loss is computed in chunks precisely so a padded batch cannot blow up the `B×T×vocab` logits.
* **Too slow** → `--no-detail` (4× fewer image tokens), `--patch 8`, smaller `--vocab`, or the `xs`
  preset. Watch the `tok/s` readout in the progress bar and size `--epoch-minutes` around it.
* **"image sequences need N tokens but max_seq_len is M"** → raise `--context` or shrink the image
  (`--img-px`/`--patch`/`--no-detail`).
* **"checkpoint was trained with a different image codec shape"** → you resumed with different
  `--img-px/--patch/--no-detail`; pass the original flags or use a fresh `--workdir`.
* **Model output is gibberish** → normal for the first few hundred steps, and for the 30-second
  smoke run. Watch `checkpoints/log.csv` and `samples/` instead of guessing.
* **`datasets` download fails / dataset is gated** → the synth corpus still trains (drop
  `--hf-dataset`/`--code-dataset`); for gated sets run `huggingface-cli login` first.
* **Page says "cannot start the worker"** → you opened `index.html` directly from disk. ES-module
  workers need an HTTP origin: run `node web/server.js` (or `serve.cmd`) and use
  <http://localhost:3000>. Any static host works too.
* **Windows console mojibake** → the progress bar is plain ASCII on purpose; run inside Windows
  Terminal for the best view.

## 10. The app: models, gallery, downloads, desktop

### Model bundles

Every folder with a `model.json` next to `web/` is a bundle: `web_model*` or `models/<name>/`
(`export_web.py --name sgm-10m` writes the latter). `node tools/make_models_json.mjs` regenerates
`web/models.json`, which is the checked-in list the picker reads, and `/api/models` additionally
scans the disk so a fresh export shows up without editing anything:

```
python src/export_web.py checkpoints_10m/latest.pt --name sgm-10m --verify
node tools/make_models_json.mjs        # SGM-10M appears in the picker
```

The names are derived from the parameter count (1.07M → `SGM-1M`, ~89M → `SGM-100M`) and the
quantisation (`SGM-1M-int4`). Presets for those sizes are in `src/model.py` (`--preset 10m`,
`--preset 100m`) with ready-made scripts: `train_10m.cmd` / `export_10m.cmd`,
`train_100m.cmd` / `export_100m.cmd`. They are real runs, not demos — size them with the table in
section 3. A bundle declared but not built yet still shows in the picker, greyed out and marked
"not exported yet", so the option is visible without pretending it works.

### Downloads

`node tools/make_downloads.mjs` pre-builds `downloads/<name>-<release>.zip` (release from the
`VERSION` file, currently `v1`); `web/server.js` builds and caches the same zips on demand:

```
GET /api/models                     available + planned bundles
GET /api/downloads                  everything you can download, with file names
GET /api/download/model/sgm-1m      the model bundle as a .zip
GET /api/download/app               the desktop build
```

Bump `VERSION` to `v2` and the same commands publish `sgm-1m-v2.zip` etc. — every artifact name
carries the release from now on. A model zip holds `model.json`, `tokenizer.json`,
`imagecodec.json`, the weights and a `README.txt`, so the downloaded folder can be dropped next to
`web/` and picked up by the picker unchanged (the HTTP check loads the zip's weights and asserts it
generates identically to the local bundle).

### Gallery

`web/gallery.js` turns the transcript into files: `sgm-1-image.png`, `sgm-2-code.html`,
`sgm-3-ascii.txt`, … numbered in creation order, plus the chat's uploads on a second tab.
*Download all (.zip)* packs them with `web/zip.js` (deflate via `CompressionStream`, standard zip
readable by Explorer/Finder/unzip — `tools/check_zip.mjs` round-trips it through Python's
`zipfile`). Nothing is added to storage: the gallery is a view over the chat that is already saved.

### Desktop app

```
cd desktop && npm install       # once (~120 MB of Electron)
npm start                       # the same UI in a window, serving web/ on a loopback port

node tools/build_desktop.mjs --platform=win32      (Windows: desktop\build.cmd)
```

The build stages a self-contained folder (`main.cjs`, `boot.cjs`, `web/`, the bundles, `VERSION`),
packages it into `desktop/dist/SGM-win32-x64/SGM.exe` and zips that to
`downloads/sgm-desktop-v1.zip`, which the web UI offers under **Downloads**. `desktop/main.cjs` is
thin on purpose: it starts `web/server.js` and loads the URL, so there is one UI, not two. Details in
`desktop/README.md`.

## 11. Files

```
src/vocab.py         token-space layout (shared with the JS runtime)
src/imagecodec.py    image <-> token codec (level-1 colour + level-2 residual)
src/ascii_art.py     deterministic image -> ASCII targets
src/bpe.py           byte-level BPE trainer + JS-compatible JSON format
src/model.py         RMSNorm/RoPE/SwiGLU decoder, chunked loss, KV cache
src/synth.py         procedural captioned scenes + web-code templates
src/data.py          corpora (synth / folder / HuggingFace datasets), epoch plans, batching
src/tasks.py         sequence builders for every task
src/train.py         trainer: progress bar, timed checkpoints, resume, validation, previews
src/generate.py      sampling + task glue (also the Python CLI)
src/export_web.py    int8 export bundle
web/engine.js        JS transformer + tokenizer + image codec (no dependencies)
web/tasks.js         task glue (combination tasks, fix task, image->ascii) shared with the CLI
web/worker.js        Web Worker wrapper (owns the loaded bundle, reports which model answered)
web/index.html       chat UI shell (model picker, tool rail, gallery + downloads modals, composer)
web/style.css        chat UI styling + dark/light themes
web/app.js           chat UI logic: routing, uploads, limits, themes, gallery, rendering
web/store.js         IndexedDB chats/uploads: quota, FIFO eviction, 24 h expiry, chat titles
web/code.js          prompt understanding: tool routing, the trained code phrasing, language notes
web/gallery.js       transcript -> created files (names, extensions, byte counts)
web/models.js        model registry helpers (names from parameter counts, manifest merge)
web/models.json      the checked-in bundle list the picker reads
web/zip.js           dependency-free zip writer/reader (gallery, downloads, build tools)
web/png.js           PNG encode/decode + bilinear upscale
web/cli.mjs          Node command line (--list shows the bundles)
web/server.js        static server + /api/models + /api/downloads (Render/HelioHost friendly)
desktop/main.cjs     Electron window + menu (thin: it starts web/server.js)
desktop/boot.cjs     root resolution + local server, Electron-free so it can be tested
VERSION              the release tag in every artifact name (v1, v2, ...)
tools/check_parity.py, tools/parity_js.mjs   Python<->JS parity check
tools/check_tasks.py   every task format builds a valid, context-sized sequence
	ools/check_web_tasks.mjs  all 11 task paths + the budget/streaming/context guards, per bundle
tools/check_store.mjs  the chat storage + chat-title rules
tools/check_gallery.mjs, tools/check_zip.mjs, tools/check_models.mjs  gallery, zip, model picker
tools/check_http.mjs   the page, the APIs and the versioned downloads (loads a downloaded zip)
tools/check_ui.mjs     HTML/JS wiring, task buttons, no-external-request promises
tools/check_code.mjs   the routing table, prompt phrasing and language notes (pure, no browser)
tools/shellcmd.mjs     how the build tools are launched (npm/.cmd/EINVAL on Windows, quoting)
tools/check_desktop.mjs  the desktop bootstrap, staged app and the built .exe
tools/make_models_json.mjs  regenerate web/models.json from the bundles on disk
tools/make_downloads.mjs    pre-build downloads/<name>-<release>.zip
tools/make_bundle.mjs       build the release archive sgm-<release>.zip (source + models + docs)
tools/build_desktop.mjs     stage + package + zip the desktop app
	ools/fix_check.py     trains on the error cases, then measures the fixes it reproduces
tools/overfit_test.py  learns 8 scenes and regenerates them from the caption alone
data/seed_code/      23 hand-written HTML/CSS/JS/Python samples with instructions
FILES.md             what to copy for training vs inference + the changed-file log
setup.cmd train.cmd train_code.cmd train_ui.cmd export.cmd serve.cmd  Windows helpers
train_10m.cmd export_10m.cmd train_100m.cmd export_100m.cmd          SGM-10M / SGM-100M
package.json         npm start = web/server.js, npm test = every check
```
