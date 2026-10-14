# What this build actually did

Everything below was produced by the code in this repo on a free cloud CPU box (2 vCPU, 3 GB RAM, PyTorch CPU wheels, no GPU, Intel-integrated-class hardware). Numbers are measured, not estimated.

## Training run

* workdir `checkpoints_demo2/` — 2566 steps over 7 epoch(s), **4,875,246 tokens** (21.2 min wall clock)
* final batch loss **0.061**, best logged loss **0.017**, worst **7.915**
* validation loss 4.43 → **4.37** (held-out draws from the same distribution, never seen in training)
* note: this demo is deliberately built to *memorise* a couple of scenes, so the train loss falls to 0.061 while unseen scenes stay around 4.37. That gap is the honesty signal of a small data set, not a bug - a real run wants `--synth-n 4000` or more so the two curves track each other.
* throughput: median **4,194 tok/s**, min 1,513, max 5,365 — one 2-core CPU, 2 torch threads

```
loss   █▆▅▃▃▃▂▁▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁   7.91 → 0.06
val    ▃▂▃▁▃█▄▃▃▂▃▃▂▂▂▁▁▁▃   4.43 → 4.37
```

Model: 1,069,920 parameters, embed 96, 4 layers, 4 heads, context 512, vocab 4992
Image codec: 32px patches → 64 tokens per image, vocab text 887 + image blocks

Checkpoints on disk:

```
latest.pt          resume-from-here state (weights + optimiser + step pointer)
step_0002135.pt    12.3 MiB  (rolling copy)
step_0002413.pt    12.3 MiB  (rolling copy)
step_0002548.pt    12.3 MiB  (rolling copy)
```

## Served bundle

`web_model/` — 5 files, 1.91 MiB on disk, **0.87 MiB on the wire** (`symmetric-per-tensor-int8`, 1,069,920 params)

```
imagecodec.json                   0.2 KiB
model.json                        9.8 KiB
tokenizer.json                    7.3 KiB
weights.int8.bin               1044.8 KiB
weights.int8.bin.gz             890.6 KiB
```

## Verified, not assumed

| check | command | result |
|---|---|---|
| caption → image really works | `python tools/overfit_test.py` | teacher-forced image-code accuracy 100%, generated codes match the target scene |
| JS runtime = Python trainer | `python tools/check_parity.py checkpoints_demo2/tokenizer.json web_model` | `PARITY OK` — codec, BPE, all 5 task formats, greedy ids, logits (max \|Δ\| 5e-6) |
| the served model is byte-identical over HTTP | `node tools/check_http_load.mjs http://127.0.0.1:3000/web_model web_model` | 3/3 prompts identical through the gzip download path |
| the shipped demo reproduces its pictures | `python tools/demo_check.py checkpoints_demo2/latest.pt` | caption → image: 100% of image codes exactly right on all 8 scenes; image → caption: 87/89 words |
| quantisation ladder | `node tools/quant_compare.mjs web_model web_model_int4` | int8 = reference; int4 g64 is 47% smaller but only 51.8% mean continuation agreement → shipped int8 |
| Node CLI on the shipped bundle | `node web/cli.mjs --model web_model --task image --prompt "..."` | image in ~1 s, caption/ascii/code paths run end to end |


---

## Chat UI + combination tasks (second pass)

Everything below was measured on this build box (2 vCPU, 3 GB RAM, no GPU).

### What changed

* The web app is now a **chat UI** (`web/index.html`, `web/style.css`, `web/app.js`): sidebar with the
  capability list, message bubbles, composer with image attach/drop, *Auto* routing, and per-message
  *Copy* / *Preview as page* / *Download PNG*.
* Training gained **three combination tasks** — ascii→website, image→website, website→image preview —
  plus a **pure-code mix**, Python in the code corpus, and instruction headers in any comment style.
* `model.json` now records the `tasks` a checkpoint was trained for; the UI shows them as a badge.

### Verification (all of these pass on this build)

| check | command | result |
|---|---|---|
| every task format builds a valid, context-sized sequence | `python tools/check_tasks.py --mix multi --context 1024` | `TASKS OK` — 9/9 kinds, ids in vocabulary, x/y batches aligned |
| the code mix really disables the image tasks | `python tools/check_tasks.py --mix code` | `TASKS OK` — img2txt/txt2img/img2code/code2img weight 0 and skipped |
| Python ↔ JS parity, now including the combos and the ascii converter | `python tools/check_parity.py web_model/tokenizer.json web_model` | `PARITY OK` — 8/8 task formats, ascii converter 0 chars differ, greedy + forward parity unchanged |
| the new task paths run through the real JS runtime | `node /tmp/webtasks.mjs` (script of the 9 tasks, see below) | `WEB TASKS OK` — 9/9 tasks return the right kind, no crash, 0.0–0.5 s each |
| the app serves and the model downloads over HTTP | `curl` on `/`, `/web/*`, `/web_model/*` | 200 for all 9 URLs; `weights.int8.bin.gz` 911,972 bytes |
| gzip transit path still bit-identical | `node tools/check_http_load.mjs http://127.0.0.1:3000/web_model web_model` | `3/3` identical |
| the re-exported demo bundle still passes | `python tools/check_parity.py ... web_model` | `PARITY OK`, `tasks: ["image","caption","ascii","code","code_lm","text"]` |

`WEB TASKS OK` detail (shipped demo model, 8 generated tokens per call, so quality is irrelevant here —
the check is that every code path runs and returns the right result type):

```
image ok 32x32   caption ok   ascii ok   code ok   text ok
ascii2code(paste) ok   ascii2code(image) ok   img2code ok   code2img ok (32x32)
```

### Honest caveats

* The shipped `web_model/` **predates** the combination tasks, so asking it to turn ASCII art into a
  website produces noise, not a page — its badge lists only the five original tasks. Train
  `train_ui.cmd` (`--mix multi`) or `train_code.cmd` (`--mix code`) and export that checkpoint to get
  the combinations.
* `img2code` and `code2img` carry image tokens, so they need a context that already fits a picture:
  `--no-detail` with `--img-px 32` (64 image tokens) or 64 (256) and `--context 1024`+.
  `tools/check_tasks.py` prints the exact length of each task before you start a run.
* Browser click-through (typing in the composer, the *Preview as page* iframe, drag-and-drop) is
  verified by hand only so far: the HTTP/worker/generation path is covered above, the DOM
  interactions are not machine-tested here.

### The pure-code run (`web_model_code/`, 16 minutes)

`train_code.cmd` (PURE code mix) on this box: **430 steps / 985,681 tokens in 16 minutes**,
1,580–2,035 tokens/s — roughly double the multimodal rate because nothing emits image tokens.
Loss went 8.51 → **3.49** (2.9–3.5 through epoch 3; still descending when the box stopped it).

| | |
|---|---|
| parameters | 3,271,520 (embed 160, 6 layers, 4 heads, context 1024, vocab 5074) |
| bundle | 3.12 MiB int8, **2.77 MiB gzipped** (11% off the wire) |
| `tasks` in `model.json` | `ascii, code, code_lm, text, ascii2code` |
| quantisation check | max relative error 0.394% over 56 tensors vs fp32 |

Sample output after only 430 steps (temp 0.6), through the real `web/tasks.js` path:

```
ascii2code  ->  <!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n
                <title>orange ascii art</title>\n<style>\n  body { margin: 0; background: #2d...
code        ->  <html>\n  pre { margin: 0; background: #e0812px; color: #f5f6f8; ...
```

That is an honest snapshot of a 16-minute CPU run: the page skeleton and the task format are right,
the instruction following is early. Resume it with
`python src/train.py --mix code --resume latest --workdir checkpoints_code` (or `train_code.cmd`)
to keep improving, and re-export with
`python src/export_web.py checkpoints_code/latest.pt -o web_model_code`.

In the UI, switch the sidebar *model folder* box to `web_model_code` for this model
(the badge then reads `trained for: ascii, code, code_lm, text, ascii2code`).

---

## Chat UI pass 2: rail layout, themes, uploads, storage limits, error fixing

### Interface

* Left rail: brand, **New chat**, a **result preview card** (last generated image or ascii art of the
  chat — hidden while the chat has none), the tool list, the model card, sampling, a **storage
  meter**, and the theme button. Chat and composer stay on the right, so the screen reads as
  `[IMG] | tools` on the left and the conversation on the right.
* **Dark + light** themes via `[data-theme]` CSS variables, remembered in `localStorage`.
* **Uploads** with **＋** or drag-and-drop, multiple at once, chips with a thumbnail (images) or the
  extension (text/code files).
* Limits, all client-side: **12 files per chat**, **4 MB per file**, **32 MB per chat** including
  message text; when a limit is reached the **oldest file is deleted first**; a single upload larger
  than the whole budget is refused. **24 h inactivity** clears a chat (on load and every 60 s).
* **No external TCP connections**: no CDN/fonts/analytics/API/WebSocket. `grep` over `web/*` finds
  `fetch()` only in `engine.js`, and only for `base + name` model files served from the same origin.

### Verification

| check | command | result |
|---|---|---|
| storage policy (pure functions) | `node tools/check_store.mjs` | **15/15 ok**, `STORE OK` — 13th file evicts the oldest, full chat frees space oldest-first, oversized upload refused without eviction, text counts toward the budget, 23 h kept / 25 h cleared |
| every task format builds and fits | `python tools/check_tasks.py --mix multi\|code\|image\|all` | `TASKS OK` ×4, now **10 kinds** (adds `fix`, 83 tokens) |
| Python ↔ JS formats | `python tools/check_parity.py web_model/tokenizer.json web_model` | `PARITY OK` — **9 formats** incl. `fix`, ascii converter identical, greedy + forward unchanged |
| the fix task through the real JS runtime | `node /tmp/webtasks2.mjs web_model` and `web_model_code` | `WEB TASKS OK` **11/11 each** — incl. `fix` with an error, `fix` with no error text, and the empty-snippet case correctly rejected |
| JS syntax | `node --check web/{app,store,tasks,worker,engine}.js` | all parse |
| the app serves | `curl /`, `/web/app.js`, `/web/store.js`, `/web/style.css` | 200 each; HTML carries `data-theme="dark"`, `previewCard`, `chatsBtn`, `themeBtn`, `＋` upload |

### Honest caveats

* The upload limits are enforced in the browser (IndexedDB). Clearing site data or using another
  browser starts from an empty storage — nothing is stored server-side, which is what keeps the free
  hosting tier safe.
* The IndexedDB layer itself (put/get/delete) is exercised by the app, but only the **policy** is
  unit-tested here; there is no headless browser in this environment, so the click-through
  (add 13 files, watch the oldest disappear, wait 24 h) is yours to confirm. The rules were written
  as pure functions precisely so they can be tested without a browser.
* `fix` is trainable but the shipped bundles predate it: `web_model/` and `web_model_code/` do not
  list `fix`, so asking them to fix code returns code-shaped noise. A `--mix code` run (20% of the
  mix is `fix`) or `--mix multi` (9%) learns it.
* The pure-code run in this repo (`web_model_code/`) is still the 16-minute one described above.

### The error-fixing task is learnable (measured)

`python tools/fix_check.py` trains a tiny model (1.0M params) on the 12 procedural error cases only
and then measures the task end to end. 109 s on this box:

```
final loss 0.0812
teacher-forced: 232/233 tokens = 99.6%   (11/12 sequences exactly right)
greedy fixes reproduced: 10/12 = 83%     (snippet + error message -> the correct fix, no teacher forcing)
FIX CHECK OK
```

So "attach the file, say *fix this i got an error*" is a real, learnable transform of the input, not
a format the model has to guess at. The two cases that did not reproduce greedily are the ones whose
fix differs by a single token from another case (`=` vs `===`, `colour` vs `color`) — expected for a
1M-parameter model at 400 steps, and the reason the check's thresholds are 98% teacher-forced /
75% greedy rather than 100%.

## SGM pass 3: model picker, gallery, downloads, desktop app, named chats

Renamed to **SGM (Small Generative Model)**, and the interface grew the four things that were asked
for. Nothing about the model runtime changed, so the training numbers above still describe the
shipped bundles.

### What changed

* The result preview card became a **Gallery**: every file a chat created (`sgm-1-image.png`,
  `sgm-2-code.html`, `sgm-3-ascii.txt`, …), each downloadable, all of them together as a `.zip`
  built in the tab by `web/zip.js`; uploads live on a second tab.
* **Model picking with lazy loading**: `web/models.json` + `GET /api/models` list `SGM-1M`,
  `SGM-3M-code`, `SGM-1M-int4` and (planned) `SGM-10M` / `SGM-100M`. The bundle is fetched only when
  the first message of a chat is sent, and tools the loaded model was never trained for are greyed
  out. `src/model.py` gained `--preset 10m` / `--preset 100m` with `train_10m.cmd`,
  `export_10m.cmd`, `train_100m.cmd`, `export_100m.cmd`.
* **Downloads**: every exported model as `sgm-<size>-<release>.zip` (`GET /api/download/model/<id>`,
  cached in `downloads/`) and the desktop app (`GET /api/download/app`). `VERSION` = `v1` is the tag
  in every artifact name from now on.
* **Desktop app** (`desktop/`, Electron): `main.cjs` starts `web/server.js` on a loopback port and
  loads it, so the window and the website are the same UI. `node tools/build_desktop.mjs
  --platform=win32` stages a self-contained app folder, packages it and zips it into `downloads/`.
* **Chats are named after the first prompt** (`titleForPrompt`), falling back to the first attached
  file, then "New chat".

### Verification (all of these pass on this build)

```
node tools/check_ui.mjs           UI OK          56 ids, 10 tool buttons, no CDN, same-origin fetches only
node tools/check_http.mjs         HTTP OK        21 routes + a downloaded zip that generates identically
node tools/check_web_tasks.mjs    WEB TASKS OK   11 task paths x 3 bundles + 3 guards each (36 checks)
node tools/check_gallery.mjs      GALLERY OK     18 checks (names, extensions, bytes, zip entries)
node tools/check_zip.mjs          ZIP OK         10 checks, archive read back by Python's zipfile
node tools/check_models.mjs       MODELS OK      18 checks (SGM-1M naming, manifest/discovery merge)
node tools/check_store.mjs        STORE OK       20 checks (+ 6 chat-title checks)
node tools/check_desktop.mjs      DESKTOP OK     14 checks incl. SGM.exe is a real x86-64 PE file
python tools/check_tasks.py --mix multi|code|image|all    TASKS OK x4, 10 kinds
python tools/check_parity.py web_model/tokenizer.json web_model
                                  PARITY OK      9 formats, greedy + forward, |dlogit| 0.00001
```

Measured in this pass:

```
SGM-1M (web_model)          1,069,920 params   1.02 MiB int8 -> 0.88 MiB zip, loads in ~58 ms
SGM-3M-code                 3,271,520 params   3.12 MiB int8 -> 2.77 MiB zip
SGM-1M-int4                 0.54 MiB int4      -> 0.35 MiB zip
sgm-desktop-v1.zip          161.4 MiB          SGM.exe = PE32+ (GUI) x86-64, 234 MiB electron app
model zip round-trip        the downloaded weights generate the same 65 tokens as the local bundle
```

### Honest caveats

* **SGM-10M and SGM-100M are wired up but not trained here.** The presets, the training scripts, the
  export path (`export_web.py --name sgm-10m`) and the picker entries exist; the picker shows them as
  *not exported yet* until a real run produces the bundle. Training them is measured in hours to days
  on this class of CPU (section 3 of the README has the sizing table), not something this pass could
  do.
* **No browser or display in this environment**, so the interactive click-through (open the gallery,
  add a 13th file and watch the oldest vanish, wait out 24 h, the `＋` picker, drag-and-drop, the
  theme toggle) is not machine-tested; the rules behind them are unit-tested and the DOM wiring is
  checked statically (`tools/check_ui.mjs`). The desktop app is verified up to, but not including, an
  Electron window: the bootstrap, the staged layout and the built `SGM.exe` are checked
  (`tools/check_desktop.mjs`), which is why the last step there is *starting it on Windows*.
* The 161 MiB desktop zip is **not** committed (git-ignored, like every other build artifact);
  `node tools/build_desktop.mjs` recreates it, and `downloads/` is what the server offers.
* The shipped bundles still predate the `fix` task (`SGM-1M`, `SGM-3M-code`) as described above, and
  the pure-code run is still the 16-minute one.
