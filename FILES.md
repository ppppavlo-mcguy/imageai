# FILES.md — what to copy, and what changed

Two independent sets of files: **training** (Python, CPU) and **inference** (plain JS in the browser).
You only need the inference set to run the chat UI, and only the training set to train.
Paths are relative to the project root — keep the folder names (`src/`, `web/`, `data/`).

---

## 1. Inference set (run the web app) — 10 web files + one model folder

```
web/index.html          chat UI shell: tool rail, result preview card, composer
web/style.css           dark + light themes (CSS variables, no external fonts)
web/app.js              chat logic: routing, uploads, storage limits, themes, rendering
web/store.js            IndexedDB chats/uploads: 32 MB + 12 files per chat, FIFO eviction, 24 h expiry
web/worker.js           Web Worker: keeps the page responsive while sampling
web/tasks.js            task prompts incl. the combination tasks and the fix task; image→ascii
web/engine.js           the transformer (int8/int4 weights, KV cache) in plain JS
web/server.js           tiny static server: node web/server.js -> http://localhost:3000
web_model/model.json        architecture, layout, quantisation, "tasks" the model was trained for
web_model/tokenizer.json    BPE merges
web_model/imagecodec.json   image token codec
web_model/weights.int8.bin.gz  the model weights (0.87 MiB, decompressed in the browser)
```
Optional in `web/`: `web/png.js` + `web/cli.mjs` (Node CLI, `node web/cli.mjs --task fix --code-file broken.js --error "SyntaxError…"`),
and the extra bundles `web_model_code/` (pure-code + ascii→code) and `web_model_int4/` (size comparison only).

Minimum to say "it works": `node web/server.js`, open `http://localhost:3000`.

### Interface rules the app implements (all local, no external requests)

* **Result preview** in the left rail shows the last generated image / ascii art of the chat; hidden
  while the chat has no such result.
* **Uploads** with the **＋** button (or drag-and-drop). **12 files per chat**, **4 MB per file**,
  **32 MB per chat** counting message text too. The 13th file (or one that would exceed the space)
  deletes the **oldest** file first — the newer upload always wins unless it alone exceeds 32 MB.
* **24 h inactivity**: a chat with no activity for 24 hours is deleted on the next load, and while the
  tab stays open a sweep runs every minute.
* **Dark and light themes** (`◐` in the header), remembered in `localStorage`.
* **No external TCP connections**: no CDN, no fonts, no analytics, no API. `web/engine.js` only
  fetches model files from the same folder the page came from. Safe on Render's free tier.

## 2. Training set (train on CPU) — `src/` + optional data

```
src/train.py        trainer: progress bar, timed checkpoints, resume, task mix (--mix)
src/data.py         corpora, epoch plans, batches, the task weights
src/tasks.py        every training sequence format (combinations, fix, caption, code, lm)
src/model.py        the transformer (RMSNorm/RoPE/SwiGLU, tied embeddings)
src/bpe.py          byte-level BPE (identical to the JS Tokenizer)
src/vocab.py        token-space layout shared with the JS runtime
src/imagecodec.py   image <-> codes codec (l1 colour + l2 detail)
src/ascii_art.py    image -> ascii art converter (mirrored by toAscii() in web/tasks.js)
src/synth.py        procedural scenes, web/Python code, combination data, error-fix pairs
src/generate.py     sampling + task glue (previews and the Python CLI)
src/export_web.py   checkpoint -> web_model/ bundle (int8, optional int4)

requirements.txt          torch, numpy, pillow, datasets
data/seed_code/*.js|html|css|py   23 instruction-tagged snippets ("make a snake game", python ...)
train_code.cmd / train_code.sh    pure-code run (--mix code, includes error fixing)
train_ui.cmd                      multi-modal run (--mix multi)
tools/check_tasks.py              every task format builds a valid, context-sized sequence
tools/check_store.mjs             the 32 MB / 12 file / FIFO / 24 h storage rules
tools/fix_check.py                proves the "fix this i got an error" task is learnable (99.6% / 83%)
tools/check_parity.py + parity_js.mjs   Python <-> JS parity (tokenizer, codec, tasks, generation)
```
`data/` is optional (the procedural corpus in `synth.py` works offline and already contains snake,
calculator, image-preview, Python and error-fix examples).

## 3. Files changed (the log you asked for)

Latest pass — chat layout, themes, storage, uploads, error fixing:

| file | lines before → after | what changed |
|---|---|---|
| `web/index.html` | 105 → 113 | left rail = result preview + tool list; chat on the right; ＋ upload; theme + chats buttons |
| `web/style.css` | 150 → 188 | dark **and light** themes via `[data-theme]`; rail, preview card, upload chips, storage meter |
| `web/app.js` | 390 → 626 | multi-file upload with limits, preview pane, theme switch, chat list, 24 h sweep, fix routing |
| `web/store.js` | new (168) | IndexedDB chats/uploads, quota policy, FIFO eviction, expiry, text-vs-binary attachments |
| `web/tasks.js` | 165 → 185 | + `fix` task (snippet + error → fixed snippet) |
| `web/engine.js` | 497 → 498 | + `INSTRUCTIONS.fix` |
| `web/worker.js` | 39 → 40 | passes `error` / `question` |
| `src/tasks.py` | 149 → 169 | + `fix_seq` / `ask_fix_seq` / `FIX_QUESTIONS` |
| `src/synth.py` | 520 → 581 | + 12 real error cases (`error_case`) with broken code, message and fix |
| `src/data.py` | 480 → 491 | + `fix` task kind (`FIX_CASES`) |
| `src/train.py` | 532 → 542 | + `--w-fix`, mixes rebalanced, fix previews |
| `src/generate.py` | 244 → 258 | + `fix` task, `--error`, `--code-file` |
| `tools/check_tasks.py` | 110 | + `fix` in the checked kinds |
| `tools/check_parity.py` / `parity_js.mjs` | 197→202 / 106→107 | + fix-sequence fixture (9 formats total) |
| `tools/check_store.mjs` | new (65) | unit test for the storage rules |
| `tools/fix_check.py` | new (140) | decisive learnability check for the fix task: 99.6% teacher-forced, 83% greedy |
| `README.md`, `RESULTS.md`, `FILES.md` | updated | this pass documented |

Earlier in the project: the chat UI itself was introduced in `web/index.html`, `web/style.css`,
`web/app.js`; the combination tasks (ascii→code, image→code, code→image) in `src/tasks.py`,
`src/data.py`, `src/synth.py`, `src/train.py`, `web/tasks.js`, `tools/check_parity.py`;
`src/export_web.py` gained the `tasks` field in `model.json`; `tools/check_tasks.py` was added.

**Unchanged** (still needed, you already have them): `src/model.py`, `src/bpe.py`, `src/vocab.py`,
`src/imagecodec.py`, `src/ascii_art.py`, `web/server.js`, `web/png.js`, `web/cli.mjs`,
`tools/demo_check.py`, `tools/overfit_test.py`, `tools/quant_report.py`, `tools/quant_compare.mjs`,
`tools/check_http_load.mjs`, `tools/make_results.py`.

## 4. Copy order (you already have `/src` and `/tools`)

1. `web/` — all 10 files (`app.js`, `store.js`, `index.html`, `style.css` are the ones that changed
   most; `engine.js`, `tasks.js`, `worker.js` have small additions).
2. `src/` — `tasks.py`, `data.py`, `train.py`, `generate.py`, `synth.py` (and `export_web.py` if you
   have not copied it yet).
3. `tools/` — `check_parity.py`, `parity_js.mjs`, `check_tasks.py`, and the new `check_store.mjs`, `fix_check.py`.
4. `data/seed_code/` — the instruction-tagged snippets.
5. Root — `train_code.cmd`, `train_ui.cmd`, `FILES.md`, `README.md`, `RESULTS.md`.

## 5. Commands

```bat
:: pure code + error fixing (web, python, ascii->website, "fix this i got an error")
train_code.cmd
:: everything (images, captions, ascii, code, three combinations, error fixing)
train_ui.cmd
:: check the formats and the storage rules without training
.venv\Scripts\python.exe tools\check_tasks.py --mix code
node tools\check_store.mjs
.venv\Scripts\python.exe tools\fix_check.py
:: bundle a checkpoint for the browser
.venv\Scripts\python.exe src\export_web.py checkpoints_code\latest.pt -o web_model_code
:: ask a code model to fix a file
node web\cli.mjs --model web_model_code --task fix --code-file broken.js --error "SyntaxError: Unexpected end of input"
:: run the chat UI
serve.cmd            ::  == node web\server.js  ->  http://localhost:3000
```
Switch the rail's `model folder` box to `web_model_code` for the pure-code model.
