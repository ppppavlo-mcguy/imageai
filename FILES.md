# FILES.md — what to copy, and what changed

SGM = **Small Generative Model**. Two independent sets of files: **training** (Python, CPU) and
**inference** (plain JS in the browser / the desktop app). You only need the inference set to run the
chat UI, and only the training set to train. Paths are relative to the project root — keep the folder
names (`src/`, `web/`, `desktop/`, `models/`).

Two extra top-level things matter now:

* `VERSION` — the release tag (`v2`) in **every** published file name: `sgm-1m-v2.zip`,
  `sgm-desktop-v2.zip`, `sgm-v2.zip`. Bump it for the next round of artifacts; nothing else needs
  editing (`web/models.json` is regenerated from it).
* `downloads/` — build output (`node tools/make_downloads.mjs`, `node tools/build_desktop.mjs`); the
  server serves that folder and the UI lists it under **Downloads**. It is git-ignored.

---

## 1. Inference set (run the web app) — 13 web files + one model folder

```
web/index.html          chat UI shell: model pill, tool rail, model/downloads panel, gallery, composer
web/style.css           dark + light themes (CSS variables, no external fonts)
web/app.js              chat logic: routing, uploads, limits, themes, models, gallery, streaming
web/store.js            IndexedDB chats/uploads: 32 MB + 12 files per chat, FIFO eviction, 24 h expiry
web/code.js             prompt understanding: tool routing, the trained code phrasing, language notes
web/gallery.js          transcript -> created files (names, extensions, byte counts)
web/models.js           model registry helpers (names from parameter counts, manifest merge)
web/models.json         the checked-in bundle list the picker reads
web/zip.js              dependency-free zip writer/reader (gallery export, downloads, build tools)
web/worker.js           Web Worker: owns the loaded bundle, keeps the page responsive
web/tasks.js            task prompts incl. the combination tasks and the fix task; image→ascii
web/engine.js           the transformer (int8/int4 weights, KV cache) in plain JS
web/server.js           static server + /api/models + /api/downloads + /api/download/*
web/png.js              PNG encode/decode (+ CLI)
web/cli.mjs             Node CLI: `node web/cli.mjs --list`, --task image|code|fix|ascii|...
```

Model bundles — **one folder per model, all selectable at once**:

```
web_model/            SGM-1M        image · caption · ascii · code · text (the shipped demo)
web_model_code/       SGM-3M-code   code · ascii→site · fix an error
web_model_int4/       SGM-1M-int4   4-bit weights, half the download
models/sgm-10m/       SGM-10M       export_web.py --name sgm-10m      (not shipped yet)
models/sgm-100m/      SGM-100M      export_web.py --name sgm-100m     (not shipped yet)
```

Minimum to say "it works": `node web/server.js`, open `http://localhost:3000` and send a message —
the model loads by itself on that first message.

### Interface rules the app implements (all local, no external requests)

* **Model picker**: `web/models.json` names the bundles, `/api/models` adds whatever is on disk, so a
  new `export_web.py --name …` shows up after `node tools/make_models_json.mjs`. A bundle the model
  was never trained for greys out its tool buttons (nothing is hidden), and a declared-but-unexported
  bundle appears as *not exported yet* instead of failing later.
* **Lazy loading**: nothing is downloaded until the first message of a chat is sent (or *Load now*).
  Switching the picker mid-chat loads the other bundle on the next message.
* **Gallery** (`Gallery`): every file this chat created, newest last — PNGs, `sgm-2-code.html`,
  `sgm-3-ascii.txt` — plus the chat's uploads on a second tab. *Download all (.zip)* packs them in the
  tab with `web/zip.js`; nothing extra is stored, the gallery is a view over the saved transcript.
* **Downloads** (`⬇`): `/api/downloads` lists every exported model as `sgm-<size>-<release>.zip` and
  the desktop build. A model zip unpacks to a folder you can drop next to `web/` and select as-is.
* **Chat names** come from the first prompt (falling back to the first attached file, then "New chat").
* **Uploads** with **＋** or drag-and-drop. **12 files per chat**, **4 MB per file**, **32 MB per
  chat** counting message text; the 13th (or one that would overflow) deletes the **oldest** file
  first. **24 h of inactivity clears a chat** (swept on load and every minute while the page is open).
* **Dark and light themes** (`◐`), remembered in `localStorage`.
* **No external TCP connections**: no CDN, no fonts, no analytics, no API, no WebSocket.
  `web/engine.js` only fetches model files from the folder the page was served from; chat data stays
  in IndexedDB. Chats/uploads never reach the host.

## 2. Desktop set (the same UI as a window / a .exe)

```
desktop/main.cjs        Electron window + menu (no node integration in the page)
desktop/boot.cjs        Electron-free bootstrap: root resolution, local server, model list
desktop/package.json    devDependencies (electron, @electron/packager) + electron-builder config
desktop/build.cmd       Windows: build the .exe and the download zip in one step
desktop/README.md       how to run, stage, package, and what ends up in the exe
tools/build_desktop.mjs stages desktop/build/app, packages it, zips it into downloads/
tools/check_desktop.mjs verifies all of that without a display
```

## 3. Training set (train on CPU) — `src/` + optional data

```
src/train.py        trainer: progress bar, timed checkpoints, resume, task mix (--mix)
src/data.py         corpora, epoch plans, batches, the task weights
src/tasks.py        every training sequence format (combinations, fix, caption, code, lm)
src/model.py        the transformer + the size presets (xs, tiny, small, base, 10m, 100m)
src/bpe.py          byte-level BPE (identical to the JS Tokenizer)
src/vocab.py        token-space layout shared with the JS runtime
src/imagecodec.py   image <-> codes codec (l1 colour + l2 detail)
src/ascii_art.py    image -> ascii art converter (mirrored by toAscii() in web/tasks.js)
src/synth.py        procedural scenes, web/Python code, combination data, error-fix pairs
src/generate.py     sampling + task glue (previews and the Python CLI)
src/export_web.py   checkpoint -> bundle: `--name sgm-10m` writes models/sgm-10m

requirements.txt                  torch, numpy, pillow, datasets
data/seed_code/*.js|html|css|py   23 instruction-tagged snippets ("make a snake game", python ...)
train_code.cmd / train_code.sh    pure-code run (--mix code, includes error fixing)
train_ui.cmd                      multi-modal run (--mix multi)
train_10m.cmd / export_10m.cmd    SGM-10M preset (embed 320, 6 layers) + its export
train_100m.cmd / export_100m.cmd  SGM-100M preset (embed 768, 12 layers) + its export
tools/check_tasks.py              every task format builds a valid, context-sized sequence
tools/check_web_tasks.mjs         all 11 task paths + budget/streaming/context guards, per bundle
tools/check_code.mjs              routing table, code phrasing and language notes (pure)
tools/shellcmd.mjs                the build command lines (npm/.cmd/EINVAL on Windows, quoting)
tools/check_store.mjs             the 32 MB / 12 file / FIFO / 24 h rules and chat titles
tools/check_gallery.mjs           transcript -> files (names, extensions, bytes)
tools/check_zip.mjs               zip round-trip (+ Python zipfile cross-check)
tools/check_models.mjs            bundle naming and manifest/discovery merge
tools/check_http.mjs              page, APIs, versioned downloads, a downloaded zip that generates
tools/check_ui.mjs                HTML/JS wiring and the no-external-request promises
tools/fix_check.py                proves the "fix this i got an error" task is learnable (99.6% / 83%)
tools/check_parity.py + parity_js.mjs   Python <-> JS parity (tokenizer, codec, tasks, generation)
tools/make_models_json.mjs        regenerate web/models.json from the bundles on disk
tools/make_downloads.mjs          pre-build downloads/<name>-<release>.zip
tools/make_bundle.mjs             build the release archive sgm-<release>.zip
```

`data/` is optional (the procedural corpus in `synth.py` works offline and already contains snake,
calculator, image-preview, Python and error-fix examples).

## 4. Files changed (the log you asked for)

Latest pass — **model selector in the composer, streaming, the 240-token cap, language handling, and
three build bugs**:

| file | what changed |
|---|---|
| `web/index.html` | the model card left the rail: the composer's right side now carries the **model pill** (`#modelPill` opens the model list), the **stream on/off** toggle and the send button; the model `<select>`, *Load now*, the custom-folder box and the downloads rows moved into the **Models & downloads** panel, which got `models` / `desktop app` tabs. The sampling panel gained **max new tokens** |
| `web/style.css` | model pill, stream toggle, selectable model rows, panel sections, the streaming caret |
| `web/app.js` | renders **your own prompt** as soon as it is sent (it used to appear only after a reload); `chooseModel` (select, list row, custom folder); `openDownloads(tab)`/`showDlTab`/`renderDlModels`/`renderDlApp`; streaming (`renderStream`, `setStream`); `maxNew` from the slider; a note under an answer that hit the cap or ignored the language; `route()` moved to `web/code.js` |
| `web/code.js` | **new**: `requestedLang`, `topicOf`, `codeInstruction` ("python to reverse a string" -> "write python for reverse a string", the phrasing the code task was trained on), `looksLikeLang`, `languageNote`, `routeTask`, `alertWords` |
| `web/engine.js` | `onProgress` now hands over the tokens produced so far (**streaming**) and the token budget is clamped to the context window, so a long prompt gets a readable error instead of `context length exceeded` mid-answer |
| `web/tasks.js` | code/fix/ascii2code/img2code budget 240/260 -> `DEFAULT_CODE_TOKENS = 512`; every text answer reports `limit`/`stopped`/`hitLimit` |
| `web/worker.js` | forwards `{type:'stream'}` token text while generating (text tasks only) and passes the limit/hitLimit through to the UI |
| `web/models.json` | regenerated; `release` now follows `VERSION` (`v2`) |
| `tools/make_models_json.mjs` | **bug**: `resolve(new URL('..', import.meta.url).pathname)` built `D:\D:\...` on Windows (that is the `ENOENT` after `export_10m.cmd`); it uses `fileURLToPath` now, and reads the release from `VERSION` |
| `tools/build_desktop.mjs` | **bug**: `execFileSync('npm.cmd', ...)` is `EINVAL` on Windows since Node 18.20/20.12 (the CVE-2024-27980 fix). It goes through `tools/shellcmd.mjs` (a shell command line) now, and names electron-packager relative to `desktop/` so a checkout path with spaces cannot break it |
| `tools/shellcmd.mjs` | **new**: `npmCommand`/`packagerCommand`/`quoteArg`/`commandLine`/`runInShell` - the pure part is unit-tested on Linux |
| `tools/check_code.mjs` | **new (17 + 16 checks)**: the routing table (including the two prompts that used to route wrong), the phrasing rewrite, the language matcher |
| `tools/check_ui.mjs` | + 7 checks: selector position, model list, streaming toggle, token slider, the prompt being rendered |
| `tools/check_web_tasks.mjs` | + 6 checks: the 512 default, a small budget honoured, `hitLimit`, streamed tokens, the context guard |
| `tools/check_desktop.mjs` | + 5 checks: the platform command names, quoting, the single-line install command, no `execFileSync` left; `web/code.js` added to the staged-app list |
| `setup.cmd`, `train.cmd`, `export.cmd`, `train_code.cmd`, `train_ui.cmd`, `train_10m.cmd`, `export_10m.cmd`, `train_100m.cmd`, `export_100m.cmd` | call `.venv\Scripts\python.exe` directly instead of `call .venv\Scripts\activate` (the *"system cannot find the path specified"* before the export ran), `cd /d "%~dp0"`, and a clear message when the venv or the checkpoint is missing |
| `VERSION` | `v1` -> **`v2`** (new artifacts: `sgm-1m-v2.zip`, `sgm-desktop-v2.zip`, `sgm-v2.zip`) |
| `package.json` | `npm test` includes `test:code` |
| `README.md`, `RESULTS.md`, `FILES.md` | this pass documented |

Earlier pass — **SGM rename, model picker, gallery, downloads, desktop app, named chats**:

| file | lines before → after | what changed |
|---|---|---|
| `web/index.html` | 113 → 152 | SGM branding; result preview card → **Gallery** button; model `<select>` + custom-folder details; downloads modal; Clear/Delete kept |
| `web/style.css` | 188 → 240 | model select, gallery grid/tiles, downloads rows, modal + tabs, `unsupported` tools, badge |
| `web/app.js` | 626 → 1038 | model registry + **lazy load on first message**, gallery (build/zip/download), downloads modal, chat titles from the prompt, clear/delete, per-model tool greying |
| `web/store.js` | 168 → 191 | + `titleForPrompt` / `DEFAULT_TITLE`; database renamed `sgm-chat` |
| `web/worker.js` | 40 → 53 | reports `dir`/`name`/`tasks` of the loaded bundle back to the UI |
| `web/server.js` | 49 → 289 | rewritten: importable `createServer`/`startServer`, `/api/models`, `/api/downloads`, `/api/download/model/<id>`, `/api/download/app`, model discovery, `VERSION` |
| `web/gallery.js` | new (87) | transcript → created files: names, extensions, byte counts, zip entries |
| `web/models.js` | new (110) | `SGM-1M`-style names from parameter counts, manifest/discovery merge, URL resolution |
| `web/models.json` | new (67) | the bundle list the picker reads (incl. planned SGM-10M / SGM-100M) |
| `web/zip.js` | new (198) | dependency-free zip writer/reader (deflate, UTF-8 names, crc32), `versionedName()` |
| `web/cli.mjs` | 62 → 72 | SGM naming; `--list` prints the bundles the picker shows |
| `desktop/main.cjs` | new (93) | Electron window + menu around the same local server |
| `desktop/boot.cjs` | new (33) | Electron-free root/server bootstrap so it can be tested headlessly |
| `desktop/package.json`, `desktop/build.cmd`, `desktop/README.md` | new | dev run, packaging, electron-builder config |
| `tools/build_desktop.mjs` | new (138) | stage → package (win32/linux/darwin) → `downloads/sgm-desktop-<release>.zip` |
| `tools/make_models_json.mjs` | new (113) | regenerate `web/models.json` (`--check` for CI) |
| `tools/make_downloads.mjs` | new (68) | pre-build every versioned model zip |
| `tools/check_ui.mjs` / `check_http.mjs` / `check_web_tasks.mjs` | new (78/137/88) | wiring, HTTP + downloads, all task paths |
| `tools/check_gallery.mjs` / `check_zip.mjs` / `check_models.mjs` / `check_desktop.mjs` | new (65/57/80/101) | the new pure logic and the desktop build |
| `tools/check_store.mjs` | 65 → 86 | + chat-title rules |
| `src/model.py` | 200 → 204 | + `10m` / `100m` presets (the sizes the picker offers) |
| `src/export_web.py` | 263 → 271 | + `--name sgm-10m` → `models/sgm-10m`, and the manifest reminder |
| `VERSION` | new | release tag used in every artifact name |
| `.gitignore` | updated | ignores `downloads/`, `desktop/dist/`, `desktop/build/` |
| `README.md`, `RESULTS.md`, `FILES.md` | updated | this pass documented (new sections 8 and 10, SGM naming) |

`web/engine.js`, `web/tasks.js`, `web/png.js` and everything in `src/` except `model.py` /
`export_web.py` are unchanged: the runtime and the task formats were already correct, this pass was
about the interface around them.

Earlier in the project: the chat UI itself; the combination tasks (ascii→code, image→code,
code→image); the fix task (`fix_seq`, 12 reproducible error cases); the storage rules; int8/int4
export.

## 5. Copy order

1. `web/` — all files (`app.js`, `server.js`, `index.html`, `style.css` changed most; `gallery.js`,
   `models.js`, `models.json`, `zip.js` are new).
2. `desktop/` — `main.cjs`, `boot.cjs`, `package.json`, `build.cmd`, `README.md`.
3. `tools/` — the new checks and the two generators, plus `check_http`/`check_ui`/`check_web_tasks`.
4. `src/` — `model.py` (presets) and `export_web.py` (`--name`) if you want the 10M/100M path.
5. Root — `VERSION`, `train_10m.cmd`, `export_10m.cmd`, `train_100m.cmd`, `export_100m.cmd`,
   `FILES.md`, `README.md`, `RESULTS.md`; delete the old `image-ai-*.zip` bundles (they are now
   `sgm-v1.zip`).

## 6. Commands

```bat
:: run the chat UI
serve.cmd                                  :: == node web/server.js -> http://localhost:3000

:: the bundles the picker will offer
node web\cli.mjs --list
node tools\make_models_json.mjs            :: refresh web\models.json after an export

:: downloads: model zips (+ the desktop build, if one has been made)
node tools\make_downloads.mjs

:: the desktop app
cd desktop && npm install
npm start                                  :: the same UI in a window
cd .. && node tools\build_desktop.mjs --platform=win32    :: or: desktop\build.cmd -> downloads\*.zip

:: a bigger bundle: train, then export it under its own name
train_10m.cmd
export_10m.cmd                             :: -> models\sgm-10m, picker shows SGM-10M

:: checks (no browser needed)
node tools\check_ui.mjs && node tools\check_code.mjs && node tools\check_http.mjs && node tools\check_web_tasks.mjs
node tools\check_gallery.mjs && node tools\check_zip.mjs && node tools\check_models.mjs
node tools\check_desktop.mjs
.venv\Scripts\python.exe tools\check_tasks.py --mix multi
.venv\Scripts\python.exe tools\check_parity.py web_model\tokenizer.json web_model
:: or simply
npm test
```

Switch the model in the composer's model list (or the `custom folder` box inside it) to run
`web_model_code` / `models\sgm-10m` instead of `web_model`.
