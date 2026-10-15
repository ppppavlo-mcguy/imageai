#!/usr/bin/env node
/* Every task the UI can route to, driven through the real browser runtime (web/tasks.js) with the
 * shipped bundles. Whatever the chat sends a worker, this sends here.
 *
 *     node tools/check_web_tasks.mjs                 # all bundles found on disk
 *     node tools/check_web_tasks.mjs web_model       # just one
 *
 * It checks the shape of each answer (image comes back as pixels, code/fix as text), that the
 * image tasks really need an image, and that an empty snippet is refused instead of invented.
 */
import { readFileSync, readdirSync, statSync } from 'node:fs';
import { join } from 'node:path';
import { Engine } from '../web/engine.js';
import { runTask, DEFAULT_CODE_TOKENS } from '../web/tasks.js';
import { PROJECT_ROOT } from '../web/server.js';

let failures = 0;
const check = (name, cond, detail = '') => {
  console.log(`  ${cond ? 'ok  ' : 'FAIL'} ${name}${detail ? '  ' + detail : ''}`);
  if (!cond) failures++;
};

function load(dir) {
  const meta = JSON.parse(readFileSync(join(dir, 'model.json'), 'utf8'));
  const tok = JSON.parse(readFileSync(join(dir, 'tokenizer.json'), 'utf8'));
  const weights = new Uint8Array(readFileSync(join(dir, meta.weight_file || 'weights.int8.bin')));
  return new Engine(meta, tok, weights);
}

/* a small synthetic image, encoded exactly like the app does it (64x64 RGBA) */
function sampleImage() {
  const size = 64;
  const rgba = new Uint8ClampedArray(size * size * 4);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const i = (y * size + x) * 4;
      rgba[i] = 40 + x * 3; rgba[i + 1] = 60 + y * 3; rgba[i + 2] = 200 - x;
      rgba[i + 3] = 255;
    }
  }
  return { rgba, width: size };
}

const only = process.argv[2];
const bundles = readdirSync(PROJECT_ROOT)
  .filter(n => /^web_model/.test(n) && statSync(join(PROJECT_ROOT, n)).isDirectory())
  .filter(n => !only || n === only || join(PROJECT_ROOT, n) === only);

const img = sampleImage();
const CASES = [
  ['image', { task: 'image', prompt: 'generate the image of a white car', maxNew: 40 }, 'image'],
  ['image (empty prompt)', { task: 'image', prompt: '', maxNew: 24 }, 'image'],
  ['caption', { task: 'caption', ...img, maxNew: 12 }, 'text'],
  ['ascii', { task: 'ascii', ...img, maxNew: 40 }, 'text'],
  ['code', { task: 'code', prompt: 'write html for a blue card', maxNew: 40 }, 'code'],
  ['fix', { task: 'fix', code: 'def f():\n    return x', error: 'NameError: name x is not defined', maxNew: 40 }, 'code'],
  ['fix (no error text)', { task: 'fix', code: '<div><p>hi</div>', maxNew: 30 }, 'code'],
  ['ascii2code', { task: 'ascii2code', ascii: ' .:-=+*\n#%@ \n', maxNew: 40 }, 'code'],
  ['img2code', { task: 'img2code', ...img, maxNew: 40 }, 'code'],
  ['code2img', { task: 'code2img', code: '<div class="card">blue</div>', maxNew: 40 }, 'image'],
  ['text', { task: 'text', prompt: 'the little robot waited', maxNew: 24 }, 'text'],
];

for (const name of bundles) {
  const dir = join(PROJECT_ROOT, name);
  console.log(`--- ${name} ---`);
  const engine = load(dir);
  for (const [label, opts, kind] of CASES) {
    try {
      const res = await runTask(engine, { temp: 0, topK: 1, seed: 11, ...opts });
      const ok = res.kind === kind && (kind === 'image'
        ? res.rgba.length === engine.codec.img_px * engine.codec.img_px * 4 && res.ids.length > 0
        : typeof res.text === 'string' && res.ids.length > 0);
      check(label, ok, `${res.kind}, ${res.ids.length} tokens` + (kind === 'image' ? '' : `, ${JSON.stringify((res.text || '').slice(0, 24))}`));
    } catch (e) {
      check(label, false, 'threw: ' + e.message);
    }
  }
  /* the guards the UI relies on */
  const throws = async (opts) => { try { await runTask(engine, opts); return false; } catch { return true; } };
  check('caption without an image is refused', await throws({ task: 'caption', maxNew: 4 }));
  check('ascii art without any input is refused',
    await throws({ task: 'ascii2code', ascii: '', maxNew: 4 }));
  check('fix with an empty snippet is refused', await throws({ task: 'fix', code: '   ', maxNew: 4 }));

  /* --- the token budget: what the UI's "max new tokens" slider and the cap warning rest on --- */
  check('the default code budget is 512, not the 240 that cut pages in half',
    DEFAULT_CODE_TOKENS === 512, `${DEFAULT_CODE_TOKENS} tokens`);
  const short = await runTask(engine, {
    task: 'code', prompt: 'write html for a blue card', maxNew: 16, temp: 0, topK: 1, seed: 4,
  });
  check('a small budget is honoured and reported',
    short.ids.length <= 16 && short.limit === 16 && typeof short.hitLimit === 'boolean',
    `${short.ids.length}/${short.limit} tokens, hitLimit=${short.hitLimit}, stopped=${short.stopped}`);
  check('an answer that ran into the cap says so',
    !short.hitLimit || short.ids.length === 16, 'hitLimit only when it really ran out');

  /* --- streaming: onProgress hands over the tokens produced so far ---------------------------- */
  const seen = [];
  const streamed = await runTask(engine, {
    task: 'code', prompt: 'write html for a blue card', maxNew: 48, temp: 0, topK: 1, seed: 4,
    onProgress: p => seen.push({ done: p.done, total: p.total, ids: p.ids ? p.ids.length : 0 }),
  });
  const grew = seen.length > 1 && seen.every((s, i) => i === 0 || s.ids >= seen[i - 1].ids);
  check('the tokens produced so far are streamed out',
    seen.length >= 2 && grew && seen[seen.length - 1].ids === streamed.ids.length,
    `${seen.length} updates, ${seen.map(s => s.ids).join('→')}`);
  check('progress reports the adjusted budget as the total',
    seen.every(s => s.total === streamed.limit), `total ${streamed.limit}`);

  /* --- a prompt that cannot fit the context window is refused before it starts --------------- */
  let longError = null;
  try { await runTask(engine, { task: 'code', prompt: 'write html for ' + 'a very long card '.repeat(400) }); }
  catch (e) { longError = e.message; }
  check('an oversized prompt is refused with a readable message',
    !!longError && /more than this model can read/.test(longError), longError || 'no error');
}

console.log(failures ? `WEB TASKS FAILED (${failures})` : 'WEB TASKS OK');
process.exit(failures ? 1 : 0);
