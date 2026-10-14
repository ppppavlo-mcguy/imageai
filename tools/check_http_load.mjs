// Load a bundle the way the browser does (HTTP + gzip transit) and check it is bit-identical to
// the raw files on disk.
//   node tools/check_http_load.mjs http://127.0.0.1:3000/web_model web_model
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { Engine, loadBundle } from '../web/engine.js';
import { runTask } from '../web/tasks.js';

const url = process.argv[2] || 'http://127.0.0.1:3000/web_model';
const dir = process.argv[3] || 'web_model';

const meta = JSON.parse(readFileSync(join(dir, 'model.json'), 'utf8'));
const tok = JSON.parse(readFileSync(join(dir, 'tokenizer.json'), 'utf8'));
const raw = new Uint8Array(readFileSync(join(dir, meta.weight_file)));
const local = new Engine(meta, tok, raw);

const t0 = Date.now();
const { engine, gz, gzBytes, ms } = await loadBundle(url);
console.log(`http  : ${url}  ${ms.toFixed(0)} ms engine build, ${Date.now() - t0} ms total`);
console.log(`transit: ${gz ? `${(gzBytes / 1048576).toFixed(2)} MiB gzipped on the wire -> ` : 'raw file, '}` +
  `${(meta.weight_bytes / 1048576).toFixed(2)} MiB in memory`);

const prompts = [
  ['image', 'generate the image of a white car'],
  ['image', 'generate the image of a big red car in front of a pink background'],
  ['code', 'write html for a blue card'],
];
let ok = 0;
for (const [task, prompt] of prompts) {
  const opts = { task, prompt, maxNew: 300, temp: 0, topK: 1, topP: 1, seed: 3 };
  const a = await runTask(local, opts);
  const b = await runTask(engine, opts);
  const sameIds = a.ids.length === b.ids.length && a.ids.every((v, i) => v === b.ids[i]);
  const samePx = a.rgba ? (a.rgba.length === b.rgba.length && a.rgba.every((v, i) => v === b.rgba[i])) : true;
  const same = sameIds && samePx;
  if (same) ok++;
  console.log(`${same ? 'SAME' : 'DIFF'}  ${task.padEnd(6)} ${JSON.stringify(prompt).slice(0, 52)}  ids=${a.ids.length}`);
}
console.log(`\nhttp-loaded model identical to the files on disk: ${ok}/${prompts.length}`);
if (ok !== prompts.length) process.exit(1);
