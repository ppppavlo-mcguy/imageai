// Compare two exported bundles through the real inference path (web/tasks.js + web/engine.js).
//   node tools/quant_compare.mjs /tmp/qi /tmp/qa            # int8 reference vs candidate
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';
import { Engine } from '../web/engine.js';
import { encodePNG, decodePNG, upscale } from '../web/png.js';
import { runTask } from '../web/tasks.js';
import { inflateSync } from 'node:zlib';

const load = (dir) => {
  const meta = JSON.parse(readFileSync(join(dir, 'model.json'), 'utf8'));
  const tok = JSON.parse(readFileSync(join(dir, 'tokenizer.json'), 'utf8'));
  const w = new Uint8Array(readFileSync(join(dir, meta.weight_file || 'weights.int8.bin')));
  const t0 = Date.now();
  return { dir, meta, engine: new Engine(meta, tok, w), loadMs: Date.now() - t0 };
};

const A = load(process.argv[2] || '/tmp/qi');
const B = load(process.argv[3] || '/tmp/qa');
const mb = x => (x.meta.weight_bytes / 1048576).toFixed(2);
console.log(`A ${A.dir}: ${mb(A)} MiB ${A.meta.quantization}   B ${B.dir}: ${mb(B)} MiB ${B.meta.quantization}`);
console.log(`B is ${(100 * (1 - B.meta.weight_bytes / A.meta.weight_bytes)).toFixed(0)}% smaller than A\n`);

const photo = decodePNG(new Uint8Array(readFileSync('samples/caption.png')), b => new Uint8Array(inflateSync(b)));
const cases = [
  ['image', 'generate the image of a big red car in front of a pink background'],
  ['image', 'generate the image of a small blue circle on a green background'],
  ['image', 'generate the image of a white car', { temp: 0.9 }],
  ['caption', 'what is on this picture', { image: true }],
  ['ascii', 'turn this into an ascii text art', { image: true }],
  ['code', 'write css for a dark page'],
  ['code', 'write html for a blue card'],
  ['text', 'tell me about a red car'],
];
mkdirSync('samples/quant', { recursive: true });
let exact = 0, pxSame = 0, images = 0, agreement = 0;
for (const [task, prompt, extra = {}] of cases) {
  const base = { task, prompt, maxNew: task === 'ascii' ? 400 : 320, temp: extra.temp ?? 0,
                 topK: 1, topP: 1, seed: 7 };
  if (extra.image) { base.rgba = photo.rgba; base.width = photo.width; }
  const ra = await runTask(A.engine, base);
  const rb = await runTask(B.engine, base);
  const idsEq = ra.ids.length === rb.ids.length && ra.ids.every((v, i) => v === rb.ids[i]);
  const txtEq = ra.text === rb.text;
  let pxEq = true;
  if (ra.rgba) {
    images++;
    pxEq = ra.rgba.length === rb.rgba.length && ra.rgba.every((v, i) => v === rb.rgba[i]);
    if (pxEq) pxSame++;
    const tag = `${task}_${prompt.replace(/[^a-z0-9]+/gi, '_').slice(0, 26)}`;
    writeFileSync(`samples/quant/${tag}_A.png`, encodePNG(upscale(ra.rgba, ra.width, ra.height, 4).rgba, ra.width * 4));
    writeFileSync(`samples/quant/${tag}_B.png`, encodePNG(upscale(rb.rgba, rb.width, rb.height, 4).rgba, rb.width * 4));
  }
  const agree = ra.ids.filter((v, i) => v === rb.ids[i]).length / Math.max(ra.ids.length, rb.ids.length);
  agreement += agree;
  const ok = ra.rgba ? pxEq : (txtEq && idsEq);
  if (ok) exact++;
  console.log(`${ok ? 'SAME' : 'DIFF'}  ${task.padEnd(8)} ${(extra.temp ? 'sampled ' : 'greedy  ')}` +
    `${(agree * 100).toFixed(1)}% tok   ${ra.rgba ? (pxEq ? 'pixels identical' : 'PIXELS DIFFER') : ''} ` +
    `${ra.text ? JSON.stringify(ra.text.slice(0, 70)) : ''}`);
}
console.log(`\nbyte-identical outputs: ${exact}/${cases.length}   images pixel-identical: ${pxSame}/${images}`);
console.log(`mean continuation agreement vs the int8 reference: ${(100 * agreement / cases.length).toFixed(1)}%`);
if (exact !== cases.length) process.exit(1);
