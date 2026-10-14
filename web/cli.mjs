#!/usr/bin/env node
/* Node CLI for SGM (Small Generative Model) - handy for testing before you deploy.
     node web/cli.mjs --list
     node web/cli.mjs --model web_model --task image --prompt "a white car" --out car.png
     node web/cli.mjs --model web_model --task ascii --image photo.png
     node web/cli.mjs --model web_model --task code --prompt "write html for a blue card"
   The browser UI (web/index.html) does the same thing with any image format. */
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { deflateSync, inflateSync } from 'node:zlib';
import { dirname, join } from 'node:path';
import { Engine } from './engine.js';
import { decodePNG, encodePNG, upscale } from './png.js';
import { runTask } from './tasks.js';

const argv = process.argv.slice(2);
const arg = (name, dflt = null) => {
  const i = argv.indexOf(`--${name}`);
  return i >= 0 ? argv[i + 1] : dflt;
};
if (argv.includes('--list')) {                 // same bundles the model picker shows
  const manifest = JSON.parse(readFileSync(new URL('./models.json', import.meta.url), 'utf8'));
  for (const m of manifest.models) {
    const ready = !m.planned && (m.params || 0) > 0;
    console.log(`${ready ? 'ready  ' : 'planned'}  ${String(m.name).padEnd(14)} ${String(m.dir).padEnd(18)} ` +
      `${((m.bytes || 0) / 1048576).toFixed(2).padStart(7)} MiB  ${(m.tasks || []).join(', ')}`);
  }
  process.exit(0);
}
const model = arg('model', 'web_model');
const task = arg('task', 'image');
const prompt = arg('prompt', 'generate the image of a white car');
const imagePath = arg('image');
const out = arg('out', task === 'image' ? 'samples/js_out.png' : null);
const maxNew = arg('max-new') ? parseInt(arg('max-new')) : null;
const temp = parseFloat(arg('temp', '0.9'));
const seed = arg('seed') ? parseInt(arg('seed')) : 1234;

const read = p => new Uint8Array(readFileSync(p));
const meta = JSON.parse(readFileSync(join(model, 'model.json'), 'utf8'));
const tok = JSON.parse(readFileSync(join(model, 'tokenizer.json'), 'utf8'));
const weights = read(join(model, meta.weight_file || 'weights.int8.bin'));
const t0 = Date.now();
const engine = new Engine(meta, tok, weights);
console.log(`loaded ${model}: ${meta.parameter_count.toLocaleString()} params, `
  + `${(meta.weight_bytes / 1048576).toFixed(2)} MiB (${meta.quantization}), ${Date.now() - t0} ms`);

let rgba = null, width = 0;
if (imagePath) {
  const png = decodePNG(read(imagePath), b => new Uint8Array(inflateSync(b)));
  rgba = png.rgba; width = png.width;
  console.log(`input image ${png.width}x${png.height} -> encoded to ${engine.codec.n1 + engine.codec.n2} tokens`);
}
const t1 = Date.now();
const res = await runTask(engine, {
  task, prompt, rgba, width, maxNew, temp, seed,
  onProgress: p => process.stdout.write(`\r  generating ${p.done}/${p.total} tokens`),
});
process.stdout.write('\r' + ' '.repeat(48) + '\r');
if (res.kind === 'image') {
  const up = upscale(res.rgba, res.width, res.height, 6);
  mkdirSync(dirname(out), { recursive: true });
  writeFileSync(out, encodePNG(up.rgba, up.width, up.height, b => new Uint8Array(deflateSync(b))));
  console.log(`image tokens: ${res.ids.length}  ->  ${out} (${up.width}x${up.height})`);
} else {
  console.log(res.text);
  if (out) {
    mkdirSync(dirname(out), { recursive: true });
    writeFileSync(out, res.text, 'utf8');
    console.log(`-> ${out}`);
  }
}
console.log(`time: ${((Date.now() - t1) / 1000).toFixed(1)}s`);
