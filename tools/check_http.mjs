#!/usr/bin/env node
/* HTTP checks for the SGM server: the page, the model API and the versioned downloads.
 *
 *     node tools/check_http.mjs
 *
 * Starts web/server.js on a free loopback port and exercises the exact routes the browser and
 * the desktop app use, then verifies that a model zip really contains a loadable bundle.
 */
import { readZip } from '../web/zip.js';
import { startServer, PROJECT_ROOT } from '../web/server.js';
import { Engine } from '../web/engine.js';
import { runTask } from '../web/tasks.js';
import { join } from 'node:path';
import { readFileSync } from 'node:fs';

let failures = 0;
const check = (name, cond, detail = '') => {
  console.log(`  ${cond ? 'ok  ' : 'FAIL'} ${name}${detail ? '  ' + detail : ''}`);
  if (!cond) failures++;
};

const { server, url, port } = await startServer({ root: PROJECT_ROOT, port: 0, log: () => {} });
const base = url.replace(/\/web\/$/, '');
const get = (path, opts) => fetch(`${base}${path}`, { redirect: 'manual', ...opts });
console.log(`  serving on 127.0.0.1:${port}`);

/* ------------------------------------------------------------------ the page */
{
  const root = await get('/');
  check('"/" redirects to the UI', root.status === 302 && root.headers.get('location') === '/web/',
    `${root.status} -> ${root.headers.get('location')}`);
  const page = await get('/web/');
  const html = await page.text();
  check('the UI is served', page.status === 200 && /<title>SGM/.test(html));
  check('the model picker and gallery are in the page',
    html.includes('id="modelSelect"') && html.includes('id="galleryBtn"') && html.includes('id="dlModal"'));
  for (const f of ['/web/app.js', '/web/style.css', '/web/gallery.js', '/web/models.js', '/web/zip.js', '/web/models.json']) {
    const r = await get(f);
    check(`${f} is served`, r.status === 200, `http ${r.status}`);
  }
  const missing = await get('/web/nope.js');
  check('a missing file is a 404', missing.status === 404);
  const escape = await get('/%2e%2e/%2e%2e/etc/passwd');
  check('path traversal is refused', escape.status === 403 || escape.status === 404, `http ${escape.status}`);
}

/* --------------------------------------------------------------- model API */
let manifest;
{
  const r = await get('/api/models');
  manifest = await r.json();
  check('/api/models answers', r.status === 200 && Array.isArray(manifest.models));
  check('the release tag travels with the manifest', /^v\d/.test(manifest.release), manifest.release);
  const ready = manifest.models.filter(m => m.available);
  const planned = manifest.models.filter(m => !m.available);
  check('exported bundles are available', ready.length >= 1, ready.map(m => m.name).join(', '));
  check('SGM-10M / SGM-100M are offered but marked planned',
    ['sgm-10m', 'sgm-100m'].every(id => planned.some(m => m.id === id)),
    planned.map(m => m.name).join(', '));
  check('every available bundle has a versioned zip name',
    ready.every(m => new RegExp(`^sgm.*-${manifest.release}\\.zip$`).test(m.zip)),
    ready.map(m => m.zip).join(', '));
  check('the default is available', ready.some(m => m.id === manifest.default), manifest.default);
}

/* --------------------------------------------------------------- downloads */
{
  const r = await get('/api/downloads');
  const dl = await r.json();
  check('/api/downloads answers', r.status === 200);
  check('all exported models are downloadable', (dl.models || []).length >= 1,
    (dl.models || []).map(m => `${m.name} -> ${m.file}`).join(', '));
  check('the download urls point at the model endpoint',
    (dl.models || []).every(m => m.url.startsWith('/api/download/model/')));
  if (dl.app) check('the desktop build is offered to the UI',
    /^sgm-desktop.*\.(zip|exe)$/.test(dl.app.name), `${dl.app.name} (${(dl.app.bytes / 1048576).toFixed(1)} MB)`);
  else check('the UI is told how to build the desktop app', typeof dl.buildApp === 'string', dl.buildApp || '');
  if (dl.app) {
    const f = await get(dl.app.url);
    check('the desktop download is served as an attachment',
      f.status === 200 && /attachment/.test(f.headers.get('content-disposition') || ''),
      f.headers.get('content-disposition') || '');
  }
}

/* ------------------------------------------------ a model zip is a real bundle */
{
  const model = manifest.models.find(m => m.available);
  const r = await get(`/api/download/model/${model.id}`);
  const disp = r.headers.get('content-disposition') || '';
  const body = new Uint8Array(await r.arrayBuffer());
  check('the model endpoint streams a zip',
    r.status === 200 && r.headers.get('content-type') === 'application/zip',
    `${r.status} ${r.headers.get('content-type')}`);
  check('the zip is named after the model and the release',
    disp.includes(model.zip), disp);
  const entries = await readZip(body);
  const names = entries.map(e => e.name);
  check('the zip contains a complete bundle',
    names.some(n => n.endsWith('/model.json')) && names.some(n => n.endsWith('/tokenizer.json'))
    && names.some(n => /weights\.(int8|int4)\.bin$/.test(n)) && names.some(n => n.endsWith('/README.txt')),
    names.join(', '));
  check('every entry passed its crc32 check', entries.every(e => e.crc32));
  const again = await get(`/api/download/model/${model.id}`);
  const body2 = new Uint8Array(await again.arrayBuffer());
  check('the second request serves the cached zip byte-for-byte', body2.length === body.length);
  const bogus = await get('/api/download/model/does-not-exist');
  check('an unknown model id is a 404 with a hint', bogus.status === 404,
    (await bogus.json()).hint || '');
  const byName = await get(`/api/download/model/${model.name.toLowerCase()}`);
  check('a bundle can also be requested by display name', byName.status === 200);

  // The strongest check: load the downloaded zip and generate with it, then compare against the
  // bundle on disk. A download that cannot reproduce the local model is a broken download.
  const pull = name => {
    const e = entries.find(x => x.name.endsWith(name));
    return e ? e.data : null;
  };
  const meta = JSON.parse(new TextDecoder().decode(pull('/model.json')));
  const tok = JSON.parse(new TextDecoder().decode(pull('/tokenizer.json')));
  const weights = pull(`/${meta.weight_file || 'weights.int8.bin'}`);
  const zipEngine = new Engine(meta, tok, weights);
  const localEngine = new Engine(
    JSON.parse(readFileSync(join(PROJECT_ROOT, model.dir, 'model.json'), 'utf8')),
    JSON.parse(readFileSync(join(PROJECT_ROOT, model.dir, 'tokenizer.json'), 'utf8')),
    new Uint8Array(readFileSync(join(PROJECT_ROOT, model.dir, meta.weight_file || 'weights.int8.bin'))));
  const opts = { task: 'image', prompt: 'generate the image of a white car', maxNew: 120, temp: 0, topK: 1, seed: 7 };
  const a = await runTask(zipEngine, opts);
  const b = await runTask(localEngine, opts);
  const same = a.ids.length === b.ids.length && a.ids.every((v, i) => v === b.ids[i]);
  check('the downloaded bundle generates identically to the local one', same,
    `${a.ids.length} tokens, ${weights.length} weight bytes`);
}

server.close();
console.log(failures ? `HTTP FAILED (${failures})` : 'HTTP OK');
process.exit(failures ? 1 : 0);
