#!/usr/bin/env node
/* Regenerate web/models.json from the bundles that are actually on disk.
 *
 *     node tools/make_models_json.mjs            # keep names/notes, add new bundles
 *     node tools/make_models_json.mjs --check    # exit 1 if the file would change (CI/verify)
 *
 * Any folder with a model.json (web_model*, models/<name>/) becomes a selectable SGM bundle.
 * Entries that are declared in the manifest but not on disk are kept as "planned", so the
 * picker can show SGM-10M / SGM-100M before they have been trained.
 */
import { readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { autoName, slugify } from '../web/models.js';
import { readRelease } from '../web/server.js';

// fileURLToPath, not URL.pathname: on Windows the latter is "/D:/dir", so resolve() would build
// "D:\\D:\\dir" and readFileSync would fail with ENOENT.
const root = resolve(fileURLToPath(new URL('..', import.meta.url)));
const MANIFEST = join(root, 'web', 'models.json');
const SKIP = new Set(['node_modules', '.venv', '.git', 'desktop', 'downloads']);

function hasBundle(dir) {
  try { return statSync(join(dir, 'model.json')).isFile(); } catch { return false; }
}

/** Bundles on disk: <root>/web_model*, <root>/models/<name>/ (also nested one level deeper). */
function scan() {
  const found = [];
  for (const name of readdirSync(root)) {
    if (SKIP.has(name) || name.startsWith('.')) continue;
    const dir = join(root, name);
    try { if (!statSync(dir).isDirectory()) continue; } catch { continue; }
    if (/^web_model/.test(name) && hasBundle(dir)) found.push({ dir: name, path: dir });
    if (name === 'models') {
      for (const sub of readdirSync(dir)) {
        const sdir = join(dir, sub);
        try { if (!statSync(sdir).isDirectory() || !hasBundle(sdir)) continue; } catch { continue; }
        found.push({ dir: `models/${sub}`, path: sdir });
      }
    }
  }
  return found.sort((a, b) => a.dir.localeCompare(b.dir));
}

function info(entry) {
  const meta = JSON.parse(readFileSync(join(entry.path, 'model.json'), 'utf8'));
  const weight = meta.weight_file || 'weights.int8.bin';
  let bytes = meta.weight_bytes || 0;
  try { bytes = statSync(join(entry.path, weight)).size; } catch {}
  const bits = meta.bits || 8;
  const suffix = /int4/.test(entry.dir) && bits === 4 ? 'int4' : '';
  return {
    params: meta.parameter_count || 0,
    bytes,
    bits,
    tasks: Array.isArray(meta.tasks) ? meta.tasks : [],
    quantized: meta.quantization || '',
    generated_base: autoName(meta.parameter_count || 0, suffix),
    note: Array.isArray(meta.tasks) && meta.tasks.length ? meta.tasks.join(' · ') : '',
    architecture: meta.architecture || null,
  };
}

const previous = JSON.parse(readFileSync(MANIFEST, 'utf8'));
const prevByDir = new Map((previous.models || []).map(m => [String(m.dir).replace(/\/+$/, ''), m]));
const onDisk = scan();
const models = [];
const seen = new Set();

for (const entry of onDisk) {
  const i = info(entry);
  const prev = prevByDir.get(entry.dir) || {};
  const name = prev.name || i.generated_base;
  models.push({
    id: prev.id || slugify(name),
    name,
    dir: entry.dir,
    ...(prev.custom ? { custom: true } : {}),
    note: i.note || prev.note || '',
    params: i.params,
    bytes: i.bytes,
    bits: i.bits,
    tasks: i.tasks,
    quantized: i.quantized,
  });
  seen.add(entry.dir);
}

for (const prev of previous.models || []) {
  const dir = String(prev.dir).replace(/\/+$/, '');
  if (seen.has(dir)) continue;
  models.push({ ...prev, dir, planned: true });        // declared but not trained/exported yet
}

// the release comes from ./VERSION (v1, v2, ...), so bumping the file moves the manifest and every
// generated artifact together instead of leaving a stale tag in here.
const out = { version: 1, release: await readRelease(root),
              default: previous.default || models[0]?.id || null, models };
const json = JSON.stringify(out, null, 2) + '\n';

if (process.argv.includes('--check')) {
  const current = readFileSync(MANIFEST, 'utf8');
  const strip = s => JSON.stringify(JSON.parse(s));
  if (strip(current) !== strip(json)) {
    console.error('web/models.json is out of date with the bundles on disk: run node tools/make_models_json.mjs');
    process.exit(1);
  }
  console.log(`models manifest ok (${models.length} entries, ${models.filter(m => !m.planned).length} on disk)`);
  process.exit(0);
}

writeFileSync(MANIFEST, json);
for (const m of models) {
  const size = m.bytes ? `${(m.bytes / 1048576).toFixed(2).padStart(6)} MiB` : '        —';
  console.log(`${m.planned ? 'planned' : 'ready  '}  ${m.name.padEnd(14)} ${m.dir.padEnd(20)} ` +
    `${size}  ${(m.tasks || []).join(', ')}`);
}
console.log(`wrote ${MANIFEST}`);
