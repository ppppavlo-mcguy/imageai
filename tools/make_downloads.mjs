#!/usr/bin/env node
/* Pre-build every downloadable artifact into downloads/ so a static host can serve them too.
 *
 *     node tools/make_downloads.mjs              # all model bundles + whatever desktop build exists
 *     node tools/make_downloads.mjs --only sgm-1m  # just one bundle (by id, name or folder)
 *     node tools/make_downloads.mjs --list
 *
 * Names carry the release from ./VERSION (v1, v2, ...): sgm-1m-v1.zip, sgm-desktop-v1.zip.
 * The server builds the same zips on demand at /api/download/model/<id> and caches them here.
 */
import { mkdirSync, readdirSync, statSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { buildModelZip, modelManifest, desktopArtifacts, readRelease, PROJECT_ROOT } from '../web/server.js';
import { versionedName } from '../web/zip.js';

const want = (() => {
  const i = process.argv.indexOf('--only');
  return i >= 0 ? (process.argv[i + 1] || '').toLowerCase() : null;
})();

const root = PROJECT_ROOT;
const outDir = join(root, 'downloads');
const release = await readRelease(root);
const manifest = await modelManifest(root);

if (process.argv.includes('--list')) {
  console.log(`release ${release}`);
  for (const m of manifest.models) {
    console.log(`${m.available ? 'ready  ' : 'planned'}  ${m.name.padEnd(14)} ${m.dir.padEnd(18)} ` +
      `${(m.bytes / 1048576).toFixed(2).padStart(7)} MiB  ${m.zip || '-'}`);
  }
  process.exit(0);
}

mkdirSync(outDir, { recursive: true });
let built = 0, skipped = 0;

for (const entry of manifest.models) {
  if (!entry.available) { skipped++; continue; }
  if (want && ![entry.id, entry.dir, entry.name].map(s => String(s).toLowerCase()).includes(want)) continue;
  const file = versionedName(entry.name, release);
  const target = join(outDir, file);
  const newest = Math.max(...entry.files.map(f => {
    try { return statSync(join(root, entry.dir, f)).mtimeMs; } catch { return 0; }
  }));
  const have = statSync(target, { throwIfNoEntry: false });
  if (have && have.mtimeMs >= newest && !process.argv.includes('--force')) {
    console.log(`up to date  downloads/${file} (${(have.size / 1048576).toFixed(2)} MiB)`);
    skipped++;
    continue;
  }
  const t0 = Date.now();
  const { bytes, entries } = await buildModelZip(root, entry, release);
  writeFileSync(target, bytes);
  console.log(`built       downloads/${file} (${(bytes.length / 1048576).toFixed(2)} MiB, ` +
    `${entries.length} files, ${((Date.now() - t0) / 1000).toFixed(1)}s)`);
  built++;
}

const apps = await desktopArtifacts(root);
for (const a of apps) console.log(`desktop     downloads/${a.name} (${(a.bytes / 1048576).toFixed(1)} MB)`);
if (!apps.length) {
  console.log('desktop     none yet — build one with: node tools/build_desktop.mjs');
}
const total = readdirSync(outDir)
  .map(n => statSync(join(outDir, n)).size)
  .reduce((a, b) => a + b, 0);
console.log(`${built} built, ${skipped} skipped · downloads/ is ${(total / 1048576).toFixed(2)} MiB · release ${release}`);
