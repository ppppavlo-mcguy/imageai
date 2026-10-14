#!/usr/bin/env node
/* Build the SGM desktop app (the same UI as the website, in an Electron window).
 *
 *     node tools/build_desktop.mjs                 # stage + package for Windows (x64) + zip it
 *     node tools/build_desktop.mjs --platform=linux
 *     node tools/build_desktop.mjs --platform=all
 *     node tools/build_desktop.mjs --stage-only    # just write desktop/build/app (no electron needed)
 *     node tools/build_desktop.mjs --skip-install  # reuse desktop/node_modules
 *
 * What it does
 *   1. stages a self-contained app folder (desktop/build/app): main.cjs, web/, web/models.json,
 *      the exported model bundles, VERSION - nothing else, so the installer stays small
 *   2. makes sure electron + @electron/packager are installed in desktop/
 *   3. packages that folder into desktop/dist/SGM-<platform>-<arch>/ (contains SGM.exe on Windows)
 *   4. zips it into downloads/sgm-desktop-<release>.zip, which the web UI offers as a download
 *
 * The staged folder also carries an electron-builder config, so on Windows
 *     cd desktop/build/app && npx electron-builder --win nsis portable
 * turns it into a single .exe installer / portable exe.
 */
import { cpSync, existsSync, mkdirSync, readdirSync, readFileSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';
import { join, resolve } from 'node:path';
import { buildZip, versionedName } from '../web/zip.js';
import { PROJECT_ROOT, modelManifest, readRelease, desktopArtifacts } from '../web/server.js';

const arg = (name, dflt = null) => {
  const hit = process.argv.find(a => a.startsWith(`--${name}=`));
  return hit ? hit.split('=')[1] : (process.argv.includes(`--${name}`) ? true : dflt);
};
const root = PROJECT_ROOT;
const desktop = join(root, 'desktop');
const stage = join(desktop, 'build', 'app');
const release = await readRelease(root);
const platform = String(arg('platform', process.platform === 'win32' ? 'win32' : 'linux'));
const platforms = platform === 'all' ? ['win32', 'linux', 'darwin'] : platform.split(',');

const say = (...a) => console.log('[desktop]', ...a);

/* ------------------------------------------------------------------ 1. stage */
function copyInto(from, to) {
  cpSync(from, to, { recursive: true, dereference: true,
                     filter: src => !/(^|[\\/])(\.git|node_modules|__pycache__)([\\/]|$)/.test(src) });
}

rmSync(join(desktop, 'build'), { recursive: true, force: true });
mkdirSync(stage, { recursive: true });
const webFiles = readdirSync(join(root, 'web')).filter(f => /\.(js|mjs|css|html|json)$/.test(f));
mkdirSync(join(stage, 'web'), { recursive: true });
for (const f of webFiles) cpSync(join(root, 'web', f), join(stage, 'web', f));
for (const f of ['main.cjs', 'boot.cjs']) cpSync(join(root, 'desktop', f), join(stage, f));
cpSync(join(root, 'VERSION'), join(stage, 'VERSION'));

const manifest = await modelManifest(root);
const shipped = [];
for (const m of manifest.models.filter(m => m.available)) {
  const src = join(root, m.dir);
  if (!existsSync(join(src, 'model.json'))) continue;
  copyInto(src, join(stage, m.dir));
  shipped.push(m);
}
const pkg = JSON.parse(readFileSync(join(desktop, 'package.json'), 'utf8'));
pkg.main = 'main.cjs';
pkg.type = 'module';              // web/*.js is ESM; main.cjs stays CommonJS on purpose
pkg.scripts = { start: 'electron .' };
pkg.build.files = ['main.cjs', 'boot.cjs', 'package.json', 'web/**', 'web_model*/**', 'models/**', 'VERSION'];
writeFileSync(join(stage, 'package.json'), JSON.stringify(pkg, null, 2) + '\n');

function dirBytes(dir) {
  let n = 0;
  for (const e of readdirSync(dir, { withFileTypes: true })) {
    const p = join(dir, e.name);
    n += e.isDirectory() ? dirBytes(p) : statSync(p).size;
  }
  return n;
}
say(`staged ${webFiles.length} web files + ${shipped.length} model bundle(s): ${shipped.map(m => m.dir).join(', ') || 'none'}`);
say(`app folder ${stage} (${(dirBytes(stage) / 1048576).toFixed(2)} MiB)`);
if (!shipped.length) say('WARNING: no model bundle was staged - the app will start but cannot generate. Train + export one first.');

if (arg('stage-only')) {
  say('done (--stage-only): run "npx electron ." inside that folder, or package it with electron-builder');
  process.exit(0);
}

/* -------------------------------------------------------- 2. tools installed */
const needInstall = !existsSync(join(desktop, 'node_modules', 'electron'));
if (needInstall && !arg('skip-install')) {
  say('installing electron + @electron/packager in desktop/ (first run downloads ~120 MB)…');
  execFileSync(process.platform === 'win32' ? 'npm.cmd' : 'npm',
    ['install', '--save-dev', 'electron', '@electron/packager'], { cwd: desktop, stdio: 'inherit' });
}
if (!existsSync(join(desktop, 'node_modules', 'electron'))) {
  say('electron is not installed - run "npm install --save-dev electron @electron/packager" in desktop/ and retry');
  process.exit(1);
}
const electronVersion = JSON.parse(
  readFileSync(join(desktop, 'node_modules', 'electron', 'package.json'), 'utf8')).version;
say(`electron ${electronVersion}`);

/* ---------------------------------------------------------------- 3. package */
const packager = join(desktop, 'node_modules', '.bin', process.platform === 'win32' ? 'electron-packager.cmd' : 'electron-packager');
rmSync(join(desktop, 'dist'), { recursive: true, force: true });
const outputs = [];
for (const plat of platforms) {
  const out = join(desktop, 'dist');
  const before = new Set(existsSync(out) ? readdirSync(out) : []);
  say(`packaging for ${plat} (the electron binary for that platform is downloaded now)…`);
  execFileSync(packager, [stage, 'SGM', `--platform=${plat}`, '--arch=x64', `--out=${out}`,
    '--overwrite', `--electron-version=${electronVersion}`, '--app-version=1.0.0',
    '--executable-name=SGM', '--prune=true', `--app-copyright=SGM (${release})`],
    { cwd: desktop, stdio: 'inherit' });
  const dir = readdirSync(out).find(n => !before.has(n) && n.startsWith('SGM-'));
  if (dir) outputs.push(join(out, dir));
  else say(`WARNING: could not find the packaged folder for ${plat} in ${out}`);
}
for (const dir of outputs) {
  const files = [];
  const walk = (base, rel = '') => {
    for (const e of readdirSync(join(base, rel), { withFileTypes: true })) {
      const next = rel ? `${rel}/${e.name}` : e.name;
      if (e.isDirectory()) walk(base, next);
      else files.push({ name: `sgm-desktop/${next}`, data: () => readFileSync(join(base, next)) });
    }
  };
  walk(dir);
  const { bytes, entries } = await buildZip(files);
  const name = dir.includes('win32') ? versionedName('sgm-desktop', release)
    : versionedName(`sgm-desktop-${dir.split('-')[1]}`, release);
  mkdirSync(join(root, 'downloads'), { recursive: true });
  writeFileSync(join(root, 'downloads', name), bytes);
  say(`built downloads/${name} (${(bytes.length / 1048576).toFixed(1)} MiB, ${entries.length} files) from ${dir}`);
  const exe = readdirSync(dir).find(f => /\.exe$/i.test(f));
  if (exe) say(`  the app is ${join(dir, exe)} - unzip the download and run it`);
}
const presented = await desktopArtifacts(root);
for (const a of presented) say(`available to the web UI: /downloads/${a.name} (${(a.bytes / 1048576).toFixed(1)} MB)`);
say('done');
