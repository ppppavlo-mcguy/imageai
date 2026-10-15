#!/usr/bin/env node
/* Desktop check without a display or an Electron download.
 *
 *     node tools/check_desktop.mjs
 *
 * It verifies the Electron-free half of the desktop app (desktop/boot.cjs):
 *   - root resolution in the development layout (desktop/ inside the project)
 *   - root resolution in the packaged/staged layout (app folder with web/ next to it)
 *   - the local server really answers with the UI and the model API from that folder
 *   - the staged app contains exactly what the exe needs (main.cjs, boot.cjs, web/, models, VERSION)
 *   - a Windows build exists with a real PE executable inside, when one has been built
 */
import { execFileSync } from 'node:child_process';
import { createRequire } from 'node:module';
import { closeSync, existsSync, openSync, readFileSync, readdirSync, readSync, statSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { join, resolve } from 'node:path';
import { PROJECT_ROOT } from '../web/server.js';
import { npmCommand, packagerCommand, packagerPath, commandLine, quoteArg } from './shellcmd.mjs';

const require = createRequire(import.meta.url);
const { resolveRoot, readVersion, startLocalServer } = require('../desktop/boot.cjs');

let failures = 0;
const check = (name, cond, detail = '') => {
  console.log(`  ${cond ? 'ok  ' : 'FAIL'} ${name}${detail ? '  ' + detail : ''}`);
  if (!cond) failures++;
};

const root = PROJECT_ROOT;
const stage = join(root, 'desktop', 'build', 'app');
const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');

/** The file index of the packed Electron app, without loading the 240 MB payload behind it.
 *  asar layout: [4][header size][4][json length][json] then the file data. Every entry carries a
 *  SHA256 of the file it points at, which is exactly the question worth asking of a build: is the
 *  UI inside the .exe the UI that is in web/ right now? */
function readAsarHeader(path) {
  const fd = openSync(path, 'r');
  try {
    const head = Buffer.alloc(16);
    readSync(fd, head, 0, 16, 0);
    const region = Buffer.alloc(head.readUInt32LE(4) + 8);
    readSync(fd, region, 0, region.length, 0);
    const jsonLen = head.readUInt32LE(12);
    return JSON.parse(region.subarray(16, 16 + jsonLen).toString('utf8'));
  } finally { closeSync(fd); }
}

/* ------------------------------------------------------------ boot.cjs itself */
check('dev layout resolves to the project root',
  resolveRoot(join(root, 'desktop')) === resolve(root), resolveRoot(join(root, 'desktop')));
check('version comes from VERSION', /^v\d/.test(readVersion(root)), readVersion(root));
check('a folder without web/ is refused', (() => {
  try { resolveRoot('/tmp'); return false; } catch { return true; }
})());

/* --------------------------------------------- the server the desktop app uses */
{
  const started = await startLocalServer({ root, port: 0, log: () => {} });
  const base = started.url.replace(/\/web\/$/, '');
  const page = await fetch(started.url);
  const html = await page.text();
  check('desktop server serves the SGM UI',
    page.status === 200 && /<title>SGM/.test(html) && html.includes('id="galleryBtn"'),
    `http ${page.status}`);
  const models = await (await fetch(`${base}/api/models`)).json();
  check('desktop server discovers the model bundles',
    models.models.filter(m => m.available).length >= 1,
    models.models.map(m => `${m.name}${m.available ? '' : ' (planned)'}`).join(', '));
  const app = await fetch(`${base}/web/app.js`);
  check('the UI files are reachable from the loopback origin', app.status === 200);
  started.server.close();
}

/* ------------------------------- the build commands (the Windows EINVAL fix) ---------------- */
check('npm is spelled per platform',
  npmCommand('win32') === 'npm.cmd' && npmCommand('linux') === 'npm' && npmCommand('darwin') === 'npm');
check('the packager is named relative to desktop/ (no spaces from the checkout path)',
  packagerCommand('win32') === 'node_modules\\.bin\\electron-packager.cmd'
  && packagerCommand('linux') === './node_modules/.bin/electron-packager'
  && packagerPath('/x/desktop', 'win32').includes('electron-packager.cmd'),
  packagerCommand('win32'));
check('arguments with spaces are quoted for the shell',
  quoteArg('--app-copyright=SGM (v1)') === '"--app-copyright=SGM (v1)"'
  && quoteArg('--arch=x64') === '--arch=x64',
  quoteArg('--app-copyright=SGM (v1)'));
const winLine = commandLine('npm.cmd', ['install', '--save-dev', 'electron', '@electron/packager']);
check('the install line is a single shell command (execSync, not execFile)',
  winLine === 'npm.cmd install --save-dev electron @electron/packager', winLine);
const packLine = commandLine(packagerCommand('win32'), ['/C/My Projects/stage', 'SGM', '--platform=win32']);
check('a staged path with a space survives the quoting',
  packLine.startsWith('node_modules\\.bin\\electron-packager.cmd "/C/My Projects/stage" SGM'), packLine);
check('build_desktop.mjs really uses the helper (no raw execFile left)',
  !/execFileSync/.test(readFileSync(join(root, 'tools', 'build_desktop.mjs'), 'utf8'))
  && /runInShell\(npmCommand\(\)/.test(readFileSync(join(root, 'tools', 'build_desktop.mjs'), 'utf8')));

/* --------------------------------------------------------- the packaged layout */
if (existsSync(stage)) {
  const needed = ['main.cjs', 'boot.cjs', 'package.json', 'VERSION', 'web/index.html', 'web/app.js',
                  'web/server.js', 'web/models.json', 'web/zip.js', 'web/gallery.js', 'web/models.js',
                  'web/code.js'];
  const missing = needed.filter(f => !existsSync(join(stage, f)));
  check('the staged app carries every file the exe needs', missing.length === 0, missing.join(', ') || 'all present');
  const bundles = readdirSync(stage).filter(n => /^web_model|^models$/.test(n));
  check('the staged app carries the model bundles', bundles.length >= 1, bundles.join(', '));
  check('the staged package.json keeps the ESM hint for web/*.js',
    JSON.parse(readFileSync(join(stage, 'package.json'), 'utf8')).type === 'module');

  const staged = await startLocalServer({ root: stage, port: 0, log: () => {} });
  const base = staged.url.replace(/\/web\/$/, '');
  const models = await (await fetch(`${base}/api/models`)).json();
  check('the staged copy serves its own models (a self-contained app)',
    models.models.some(m => m.available), models.models.filter(m => m.available).map(m => m.name).join(', '));
  const dl = await (await fetch(`${base}/api/downloads`)).json();
  check('the staged copy offers its model downloads', (dl.models || []).length >= 1);
  staged.server.close();
} else {
  console.log('  skip  staged app (run: node tools/build_desktop.mjs --stage-only)');
}

/* ------------------------------------------------------------- a real .exe build */
{
  const dist = join(root, 'desktop', 'dist');
  const dirs = existsSync(dist) ? readdirSync(dist).filter(n => n.startsWith('SGM-')) : [];
  const win = dirs.find(d => d.includes('win32'));
  if (!win) {
    console.log('  skip  windows build (run: node tools/build_desktop.mjs --platform=win32)');
  } else {
    const exe = join(dist, win, 'SGM.exe');
    const info = existsSync(exe) ? execFileSync('file', ['-b', exe], { encoding: 'utf8' }).trim() : '';
    check('the windows build contains a real x86-64 PE executable',
      /PE32\+ executable \(GUI\) x86-64/.test(info), info || 'no SGM.exe');
    check('SGM.exe is a full Electron app (not a stub)', existsSync(exe) && statSync(exe).size > 100 * 1048576,
      existsSync(exe) ? `${(statSync(exe).size / 1048576).toFixed(0)} MiB` : '-');
    const zips = existsSync(join(root, 'downloads'))
      ? readdirSync(join(root, 'downloads')).filter(n => /^sgm-desktop.*\.zip$/.test(n)) : [];
    check('the download the web UI offers exists', zips.length >= 1, zips.join(', '));

    /* The packaged app has to carry the UI as it is now, not the one from the last build. */
    const asar = join(dist, win, 'resources', 'app.asar');
    if (!existsSync(asar)) {
      console.log('  skip  the packaged app.asar (rebuild with: node tools/build_desktop.mjs)');
    } else {
      const header = readAsarHeader(asar);
      const files = (header.files && header.files.web && header.files.web.files) || {};
      const wanted = ['app.js', 'code.js', 'index.html', 'style.css', 'worker.js', 'tasks.js',
                      'engine.js', 'models.json', 'server.js'];
      const stale = wanted.filter(f => !files[f] || files[f].integrity.hash !== sha256(readFileSync(join(root, 'web', f))));
      check('the packaged .exe carries web/ exactly as it is now (asar hashes)', stale.length === 0,
        stale.length ? `stale: ${stale.join(', ')}` : `${wanted.length} files, incl. web/code.js`);
      const ver = header.files.VERSION;
      check('the packaged app reports the same release as VERSION',
        !!ver && ver.integrity.hash === sha256(readFileSync(join(root, 'VERSION'))),
        readFileSync(join(root, 'VERSION'), 'utf8').trim());
      const bundles = Object.keys(header.files).filter(n => /^web_model|^models$/.test(n));
      check('the packaged app carries the model bundles', bundles.length >= 1, bundles.join(', '));
    }
  }
}

console.log(failures ? `DESKTOP FAILED (${failures})` : 'DESKTOP OK');
process.exit(failures ? 1 : 0);
