#!/usr/bin/env node
/* Tiny static server + small JSON API for local use and for free hosts (Render / HelioHost /
 * any Node host).
 *
 *     node web/server.js              # http://localhost:3000
 *     PORT=8080 node web/server.js
 *
 * It serves the project root, so /web/* and the model folders are reachable; "/" redirects to
 * the UI. On top of the static files it answers:
 *
 *     GET /api/models                     which SGM bundles exist on this machine
 *     GET /api/downloads                  what can be downloaded, with versioned file names
 *     GET /api/download/model/<id>        zips a model folder (cached in downloads/)
 *     GET /api/download/app               the desktop build, when it has been compiled
 *
 * The desktop app (desktop/main.cjs) imports createServer() and serves the very same files.
 */
import { createServer as createHttpServer } from 'node:http';
import { createReadStream, existsSync } from 'node:fs';
import { readFile, stat, writeFile, mkdir, readdir } from 'node:fs/promises';
import { extname, join, normalize, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildZip, versionedName } from './zip.js';
import { autoName, mergeModels, slugify } from './models.js';

export const HERE = fileURLToPath(new URL('.', import.meta.url));
export const PROJECT_ROOT = resolve(HERE, '..');

const MIME = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8', '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8', '.bin': 'application/octet-stream',
  '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.svg': 'image/svg+xml',
  '.txt': 'text/plain; charset=utf-8', '.md': 'text/markdown; charset=utf-8',
  '.ico': 'image/x-icon', '.pt': 'application/octet-stream', '.zip': 'application/zip',
  '.exe': 'application/vnd.microsoft.portable-executable',
};

const SKIP_DIRS = new Set(['node_modules', '.venv', '.git', 'desktop', 'downloads']);

/** The release tag every packaged file carries (VERSION file, default v1). */
export async function readRelease(root = PROJECT_ROOT) {
  try {
    const text = (await readFile(join(root, 'VERSION'), 'utf8')).trim();
    if (/^v?[0-9]/.test(text)) return text.startsWith('v') ? text : `v${text}`;
  } catch { /* no VERSION file yet */ }
  return 'v1';
}

/** Every folder on disk that holds a model.json, with the numbers the picker shows. */
export async function discoverModels(root = PROJECT_ROOT) {
  const dirs = [];
  let names = [];
  try { names = await readdir(root); } catch { return []; }
  for (const name of names.sort()) {
    if (SKIP_DIRS.has(name) || name.startsWith('.')) continue;
    const dir = join(root, name);
    const st = await stat(dir).catch(() => null);
    if (!st || !st.isDirectory()) continue;
    if (/^web_model/.test(name)) dirs.push({ dir: name, path: dir });
    if (name === 'models') {
      for (const sub of (await readdir(dir).catch(() => [])).sort()) {
        const sdir = join(dir, sub);
        const sst = await stat(sdir).catch(() => null);
        if (sst && sst.isDirectory()) dirs.push({ dir: `models/${sub}`, path: sdir });
      }
    }
  }
  const out = [];
  for (const entry of dirs) {
    const metaPath = join(entry.path, 'model.json');
    if (!existsSync(metaPath)) continue;
    const meta = JSON.parse(await readFile(metaPath, 'utf8'));
    const weight = meta.weight_file || 'weights.int8.bin';
    let bytes = meta.weight_bytes || 0;
    const wst = await stat(join(entry.path, weight)).catch(() => null);
    if (wst) bytes = wst.size;
    const suffix = meta.bits === 4 ? 'int4' : '';
    out.push({
      id: slugify(autoName(meta.parameter_count || 0, suffix)),
      name: autoName(meta.parameter_count || 0, suffix),
      suffix,
      dir: entry.dir,
      params: meta.parameter_count || 0,
      bytes,
      bits: meta.bits || 8,
      tasks: Array.isArray(meta.tasks) ? meta.tasks : [],
      quantized: meta.quantization || '',
      architecture: meta.architecture || null,
      files: ['model.json', 'tokenizer.json', 'imagecodec.json', weight]
        .filter(f => existsSync(join(entry.path, f))),
    });
  }
  return out;
}

/** The manifest the picker consumes: declared names + everything found on disk. */
export async function modelManifest(root = PROJECT_ROOT) {
  const manifest = JSON.parse(await readFile(join(root, 'web', 'models.json'), 'utf8').catch(() => '{}'));
  const found = await discoverModels(root);
  const release = await readRelease(root);
  const { models, default: def } = mergeModels(manifest, found,
    dir => `/${String(dir).replace(/^\.?\//, '')}`);
  for (const m of models) {
    if (m.available) m.zip = versionedName(slugify(m.name), release);
    if (m.quantized) m.quantized = m.quantized;
  }
  return { version: 1, release, default: def, models };
}

/** The desktop artifacts that have been compiled (downloads/sgm-desktop-*.exe|.zip). */
export async function desktopArtifacts(root = PROJECT_ROOT) {
  const dir = join(root, 'downloads');
  const names = await readdir(dir).catch(() => []);
  const out = [];
  for (const name of names.sort()) {
    if (!/^sgm-desktop.*\.(exe|zip|msi|AppImage|dmg)$/i.test(name)) continue;
    const st = await stat(join(dir, name)).catch(() => null);
    if (st) out.push({ name, bytes: st.size, url: `/downloads/${name}`, mtime: st.mtimeMs });
  }
  return out.sort((a, b) => b.mtime - a.mtime);
}

/** Pack one bundle (model.json + tokenizer + codec + weights + a README) into a versioned .zip. */
export async function buildModelZip(root, entry, release) {
  const dir = join(root, entry.dir);
  const dirName = String(entry.dir).split('/').pop();
  const files = [];
  for (const name of entry.files) {
    const p = join(dir, name);
    const st = await stat(p).catch(() => null);
    if (st && st.isFile()) files.push({ name: `${dirName}/${name}`, data: new Uint8Array(await readFile(p)) });
  }
  if (!files.length) throw new Error(`${entry.dir} has no files to zip`);
  files.push({
    name: `${dirName}/README.txt`,
    data: `SGM (Small Generative Model) bundle: ${entry.name}\r\n` +
      `parameters: ${entry.params}\r\nweights: ${entry.files.join(', ')}\r\n\r\n` +
      `How to use it\r\n  1. copy this folder next to web/ in an SGM checkout\r\n` +
      `  2. add it to web/models.json (or run: node tools/make_models_json.mjs)\r\n` +
      `  3. node web/server.js, open http://localhost:3000, pick "${entry.name}" in the model box\r\n` +
      `  4. everything runs in your browser tab: no API keys, no external requests\r\n`,
  });
  const { bytes, entries } = await buildZip(files);
  return { bytes, entries, file: versionedName(slugify(entry.name), release) };
}

const send = (res, code, type, body, extra = {}) => {
  res.writeHead(code, { 'content-type': type, 'content-length': body.length, ...extra });
  res.end(body);
};
const json = (res, code, obj, extra = {}) =>
  send(res, code, 'application/json; charset=utf-8', Buffer.from(JSON.stringify(obj, null, 2)), extra);

/**
 * The whole site as an http.Server (no listen call), so the desktop app can use the exact
 * same files the browser gets.
 */
export function createServer({ root = PROJECT_ROOT, log = console.log } = {}) {
  const downloads = join(root, 'downloads');

  async function api(req, res, url) {
    const release = await readRelease(root);
    const parts = url.pathname.split('/').filter(Boolean);      // ['api','download','model','id']

    if (url.pathname === '/api/models') {
      return json(res, 200, await modelManifest(root), { 'cache-control': 'no-cache' });
    }

    if (url.pathname === '/api/downloads') {
      const manifest = await modelManifest(root);
      const app = (await desktopArtifacts(root))[0] || null;
      return json(res, 200, {
        release,
        models: manifest.models.filter(m => m.available).map(m => ({
          id: m.id, name: m.name, dir: m.dir, bytes: m.bytes, file: m.zip,
          url: `/api/download/model/${m.id}`,
        })),
        app,
        buildApp: app ? null : 'node tools/build_desktop.mjs   (or desktop\\build.cmd on Windows)',
      }, { 'cache-control': 'no-cache' });
    }

    if (parts[1] === 'download' && parts[2] === 'app') {
      const app = (await desktopArtifacts(root))[0];
      if (!app) {
        return json(res, 404, { error: 'the desktop app has not been built yet',
                                hint: 'run: node tools/build_desktop.mjs' }, { 'cache-control': 'no-cache' });
      }
      return file(res, join(downloads, app.name), { 'content-disposition': `attachment; filename="${app.name}"` });
    }

    if (parts[1] === 'download' && parts[2] === 'model') {
      const wanted = decodeURIComponent(parts[3] || '').toLowerCase();
      const manifest = await modelManifest(root);
      const entry = manifest.models.find(m => m.available &&
        (m.id.toLowerCase() === wanted || slugify(m.name) === wanted ||
         slugify(String(m.dir).split('/').pop()) === wanted));
      if (!entry) return json(res, 404, { error: `no exported bundle matches "${wanted}"`,
                                          hint: 'GET /api/models lists the available ids' });
      const zipName = versionedName(slugify(entry.name), release);
      const cached = join(downloads, zipName);
      const metaPath = join(root, entry.dir, 'model.json');
      const fresh = existsSync(cached) && (await stat(cached)).mtimeMs >= (await stat(metaPath)).mtimeMs;
      if (!fresh) {
        const t0 = Date.now();
        const built = await buildModelZip(root, entry, release);
        await mkdir(downloads, { recursive: true });
        await writeFile(join(downloads, built.file), built.bytes);
        log(`zipped ${entry.dir} -> downloads/${built.file} ` +
            `(${(built.bytes.length / 1048576).toFixed(2)} MiB, ${built.entries.length} files, ` +
            `${((Date.now() - t0) / 1000).toFixed(1)}s)`);
      }
      return file(res, cached, { 'content-disposition': `attachment; filename="${zipName}"` });
    }

    return json(res, 404, { error: 'unknown api route', path: url.pathname });
  }

  function file(res, path, extra = {}) {
    res.writeHead(200, { 'content-type': MIME[extname(path).toLowerCase()] || 'application/octet-stream',
                         'cache-control': 'no-cache', ...extra });
    createReadStream(path).pipe(res);
  }

  return createHttpServer(async (req, res) => {
    try {
      const url = new URL(req.url, 'http://localhost');
      if (url.pathname.startsWith('/api/')) return await api(req, res, url);

      let path = decodeURIComponent(url.pathname);
      if (path === '/' || path === '') {
        res.writeHead(302, { location: '/web/' }).end();
        return;
      }
      if (path.endsWith('/')) path += 'index.html';
      const target = normalize(join(root, path));
      if (!target.startsWith(root)) { send(res, 403, 'text/plain', Buffer.from('forbidden')); return; }
      const info = await stat(target).catch(() => null);
      if (!info || !info.isFile()) {
        send(res, 404, 'text/plain', Buffer.from('not found: ' + path));
        return;
      }
      if (target.startsWith(downloads)) {                    // build output is always a download
        return file(res, target, { 'content-disposition': `attachment; filename="${target.split('/').pop()}"` });
      }
      const body = await readFile(target);
      const type = MIME[extname(target).toLowerCase()] || 'application/octet-stream';
      const cache = /\.(bin|png|jpg|jpeg|pt)$/.test(target) ? 'public, max-age=86400' : 'no-cache';
      send(res, 200, type, body, { 'cache-control': cache, 'cross-origin-opener-policy': 'same-origin' });
    } catch (err) {
      send(res, 500, 'text/plain', Buffer.from('server error: ' + (err && err.message ? err.message : err)));
    }
  });
}

/** Listen on a port (0 = pick a free one) and resolve once it is up. */
export async function startServer({ root = PROJECT_ROOT, port = 3000, host = '0.0.0.0',
                                    log = console.log } = {}) {
  const server = createServer({ root, log });
  await new Promise((res, rej) => {
    server.once('error', rej);
    server.listen(port, host, res);
  });
  const actual = server.address().port;
  return { server, port: actual, url: `http://127.0.0.1:${actual}/web/`, root };
}

const isMain = process.argv[1] && resolve(process.argv[1]) === resolve(fileURLToPath(import.meta.url));
if (isMain) {
  const root = PROJECT_ROOT;
  const { port, url } = await startServer({
    root, port: parseInt(process.env.PORT || '3000', 10), host: process.env.HOST || '0.0.0.0',
  });
  console.log(`serving ${root}`);
  console.log(`open ${url}  (model bundles are discovered automatically)`);
  const listed = (await modelManifest(root)).models;
  for (const m of listed) console.log(`  ${(m.available ? m.name : m.name + ' (planned)').padEnd(22)} ` +
    `${String(m.dir).padEnd(18)} ${m.available ? (m.bytes / 1048576).toFixed(2) + ' MiB  ' + m.tasks.join(', ') : m.note || ''}`);
  const models = listed.filter(m => m.available);
  if (!models.length) {
    console.warn('WARNING: no model bundle found - train one (train.cmd) and export it ' +
                 '(export.cmd), the page cannot generate anything without one.');
  }
  const release = await readRelease(root);
  console.log(`release ${release} · downloads: /api/downloads`);
  if (!existsSync(join(root, 'downloads'))) {
    console.log('tip: node tools/make_downloads.mjs pre-builds the versioned model zips');
  }
}
