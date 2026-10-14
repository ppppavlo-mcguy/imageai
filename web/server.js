#!/usr/bin/env node
/* Tiny static server for local use and for free hosts (Render / HelioHost / any Node host).
     node web/server.js            # http://localhost:3000
     PORT=8080 node web/server.js
   It serves the project root, so /web/* and /web_model/* are reachable; "/" shows the UI. */
import { createServer } from 'node:http';
import { readFile, stat } from 'node:fs/promises';
import { extname, join, normalize, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = fileURLToPath(new URL('.', import.meta.url));
const root = resolve(here, '..');
const port = parseInt(process.env.PORT || '3000', 10);
const host = process.env.HOST || '0.0.0.0';

const MIME = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8',
  '.mjs': 'text/javascript; charset=utf-8', '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8', '.bin': 'application/octet-stream',
  '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.svg': 'image/svg+xml',
  '.txt': 'text/plain; charset=utf-8', '.md': 'text/markdown; charset=utf-8',
  '.ico': 'image/x-icon', '.pt': 'application/octet-stream',
};

createServer(async (req, res) => {
  try {
    const url = new URL(req.url, 'http://localhost');
    let path = decodeURIComponent(url.pathname);
    if (path === '/' || path === '') path = '/web/index.html';
    if (path.endsWith('/')) path += 'index.html';
    const target = normalize(join(root, path));
    if (!target.startsWith(root)) { res.writeHead(403).end('forbidden'); return; }
    const info = await stat(target).catch(() => null);
    if (!info || !info.isFile()) { res.writeHead(404, { 'content-type': 'text/plain' }).end('not found: ' + path); return; }
    const body = await readFile(target);
    const type = MIME[extname(target).toLowerCase()] || 'application/octet-stream';
    const cache = /\.(bin|png|jpg|jpeg|pt)$/.test(target) ? 'public, max-age=86400' : 'no-cache';
    res.writeHead(200, {
      'content-type': type, 'content-length': body.length, 'cache-control': cache,
      'cross-origin-opener-policy': 'same-origin',
    });
    res.end(body);
  } catch (err) {
    res.writeHead(500, { 'content-type': 'text/plain' }).end('server error: ' + err.message);
  }
}).listen(port, host, () => {
  console.log(`serving ${root}`);
  console.log(`open http://localhost:${port}/  (model folder default: web_model)`);
});
