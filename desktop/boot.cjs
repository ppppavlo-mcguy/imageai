/* The Electron-free half of the desktop app: work out where the app lives, start web/server.js
 * on a loopback port and read the model list. Kept separate from main.cjs so it can be tested
 * with plain Node (tools/check_desktop.mjs) - no display, no Electron download.
 */
const { existsSync, readFileSync } = require('node:fs');
const { join, resolve } = require('node:path');
const { pathToFileURL } = require('node:url');

/**
 * Where are web/ and the model folders?
 *  - packaged/staged: this file sits next to web/
 *  - development:     this file is desktop/boot.cjs inside the project
 */
function resolveRoot(from = __dirname) {
  if (existsSync(join(from, 'web', 'server.js'))) return from;
  const parent = resolve(from, '..');
  if (existsSync(join(parent, 'web', 'server.js'))) return parent;
  throw new Error(`no web/server.js found next to ${from} or ${parent}`);
}

function readVersion(root = resolveRoot()) {
  try { return readFileSync(join(root, 'VERSION'), 'utf8').trim() || 'v1'; } catch { return 'v1'; }
}

/** Start the bundled static server (the same one the website uses) on a free port. */
async function startLocalServer({ root = resolveRoot(), host = '127.0.0.1', port = 0, log } = {}) {
  const mod = await import(pathToFileURL(join(root, 'web', 'server.js')).href);
  const started = await mod.startServer({ root, port, host, log: log || (() => {}) });
  const models = await mod.discoverModels(root);
  return { ...started, models, version: readVersion(root), module: mod };
}

module.exports = { resolveRoot, readVersion, startLocalServer };
