/* SGM desktop shell.
 *
 * It is deliberately thin: the very same files the website serves (web/*, the model bundles and
 * web/models.json) are served by web/server.js on a loopback port and shown in an Electron
 * window, so there is exactly one UI to maintain. No node integration reaches the page.
 *
 *   dev      : npm start            (from desktop/, after: npm install)
 *   packaged : SGM.exe runs this file with web/ and the model folders next to it
 */
const { app, BrowserWindow, Menu, dialog, shell } = require('electron');
const { join } = require('node:path');
const { resolveRoot, readVersion, startLocalServer } = require('./boot.cjs');

// Packaged: this file sits next to web/. In development: desktop/ lives inside the project.
const ROOT = resolveRoot();

let win = null;
let server = null;

async function boot() {
  const started = await startLocalServer({ root: ROOT, log: (...a) => console.log('[sgm]', ...a) });
  server = started.server;
  const models = started.models;
  console.log(`[sgm] serving ${ROOT} at ${started.url}`);
  for (const m of models) console.log(`[sgm] model ${m.name} (${m.dir})`);
  if (!models.length) {
    dialog.showErrorBox('No model bundle found',
      `SGM could not find a model bundle next to the app.\nExpected a folder with model.json in:\n${ROOT}`);
  }

  win = new BrowserWindow({
    width: 1320, height: 880, minWidth: 900, minHeight: 600,
    backgroundColor: '#0f1115', title: `SGM — Small Generative Model (${started.version})`,
    autoHideMenuBar: false,
    webPreferences: { contextIsolation: true, nodeIntegration: false, sandbox: true, spellcheck: false },
  });
  win.on('closed', () => { win = null; });
  await win.loadURL(started.url);

  Menu.setApplicationMenu(Menu.buildFromTemplate([
    {
      label: 'File',
      submenu: [
        { label: 'New chat', accelerator: 'CmdOrCtrl+N', click: () => win && win.webContents.executeJavaScript('document.getElementById("newChat").click()') },
        { label: 'Gallery', accelerator: 'CmdOrCtrl+G', click: () => win && win.webContents.executeJavaScript('window.SGM.openGallery("created")') },
        { label: 'Downloads', accelerator: 'CmdOrCtrl+D', click: () => win && win.webContents.executeJavaScript('window.SGM.openDownloads()') },
        { type: 'separator' },
        { label: 'Open model folder', click: () => shell.openPath(ROOT) },
        { label: 'Open downloads folder', click: () => shell.openPath(join(ROOT, 'downloads')) },
        { type: 'separator' },
        { role: 'quit' },
      ],
    },
    {
      label: 'View',
      submenu: [
        { role: 'reload' }, { role: 'forceReload' }, { role: 'toggleDevTools' },
        { type: 'separator' },
        { role: 'resetZoom' }, { role: 'zoomIn' }, { role: 'zoomOut' },
        { type: 'separator' }, { role: 'togglefullscreen' },
      ],
    },
    {
      label: 'Tools',
      submenu: [
        {
          label: 'Which models are here?',
          click: () => dialog.showMessageBox(win, {
            type: 'info', title: 'SGM models', message: 'Export folders found next to the app',
            detail: models.length
              ? models.map(m => `${m.name}  ·  ${(m.bytes / 1048576).toFixed(2)} MiB  ·  ${m.tasks.join(', ') || 'no task list'}\n   ${m.dir}`).join('\n\n')
              : 'none — train one (train.cmd) and export it (export.cmd)',
          }),
        },
        {
          label: 'About SGM',
          click: () => dialog.showMessageBox(win, {
            type: 'info', title: 'SGM', message: 'SGM — Small Generative Model',
            detail: `release ${started.version}\neverything runs locally: no API keys, no external requests, ` +
                    `chats and uploads stay in this machine's browser storage`,
          }),
        },
      ],
    },
  ]));
}

app.whenReady().then(boot).catch(err => {
  dialog.showErrorBox('SGM failed to start', String(err && err.stack ? err.stack : err));
  app.quit();
});
app.on('window-all-closed', () => app.quit());
app.on('before-quit', () => { if (server) server.close(); });
