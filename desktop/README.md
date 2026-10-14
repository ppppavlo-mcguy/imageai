# SGM desktop app

The same chat UI as the website, in its own window. There is only one UI to maintain: the desktop
shell starts `web/server.js` on a loopback port and loads the exact files `web/` serves, so a change
to the chat UI shows up in both.

```
desktop/
  main.cjs      the Electron window + menu (no node integration in the page)
  boot.cjs      Electron-free bootstrap: where the app lives, start the local server, list models
  package.json  devDependencies (electron + @electron/packager) and the electron-builder config
  build.cmd     Windows: build the .exe and the download zip in one step
```

## Run it from a checkout

```
# once per machine (downloads ~120 MB of Electron)
cd desktop && npm install

# start it (needs a model bundle next to web/, i.e. web_model/ etc.)
npm start
```

`npm start` uses `desktop/main.cjs`, which finds the project root automatically and logs the models
it found. Menu: **File** (New chat, Gallery, Downloads, Open folders, Quit), **View** (reload,
devtools, zoom), **Tools** (which models are here, about).

## Build the Windows .exe

```
node tools\build_desktop.mjs --platform=win32      (or: desktop\build.cmd)
```

1. stages a self-contained app folder (`desktop/build/app`): `main.cjs`, `boot.cjs`, `web/`, every
   exported model bundle and `VERSION` - nothing else, so the app stays small
2. packages it with `@electron/packager` into `desktop/dist/SGM-win32-x64/` (**this is the .exe**
   plus Electron's DLLs)
3. zips that folder to `downloads/sgm-desktop-<release>.zip`, which the web UI offers under
   **Downloads** (`⬇` in the header, `/api/download/app`)

Other targets: `--platform=linux`, `--platform=darwin`, `--platform=all`, `--stage-only` (no
Electron download; good for checking the layout).

A single-file installer instead of a folder: on Windows, in the staged folder,

```
cd desktop\build\app
npx electron-builder --win nsis portable --x64
```

which produces `sgm-desktop-1.0.0-x64.exe` (portable) / an NSIS setup from the `build` block in
`desktop/package.json`.

## Notes

* Chat history and uploads live in the app's own IndexedDB, per machine - nothing is sent anywhere.
* Model downloads are served by the local server: `Downloads → SGM-1M → sgm-1m-v1.zip`.
* `node tools/check_desktop.mjs` verifies this folder without a display: root resolution, the local
  server, the staged layout and (when built) that `SGM.exe` is a real x86-64 PE executable.
