/* Running the build tools (npm, electron-packager) on every platform.
 *
 * Why this is its own module: `node tools/build_desktop.mjs` died on Windows with
 *
 *     Error: spawnSync npm.cmd EINVAL
 *     at execFileSync (node:child_process:954)
 *
 * npm and electron-packager are .cmd shims there, and since the CVE-2024-27980 fix (Node
 * 18.20 / 20.12 / 21.7) spawn()/execFile() refuse to launch a .cmd without a shell. So the build
 * runs its tools through the shell (execSync) instead. Two details matter and are tested in
 * tools/check_desktop.mjs without needing a Windows box:
 *
 *   - the executable name: npm.cmd / electron-packager.cmd on Windows, npm / electron-packager
 *     elsewhere, and the packager is named relative to desktop/ so a space in the checkout path
 *     never reaches the command line,
 *   - quoting: arguments with spaces (electron-packager's --app-copyright=SGM (v1)) are quoted,
 *     because with a shell the whole line is one string.
 */
import { execSync } from 'node:child_process';
import { join } from 'node:path';

/** npm, as it has to be spelled on this platform. */
export const npmCommand = (platform = process.platform) => (platform === 'win32' ? 'npm.cmd' : 'npm');

/** electron-packager, relative to the desktop/ folder (no spaces, whatever the checkout is called). */
export const packagerCommand = (platform = process.platform) =>
  (platform === 'win32' ? 'node_modules\\.bin\\electron-packager.cmd' : './node_modules/.bin/electron-packager');

/** electron-packager's real path, for the "is it installed?" check. */
export const packagerPath = (desktop, platform = process.platform) =>
  join(desktop, platform === 'win32' ? 'node_modules\\.bin\\electron-packager.cmd' : 'node_modules/.bin/electron-packager');

/** Quote one argument for a shell command line (only when it needs it). */
export function quoteArg(value) {
  const s = String(value);
  return /[\s"]/.test(s) ? `"${s}"` : s;
}

/** The exact line a shell would run. */
export function commandLine(file, args = []) {
  return [file, ...args].map(quoteArg).join(' ');
}

/** Run through the shell, with the output going straight to the terminal. */
export function runInShell(file, args, cwd) {
  return execSync(commandLine(file, args), { cwd, stdio: 'inherit' });
}
