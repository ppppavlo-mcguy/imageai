#!/usr/bin/env node
/* Static UI checks - the parts of web/index.html + web/app.js that a browser would otherwise
 * have to prove by hand. No DOM needed: the markup and the driver are read as text.
 *
 *     node tools/check_ui.mjs
 */
import { readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import { PROJECT_ROOT } from '../web/server.js';

let failures = 0;
const check = (name, cond, detail = '') => {
  console.log(`  ${cond ? 'ok  ' : 'FAIL'} ${name}${detail ? '  ' + detail : ''}`);
  if (!cond) failures++;
};

const html = readFileSync(join(PROJECT_ROOT, 'web', 'index.html'), 'utf8');
const app = readFileSync(join(PROJECT_ROOT, 'web', 'app.js'), 'utf8');
const css = readFileSync(join(PROJECT_ROOT, 'web', 'style.css'), 'utf8');
const ids = new Set([...html.matchAll(/\bid="([^"]+)"/g)].map(m => m[1]));
const used = [...app.matchAll(/\$\('([^']+)'\)/g)].map(m => m[1]);

const missing = [...new Set(used)].filter(id => !ids.has(id));
check('every element app.js grabs exists in the markup', missing.length === 0, missing.join(', ') || `${ids.size} ids`);

/* ---------------------------------------------------------- tools and routing */
const buttons = [...html.matchAll(/data-task="([^"]+)"/g)].map(m => m[1]);
const tasksBlock = app.match(/const TASKS = \{([\s\S]*?)\n\};/)[1];
const tasks = [...tasksBlock.matchAll(/^\s{2}(\w+): \{/gm)].map(m => m[1]);
check('every tool button has a task entry', buttons.every(t => tasks.includes(t)), buttons.join(', '));
check('every task entry has a button', tasks.every(t => buttons.includes(t)),
  tasks.filter(t => !buttons.includes(t)).join(', ') || 'all wired');
check('the active tool is marked in the markup', /data-task="auto" class="active"/.test(html));
check('unsupported tools are greyed out, never hidden',
  app.includes('markUnsupportedTools') && /classList\.toggle\('unsupported'/.test(app)
  && /\.unsupported/.test(css));

/* -------------------------------------------------------------- new features */
check('the gallery replaced the old single preview card',
  !html.includes('previewCard') && !app.includes('previewCard') && /galleryBtn/.test(html));
check('the gallery can zip what the chat created',
  app.includes('downloadGalleryZip') && app.includes('buildZip') && /galleryZip/.test(html));
check('the model picker is a select, plus a manual folder', /id="modelSelect"/.test(html)
  && /id="modelDir"/.test(html) && app.includes('ensureModel'));
const composer = html.match(/<footer id="composer">([\s\S]*?)<\/footer>/)[1];
const sideBottom = html.match(/<div class="sideBottom">([\s\S]*?)<\/aside>/)[1];
check('the model selector sits in the composer, on the right',
  /id="composer"/.test(html) && composer.includes('id="modelPill"')
  && composer.indexOf('id="modelSelect"') < 0
  && composer.indexOf('id="modelPill"') > composer.indexOf('id="prompt"')
  && composer.indexOf('id="modelPill"') < composer.indexOf('id="send"'),
  'pill between the prompt box and send');
check('the selector is no longer at the bottom of the tools list',
  !sideBottom.includes('modelSelect') && !sideBottom.includes('modelPill'),
  'rail keeps sampling + storage only');
check('the selector opens the model list, not a bare dropdown',
  /\$\('modelPill'\)\.onclick = \(\) => openDownloads\('models'\)/.test(app)
  && /id="dlTabs"/.test(html) && /data-tab="models"/.test(html) && /data-tab="app"/.test(html)
  && app.includes('showDlTab') && app.includes('renderDlModels'));
check('a model row in the list can be chosen, not only downloaded',
  app.includes('chooseModel') && /textContent = 'use'/.test(app)
  && /dlModelRows/.test(html) && app.includes('selected'));
check('answers can stream as they are written',
  /id="streamBtn"/.test(html) && app.includes('setStream') && app.includes("type === 'stream'")
  && app.includes('renderStream'));
check('the token budget is adjustable instead of a hidden 240',
  /id="maxNew"/.test(html) && app.includes("parseInt($('maxNew').value")
  && /maxNew: parseInt/.test(app) && /maxNew/.test(html.match(/<details class="settings">([\s\S]*?)<\/details>/)[1]));
check('your own prompt appears in the chat when it is sent',
  /const userMsg = \{ role: 'user'/.test(app) && /renderInto\(null, userMsg\)/.test(app));
check('asking for another language is not silently answered in html',
  app.includes("from './code.js'") && app.includes('codeInstruction(') && app.includes('languageNote(')
  && readFileSync(join(PROJECT_ROOT, 'web', 'code.js'), 'utf8').includes('export function codeInstruction'),
  'web/code.js shapes the instruction and warns on a mismatch');
check('the model loads on the first message, not at page load',
  /ensureModel\(\)/.test(app) && !/loadModel\(\);/.test(app)
  && app.includes("loads on your first message"));
check('a chat is named after its first prompt',
  app.includes('titleForPrompt') && html.includes('id="roomName"'));
check('clear and delete are wired',
  /id="clearBtn"/.test(html) && /id="deleteBtn"/.test(html)
  && app.includes("$('clearBtn').onclick") && app.includes("$('deleteBtn').onclick"));
check('downloads are offered from the header', /id="dlBtn"/.test(html) && app.includes('openDownloads'));
check('the branding is SGM everywhere it shows',
  /SGM/.test(html) && !/ImgCode/.test(html) && !/ImgCode/.test(app),
  [...html.matchAll(/SGM[^<\n]{0,20}/g)].slice(0, 2).map(m => m[0]).join(' | '));
check('the desktop app can drive the same DOM hooks',
  html.includes('id="newChat"') && app.includes('window.SGM'));

/* ------------------------------------------------------- offline / no network */
const webFiles = readdirSync(join(PROJECT_ROOT, 'web')).filter(f => /\.(js|mjs)$/.test(f));
const offenders = [];
for (const f of webFiles) {
  const text = readFileSync(join(PROJECT_ROOT, 'web', f), 'utf8');
  for (const m of text.matchAll(/https?:\/\/[^\s'"`)]+/g)) {
    if (/^https?:\/\/(localhost|127\.0\.0\.1|www\.w3\.org|example\.com)/.test(m[0])) continue;
    offenders.push(`${f}: ${m[0]}`);
  }
}
check('no external URLs are fetched from web/*.js', offenders.length === 0, offenders.join(', ') || `${webFiles.length} files`);
const remote = [...html.matchAll(/(src|href)="(https?:)?\/\//g)];
check('the page loads nothing from a CDN', remote.length === 0 && !/@import\s+(url\()?["']?https?:/.test(css));
check('only same-origin fetches are used',
  [...app.matchAll(/fetch\('([^']+)'/g)].every(m => m[1].startsWith('./') || m[1].startsWith('/')),
  [...app.matchAll(/fetch\('([^']+)'/g)].map(m => m[1]).join(', '));

console.log(failures ? `UI FAILED (${failures})` : 'UI OK');
process.exit(failures ? 1 : 0);
