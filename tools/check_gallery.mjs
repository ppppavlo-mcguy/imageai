/* Unit tests for the gallery rules in web/gallery.js (pure - no browser needed).
     node tools/check_gallery.mjs
   The gallery is what replaced the old single-image preview: it lists every file a chat created
   (images, ascii art, code) and can zip them. These tests pin the names, extensions and bytes. */
import { artifactsFromMessages, artifactName, codeLang, dataUrlBytes, galleryEntries, totalBytes, LANG_EXT }
  from '../web/gallery.js';

let failures = 0;
const check = (name, cond, detail = '') => {
  console.log(`  ${cond ? 'ok  ' : 'FAIL'} ${name}${detail ? '  ' + detail : ''}`);
  if (!cond) failures++;
};

const PNG_1PX = // 1x1 transparent png, base64 payload only
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFAAH/q842iQAAAABJRU5ErkJggg==';
const img = data => `data:image/png;base64,${data}`;

const chat = [
  { role: 'user', kind: 'text', text: 'a blue card and some code' },
  { role: 'bot', kind: 'image', imageData: img(PNG_1PX), width: 64, height: 64, meta: { ms: 1200, tokens: 260 } },
  { role: 'bot', kind: 'info', text: 'model loaded' },
  { role: 'bot', kind: 'code', text: '<!doctype html>\n<div>card</div>', meta: { ms: 300, tokens: 40 } },
  { role: 'bot', kind: 'ascii', text: ' .:-=+*\n#%@ \n', meta: { ms: 200, tokens: 12 } },
  { role: 'bot', kind: 'code', text: 'def go():\n    return 1', meta: { ms: 200, tokens: 9 } },
  { role: 'bot', kind: 'text', text: 'once upon a time' },
  { role: 'bot', kind: 'code', text: 'body { color: red; }', meta: { ms: 100, tokens: 7 } },
];

const arts = artifactsFromMessages(chat);
check('only finished bot answers become files', arts.length === 5, `${arts.length} artifacts`);
check('info + prose messages are skipped',
  !arts.some(a => a.text === 'model loaded') && !arts.some(a => a.text === 'once upon a time'));
check('names are numbered in creation order',
  arts.map(a => a.name).join(',') === 'sgm-1-image.png,sgm-2-code.html,sgm-3-ascii.txt,sgm-4-code.py,sgm-5-code.css',
  arts.map(a => a.name).join(','));
check('images keep their data URL', arts[0].kind === 'image' && arts[0].dataUrl.startsWith('data:image/png'));
check('image bytes come from the base64 payload', arts[0].bytes === dataUrlBytes(img(PNG_1PX)),
  `${arts[0].bytes} bytes`);
check('ascii art is a .txt file', arts[2].kind === 'ascii' && arts[2].bytes === 13, `${arts[2].bytes} bytes`);
check('generation time and token count are kept', arts[1].ms === 300 && arts[1].tokens === 40);

check('html is detected', codeLang('<!doctype html><div>x</div>') === 'html');
check('python is detected', codeLang('def f():\n    return 1') === 'python');
check('css is detected', codeLang('body { color: red; }') === 'css');
check('javascript is detected', codeLang('const a = 1; function b() {}') === 'javascript');
check('everything else is plain text', codeLang('hello there') === 'text');
check('every language has an extension', ['html', 'css', 'javascript', 'python', 'json', 'text']
  .every(l => LANG_EXT[l]));

check('an empty transcript has no files', artifactsFromMessages([]).length === 0);
check('artifactName picks the extension from the kind',
  artifactName({ kind: 'ascii' }, 0) === 'sgm-1-ascii.txt' && artifactName({ kind: 'code', lang: 'css' }, 4) === 'sgm-5-code.css');

const uploads = [{ name: 'photo.png', size: 10 }, { name: 'photo.png', size: 20 }];
const entries = galleryEntries(arts, uploads);
check('the zip keeps created files first and uploads after',
  entries.slice(0, 5).every((e, i) => e.name === arts[i].name) && entries.length === 7,
  `${entries.length} entries`);
check('duplicate upload names are made unique',
  entries[5].name === 'photo.png' && entries[6].name === 'photo-2.png',
  entries.slice(5).map(e => e.name).join(','));
check('totalBytes adds up', totalBytes(arts) === arts.reduce((n, a) => n + a.bytes, 0));

console.log(failures ? `GALLERY FAILED (${failures})` : 'GALLERY OK');
process.exit(failures ? 1 : 0);
