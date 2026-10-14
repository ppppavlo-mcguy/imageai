/* SGM chat driver.
   Everything is local: generation runs in ./worker.js, chats and uploads live in IndexedDB
   (./store.js) and every created file shows up in the gallery (./gallery.js).
   No external requests are made anywhere in this app. */
import {
  LIMITS, planEvictions, readAttachment, attachmentText, fmtBytes, chatBytes,
  sweepExpired, trimChats, listChats, getChat, putChat, deleteChat, filesFor, putFile,
  deleteFile, dropFiles, titleForPrompt, DEFAULT_TITLE,
} from './store.js';
import { artifactsFromMessages, codeLang, LANG_EXT } from './gallery.js';
import { buildZip, versionedName } from './zip.js';
import { mergeModels, modelUrl, formatParams } from './models.js';

const $ = id => document.getElementById(id);
const TASKS = {
  auto: { label: 'auto' },
  image: { label: 'image', model: 'image', examples: ['a big yellow car on a blue background', 'a small red tree on a pink background'] },
  code: { label: 'code', model: 'code', examples: ['make a snake game in javascript', 'write css for a dark page', 'python to reverse a string'] },
  fix: { label: 'fix an error', model: 'code', examples: ['i got this error: SyntaxError: Unexpected end of input', 'fix this, i got an error: NameError: name is not defined'] },
  ascii2code: { label: 'ascii → website', model: 'ascii2code', examples: ['turn this ascii art into a website'] },
  caption: { label: 'describe image', model: 'caption', examples: ['what is on this picture?'] },
  img2code: { label: 'image → website', model: 'img2code', examples: ['build a web page that looks like this'] },
  ascii: { label: 'ascii art', model: 'ascii', examples: ['turn this into an ascii text art'] },
  code2img: { label: 'website → image', examples: ['turn this website into an image preview'] },
  text: { label: 'text', examples: ['the little robot waited by the window'] },
};
const NEEDS_IMAGE = new Set(['caption', 'ascii', 'img2code']);

const state = {
  task: 'auto', busy: false, trained: [],
  chat: null, attachments: [], worker: null, seq: 0, theme: 'dark',
  models: [], modelId: null, release: 'v1', loadedDir: null, loadedName: null,
  loadPromise: null, loadResolve: null, loadReject: null, ready: false,
  galleryTab: 'created', downloads: null,
};
const nodes = new Map();

/* ------------------------------------------------------------------ theme */
function applyTheme(theme) {
  state.theme = theme;
  document.documentElement.dataset.theme = theme;
  try { localStorage.setItem('sgm.theme', theme); } catch {}
}
function toggleTheme() { applyTheme(state.theme === 'dark' ? 'light' : 'dark'); }
function remember(key, value) { try { localStorage.setItem(key, value); } catch {} }
function recall(key, dflt = null) { try { return localStorage.getItem(key) ?? dflt; } catch { return dflt; } }

/* ------------------------------------------------------------- attachments */
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
function renderAttachments() {
  const box = $('attachList');
  box.innerHTML = '';
  box.classList.toggle('hidden', state.attachments.length === 0);
  for (const f of state.attachments) {
    const chip = document.createElement('div');
    chip.className = 'attach';
    if (f.preview) {
      const img = document.createElement('img');
      img.src = f.preview;
      chip.appendChild(img);
    } else {
      const t = document.createElement('div');
      t.className = 'txt';
      t.textContent = (f.name.split('.').pop() || '?').slice(0, 4);
      chip.appendChild(t);
    }
    const name = document.createElement('div');
    name.className = 'attachName';
    name.innerHTML = `<b>${escapeHtml(f.name)}</b><span class="tiny muted">${fmtBytes(f.size)}</span>`;
    chip.appendChild(name);
    const rm = document.createElement('button');
    rm.className = 'ghost tiny';
    rm.textContent = '×';
    rm.onclick = () => removeAttachment(f.id);
    chip.appendChild(rm);
    box.appendChild(chip);
  }
  updateUsage();
  updateGalleryBadge();
}
async function removeAttachment(id) {
  state.attachments = state.attachments.filter(f => f.id !== id);
  try { await deleteFile(id); } catch {}
  renderAttachments();
}
function messageBytes() {
  return chatBytes([], state.chat ? state.chat.messages : []);
}
async function addFiles(fileList) {
  const files = [...fileList];
  for (const file of files) {
    let rec;
    try { rec = await readAttachment(file); }
    catch (e) { note(`could not read ${file.name}: ${e.message}`); continue; }
    const plan = planEvictions(state.attachments, { name: rec.name, size: rec.size }, LIMITS, messageBytes());
    if (plan.rejected) { note(plan.rejected); continue; }
    for (const gone of plan.evict) {
      state.attachments = state.attachments.filter(f => f.id !== gone.id);
      try { await dropFiles([gone.id]); } catch {}
      note(`limit reached — dropped the oldest attachment (${gone.name}) to fit ${rec.name}`);
    }
    if (rec.blob && /^image\//.test(rec.type)) rec.preview = await thumbnail(rec.blob);
    try { await putFile({ ...rec, chatId: state.chat.id, blob: rec.blob }); }
    catch (e) { note(`could not store ${rec.name}: ${e.message}`); continue; }
    state.attachments.push(rec);
  }
  renderAttachments();
}
async function thumbnail(blob) {
  try {
    const bmp = await createImageBitmap(blob);
    const c = document.createElement('canvas');
    c.width = c.height = 48;
    c.getContext('2d').drawImage(bmp, 0, 0, 48, 48);
    return c.toDataURL('image/png');
  } catch { return null; }
}
async function imageToRgba(rec) {
  const blob = rec.blob;
  if (!blob) return null;
  const bmp = await createImageBitmap(blob);
  const size = 64;
  const c = document.createElement('canvas');
  c.width = c.height = size;
  c.getContext('2d').drawImage(bmp, 0, 0, size, size);
  return { rgba: c.getContext('2d').getImageData(0, 0, size, size).data, width: size };
}

/* ------------------------------------------------------------------ chats */
function newChat({ persist = true } = {}) {
  state.chat = { id: `c${Date.now().toString(36)}`, title: DEFAULT_TITLE, createdAt: Date.now(),
                 updatedAt: Date.now(), messages: [] };
  state.attachments = [];
  nodes.clear();
  $('messages').innerHTML = '';
  hero();
  $('roomName').textContent = DEFAULT_TITLE;
  renderAttachments();
  updateGalleryBadge();
  if (persist) saveChat();
  return state.chat;
}
function saveChat() {
  if (!state.chat) return null;
  state.chat.updatedAt = Date.now();
  remember('sgm.chat', state.chat.id);
  const payload = { ...state.chat, files: undefined };
  return putChat(payload).catch(() => null).then(() => updateUsage());
}
function pushMessage(msg) {
  state.chat.messages.push(msg);
  // A chat is named after the first thing you asked for, not "New chat" forever.
  if ((!state.chat.title || state.chat.title === DEFAULT_TITLE) && msg.role === 'user') {
    const name = titleForPrompt(msg.text, msg.files || []);
    if (name && name !== DEFAULT_TITLE) {
      state.chat.title = name;
      $('roomName').textContent = name;
    }
  }
  return saveChat();
}
async function useChat(id) {
  const chat = await getChat(id);
  if (!chat) return;
  state.chat = chat;
  const files = await filesFor(id);
  for (const f of files) if (f.blob && /^image\//.test(f.type)) f.preview = await thumbnail(f.blob);
  state.attachments = files;
  nodes.clear();
  $('messages').innerHTML = '';
  if (!chat.messages.length) hero();
  for (const m of chat.messages) renderInto(null, m);
  $('roomName').textContent = chat.title || DEFAULT_TITLE;
  renderAttachments();
  updateGalleryBadge();
  hideChatList();
}
async function renderChatList() {
  const box = $('chatList');
  const chats = await listChats();
  box.innerHTML = '';
  if (!chats.length) box.innerHTML = '<p class="tiny muted" style="margin:8px">no saved chats yet</p>';
  const now = Date.now();
  for (const c of chats) {
    const row = document.createElement('div');
    row.className = 'chatRow';
    const age = Math.round((now - (c.updatedAt || 0)) / 60000);
    const open = document.createElement('button');
    open.innerHTML = `<b>${escapeHtml(c.title || DEFAULT_TITLE)}</b><br><span class="tiny muted">${c.messages.length} messages · ${age < 60 ? age + ' min' : Math.round(age / 60) + ' h'} ago</span>`;
    open.onclick = () => useChat(c.id);
    const del = document.createElement('span');
    del.className = 'del';
    del.textContent = 'delete';
    del.onclick = async () => {
      for (const f of await filesFor(c.id)) await deleteFile(f.id);
      await deleteChat(c.id);
      if (state.chat && state.chat.id === c.id) newChat();
      renderChatList();
    };
    row.append(open, del);
    box.appendChild(row);
  }
  box.classList.toggle('hidden');
}
const hideChatList = () => $('chatList').classList.add('hidden');

function clearChat() {
  state.chat.messages = [];
  state.chat.title = DEFAULT_TITLE;
  for (const f of state.attachments) deleteFile(f.id).catch(() => {});
  state.attachments = [];
  nodes.clear();
  $('messages').innerHTML = '';
  hero();
  $('roomName').textContent = DEFAULT_TITLE;
  renderAttachments();
  updateGalleryBadge();
  saveChat();
}
async function deleteChatAndReset() {
  const id = state.chat.id;
  for (const f of await filesFor(id)) await deleteFile(f.id);
  await deleteChat(id);
  newChat();
}

/* ----------------------------------------------------------------- models */
function currentModel() {
  return state.models.find(m => m.id === state.modelId) || state.models.find(m => m.available) || null;
}
function renderModelSelect() {
  const sel = $('modelSelect');
  sel.innerHTML = '';
  if (!state.models.length) {
    sel.innerHTML = '<option value="">no bundles found</option>';
    $('modelNote').textContent = 'train one with train.cmd, then export.cmd';
    $('modelInfo').textContent = 'no model bundle found next to web/';
    return;
  }
  for (const m of state.models) {
    const o = document.createElement('option');
    o.value = m.id;
    o.disabled = !m.available;
    o.textContent = m.available
      ? `${m.name} · ${formatParams(m.params)} · ${(m.bytes / 1048576).toFixed(2)} MiB`
      : `${m.name} — not exported yet`;
    sel.appendChild(o);
  }
  const usable = state.models.filter(m => m.available);
  if (!usable.some(m => m.id === state.modelId)) state.modelId = usable.length ? usable[0].id : null;
  if (state.modelId) sel.value = state.modelId;
  const m = currentModel();
  if (m) {
    $('modelNote').textContent = m.note || '';
    $('modelInfo').innerHTML = `<b>${escapeHtml(m.name)}</b> · ${(m.bytes / 1048576).toFixed(2)} MiB · `
      + `${m.bits}-bit${m.tasks && m.tasks.length ? ' · ' + escapeHtml(m.tasks.slice(0, 6).join('/')) : ''}`
      + (m.available ? '' : '<br>this bundle has not been exported yet');
  }
  $('modelDir').value = m && m.dir ? m.dir : 'web_model';
  markUnsupportedTools();
}
/** Grey out the tools the loaded model was never trained for (nothing is hidden). */
function markUnsupportedTools() {
  const tasks = state.ready ? state.trained : [];
  for (const b of $('capList').querySelectorAll('button[data-task]')) {
    const need = TASKS[b.dataset.task] && TASKS[b.dataset.task].model;
    const unsupported = !!need && tasks.length > 0 && !tasks.includes(need) && !tasks.includes(b.dataset.task);
    b.classList.toggle('unsupported', unsupported);
    b.title = unsupported ? `${state.loadedName || 'this model'} was not trained for ${b.dataset.task}` : '';
  }
}
async function refreshModels() {
  let manifest = null, found = [];
  try { manifest = await (await fetch('./models.json', { cache: 'no-store' })).json(); } catch {}
  try {
    const r = await fetch('/api/models', { cache: 'no-store' });
    if (r.ok) {
      const data = await r.json();
      found = data.models || [];
      state.release = data.release || state.release;
    }
  } catch { /* static host without the bundled server: the manifest alone is enough */ }
  const merged = mergeModels(manifest || {}, found, modelUrl);
  state.models = merged.models;
  const remembered = recall('sgm.model');
  state.modelId = merged.models.some(m => m.id === remembered && m.available) ? remembered
    : merged.default;
  renderModelSelect();
}
function startWorker() {
  if (state.worker) return true;
  try {
    state.worker = new Worker('./worker.js', { type: 'module' });
    state.worker.onmessage = onWorker;
    state.worker.onerror = e => {
      setStatus('worker error: ' + e.message, 'err');
      if (state.loadReject) state.loadReject(new Error(e.message));
      botError(nodes.get(state.seq), String(e.message));
    };
    return true;
  } catch (e) {
    setStatus('no module workers — run "node web/server.js"', 'err');
    return false;
  }
}
/** Load the selected bundle if it is not already in memory (this is what "first message" does). */
async function ensureModel() {
  const m = currentModel();
  if (!m) throw new Error('no model bundle found — train and export one first (train.cmd, export.cmd)');
  if (state.ready && state.loadedDir === m.url) return m;
  if (!startWorker()) throw new Error('this browser needs module workers — serve the app with "node web/server.js"');
  if (state.loadPromise && state.loadedDir !== m.url) await state.loadPromise.catch(() => {});
  if (state.ready && state.loadedDir === m.url) return m;
  state.ready = false;
  $('modelState').textContent = `loading ${m.name}…`;
  setStatus(`loading ${m.name}…`);
  state.loadPromise = new Promise((resolve, reject) => { state.loadResolve = resolve; state.loadReject = reject; });
  state.worker.postMessage({ cmd: 'load', dir: m.url, id: m.id, name: m.name });
  await state.loadPromise;
  return m;
}
function setStatus(t, kind = '') {
  $('status').textContent = t;
  $('dot').className = 'dot' + (kind === 'ok' ? ' ok' : kind === 'err' ? ' err' : '');
}
function onWorker(ev) {
  const m = ev.data;
  if (m.type === 'loaded') {
    state.ready = true;
    state.loadedDir = m.dir || null;
    state.loadedName = m.name || 'model';
    state.trained = Array.isArray(m.tasks) ? m.tasks : [];
    state.modelId = state.models.find(x => x.url === m.dir)?.id || state.modelId;
    $('modelState').textContent = `${state.loadedName} ready`;
    const a = m.meta.architecture;
    const gz = m.meta.weight_bytes_gz ? ` · ${(m.meta.weight_bytes_gz / 1048576).toFixed(2)} MiB gz` : '';
    $('modelInfo').textContent = `${m.meta.parameter_count.toLocaleString()} params · `
      + `${(m.meta.weight_bytes / 1048576).toFixed(2)} MiB int8${gz} · ${a.embed_size}d/${a.num_heads}h/${a.num_layers}L`;
    $('trainedPill').hidden = !state.trained.length;
    if (state.trained.length) $('trainedPill').textContent = 'trained for: ' + state.trained.join(', ');
    setStatus(`ready in ${m.ms.toFixed(0)} ms`, 'ok');
    markUnsupportedTools();
    if (state.loadResolve) { const r = state.loadResolve; state.loadResolve = state.loadReject = null; r(m); }
    return;
  }
  if (m.type === 'progress') {
    $('progress').style.width = Math.round(100 * m.done / m.total) + '%';
    const node = nodes.get(state.seq);
    if (node && node.thinking) node.thinking.textContent = `generating… ${m.done}/${m.total} tokens`;
  } else if (m.type === 'result') {
    $('progressWrap').classList.add('hidden');
    $('progress').style.width = '0';
    state.busy = false;
    $('send').disabled = false;
    setStatus(`done in ${(m.ms / 1000).toFixed(1)} s`, 'ok');
    const msg = resultToMessage(m);
    pushMessage(msg);
    renderInto(nodes.get(state.seq), msg);
    updateGalleryBadge();
  } else if (m.type === 'error') {
    $('progressWrap').classList.add('hidden');
    state.busy = false;
    $('send').disabled = false;
    setStatus('error', 'err');
    if (state.loadReject && !state.ready) {
      const reject = state.loadReject;
      state.loadResolve = state.loadReject = null;
      reject(new Error(m.message));
    }
    botError(nodes.get(state.seq), m.message);
  }
}
function resultToMessage(m) {
  if (m.kind === 'image') {
    const px = new Uint8ClampedArray(m.rgba);
    const src = document.createElement('canvas');
    src.width = m.width; src.height = m.height;
    src.getContext('2d').putImageData(new ImageData(px, m.width, m.height), 0, 0);
    const big = document.createElement('canvas');
    big.width = big.height = 128;
    const ctx = big.getContext('2d');
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(src, 0, 0, 128, 128);
    return { role: 'bot', kind: 'image', text: '', imageData: big.toDataURL('image/png'),
             width: m.width, height: m.height, model: state.loadedName,
             meta: { ms: m.ms, tokens: m.tokens } };
  }
  const kind = m.kind === 'code' ? 'code' : (m.kind === 'ascii' ? 'ascii' : 'text');
  return { role: 'bot', kind, text: m.text || '', model: state.loadedName,
           meta: { ms: m.ms, tokens: m.tokens } };
}

/* --------------------------------------------------------------- rendering */
function hero() {
  const h = document.createElement('div');
  h.className = 'hero';
  h.innerHTML = `<h2>What should SGM make?</h2>
    <p>A small transformer running in this browser tab — no server compute, no API keys, no external
    requests. Pick a model on the left, ask for an image or code, attach a file, or paste code and
    say “fix this, i got an error”.</p>
    <p><code>a big white car on a blue background</code> · <code>make a snake game</code> ·
    <code>turn this ASCII art into a website</code></p>`;
  $('messages').appendChild(h);
}
function shell(role, who) {
  const wrap = document.createElement('div');
  wrap.className = 'msg ' + role;
  const av = document.createElement('div');
  av.className = 'avatar';
  av.textContent = role === 'user' ? 'you' : '◈';
  const body = document.createElement('div');
  body.className = 'bubbleWrap';
  const w = document.createElement('div');
  w.className = 'who';
  w.textContent = who;
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  body.append(w, bubble);
  wrap.append(av, body);
  return { wrap, bubble, body };
}
const scrollDown = () => { const m = $('messages'); m.scrollTop = m.scrollHeight; };

/** The gallery file a given message became (so the download link carries the same name). */
function artifactForMessage(msg) {
  const msgs = state.chat ? state.chat.messages : [];
  const idx = msgs.indexOf(msg);
  const slice = idx >= 0 ? msgs.slice(0, idx + 1) : msgs;
  const list = artifactsFromMessages(slice);
  return list.length ? list[list.length - 1] : null;
}

function renderInto(node, msg) {
  if (!node) {
    node = shell(msg.role === 'user' ? 'user' : 'bot', msg.role === 'user' ? 'you' : 'SGM');
    $('messages').appendChild(node.wrap);
  }
  const bubble = node.bubble;
  bubble.innerHTML = '';
  if (msg.role === 'user') {
    if (msg.files && msg.files.length) {
      const fileNote = document.createElement('div');
      fileNote.className = 'attachedNote';
      fileNote.textContent = '📎 ' + msg.files.join(', ');
      bubble.parentNode.insertBefore(fileNote, bubble);
    }
    bubble.textContent = msg.text || '(attachment only)';
    scrollDown();
    return node;
  }
  if (msg.kind === 'info') { bubble.textContent = msg.text; scrollDown(); return node; }
  if (msg.kind === 'image' && msg.imageData) {
    const img = document.createElement('img');
    img.className = 'resultImage';
    img.src = msg.imageData;
    img.style.width = 'min(420px, 100%)';
    img.alt = 'generated image';
    bubble.appendChild(img);
    const dl = document.createElement('div');
    dl.className = 'rowActions';
    const a = document.createElement('a');
    const art = artifactForMessage(msg);
    a.href = msg.imageData;
    a.download = art ? art.name : 'sgm-image.png';
    a.textContent = 'Download PNG';
    dl.appendChild(a);
    const gal = document.createElement('button');
    gal.textContent = 'Show in gallery';
    gal.onclick = () => openGallery('created');
    dl.appendChild(gal);
    bubble.appendChild(dl);
    const meta = document.createElement('div');
    meta.className = 'meta';
    meta.textContent = `${msg.width}×${msg.height} output${msg.model ? ' · ' + msg.model : ''}`
      + `${msg.meta ? ` · ${(msg.meta.ms / 1000).toFixed(1)}s` : ''}`;
    bubble.appendChild(meta);
    scrollDown();
    return node;
  }
  const text = msg.text || '';
  const looksHtml = /<!doctype html|<html|<body|<div|<h1|<style|<script|<button/i.test(text);
  if (msg.kind === 'ascii') {
    const pre = document.createElement('pre');
    pre.className = 'asciiOut';
    pre.textContent = text;
    bubble.appendChild(pre);
  } else if (msg.kind === 'code' || looksHtml) {
    const card = document.createElement('div');
    card.className = 'codeCard';
    const head = document.createElement('div');
    head.className = 'codeHead';
    const lang = codeLang(text);
    head.innerHTML = `<span>${lang}</span><span class="spacer"></span><span>${text.length} chars</span>`;
    const pre = document.createElement('pre');
    pre.textContent = text;
    card.append(head, pre);
    bubble.appendChild(card);
  } else {
    bubble.textContent = text || '(empty)';
  }
  const actions = document.createElement('div');
  actions.className = 'rowActions';
  const copy = document.createElement('button');
  copy.textContent = 'Copy';
  copy.onclick = async () => {
    try { await navigator.clipboard.writeText(text); copy.textContent = 'Copied ✓'; }
    catch { copy.textContent = 'select + copy'; }
    setTimeout(() => { copy.textContent = 'Copy'; }, 1400);
  };
  actions.appendChild(copy);
  if (looksHtml || msg.kind === 'code') {
    const pv = document.createElement('button');
    pv.textContent = 'Preview as page';
    pv.onclick = () => {
      const old = bubble.querySelector('.pagePreview');
      if (old) { old.remove(); pv.textContent = 'Preview as page'; return; }
      const frame = document.createElement('iframe');
      frame.className = 'pagePreview';
      frame.sandbox = 'allow-scripts allow-forms';
      frame.srcdoc = /<html/i.test(text) ? text
        : `<!doctype html><html><head><meta charset="utf-8"><style>body{font-family:system-ui;padding:16px}${/body|\.|#/.test(text) ? text : ''}</style></head><body>${text}</body></html>`;
      bubble.appendChild(frame);
      pv.textContent = 'Hide preview';
    };
    actions.appendChild(pv);
  }
  const gal = document.createElement('button');
  gal.textContent = 'Show in gallery';
  gal.onclick = () => openGallery('created');
  actions.appendChild(gal);
  bubble.appendChild(actions);
  if (msg.meta) {
    const meta = document.createElement('div');
    meta.className = 'meta';
    meta.textContent = `${(msg.meta.ms / 1000).toFixed(1)}s · ${msg.meta.tokens} tokens`
      + `${msg.model ? ' · ' + msg.model : ''}`;
    bubble.appendChild(meta);
  }
  scrollDown();
  return node;
}
function note(text) {
  pushMessage({ role: 'bot', kind: 'info', text: '⚠ ' + text });
  renderInto(null, state.chat.messages[state.chat.messages.length - 1]);
}
function botError(node, message) {
  if (!node) return;
  node.bubble.innerHTML = '';
  const e = document.createElement('div');
  e.className = 'errBox';
  e.textContent = '⚠ ' + message;
  node.bubble.appendChild(e);
  scrollDown();
}

/* ----------------------------------------------------------------- gallery */
const chatArtifacts = () => artifactsFromMessages(state.chat ? state.chat.messages : []);
function updateGalleryBadge() {
  const created = chatArtifacts().length;
  const total = created + state.attachments.length;
  const badge = $('galleryCount');
  badge.textContent = String(total);
  badge.classList.toggle('live', total > 0);
  const cell = document.querySelector('.galleryBtn');
  if (cell) cell.title = `${created} generated file(s) · ${state.attachments.length} upload(s)`;
}
function openGallery(tab = 'created') {
  state.galleryTab = tab;
  renderGallery();
  $('galleryModal').classList.remove('hidden');
}
const closeGallery = () => $('galleryModal').classList.add('hidden');

function tileFor(art) {
  const tile = document.createElement('div');
  tile.className = 'tile';
  const thumb = document.createElement('div');
  thumb.className = 'thumb';
  if (art.kind === 'image') {
    const img = document.createElement('img');
    img.src = art.dataUrl;
    img.alt = art.name;
    thumb.appendChild(img);
  } else if (art.kind === 'ascii') {
    const pre = document.createElement('pre');
    pre.textContent = art.text.split('\n').slice(0, 16).join('\n');
    thumb.appendChild(pre);
  } else {
    const ext = document.createElement('div');
    ext.className = 'ext';
    ext.textContent = '.' + (LANG_EXT[art.lang] || 'txt');
    thumb.appendChild(ext);
  }
  const name = document.createElement('div');
  name.className = 'tileName';
  name.textContent = art.name;
  const info = document.createElement('div');
  info.className = 'tiny muted';
  info.textContent = `${fmtBytes(art.bytes)}${art.width ? ` · ${art.width}×${art.height}` : ''}`
    + `${art.tokens ? ` · ${art.tokens} tokens` : ''}`;
  const row = document.createElement('div');
  row.className = 'tileRow';
  row.appendChild(downloadLink(art.name, art.kind === 'image' ? art.dataUrl : art.text));
  if (art.kind !== 'image') {
    const copy = document.createElement('button');
    copy.textContent = 'copy';
    copy.onclick = async () => {
      try { await navigator.clipboard.writeText(art.text); copy.textContent = 'copied ✓'; }
      catch { copy.textContent = 'select it'; }
      setTimeout(() => { copy.textContent = 'copy'; }, 1400);
    };
    row.appendChild(copy);
  }
  tile.append(thumb, name, info, row);
  return tile;
}
function downloadLink(name, data) {
  const a = document.createElement('a');
  a.href = typeof data === 'string' && data.startsWith('data:')
    ? data
    : URL.createObjectURL(new Blob([data], { type: 'text/plain;charset=utf-8' }));
  a.download = name;
  a.textContent = 'download';
  return a;
}
function renderGallery() {
  const grid = $('galleryGrid');
  grid.innerHTML = '';
  for (const b of document.querySelectorAll('#galleryModal .tabs button')) {
    b.classList.toggle('active', b.dataset.tab === state.galleryTab);
  }
  const created = chatArtifacts();
  if (state.galleryTab === 'created') {
    if (!created.length) {
      grid.innerHTML = '<p class="empty">nothing generated in this chat yet — ask SGM for an '
        + 'image, some code or an ASCII art and it will show up here.</p>';
    }
    for (const art of created) grid.appendChild(tileFor(art));
    $('galleryMeta').textContent = `${created.length} file(s) · ${fmtBytes(created.reduce((n, a) => n + a.bytes, 0))}`;
  } else {
    if (!state.attachments.length) {
      grid.innerHTML = '<p class="empty">no files attached to this chat.</p>';
    }
    for (const f of state.attachments) {
      const tile = document.createElement('div');
      tile.className = 'tile';
      const thumb = document.createElement('div');
      thumb.className = 'thumb';
      if (f.preview) {
        const img = document.createElement('img');
        img.src = f.preview;
        thumb.appendChild(img);
      } else {
        const ext = document.createElement('div');
        ext.className = 'ext';
        ext.textContent = '.' + (f.name.split('.').pop() || '?').slice(0, 5);
        thumb.appendChild(ext);
      }
      const name = document.createElement('div');
      name.className = 'tileName';
      name.textContent = f.name;
      const info = document.createElement('div');
      info.className = 'tiny muted';
      info.textContent = `${fmtBytes(f.size)} · ${f.text ? 'read as text' : 'stored as file'}`;
      const row = document.createElement('div');
      row.className = 'tileRow';
      const a = document.createElement('a');
      if (f.text != null) {
        a.href = URL.createObjectURL(new Blob([f.text], { type: 'text/plain;charset=utf-8' }));
      } else {
        a.href = URL.createObjectURL(f.blob);
      }
      a.download = f.name;
      a.textContent = 'download';
      const rm = document.createElement('button');
      rm.textContent = 'delete';
      rm.onclick = async () => { await removeAttachment(f.id); renderGallery(); };
      row.append(a, rm);
      tile.append(thumb, name, info, row);
      grid.appendChild(tile);
    }
    $('galleryMeta').textContent = `${state.attachments.length} upload(s) · `
      + fmtBytes(state.attachments.reduce((n, f) => n + f.size, 0));
  }
}
/** base64 data: URL -> bytes (for packing generated images into the zip). */
function dataUrlToBytes(dataUrl) {
  const b64 = String(dataUrl).slice(String(dataUrl).indexOf(',') + 1);
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}
async function artifactBytes(art) {
  if (art.kind === 'image') return dataUrlToBytes(art.dataUrl);
  return new TextEncoder().encode(art.text || '');
}
async function downloadGalleryZip() {
  const created = chatArtifacts();
  const files = [];
  for (const art of created) files.push({ name: art.name, data: await artifactBytes(art) });
  for (const f of state.attachments) {
    const data = f.text != null ? new TextEncoder().encode(f.text)
      : new Uint8Array(await f.blob.arrayBuffer());
    let name = f.name, i = 2;
    while (files.some(x => x.name === name)) name = f.name.replace(/(\.[^.]+)?$/, `-${i++}$1`);
    files.push({ name, data });
  }
  if (!files.length) { note('nothing to zip yet — generate something first'); return; }
  const title = (state.chat.title || 'chat').slice(0, 40);
  files.push({ name: 'README.txt',
    data: `SGM (Small Generative Model) chat export\r\ncreated: ${new Date().toISOString()}\r\n` +
      `model: ${state.loadedName || 'not loaded'}\r\nfiles: ${files.length}\r\n` +
      `Everything here was generated in a browser tab by ${state.loadedName || 'SGM'}.\r\n` });
  const btn = $('galleryZip');
  btn.disabled = true;
  btn.textContent = 'zipping…';
  try {
    const { bytes } = await buildZip(files);
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([bytes], { type: 'application/zip' }));
    a.download = versionedName(`sgm-${title}`, state.release);
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 30_000);
    btn.textContent = `zipped ${files.length} files`;
  } catch (e) {
    btn.textContent = 'zip failed';
    note('could not build the zip: ' + e.message);
  }
  setTimeout(() => { btn.disabled = false; btn.textContent = 'Download all (.zip)'; }, 1600);
}

/* --------------------------------------------------------------- downloads */
async function openDownloads() {
  $('dlModal').classList.remove('hidden');
  $('dlRelease').textContent = `release ${state.release}`;
  const body = $('dlBody');
  body.innerHTML = '<p class="tiny muted">checking what is available…</p>';
  let data = null;
  try {
    const r = await fetch('/api/downloads', { cache: 'no-store' });
    if (r.ok) data = await r.json();
  } catch { /* static host */ }
  state.downloads = data;
  body.innerHTML = '';
  if (!data) {
    body.innerHTML = '<p class="tiny muted">the download API is served by <code>node web/server.js</code>. '
      + 'On a static host, fetch the zips from <code>/downloads/</code> after running '
      + '<code>node tools/make_downloads.mjs</code>.</p>';
    return;
  }
  const app = document.createElement('div');
  app.className = 'dlRow';
  app.innerHTML = `<b>SGM desktop app</b><span class="tiny muted">${
    data.app ? `${(data.app.bytes / 1048576).toFixed(1)} MB · ${data.app.name}` : 'not compiled yet'
  }</span><span class="spacer"></span>`;
  if (data.app) {
    const a = document.createElement('a');
    a.className = 'button';
    a.href = data.app.url;
    a.textContent = 'Download .exe';
    app.appendChild(a);
  } else {
    const hint = document.createElement('span');
    hint.className = 'tiny muted';
    hint.textContent = 'build it with: ' + (data.buildApp || 'node tools/build_desktop.mjs');
    app.appendChild(hint);
  }
  body.appendChild(app);

  const head = document.createElement('p');
  head.className = 'tiny muted';
  head.style.margin = '6px 0 0';
  head.textContent = 'model bundles (.zip) — unzip, drop the folder next to web/, then pick it in the model box';
  body.appendChild(head);
  for (const m of data.models || []) {
    const row = document.createElement('div');
    row.className = 'dlRow';
    row.innerHTML = `<b>${escapeHtml(m.name)}</b>`
      + `<span class="tiny muted">${(m.bytes / 1048576).toFixed(2)} MiB · ${escapeHtml(m.dir)}</span>`
      + `<span class="spacer"></span>`;
    const a = document.createElement('a');
    a.className = 'button secondary';
    a.href = m.url;
    a.textContent = `download ${m.file}`;
    row.appendChild(a);
    body.appendChild(row);
  }
}

/* ------------------------------------------------------------- task + send */
function selectTask(task) {
  state.task = task;
  [...$('capList').querySelectorAll('button')].forEach(b => b.classList.toggle('active', b.dataset.task === task));
  $('taskPill').textContent = TASKS[task].label;
  const ex = $('examples');
  ex.innerHTML = '';
  for (const e of (TASKS[task].examples || TASKS.image.examples)) {
    const b = document.createElement('button');
    b.textContent = e;
    b.onclick = () => { $('prompt').value = e; $('prompt').focus(); autosize(); };
    ex.appendChild(b);
  }
}
function alertWords(t) { return /error|traceback|exception|fails?|broken|not work|crash|bug/.test(t); }
function route(text) {
  if (state.task !== 'auto') return state.task;
  const t = text.toLowerCase();
  const hasText = state.attachments.some(f => f.text);
  const hasImage = state.attachments.some(f => f.blob && /^image\//.test(f.type));
  if (alertWords(t) && (hasText || /code|function|def |import |<|>|\{|\}/.test(text))) return 'fix';
  if (hasImage) return /ascii|text art/.test(t) ? 'ascii' : /website|html|page|build/.test(t) ? 'img2code' : 'caption';
  if (/ascii art|ascii/.test(t) && /turn|convert|into|website|page|html/.test(t)) return 'ascii2code';
  if (/^(make|write|build|create|code|give me|generate).*(snake|game|app|page|html|css|javascript|js|python|function|script|website|site)/.test(t)) return 'code';
  if (/preview image|image preview|render.*image/.test(t)) return 'code2img';
  if (/continue|story|poem|paragraph|essay/.test(t)) return 'text';
  return 'image';
}
function autosize() {
  const p = $('prompt');
  p.style.height = 'auto';
  p.style.height = Math.min(180, p.scrollHeight) + 'px';
}
async function send() {
  if (state.busy) return;
  const text = $('prompt').value.trim();
  const task = route(text);
  const imageRec = state.attachments.find(f => f.blob && /^image\//.test(f.type));
  if (NEEDS_IMAGE.has(task) && !imageRec) { note('that tool needs an image — attach one with the ＋ button'); return; }
  if (!text && !state.attachments.length) return;
  if (task === 'fix' && !state.attachments.some(f => f.text) && !text) { note('paste the snippet or attach the file that failed'); return; }
  if (!currentModel()) { note('no model bundle found — train one, export it, then reload this page'); return; }

  const fileNames = state.attachments.map(f => f.name);
  $('prompt').value = '';
  autosize();
  state.busy = true;
  $('send').disabled = true;
  pushMessage({ role: 'user', kind: 'text', text, files: fileNames });

  const node = shell('bot', 'SGM');
  $('messages').appendChild(node.wrap);
  node.thinking = document.createElement('span');
  node.thinking.className = 'thinking';
  node.thinking.innerHTML = 'thinking<i>…</i>';
  node.bubble.appendChild(node.thinking);
  scrollDown();
  const id = ++state.seq;
  nodes.set(id, node);
  $('progressWrap').classList.remove('hidden');
  $('progress').style.width = '0';

  const model = currentModel();
  if (!state.ready || state.loadedDir !== model.url) {
    node.thinking.textContent = `loading ${model.name}…`;
    try { await ensureModel(); }
    catch (e) {
      node.thinking.textContent = '';
      botError(node, `could not load ${model.name}: ${e.message}`);
      state.busy = false;
      $('send').disabled = false;
      $('progressWrap').classList.add('hidden');
      return;
    }
    node.thinking.textContent = 'thinking…';
  }
  setStatus('generating…');

  const attached = attachmentText(state.attachments);
  const code = [attached, /[<>{}=;()]/.test(text) ? text : ''].filter(Boolean).join('\n\n');
  let prompt = text, ascii = null, error = null, question = null;
  if (task === 'ascii2code') {
    if (imageRec) prompt = text; else { ascii = text; prompt = ''; }
  } else if (task === 'code2img') {
    prompt = ''; question = text;
  } else if (task === 'fix') {
    // The trained format is "i got this error: <message>", so pull the message out of the text
    // (or the first error-looking line of the attached file) and let tasks.js build the question.
    const fromText = text.match(/error[:\s]+(.+)$/i);
    const fromFile = (attached.match(/^.*(error|exception|traceback|undefined|null)[^\n]*$/im) || [])[0];
    error = (fromText ? fromText[1] : fromFile || text || 'unknown error').trim().slice(0, 200);
    prompt = '';
  } else if (task === 'code' && !/^(make|write|build|create|generate|code|can you|please|give)/i.test(text)) {
    prompt = 'write code for ' + text;
  } else if (task === 'img2code' && !/website|html|page/i.test(text)) {
    prompt = 'build a web page that looks like this';
  }

  if (attached && ['code', 'text', 'ascii2code', 'img2code'].includes(task)) {
    prompt = (prompt ? prompt + '\n\n' : '') + attached.slice(0, 900);
  }

  let rgba = null, width = 0;
  if (imageRec) {
    const img = await imageToRgba(imageRec);
    if (img) { rgba = img.rgba; width = img.width; }
  }
  state.worker.postMessage({
    cmd: 'run', task, prompt, ascii, code: task === 'code2img' ? (question || text) : code,
    error, question, rgba, width,
    temp: parseFloat($('temp').value), topK: parseInt($('topk').value),
    topP: parseFloat($('topP').value), seed: parseInt($('seed').value) || 0,
  });
}

/* ------------------------------------------------------------ storage meter */
function updateUsage() {
  const files = state.attachments.reduce((n, f) => n + (f.size || 0), 0);
  const text = messageBytes();
  const total = files + text;
  const pct = Math.min(100, Math.round(100 * total / LIMITS.chatBytes));
  $('usageBar').style.width = pct + '%';
  $('usageText').textContent = `storage: ${fmtBytes(total)} / ${fmtBytes(LIMITS.chatBytes)} · `
    + `${state.attachments.length}/${LIMITS.fileCount} files`;
}

/* ------------------------------------------------------------------ wiring */
$('send').onclick = send;
$('prompt').addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
});
$('prompt').addEventListener('input', autosize);
$('capList').addEventListener('click', e => {
  const b = e.target.closest('button[data-task]');
  if (!b) return;
  if (b.classList.contains('unsupported')) {
    note(`${state.loadedName || 'this model'} was not trained for "${b.dataset.task}" — pick another tool or load a model that has it`);
    return;
  }
  selectTask(b.dataset.task);
});
$('newChat').onclick = () => { newChat(); hideChatList(); $('prompt').focus(); };
$('themeBtn').onclick = toggleTheme;
$('chatsBtn').onclick = async () => {
  if ($('chatList').classList.contains('hidden')) await renderChatList();
  else hideChatList();
};
$('clearBtn').onclick = () => { clearChat(); note('chat cleared — the gallery went with it'); setTimeout(() => { $('messages').innerHTML = ''; hero(); updateGalleryBadge(); }, 900); };
$('deleteBtn').onclick = async () => { await deleteChatAndReset(); note('chat deleted'); setTimeout(() => { $('messages').innerHTML = ''; hero(); updateGalleryBadge(); }, 900); };
$('galleryBtn').onclick = () => openGallery('created');
$('galleryClose').onclick = closeGallery;
$('galleryZip').onclick = downloadGalleryZip;
$('galleryModal').addEventListener('click', e => { if (e.target.id === 'galleryModal') closeGallery(); });
document.querySelectorAll('#galleryModal .tabs button').forEach(b => {
  b.onclick = () => { state.galleryTab = b.dataset.tab; renderGallery(); };
});
$('dlBtn').onclick = openDownloads;
$('dlClose').onclick = () => $('dlModal').classList.add('hidden');
$('dlModal').addEventListener('click', e => { if (e.target.id === 'dlModal') $('dlModal').classList.add('hidden'); });
document.addEventListener('keydown', e => { if (e.key === 'Escape') { closeGallery(); $('dlModal').classList.add('hidden'); } });
$('load').onclick = async () => {
  const model = currentModel();
  if (!model) { note('no model bundle found — train and export one first'); return; }
  try { await ensureModel(); note(`${state.loadedName} is loaded — the next message generates right away`); }
  catch (e) { note(`could not load ${model.name}: ${e.message}`); }
};
$('loadDir').onclick = async () => {
  const dir = $('modelDir').value.trim();
  if (!dir) return;
  const url = modelUrl(dir);
  state.models = [...state.models.filter(m => m.id !== 'custom'),
    { id: 'custom', name: dir.split('/').pop() || dir, dir, url, available: true, params: 0, bytes: 0,
      bits: 8, tasks: [], note: 'custom folder' }];
  state.modelId = 'custom';
  renderModelSelect();
  try { await ensureModel(); } catch (e) { note(`could not load ${dir}: ${e.message}`); }
};
$('modelSelect').onchange = () => {
  state.modelId = $('modelSelect').value;
  remember('sgm.model', state.modelId);
  const m = currentModel();
  renderModelSelect();
  if (m && state.loadedDir !== m.url) {
    setStatus(`will load ${m.name} on the next message`);
    $('modelState').textContent = `${state.loadedName || 'no model'} loaded`;
  }
};
$('pick').onclick = () => $('files').click();
$('files').addEventListener('change', e => { addFiles(e.target.files); e.target.value = ''; });

const fmt = (id, v) => { $(id).textContent = /topk/.test(id) ? String(parseInt(v)) : parseFloat(v).toFixed(2); };
for (const [id, out] of [['temp', 'tempV'], ['topk', 'topkV'], ['topP', 'topPV']]) {
  $(id).addEventListener('input', () => fmt(out, $(id).value));
  fmt(out, $(id).value);
}
const chat = $('chat');
['dragenter', 'dragover'].forEach(ev => chat.addEventListener(ev, e => {
  e.preventDefault(); $('drop').classList.remove('hidden'); document.querySelector('.composerRow').classList.add('over');
}));
['dragleave', 'drop'].forEach(ev => chat.addEventListener(ev, e => {
  if (ev === 'dragleave' && chat.contains(e.relatedTarget)) return;
  e.preventDefault(); $('drop').classList.add('hidden'); document.querySelector('.composerRow').classList.remove('over');
}));
chat.addEventListener('drop', e => { if (e.dataTransfer.files.length) addFiles(e.dataTransfer.files); });

/* ----------------------------------------------------------------- boot up */
async function boot() {
  applyTheme(recall('sgm.theme') === 'light' ? 'light' : 'dark');

  try {
    const swept = await sweepExpired();
    await trimChats();
    if (swept.length) {
      // Chats idle for more than 24 h are gone; say so instead of silently starting over.
      setTimeout(() => note(`cleared ${swept.length} chat(s) that had been idle for over 24 hours`), 400);
    }
  } catch (e) { setStatus('storage unavailable: ' + e.message, 'err'); }

  const last = recall('sgm.chat');
  const existing = last ? await getChat(last) : null;
  if (existing) await useChat(existing.id);
  else newChat();
  remember('sgm.chat', state.chat.id);

  await refreshModels();
  startWorker();
  const m = currentModel();
  setStatus(m ? `${m.name} loads on your first message` : 'no model bundle found');
  if (m) $('modelState').textContent = `${m.name} not loaded`;
  updateUsage();
  updateGalleryBadge();
  setInterval(async () => {                        // keep the 24 h rule honest while the tab is open
    if (state.chat && Date.now() - state.chat.updatedAt > LIMITS.ttlMs) {
      note('this chat reached 24 h of inactivity and was cleared');
      newChat();
    }
    try { await sweepExpired(); } catch {}
  }, 60_000);
}
boot();

// Handy in the console / for the headless smoke test in tools/check_ui.mjs
window.SGM = {
  state, ensureModel, refreshModels, openGallery, openDownloads, artifacts: chatArtifacts,
  mergeModels, modelUrl, titleForPrompt,
};
