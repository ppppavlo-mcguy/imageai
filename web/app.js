/* Chat UI driver.
   Everything is local: generation runs in ./worker.js, chats and uploads live in IndexedDB
   (./store.js). No external requests are made anywhere in this app. */
import {
  LIMITS, planEvictions, readAttachment, isTextFile, fmtBytes, chatBytes, attachmentText,
  sweepExpired, trimChats, listChats, getChat, putChat, deleteChat, filesFor, putFile,
  deleteFile, dropFiles,
} from './store.js';

const $ = id => document.getElementById(id);
const TASKS = {
  auto: { label: 'auto' },
  image: { label: 'image', examples: ['a big yellow car on a blue background', 'a small red tree on a pink background'] },
  code: { label: 'code', examples: ['make a snake game in javascript', 'write css for a dark page', 'python to reverse a string'] },
  fix: { label: 'fix an error', examples: ['i got this error: SyntaxError: Unexpected end of input', 'fix this, i got an error: NameError: name is not defined'] },
  ascii2code: { label: 'ascii → website', examples: ['turn this ascii art into a website'] },
  caption: { label: 'describe image', examples: ['what is on this picture?'] },
  img2code: { label: 'image → website', examples: ['build a web page that looks like this'] },
  ascii: { label: 'ascii art', examples: ['turn this into an ascii text art'] },
  code2img: { label: 'website → image', examples: ['turn this website into an image preview'] },
  text: { label: 'text', examples: ['the little robot waited by the window'] },
};
const NEEDS_IMAGE = new Set(['caption', 'ascii', 'img2code']);

const state = {
  task: 'auto', ready: false, busy: false, trained: [],
  chat: null, attachments: [], worker: null, seq: 0, theme: 'dark',
};
const nodes = new Map();

/* ------------------------------------------------------------------ theme */
function applyTheme(theme) {
  state.theme = theme;
  document.documentElement.dataset.theme = theme;
  try { localStorage.setItem('imgcode.theme', theme); } catch {}
}
function toggleTheme() { applyTheme(state.theme === 'dark' ? 'light' : 'dark'); }

/* ------------------------------------------------------------- attachments */
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
}
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
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
  state.chat = { id: `c${Date.now().toString(36)}`, title: 'new chat', createdAt: Date.now(),
                 updatedAt: Date.now(), messages: [] };
  state.attachments = [];
  nodes.clear();
  $('messages').innerHTML = '';
  hero();
  $('roomName').textContent = 'new chat';
  renderAttachments();
  showPreview(null);
  if (persist) saveChat();
  return state.chat;
}
function saveChat() {
  if (!state.chat) return null;
  state.chat.updatedAt = Date.now();
  try { localStorage.setItem('imgcode.chat', state.chat.id); } catch {}
  const payload = { ...state.chat, files: undefined };
  return putChat(payload).catch(() => null).then(() => updateUsage());
}
function pushMessage(msg) {
  state.chat.messages.push(msg);
  if (state.chat.title === 'new chat' && msg.role === 'user' && msg.text) {
    state.chat.title = msg.text.slice(0, 42);
    $('roomName').textContent = state.chat.title;
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
  $('roomName').textContent = chat.title || 'chat';
  renderAttachments();
  showPreview([...chat.messages].reverse().find(m => m.kind === 'image' || m.kind === 'ascii') || null);
  hideChatList();
}
async function renderChatList() {
  const box = $('chatList');
  const chats = await listChats();
  box.innerHTML = '';
  if (!chats.length) { box.innerHTML = '<p class="tiny muted" style="margin:8px">no saved chats yet</p>'; }
  const now = Date.now();
  for (const c of chats) {
    const row = document.createElement('div');
    row.className = 'chatRow';
    const age = Math.round((now - (c.updatedAt || 0)) / 60000);
    const open = document.createElement('button');
    open.innerHTML = `<b>${escapeHtml(c.title || 'chat')}</b><br><span class="tiny muted">${c.messages.length} messages · ${age < 60 ? age + ' min' : Math.round(age / 60) + ' h'} ago</span>`;
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

/* ----------------------------------------------------------------- worker */
function startWorker() {
  try {
    state.worker = new Worker('./worker.js', { type: 'module' });
    state.worker.onmessage = onWorker;
    state.worker.onerror = e => { setStatus('worker error: ' + e.message, 'err'); botError(nodes.get(state.seq), String(e.message)); };
    return true;
  } catch (e) {
    setStatus('no module workers — run "node web/server.js"', 'err');
    return false;
  }
}
function setStatus(t, kind = '') {
  $('status').textContent = t;
  $('dot').className = 'dot' + (kind === 'ok' ? ' ok' : kind === 'err' ? ' err' : '');
}
function onWorker(ev) {
  const m = ev.data;
  if (m.type === 'loaded') {
    state.ready = true;
    state.trained = Array.isArray(m.meta.tasks) ? m.meta.tasks : [];
    $('modelState').textContent = 'model loaded';
    const a = m.meta.architecture;
    const gz = m.meta.weight_bytes_gz ? ` · ${(m.meta.weight_bytes_gz / 1048576).toFixed(2)} MiB gz` : '';
    $('modelInfo').textContent = `${m.meta.parameter_count.toLocaleString()} params · ${(m.meta.weight_bytes / 1048576).toFixed(2)} MiB int8${gz} · ${a.embed_size}d/${a.num_heads}h/${a.num_layers}L`;
    $('trainedPill').hidden = !state.trained.length;
    if (state.trained.length) $('trainedPill').textContent = 'trained for: ' + state.trained.join(', ');
    setStatus(`ready in ${m.ms.toFixed(0)} ms`, 'ok');
    pushMessage({ role: 'bot', kind: 'info', text: `Model loaded — ${m.meta.parameter_count.toLocaleString()} parameters, ${(m.meta.weight_bytes / 1048576).toFixed(2)} MiB int8, running entirely in this tab.` });
    renderInto(null, state.chat.messages[state.chat.messages.length - 1]);
  } else if (m.type === 'progress') {
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
    showPreview(msg);
  } else if (m.type === 'error') {
    $('progressWrap').classList.add('hidden');
    state.busy = false;
    $('send').disabled = false;
    setStatus('error', 'err');
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
             width: m.width, height: m.height, meta: { ms: m.ms, tokens: m.tokens } };
  }
  const kind = m.kind === 'code' ? 'code' : (m.kind === 'ascii' ? 'ascii' : 'text');
  return { role: 'bot', kind, text: m.text || '', meta: { ms: m.ms, tokens: m.tokens } };
}

/* --------------------------------------------------------------- rendering */
function hero() {
  const h = document.createElement('div');
  h.className = 'hero';
  h.innerHTML = `<h2>What should I make?</h2>
    <p>A small transformer running in this browser tab — no server compute, no API keys, no external
    requests. Ask for an image, code, ASCII art, attach a file, or paste code and say
    “fix this, i got an error”.</p>
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

function renderInto(node, msg) {
  if (!node) {
    node = shell(msg.role === 'user' ? 'user' : 'bot', msg.role === 'user' ? 'you' : 'ImgCode');
    $('messages').appendChild(node.wrap);
  }
  const bubble = node.bubble;
  bubble.innerHTML = '';
  if (msg.role === 'user') {
    if (msg.files && msg.files.length) {
      const note = document.createElement('div');
      note.className = 'attachedNote';
      note.textContent = '📎 ' + msg.files.join(', ');
      bubble.parentNode.insertBefore(note, bubble);
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
    a.href = msg.imageData;
    a.download = 'generated.png';
    a.textContent = 'Download PNG';
    dl.appendChild(a);
    bubble.appendChild(dl);
    const meta = document.createElement('div');
    meta.className = 'meta';
    meta.textContent = `${msg.width}×${msg.height} output${msg.meta ? ` · ${(msg.meta.ms / 1000).toFixed(1)}s` : ''}`;
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
    const lang = /^\s*(def |import |from )/m.test(text) ? 'python'
      : /<\/?(html|body|div|head)/i.test(text) ? 'html'
      : /\{[\s\S]*:[^}]*;/.test(text) ? 'css'
      : /(const|let|function|=>)/.test(text) ? 'javascript' : 'text';
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
  bubble.appendChild(actions);
  if (msg.meta) {
    const meta = document.createElement('div');
    meta.className = 'meta';
    meta.textContent = `${(msg.meta.ms / 1000).toFixed(1)}s · ${msg.meta.tokens} tokens`;
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

/* --------------------------------------------------------- result preview */
function showPreview(msg) {
  const card = $('previewCard');
  const canvas = $('previewCanvas');
  const ascii = $('previewAscii');
  const dl = $('previewDownload');
  canvas.classList.add('hidden');
  ascii.classList.add('hidden');
  dl.classList.add('hidden');
  if (!msg) { card.classList.add('hidden'); $('previewMeta').textContent = ''; return; }
  card.classList.remove('hidden');
  if (msg.kind === 'image' && msg.imageData) {
    const img = new Image();
    img.onload = () => {
      canvas.width = canvas.height = 96;
      canvas.getContext('2d').drawImage(img, 0, 0, 96, 96);
    };
    img.src = msg.imageData;
    canvas.classList.remove('hidden');
    dl.href = msg.imageData;
    dl.classList.remove('hidden');
    $('previewMeta').textContent = `image ${msg.width}×${msg.height}`;
  } else if (msg.kind === 'ascii') {
    ascii.textContent = msg.text;
    ascii.classList.remove('hidden');
    $('previewMeta').textContent = `ascii art · ${msg.text.split('\n').length} lines`;
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
  if (!state.ready) { note('the model is still loading (or failed) — check the model folder in the rail, then press Load'); return; }

  const fileNames = state.attachments.map(f => f.name);
  $('prompt').value = '';
  autosize();
  state.busy = true;
  $('send').disabled = true;
  pushMessage({ role: 'user', kind: 'text', text, files: fileNames });

  const node = shell('bot', 'ImgCode');
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
  if (b) selectTask(b.dataset.task);
});
$('newChat').onclick = () => { newChat(); hideChatList(); $('prompt').focus(); };
$('themeBtn').onclick = toggleTheme;
$('chatsBtn').onclick = async () => {
  if ($('chatList').classList.contains('hidden')) await renderChatList();
  else hideChatList();
};
$('previewHide').onclick = () => $('previewCard').classList.add('hidden');
$('load').onclick = loadModel;
function loadModel() {
  if (!state.worker && !startWorker()) return;
  $('modelState').textContent = 'loading…';
  setStatus('loading model…');
  state.worker.postMessage({ cmd: 'load', dir: $('modelDir').value.trim() || 'web_model' });
}
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
  let saved = null;
  try { saved = localStorage.getItem('imgcode.theme'); } catch {}
  applyTheme(saved === 'light' ? 'light' : 'dark');

  try {
    const swept = await sweepExpired();
    await trimChats();
    if (swept.length) {
      // Chats idle for more than 24 h are gone; say so instead of silently starting over.
      setTimeout(() => note(`cleared ${swept.length} chat(s) that had been idle for over 24 hours`), 400);
    }
  } catch (e) { setStatus('storage unavailable: ' + e.message, 'err'); }

  let last = null;
  try { last = localStorage.getItem('imgcode.chat'); } catch {}
  const existing = last ? await getChat(last) : null;
  if (existing) await useChat(existing.id);
  else newChat();
  try { localStorage.setItem('imgcode.chat', state.chat.id); } catch {}

  startWorker();
  loadModel();
  updateUsage();
  setInterval(async () => {                        // keep the 24 h rule honest while the tab is open
    if (state.chat && Date.now() - state.chat.updatedAt > LIMITS.ttlMs) {
      note('this chat reached 24 h of inactivity and was cleared');
      newChat();
    }
    try { await sweepExpired(); } catch {}
  }, 60_000);
}
boot();
