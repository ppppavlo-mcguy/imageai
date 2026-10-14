/* Per-chat storage: IndexedDB only, no network, nothing leaves the tab.
 *
 * Between them, `LIMITS` and the pure policy functions (planEvictions / isExpired / chatBytes)
 * implement the interface rules:
 *   - a chat may hold at most 12 uploaded files   (the 13th evicts the oldest, FIFO)
 *   - a chat may hold at most 32 MB of file bytes (a bigger upload evicts oldest files first)
 *   - a single upload larger than the whole budget is refused (it can never fit)
 *   - a chat with no activity for 24 h is deleted on the next load / sweep
 *
 * The pure functions are exported so tools/check_store.mjs can test the rules without a browser.
 */

export const LIMITS = {
  chatBytes: 32 * 1024 * 1024,   // 25-50 MB per chat, as requested
  fileCount: 12,                 // file amount limit (not size)
  fileBytes: 4 * 1024 * 1024,    // a single attachment cap
  ttlMs: 24 * 60 * 60 * 1000,    // clear a chat after 24 h of inactivity
  keepChats: 8,                  // how many chats the sidebar remembers
};

/* ------------------------------------------------------------------ policy */

/** Bytes a chat occupies: uploaded files plus its message text (UTF-8). */
export function chatBytes(files = [], messages = []) {
  let total = files.reduce((n, f) => n + (f.size || 0), 0);
  for (const m of messages) total += new TextEncoder().encode(String(m.text || '')).length;
  return total;
}

/**
 * Decide what the store keeps when `incoming` is added to `files` (oldest first).
 * Returns {keep, evict, rejected, bytes}: `keep` are the attachments to retain, `evict` the ids to
 * delete, and `rejected` is a message when the incoming file can never fit.
 */
export function planEvictions(files, incoming, limits = LIMITS, reserved = 0) {
  const size = Number(incoming.size) || 0;
  if (size > limits.chatBytes - reserved) {
    return { keep: files.slice(), evict: [], rejected:
      `${incoming.name} is ${fmtBytes(size)} — bigger than the ${fmtBytes(limits.chatBytes)} a chat can hold` };
  }
  const keep = files.slice();                       // oldest first
  const evict = [];
  const bytesOf = list => list.reduce((n, f) => n + (f.size || 0), 0);
  while (keep.length && (keep.length >= limits.fileCount
                         || reserved + bytesOf(keep) + size > limits.chatBytes)) {
    evict.push(keep.shift());                       // delete the oldest, then retry
  }
  return { keep, evict, rejected: null, bytes: reserved + bytesOf(keep) + size };
}

/** A chat is dropped 24 h after the last message/attachment. */
export function isExpired(chat, now = Date.now(), ttl = LIMITS.ttlMs) {
  return !chat || now - (chat.updatedAt || 0) > ttl;
}

export function fmtBytes(n) {
  if (!n) return '0 B';
  if (n < 1024) return `${n} B`;
  if (n < 1048576) return `${(n / 1024).toFixed(1)} kB`;
  return `${(n / 1048576).toFixed(2)} MB`;
}

/* ------------------------------------------------------------- IndexedDB */

const DB_NAME = 'imgcode-chat';
const DB_VERSION = 1;
const CHATS = 'chats';
const FILES = 'files';

function request(req) {
  return new Promise((resolve, reject) => {
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error || new Error('indexeddb request failed'));
  });
}

export async function openDb() {
  if (typeof indexedDB === 'undefined') throw new Error('this browser has no IndexedDB');
  return await new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(CHATS)) db.createObjectStore(CHATS, { keyPath: 'id' });
      if (!db.objectStoreNames.contains(FILES)) {
        const files = db.createObjectStore(FILES, { keyPath: 'id' });
        files.createIndex('chatId', 'chatId');
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error || new Error('could not open IndexedDB'));
  });
}

async function tx(store, mode, fn) {
  const db = await openDb();
  try {
    const t = db.transaction(store, mode);
    const result = await fn(t.objectStore(store));
    await new Promise((res, rej) => { t.oncomplete = res; t.onerror = () => rej(t.error); });
    return result;
  } finally {
    db.close();
  }
}

export const listChats = () => tx(CHATS, 'readonly', s => request(s.getAll()))
  .then(rows => rows.sort((a, b) => (b.updatedAt || 0) - (a.updatedAt || 0)));
export const getChat = id => tx(CHATS, 'readonly', s => request(s.get(id)));
export const putChat = chat => tx(CHATS, 'readwrite', s => request(s.put(chat)));
export const deleteChat = id => tx(CHATS, 'readwrite', s => request(s.delete(id)));

export async function filesFor(chatId) {
  const rows = await tx(FILES, 'readonly', s => request(s.index('chatId').getAll(chatId)));
  return rows.sort((a, b) => (a.addedAt || 0) - (b.addedAt || 0));      // oldest first
}
export const putFile = file => tx(FILES, 'readwrite', s => request(s.put(file)));
export const deleteFile = id => tx(FILES, 'readwrite', s => request(s.delete(id)));

/** Delete every chat (and its files) that has been idle for longer than the TTL. */
export async function sweepExpired(now = Date.now()) {
  const chats = await listChats();
  const dead = chats.filter(c => isExpired(c, now));
  for (const c of dead) {
    for (const f of await filesFor(c.id)) await deleteFile(f.id);
    await deleteChat(c.id);
  }
  return dead.map(c => c.id);
}

export async function dropFiles(ids) {
  for (const id of ids) await deleteFile(id);
}

export async function trimChats(keep = LIMITS.keepChats) {
  const chats = await listChats();
  for (const c of chats.slice(keep)) {
    for (const f of await filesFor(c.id)) await deleteFile(f.id);
    await deleteChat(c.id);
  }
}

/* Text-ish attachments are stored as text (much smaller and directly usable as model input). */
export const TEXT_EXT = /\.(txt|md|json|csv|log|html?|css|js|mjs|jsx|ts|tsx|py|rb|go|rs|java|c|cpp|h|sh|yml|yaml|xml|svg|ini|toml|sql|env)$/i;
export const isTextFile = (name, type = '') =>
  TEXT_EXT.test(name || '') || /^text\/|json|javascript|xml|csv|x-sh|yaml/.test(type || '');

/** Read a File into the record shape the store keeps. */
export async function readAttachment(file) {
  const base = { id: `f${Date.now().toString(36)}${Math.random().toString(36).slice(2, 7)}`,
                 name: file.name || 'file', type: file.type || '', size: file.size || 0,
                 addedAt: Date.now(), text: null, blob: null };
  if (isTextFile(file.name, file.type) && file.size < 512 * 1024) {
    base.text = await file.text();
  } else {
    base.blob = file;
  }
  return base;
}

/** The text the model should read when this attachment is part of the prompt. */
export function attachmentText(files, limit = 1200) {
  const parts = [];
  for (const f of files) {
    if (!f.text) continue;
    parts.push(`--- ${f.name} ---\n${f.text.slice(0, limit)}`);
  }
  return parts.join('\n\n');
}
