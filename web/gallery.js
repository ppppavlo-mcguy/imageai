/* Turn a chat's messages into the "created files" the gallery shows and can download.
 *
 * The gallery is a view over the transcript: every finished answer becomes one file
 * (image -> .png, ascii art -> .txt, code -> .html/.js/.css/.py, prose -> .txt), so nothing
 * extra has to be stored and the files survive a reload with the chat itself.
 *
 * The logic is pure (no DOM), so tools/check_gallery.mjs can verify the names and byte counts
 * without a browser.
 */

/** Which extension a piece of generated code deserves. Mirrors the detector in the chat renderer. */
export function codeLang(text = '') {
  if (/^\s*(def |import |from )/m.test(text) || /\bprint\(/.test(text)) return 'python';
  if (/<\s*\/?\s*(html|body|div|head|script|style|button|p)\b/i.test(text)) return 'html';
  if (/\{[^}]*:[^}]*;/.test(text) && !/\{[\s\S]*\}/.test(text.replace(/\{[^}]*:[^}]*;/, ''))) return 'css';
  if (/\b(const|let|var|function|=>|class)\b/.test(text)) return 'javascript';
  if (/^\s*[\[{]/.test(text)) return 'json';
  return 'text';
}

export const LANG_EXT = {
  html: 'html', css: 'css', javascript: 'js', python: 'py', json: 'json', text: 'txt',
};

/** bytes a data: URL payload holds (base64 -> bytes) - used for the gallery's size column. */
export function dataUrlBytes(dataUrl) {
  const s = String(dataUrl || '');
  const comma = s.indexOf(',');
  if (comma < 0) return 0;
  const b64 = s.slice(comma + 1);
  const pad = b64.endsWith('==') ? 2 : b64.endsWith('=') ? 1 : 0;
  return Math.max(0, Math.floor(b64.length * 3 / 4) - pad);
}

/** "sgm-3-image.png" - the file the gallery offers, numbered in creation order. */
export function artifactName(artifact, index) {
  const n = index + 1;
  const ext = artifact.kind === 'image' ? 'png'
    : artifact.kind === 'ascii' ? 'txt'
    : LANG_EXT[artifact.lang || codeLang(artifact.text)] || 'txt';
  const tag = artifact.kind === 'image' ? 'image' : artifact.kind === 'ascii' ? 'ascii' : 'code';
  return `sgm-${n}-${tag}.${ext}`;
}

/**
 * Every file a chat produced, oldest first.
 * @param {Array} messages chat transcript ({ role, kind, text, imageData, width, height, meta })
 * @param {{prefix?: string}} [opts]
 */
export function artifactsFromMessages(messages = [], opts = {}) {
  const out = [];
  for (const m of messages) {
    if (!m || m.role !== 'bot') continue;
    const base = { text: m.text || '', width: m.width || 0, height: m.height || 0,
                   ms: (m.meta && m.meta.ms) || 0, tokens: (m.meta && m.meta.tokens) || 0 };
    if (m.kind === 'image' && m.imageData) {
      out.push({ ...base, kind: 'image', dataUrl: m.imageData, bytes: dataUrlBytes(m.imageData) });
    } else if (m.kind === 'ascii' && m.text) {
      out.push({ ...base, kind: 'ascii', lang: 'text',
                 bytes: new TextEncoder().encode(m.text).length });
    } else if ((m.kind === 'code' || m.kind === 'text') && m.text) {
      if (m.kind === 'text') continue;                     // prose answers are not "files"
      out.push({ ...base, kind: 'code', lang: codeLang(m.text),
                 bytes: new TextEncoder().encode(m.text).length });
    }
  }
  return out.map((a, i) => ({ ...a, index: i, name: artifactName(a, i) }));
}

/**
 * The entries a "download all" zip contains for a chat: the created files first, then the
 * files the user attached (so the archive is a complete record of the conversation).
 */
export function galleryEntries(artifacts = [], uploads = []) {
  const entries = artifacts.map(a => ({ name: a.name, artifact: a }));
  for (const f of uploads) {
    const name = String(f.name || 'file');
    let n = name, i = 2;
    while (entries.some(e => e.name === n)) n = name.replace(/(\.[^.]+)?$/, `-${i++}$1`);
    entries.push({ name: n, file: f });
  }
  return entries;
}

export function totalBytes(list = []) {
  return list.reduce((n, x) => n + (x.bytes || 0), 0);
}
