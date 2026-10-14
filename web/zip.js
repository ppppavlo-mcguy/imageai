/* Minimal ZIP writer (and raw-inflate helper) with no dependencies.
 *
 * Used in three places:
 *   - the browser gallery: "Download all (.zip)" packs the chat's created files,
 *   - web/server.js /api/download/*: packs a model folder for download,
 *   - tools/make_downloads.mjs: pre-builds the versioned zips.
 *
 * It writes a plain .zip (local file header + central directory + EOCD) with method 8 (deflate)
 * when a deflater is available and method 0 (store) otherwise, so the produced file opens in
 * Explorer, Finder, `unzip` and Python's zipfile. No zip64 - files here are far below 4 GiB.
 */

const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xEDB88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();

/** CRC-32 of a byte array (what ZIP stores per entry). */
export function crc32(bytes) {
  let c = 0xFFFFFFFF;
  for (let i = 0; i < bytes.length; i++) c = CRC_TABLE[(c ^ bytes[i]) & 0xFF] ^ (c >>> 8);
  return (c ^ 0xFFFFFFFF) >>> 0;
}

let nodeZlib;                       // false = checked and unavailable
async function zlib() {
  if (nodeZlib !== undefined) return nodeZlib;
  nodeZlib = false;
  if (typeof process !== 'undefined' && process.versions && process.versions.node) {
    try { nodeZlib = await import('node:zlib'); } catch { nodeZlib = false; }
  }
  return nodeZlib;
}

/** deflate-raw a buffer: node zlib in node, CompressionStream in the browser, null if neither. */
export async function deflateRaw(bytes) {
  const z = await zlib();
  if (z && typeof z.deflateRawSync === 'function') return new Uint8Array(z.deflateRawSync(bytes, { level: 9 }));
  if (typeof CompressionStream === 'function') {
    try {
      const s = new Blob([bytes]).stream().pipeThrough(new CompressionStream('deflate-raw'));
      return new Uint8Array(await new Response(s).arrayBuffer());
    } catch { /* fall through to store */ }
  }
  return null;
}

/** inflate-raw a buffer: the inverse of deflateRaw (tests, CLI and the desktop build use it). */
export async function inflateRaw(bytes) {
  const z = await zlib();
  if (z && typeof z.inflateRawSync === 'function') return new Uint8Array(z.inflateRawSync(bytes));
  if (typeof DecompressionStream === 'function') {
    const s = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('deflate-raw'));
    return new Uint8Array(await new Response(s).arrayBuffer());
  }
  throw new Error('no deflate implementation available');
}

const UTF8_FLAG = 0x0800;       // general purpose bit 11: file names are UTF-8
const enc = new TextEncoder();

function dosTime(d = new Date()) {
  const time = ((d.getHours() & 31) << 11) | ((d.getMinutes() & 63) << 5) | ((d.getSeconds() / 2) & 31);
  const date = (((d.getFullYear() - 1980) & 127) << 9) | (((d.getMonth() + 1) & 15) << 5) | (d.getDate() & 31);
  return { time, date };
}

/**
 * Build a zip archive.
 * @param {{name: string, data: Uint8Array|string}[]} files  entries (data may be a string)
 * @param {{compress?: boolean, date?: Date}} [opts]
 * @returns {Promise<{bytes: Uint8Array, entries: {name: string, size: number, compressed: number, method: number, crc: number}[]}>}
 */
export async function buildZip(files, opts = {}) {
  const compress = opts.compress !== false;
  const { time, date } = dosTime(opts.date || new Date());
  const entries = [];
  const parts = [];
  let offset = 0;

  for (const file of files) {
    const name = enc.encode(String(file.name).replace(/\\/g, '/'));
    // `data` may be a thunk, so packing a 250 MB folder never holds more than one file at a time
    const raw = typeof file.data === 'function' ? file.data() : file.data;
    const data = typeof raw === 'string' ? enc.encode(raw) : new Uint8Array(raw);
    const crc = crc32(data);
    let method = 0, payload = data;
    if (compress && data.length) {
      const deflated = await deflateRaw(data);
      if (deflated && deflated.length < data.length) { method = 8; payload = deflated; }
    }
    const head = new Uint8Array(30 + name.length);
    const dv = new DataView(head.buffer);
    dv.setUint32(0, 0x04034b50, true);      // local file header
    dv.setUint16(4, 20, true);              // version needed
    dv.setUint16(6, UTF8_FLAG, true);       // names are UTF-8 (bit 11), so unzip tools agree
    dv.setUint16(8, method, true);
    dv.setUint16(10, time, true);
    dv.setUint16(12, date, true);
    dv.setUint32(14, crc, true);
    dv.setUint32(18, payload.length, true);
    dv.setUint32(22, data.length, true);
    dv.setUint16(26, name.length, true);
    dv.setUint16(28, 0, true);
    head.set(name, 30);
    parts.push(head, payload);
    entries.push({ name: String(file.name).replace(/\\/g, '/'), size: data.length,
                   compressed: payload.length, method, crc, offset });
    offset += head.length + payload.length;
  }

  const cdStart = offset;
  for (const e of entries) {
    const name = enc.encode(e.name);
    const cd = new Uint8Array(46 + name.length);
    const dv = new DataView(cd.buffer);
    dv.setUint32(0, 0x02014b50, true);      // central directory header
    dv.setUint16(4, 20, true);              // version made by
    dv.setUint16(6, 20, true);              // version needed
    dv.setUint16(8, UTF8_FLAG, true);       // flags (UTF-8 names)
    dv.setUint16(10, e.method, true);
    dv.setUint16(12, time, true);
    dv.setUint16(14, date, true);
    dv.setUint32(16, e.crc, true);
    dv.setUint32(20, e.compressed, true);
    dv.setUint32(24, e.size, true);
    dv.setUint16(28, name.length, true);
    dv.setUint16(30, 0, true);              // extra
    dv.setUint16(32, 0, true);              // comment
    dv.setUint16(34, 0, true);              // disk
    dv.setUint16(36, 0, true);              // internal attrs
    dv.setUint32(38, 0, true);              // external attrs
    dv.setUint32(42, e.offset, true);
    cd.set(name, 46);
    parts.push(cd);
    offset += cd.length;
  }

  const eocd = new Uint8Array(22);
  const dv = new DataView(eocd.buffer);
  dv.setUint32(0, 0x06054b50, true);
  dv.setUint16(8, entries.length, true);
  dv.setUint16(10, entries.length, true);
  dv.setUint32(12, offset - cdStart, true);
  dv.setUint32(16, cdStart, true);
  parts.push(eocd);

  const total = parts.reduce((n, p) => n + p.length, 0);
  const bytes = new Uint8Array(total);
  let at = 0;
  for (const p of parts) { bytes.set(p, at); at += p.length; }
  return { bytes, entries };
}

/** Read an archive back (used by tools/check_zip.mjs to prove what we write is readable). */
export async function readZip(bytes) {
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let p = bytes.length - 22;
  while (p >= 0 && dv.getUint32(p, true) !== 0x06054b50) p--;
  if (p < 0) throw new Error('not a zip: no end-of-central-directory record');
  const count = dv.getUint16(p + 10, true);
  let at = dv.getUint32(p + 16, true);
  const out = [];
  const dec = new TextDecoder();
  for (let i = 0; i < count; i++) {
    if (dv.getUint32(at, true) !== 0x02014b50) throw new Error('bad central directory entry');
    const method = dv.getUint16(at + 10, true);
    const crc = dv.getUint32(at + 16, true);
    const comp = dv.getUint32(at + 20, true);
    const size = dv.getUint32(at + 24, true);
    const nameLen = dv.getUint16(at + 28, true);
    const extraLen = dv.getUint16(at + 30, true);
    const commentLen = dv.getUint16(at + 32, true);
    const localOff = dv.getUint32(at + 42, true);
    const name = dec.decode(bytes.subarray(at + 46, at + 46 + nameLen));
    const lNameLen = dv.getUint16(localOff + 26, true);
    const lExtraLen = dv.getUint16(localOff + 28, true);
    const dataAt = localOff + 30 + lNameLen + lExtraLen;
    const raw = bytes.subarray(dataAt, dataAt + comp);
    const data = method === 0 ? raw : await inflateRaw(raw);
    const nameCheck = crc32(data) === crc && data.length === size;
    out.push({ name, method, size, compressed: comp, data, crc32: nameCheck });
    at += 46 + nameLen + extraLen + commentLen;
  }
  return out;
}

/** "sgm-1m" + "v1" -> "sgm-1m-v1.zip" (every artifact we publish carries a version). */
export function versionedName(base, version = 'v1', ext = 'zip') {
  const clean = String(version || 'v1').replace(/[^0-9a-zA-Z._-]/g, '');
  const stem = String(base).replace(/\.zip$/i, '').replace(/[^0-9a-zA-Z._-]/g, '-').toLowerCase();
  return `${stem}-${clean.startsWith('v') ? clean : 'v' + clean}.${ext}`;
}
