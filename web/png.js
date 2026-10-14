/* Minimal PNG writer (zlib from Node, or a tiny stored-deflate fallback in the browser). */
const CRC = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xEDB88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();
function crc32(bytes) {
  let c = 0xFFFFFFFF;
  for (let i = 0; i < bytes.length; i++) c = CRC[(c ^ bytes[i]) & 0xFF] ^ (c >>> 8);
  return (c ^ 0xFFFFFFFF) >>> 0;
}
function chunk(type, data) {
  const out = new Uint8Array(8 + data.length + 4);
  const dv = new DataView(out.buffer);
  dv.setUint32(0, data.length);
  out.set(new TextEncoder().encode(type), 4);
  out.set(data, 8);
  dv.setUint32(8 + data.length, crc32(out.subarray(4, 8 + data.length)));
  return out;
}
export function encodePNG(rgba, width, height, deflate) {
  const raw = new Uint8Array((width * 3 + 1) * height);
  for (let y = 0; y < height; y++) {
    raw[y * (width * 3 + 1)] = 0;
    for (let x = 0; x < width; x++) {
      const src = (y * width + x) * 4, dst = y * (width * 3 + 1) + 1 + x * 3;
      raw[dst] = rgba[src]; raw[dst + 1] = rgba[src + 1]; raw[dst + 2] = rgba[src + 2];
    }
  }
  const body = deflate ? deflate(raw) : raw;
  const ihdr = new Uint8Array(13);
  const dv = new DataView(ihdr.buffer);
  dv.setUint32(0, width); dv.setUint32(4, height);
  ihdr[8] = 8; ihdr[9] = 2; ihdr[10] = 0; ihdr[11] = 0; ihdr[12] = 0;
  const sig = Uint8Array.of(137, 80, 78, 71, 13, 10, 26, 10);
  const parts = [sig, chunk('IHDR', ihdr), chunk('IDAT', body), chunk('IEND', new Uint8Array(0))];
  const total = parts.reduce((a, p) => a + p.length, 0);
  const out = new Uint8Array(total);
  let off = 0;
  for (const p of parts) { out.set(p, off); off += p.length; }
  return out;
}

/* PNG decode for the Node CLI: 8-bit RGB/RGBA, no interlace (what this project and most
   screenshots produce). Browsers do not need this - they use canvas instead. */
export function decodePNG(bytes, inflate) {
  const dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  if (dv.getUint32(0) !== 0x89504E47) throw new Error('not a PNG');
  let off = 8, width = 0, height = 0, colorType = 6, bitDepth = 8, interlace = 0, idat = [];
  while (off < bytes.length) {
    const len = dv.getUint32(off);
    const type = String.fromCharCode(bytes[off + 4], bytes[off + 5], bytes[off + 6], bytes[off + 7]);
    const data = bytes.subarray(off + 8, off + 8 + len);
    if (type === 'IHDR') {
      width = dv.getUint32(off + 8); height = dv.getUint32(off + 12);
      bitDepth = data[8]; colorType = data[9]; interlace = data[12];
    } else if (type === 'IDAT') idat.push(data);
    else if (type === 'IEND') break;
    off += 12 + len;
  }
  if (bitDepth !== 8) throw new Error('only 8-bit PNG is supported');
  if (interlace) throw new Error('interlaced PNG is not supported');
  const channels = colorType === 6 ? 4 : colorType === 2 ? 3 : colorType === 0 ? 1 : 0;
  if (!channels) throw new Error('unsupported PNG colour type ' + colorType);
  const total = idat.reduce((a, d) => a + d.length, 0);
  const merged = new Uint8Array(total);
  let o = 0;
  for (const d of idat) { merged.set(d, o); o += d.length; }
  const raw = inflate(merged);
  const stride = width * channels;
  const out = new Uint8ClampedArray(width * height * 4);
  const prev = new Uint8Array(stride);
  const cur = new Uint8Array(stride);
  let p = 0;
  for (let y = 0; y < height; y++) {
    const filter = raw[p++];
    for (let i = 0; i < stride; i++) {
      const x = raw[p++];
      const a = i >= channels ? cur[i - channels] : 0;
      const b = prev[i];
      const c = i >= channels ? prev[i - channels] : 0;
      let v = x;
      if (filter === 1) v = x + a;
      else if (filter === 2) v = x + b;
      else if (filter === 3) v = x + ((a + b) >> 1);
      else if (filter === 4) {
        const pp = a + b - c, pa = Math.abs(pp - a), pb = Math.abs(pp - b), pc = Math.abs(pp - c);
        v = x + (pa <= pb && pa <= pc ? a : pb <= pc ? b : c);
      }
      cur[i] = v & 0xFF;
    }
    for (let x = 0; x < width; x++) {
      const d = (y * width + x) * 4, s = x * channels;
      out[d] = cur[s];
      out[d + 1] = channels >= 3 ? cur[s + 1] : cur[s];
      out[d + 2] = channels >= 3 ? cur[s + 2] : cur[s];
      out[d + 3] = channels === 4 ? cur[s + 3] : 255;
    }
    prev.set(cur);
  }
  return { rgba: out, width, height };
}

/** Bilinear upscale of RGBA (used by the CLI so exported PNGs are viewable). */
export function upscale(rgba, width, height, factor) {
  const w = width * factor, h = height * factor;
  const out = new Uint8ClampedArray(w * h * 4);
  for (let y = 0; y < h; y++) {
    const sy = Math.min(height - 1.001, (y + 0.5) / factor - 0.5);
    const y0 = Math.max(0, Math.floor(sy)), y1 = Math.min(height - 1, y0 + 1), fy = Math.max(0, sy - y0);
    for (let x = 0; x < w; x++) {
      const sx = Math.min(width - 1.001, (x + 0.5) / factor - 0.5);
      const x0 = Math.max(0, Math.floor(sx)), x1 = Math.min(width - 1, x0 + 1), fx = Math.max(0, sx - x0);
      const o = (y * w + x) * 4;
      for (let k = 0; k < 3; k++) {
        const p00 = rgba[(y0 * width + x0) * 4 + k], p01 = rgba[(y0 * width + x1) * 4 + k];
        const p10 = rgba[(y1 * width + x0) * 4 + k], p11 = rgba[(y1 * width + x1) * 4 + k];
        out[o + k] = (p00 * (1 - fx) + p01 * fx) * (1 - fy) + (p10 * (1 - fx) + p11 * fx) * fy;
      }
      out[o + 3] = 255;
    }
  }
  return { rgba: out, width: w, height: h };
}
