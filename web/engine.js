/* Local int8 transformer runtime: weights + tokenizer + image codec in plain JS.
   No dependencies. Runs in a browser worker and in Node (see web/cli.mjs).
   Maths mirrors src/model.py, src/bpe.py and src/imagecodec.py exactly:
   RoPE is the half-split rotation, RMSNorm has eps 1e-5, MLP is SwiGLU w2(silu(w1x)*w3x),
   and the output head is the (tied) token embedding matrix. */

export const SPECIALS = ["<pad>", "<bos>", "<eos>", "<sep>", "<img>", "<cap>", "<ascii>", "<code>", "<ask>"];
export const PAD = 0, BOS = 1, EOS = 2, SEP = 3, IMG = 4, CAP = 5, ASCII = 6, CODE = 7, ASK = 8;
const WORD_RE = /\s+|[A-Za-z]+|[0-9]+|[^A-Za-z0-9\s]+/g;   // must match WORD_RE in src/bpe.py

export const QUESTIONS = {
  caption: "what is on this picture?",
  ascii: "turn this into an ascii text art",
};

/* Default instructions for the combination tasks (mirrors src/tasks.py). */
export const INSTRUCTIONS = {
  ascii2code: "turn this ascii art into a website",
  img2code: "build a web page that looks like this",
  code2img: "turn this website into an image preview",
  fix: "i got this error",
};

/* ------------------------------------------------------------------ tokenizer */
export class Tokenizer {
  constructor(tok) {
    this.base = tok.base ?? SPECIALS.length;
    this.tokens = new Array(256);
    for (let i = 0; i < 256; i++) this.tokens[i] = Uint8Array.of(i);
    this.rank = new Map();
    (tok.merges || []).forEach(([a, b], i) => {
      this.rank.set(a * 1e6 + b, i);
      const A = this.tokens[a], B = this.tokens[b];
      const c = new Uint8Array(A.length + B.length);
      c.set(A, 0); c.set(B, A.length);
      this.tokens.push(c);
    });
    this.nText = this.tokens.length;
    this.enc = new TextEncoder();
    this.dec = new TextDecoder('utf-8', { fatal: false });
    this.cache = new Map();
  }
  encodeWord(word) {
    const hit = this.cache.get(word);
    if (hit) return hit;
    const parts = Array.from(this.enc.encode(word));
    while (parts.length > 1) {
      let bestRank = Infinity, bestI = -1;
      for (let i = 0; i < parts.length - 1; i++) {
        const r = this.rank.get(parts[i] * 1e6 + parts[i + 1]);
        if (r !== undefined && r < bestRank) { bestRank = r; bestI = i; }
      }
      if (bestI < 0) break;
      parts.splice(bestI, 2, 256 + bestRank);
    }
    const ids = parts.map(p => this.base + p);
    if (this.cache.size < 200000) this.cache.set(word, ids);
    return ids;
  }
  encode(text) {
    const ids = [];
    for (const w of (text.match(WORD_RE) || [])) ids.push(...this.encodeWord(w));
    return ids;
  }
  decode(ids) {
    const chunks = [];
    let n = 0;
    for (const id of ids) {
      if (id < this.base) continue;
      const t = this.tokens[id - this.base];
      if (!t) continue;
      chunks.push(t); n += t.length;
    }
    const all = new Uint8Array(n);
    let off = 0;
    for (const c of chunks) { all.set(c, off); off += c.length; }
    return this.dec.decode(all).replace(/\u0000/g, '');
  }
}

/* --------------------------------------------------------------- image codec */
export class ImageCodec {
  constructor(c) { Object.assign(this, c); }

  /** codes -> Uint8ClampedArray RGBA of img_px x img_px (mirrors Codec.decode) */
  decode(l1, l2) {
    const px = this.img_px, g = this.grid, p = this.patch, L = this.l1_levels;
    const out = new Uint8ClampedArray(px * px * 4);
    const base = new Float32Array(g * g * 3);
    for (let i = 0; i < g * g; i++) {
      const c = l1[i] || 0;
      base[i * 3] = ((c >> 8) & 15) * 255 / L;
      base[i * 3 + 1] = ((c >> 4) & 15) * 255 / L;
      base[i * 3 + 2] = (c & 15) * 255 / L;
    }
    const det = (this.detail && l2 && l2.length) ? new Float32Array(l2.length * 3) : null;
    if (det) {
      for (let i = 0; i < l2.length; i++) {
        const q = l2[i];
        det[i * 3] = (Math.floor(q / 49) + this.l2_min) * this.det_step;
        det[i * 3 + 1] = ((Math.floor(q / 7) % 7) + this.l2_min) * this.det_step;
        det[i * 3 + 2] = ((q % 7) + this.l2_min) * this.det_step;
      }
    }
    for (let gy = 0; gy < g; gy++) {
      for (let gx = 0; gx < g; gx++) {
        const pi = gy * g + gx;
        for (let py = 0; py < p; py++) {
          for (let q = 0; q < p; q++) {
            let r = base[pi * 3], gc = base[pi * 3 + 1], b = base[pi * 3 + 2];
            if (det) {
              const sy = (py / 2) | 0, sx = (q / 2) | 0;
              const di = ((pi * this.sub + sy) * this.sub + sx) * 3;
              r += det[di]; gc += det[di + 1]; b += det[di + 2];
            }
            const o = ((gy * p + py) * px + (gx * p + q)) * 4;
            out[o] = r; out[o + 1] = gc; out[o + 2] = b; out[o + 3] = 255;
          }
        }
      }
    }
    return out;
  }

  /** RGBA pixels (width = img_px square) -> {l1, l2}; mirrors Codec.encode */
  encode(rgba, width) {
    const px = this.img_px, g = this.grid, p = this.patch, L = this.l1_levels;
    const scale = width / px;
    const at = (x, y) => {
      const sxp = Math.min(width - 1, Math.floor(x * scale));
      const syp = Math.min(width - 1, Math.floor(y * scale));
      return (syp * width + sxp) * 4;
    };
    const mean = new Float32Array(g * g * 3);
    for (let gy = 0; gy < g; gy++) for (let gx = 0; gx < g; gx++) {
      let r = 0, gg = 0, b = 0, n = 0;
      for (let y = 0; y < p; y++) for (let x = 0; x < p; x++) {
        const o = at(gx * p + x, gy * p + y);
        r += rgba[o]; gg += rgba[o + 1]; b += rgba[o + 2]; n++;
      }
      const pi = (gy * g + gx) * 3;
      mean[pi] = r / n; mean[pi + 1] = gg / n; mean[pi + 2] = b / n;
    }
    const l1 = [], l2 = [];
    const q = v => Math.max(0, Math.min(L, Math.round(v * L / 255)));
    for (let i = 0; i < g * g; i++) {
      l1.push(q(mean[i * 3]) * 256 + q(mean[i * 3 + 1]) * 16 + q(mean[i * 3 + 2]));
    }
    if (this.detail) {
      const s = this.sub;
      for (let gy = 0; gy < g; gy++) for (let gx = 0; gx < g; gx++) {
        for (let sy = 0; sy < s; sy++) for (let sx = 0; sx < s; sx++) {
          let r = 0, gg = 0, b = 0, n = 0;
          for (let y = 0; y < 2; y++) for (let x = 0; x < 2; x++) {
            const o = at(gx * p + sx * 2 + x, gy * p + sy * 2 + y);
            r += rgba[o]; gg += rgba[o + 1]; b += rgba[o + 2]; n++;
          }
          const pi = (gy * g + gx) * 3;
          const qd = (v, base) => Math.max(this.l2_min, Math.min(this.l2_max,
            Math.round((v / n - base) / this.det_step)));
          l2.push((qd(r, mean[pi]) - this.l2_min) * 49 + (qd(gg, mean[pi + 1]) - this.l2_min) * 7
            + (qd(b, mean[pi + 2]) - this.l2_min));
        }
      }
    }
    return { l1, l2 };
  }
}

/* ------------------------------------------------------------------- helpers */
function f16(u16) {                       // IEEE half -> float32 (for stored int4 scales)
  const s = (u16 & 0x8000) ? -1 : 1, e = (u16 >> 10) & 0x1F, f = u16 & 0x3FF;
  if (e === 0) return s * Math.pow(2, -14) * (f / 1024);
  if (e === 31) return f ? NaN : s * Infinity;
  return s * Math.pow(2, e - 15) * (1 + f / 1024);
}

/** Unpack one tensor. int8 = 1 byte/value, int4 = 4 bits/value + one scale per group. */
function dequant(blob, m) {
  const n = m.shape.reduce((a, b) => a * b, 1);
  const out = new Float32Array(n);
  if (!m.quantized) {
    const dv = new DataView(blob.buffer, blob.byteOffset + m.offset, m.length);
    for (let i = 0; i < n; i++) out[i] = dv.getFloat32(i * 4, true);
    return out;
  }
  if ((m.bits || 8) === 8) {
    const off = m.offset, sc = m.scale;
    for (let i = 0; i < n; i++) {
      const v = blob[off + i];
      out[i] = (v > 127 ? v - 256 : v) * sc;
    }
    return out;
  }
  const gs = m.group_size, groups = m.groups;
  const scales = new Float32Array(groups);
  const dv = new DataView(blob.buffer, blob.byteOffset + m.scale_offset,
                          groups * (m.scale_dtype === 'fp16' ? 2 : 4));
  for (let g = 0; g < groups; g++) {
    scales[g] = m.scale_dtype === 'fp16' ? f16(dv.getUint16(g * 2, true)) : dv.getFloat32(g * 4, true);
  }
  const dataOff = m.offset;
  const count = Math.min(n, groups * gs);
  for (let i = 0; i < count; i++) {
    const byte = blob[dataOff + (i >> 1)];
    const nib = (i & 1) ? ((byte >> 4) & 0x0F) : (byte & 0x0F);
    out[i] = (nib - 8) * scales[(i / gs) | 0];
  }
  return out;
}
function rmsnorm(x, w, d, eps = 1e-5) {
  let ss = 0;
  for (let i = 0; i < d; i++) ss += x[i] * x[i];
  const inv = 1 / Math.sqrt(ss / d + eps);
  const out = new Float32Array(d);
  for (let i = 0; i < d; i++) out[i] = x[i] * inv * w[i];
  return out;
}
function matvec(W, x, outDim, inDim, out) {
  out = out || new Float32Array(outDim);
  for (let o = 0; o < outDim; o++) {
    let acc = 0;
    const base = o * inDim;
    for (let i = 0; i < inDim; i++) acc += W[base + i] * x[i];
    out[o] = acc;
  }
  return out;
}
function ropeTables(hd, len, base) {
  const half = hd / 2;
  const cos = new Float32Array(len * half), sin = new Float32Array(len * half);
  for (let t = 0; t < len; t++) {
    for (let i = 0; i < half; i++) {
      const freq = 1 / Math.pow(base, (2 * i) / hd);
      cos[t * half + i] = Math.cos(t * freq);
      sin[t * half + i] = Math.sin(t * freq);
    }
  }
  return { cos, sin, half };
}
function mulberry32(a) {
  return function () {
    a |= 0; a = a + 0x6D2B79F5 | 0;
    let t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

/* --------------------------------------------------------------------- model */
export class Engine {
  constructor(model, tokenizer, weightBytes) {
    this.meta = model;
    this.cfg = model.architecture;
    this.quant = model.quantization || 'symmetric-per-tensor-int8';
    this.layout = model.layout;
    this.codec = new ImageCodec(model.codec);
    this.tok = new Tokenizer(tokenizer);
    this.d = this.cfg.embed_size;
    this.h = this.cfg.num_heads;
    this.hd = this.d / this.h;
    this.inner = this.cfg.expansion * this.d;
    this.ctx = this.cfg.context_length;
    this.V = this.cfg.vocab_size;
    const blob = weightBytes instanceof Uint8Array ? weightBytes : new Uint8Array(weightBytes);
    this.W = new Map();
    for (const [name, m] of Object.entries(model.tensors)) this.W.set(name, dequant(blob, m));
    this.rope = ropeTables(this.hd, this.ctx, this.cfg.rope_base || 10000);
    this.reset();
  }
  reset() {
    const { h, hd, ctx } = this;
    this.cache = [];
    for (let i = 0; i < this.cfg.num_layers; i++) {
      this.cache.push({ k: new Float32Array(ctx * h * hd), v: new Float32Array(ctx * h * hd), len: 0 });
    }
  }
  _rotate(vec, headOff, pos) {
    const hd = this.hd, half = hd / 2, co = this.rope.cos, si = this.rope.sin;
    for (let i = 0; i < half; i++) {
      const j = headOff + i;
      const a = vec[j], b = vec[j + half];
      const c = co[pos * half + i], s = si[pos * half + i];
      vec[j] = a * c - b * s;
      vec[j + half] = a * s + b * c;
    }
  }
  /** forward one token: returns Float32Array of logits in [lo,hi); lo defaults to 0 */
  step(tokId, pos, lo = 0, hi = this.V) {
    if (pos >= this.ctx) throw new Error('context length exceeded');
    const d = this.d, h = this.h, hd = this.hd;
    const emb = this.W.get('token_embedding.weight');
    let x = emb.slice(tokId * d, tokId * d + d);
    const q = new Float32Array(d), k = new Float32Array(d), v = new Float32Array(d);
    const attn = new Float32Array(d), scores = new Float32Array(this.ctx);
    for (let li = 0; li < this.cfg.num_layers; li++) {
      const pre = `blocks.${li}.`;
      const xn = rmsnorm(x, this.W.get(pre + 'norm1.weight'), d);
      matvec(this.W.get(pre + 'attn.wq.weight'), xn, d, d, q);
      matvec(this.W.get(pre + 'attn.wk.weight'), xn, d, d, k);
      matvec(this.W.get(pre + 'attn.wv.weight'), xn, d, d, v);
      const c = this.cache[li];
      const slot = c.len * h * hd;
      for (let head = 0; head < h; head++) {
        const ho = head * hd;
        this._rotate(q, ho, pos);
        this._rotate(k, ho, pos);
        for (let i = 0; i < hd; i++) {
          c.k[slot + ho + i] = k[ho + i];
          c.v[slot + ho + i] = v[ho + i];
        }
      }
      const nKeys = c.len + 1;
      let maxScore = -Infinity;
      for (let head = 0; head < h; head++) {
        const ho = head * hd;
        for (let j = 0; j < nKeys; j++) {
          let acc = 0;
          const ko = j * h * hd + ho;
          for (let i = 0; i < hd; i++) acc += q[ho + i] * c.k[ko + i];
          const sc = acc / Math.sqrt(hd);
          scores[j] = sc;
          if (sc > maxScore) maxScore = sc;
        }
        let sum = 0;
        for (let j = 0; j < nKeys; j++) { const e = Math.exp(scores[j] - maxScore); scores[j] = e; sum += e; }
        for (let i = 0; i < hd; i++) attn[ho + i] = 0;
        for (let j = 0; j < nKeys; j++) {
          const w = scores[j] / sum, vo = j * h * hd + ho;
          for (let i = 0; i < hd; i++) attn[ho + i] += w * c.v[vo + i];
        }
      }
      const proj = matvec(this.W.get(pre + 'attn.wo.weight'), attn, d, d);
      const x2 = new Float32Array(d);
      for (let i = 0; i < d; i++) x2[i] = x[i] + proj[i];
      const xn2 = rmsnorm(x2, this.W.get(pre + 'norm2.weight'), d);
      const g1 = matvec(this.W.get(pre + 'mlp.w1.weight'), xn2, this.inner, d);
      const g3 = matvec(this.W.get(pre + 'mlp.w3.weight'), xn2, this.inner, d);
      const act = new Float32Array(this.inner);
      for (let i = 0; i < this.inner; i++) {
        const s = g1[i] / (1 + Math.exp(-g1[i]));      // silu
        act[i] = s * g3[i];
      }
      const down = matvec(this.W.get(pre + 'mlp.w2.weight'), act, d, this.inner);
      x = new Float32Array(d);
      for (let i = 0; i < d; i++) x[i] = x2[i] + down[i];
      c.len = nKeys;
    }
    const xf = rmsnorm(x, this.W.get('norm.weight'), d);
    const out = new Float32Array(hi - lo);
    const embRow = emb;
    for (let o = 0; o < out.length; o++) {
      let acc = 0;
      const base = (lo + o) * d;
      for (let i = 0; i < d; i++) acc += embRow[base + i] * xf[i];
      out[o] = acc;
    }
    return out;
  }
}

/* ------------------------------------------------------------------- sampling */
export function makeRegion(layout, codec) {
  return (i) => {
    if (i < codec.n1) return { lo: layout.img1_base, hi: layout.img1_base + layout.l1 };
    if (i < codec.n1 + codec.n2) return { lo: layout.img2_base, hi: layout.img2_base + layout.l2 };
    return EOS;
  };
}

export class Runtime {
  constructor(engine, opts = {}) {
    this.e = engine;
    this.rng = mulberry32(opts.seed ?? 12345);
  }
  _sample(scores, lo, produced, temp, topK, topP, repPenalty) {
    const logits = Float32Array.from(scores);
    if (repPenalty && repPenalty !== 1) {
      for (const t of new Set(produced.slice(-64))) {
        const i = t - lo;
        if (i >= 0 && i < logits.length) logits[i] = logits[i] > 0 ? logits[i] / repPenalty : logits[i] * repPenalty;
      }
    }
    let order = Array.from(logits.keys());
    if (topK && topK > 0 && topK < order.length) {
      order.sort((a, b) => logits[b] - logits[a]);
      order = order.slice(0, topK);
    }
    const t = temp || 1;
    let maxv = -Infinity;
    const ex = new Float64Array(order.length);
    for (let k = 0; k < order.length; k++) { ex[k] = logits[order[k]] / t; if (ex[k] > maxv) maxv = ex[k]; }
    let sum = 0;
    for (let k = 0; k < ex.length; k++) { ex[k] = Math.exp(ex[k] - maxv); sum += ex[k]; }
    if (topP && topP < 1) {
      const pairs = order.map((id, k) => [id, ex[k] / sum]).sort((a, b) => b[1] - a[1]);
      let cum = 0;
      const keep = [];
      for (const pr of pairs) { keep.push(pr); cum += pr[1]; if (cum >= topP) break; }
      order = keep.map(pr => pr[0]);
      let s2 = 0;
      const e2 = keep.map(pr => pr[1]);
      for (const v of e2) s2 += v;
      for (let k = 0; k < e2.length; k++) e2[k] /= s2;
      let acc = 0;
      const pick = this.rng();
      for (let k = 0; k < order.length; k++) { acc += e2[k]; if (pick <= acc) return order[k] + lo; }
      return order[0] + lo;
    }
    let acc = 0;
    const pick = this.rng();
    for (let k = 0; k < order.length; k++) {
      acc += ex[k] / sum;
      if (pick <= acc) return order[k] + lo;
    }
    return order[order.length - 1] + lo;
  }

  /** Autoregressive sampling with the KV cache. `region(step)` returns {lo,hi} or EOS to stop.
   *
   *  `onProgress({step, done, total, ids})` is called every few tokens; `ids` is the live array of
   *  tokens produced so far, which is what the UI streams into the answer bubble. `total` is the
   *  token budget *after* it was fitted to the model's context window (`maxNew` is clamped here so
   *  a long prompt can never walk into "context length exceeded" mid-answer). The effective budget
   *  is left on `this.maxNew` for callers that report whether an answer was cut off. */
  async generate(promptIds, { maxNew = 420, temp = 0.9, topK = 64, topP = 0.95, repPenalty = 1.1,
                              region = null, stopIds = [EOS], onProgress = null, seed = null,
                              minNew = 0 } = {}) {
    if (seed !== null && seed !== undefined) this.rng = mulberry32(seed);
    const e = this.e;
    e.reset();
    const produced = [];
    if (!promptIds.length) promptIds = [BOS];
    const room = e.ctx - promptIds.length;          // one slot is the token we sample next
    if (room < 1) {
      throw new Error(`the prompt is ${promptIds.length} tokens, more than this model can read ` +
                      `(${e.ctx}) - shorten it or attach a smaller file`);
    }
    this.maxNew = Math.max(1, Math.min(maxNew, room));
    const budget = this.maxNew;
    let pos = 0;
    for (let i = 0; i < promptIds.length - 1; i++) e.step(promptIds[i], pos++, 0, 1);
    let cur = promptIds[promptIds.length - 1];
    for (let step = 0; step < budget; step++) {
      const allowed = region ? region(step) : null;
      if (allowed === EOS || typeof allowed === 'number') { produced.push(EOS); break; }
      const lo = allowed ? allowed.lo : 0;
      const hi = allowed ? allowed.hi : e.V;
      const scores = e.step(cur, pos, lo, hi);
      pos++;
      if (minNew && step < minNew && EOS >= lo && EOS < hi) scores[EOS - lo] = -Infinity;
      const tok = this._sample(scores, lo, produced, temp, topK, topP, repPenalty);
      produced.push(tok);
      if (stopIds.includes(tok)) break;
      cur = tok;
      if (onProgress && (step % 8 === 0 || step === budget - 1)) {
        onProgress({ step, done: produced.length, total: budget, ids: produced });
        await new Promise(r => setTimeout(r, 0));
      }
    }
    return produced;
  }
}

/* ------------------------------------------------------------------ bundle loading */
/* Fetch a bundle (model.json + tokenizer.json + weights) and return a ready Engine.
   The weight file may be shipped gzipped as `<weight_file>.gz` - it is decompressed in the
   browser, so the numbers the model computes are bit-identical to the raw int8 file while the
   download is ~12% smaller. Works with a host that adds Content-Encoding: gzip itself (we sniff
   the gzip magic) and falls back to the raw file when the .gz is missing or the wrong size. */
export async function fetchWeights(base, meta) {
  const raw = meta.weight_file || 'weights.int8.bin';
  const gzName = meta.weight_file_gz || `${raw}.gz`;
  const expect = meta.weight_bytes || 0;
  const get = async (name) => {
    const r = await fetch(base + name);
    if (!r.ok) return null;
    return new Uint8Array(await r.arrayBuffer());
  };
  const gunzip = async (bytes) => {
    if (typeof DecompressionStream !== 'function') return null;
    try {
      const s = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
      return new Uint8Array(await new Response(s).arrayBuffer());
    } catch { return null; }
  };
  const gz = await get(gzName);
  if (gz && gz.length > 2) {
    const gzBytes = gz.length;
    const bytes = (gz[0] === 0x1f && gz[1] === 0x8b) ? await gunzip(gz) : gz;
    if (bytes && (!expect || bytes.length === expect)) return { bytes, gz: true, gzBytes };
  }
  const plain = await get(raw);
  if (!plain) throw new Error(`${raw}: file not found next to model.json`);
  return { bytes: plain, gz: false, gzBytes: 0 };
}

export async function loadBundle(dir) {
  const base = dir.replace(/\/+$/, '') + '/';
  const getJson = async (name) => {
    const r = await fetch(base + name);
    if (!r.ok) throw new Error(`${name}: HTTP ${r.status}`);
    return r.json();
  };
  const meta = await getJson('model.json');
  const [tok, weights] = await Promise.all([getJson('tokenizer.json'), fetchWeights(base, meta)]);
  const t0 = (typeof performance !== 'undefined' ? performance.now() : Date.now());
  const engine = new Engine(meta, tok, weights.bytes);
  const ms = (typeof performance !== 'undefined' ? performance.now() : Date.now()) - t0;
  return { engine, meta, tok, ms, gz: weights.gz, gzBytes: weights.gzBytes };
}
