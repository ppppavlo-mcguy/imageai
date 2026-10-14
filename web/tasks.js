/* Task glue shared by the browser UI and the Node CLI. */
import { Runtime, makeRegion, QUESTIONS, INSTRUCTIONS, ASK, SEP, CODE, IMG, ASCII, BOS, EOS } from './engine.js';

/* Image -> ascii art. Mirrors src/ascii_art.py character for character (the ascii->code task
   feeds this text into the model, so both languages must produce the same string). */
export const RAMP = ' .:-=+*#%@';
export function toAscii(rgba, width = 48, aspect = 0.5) {
  const size = Math.round(Math.sqrt(rgba.length / 4));   // square: the codec only stores squares
  const lum = (x, y) => {
    const i = (y * size + x) * 4;
    return 0.299 * rgba[i] + 0.587 * rgba[i + 1] + 0.114 * rgba[i + 2];
  };
  const outW = Math.max(8, Math.floor(width));
  const outH = Math.max(2, Math.round(outW * (size / size) * aspect));
  const rows = [];
  for (let j = 0; j < outH; j++) {
    const y0 = Math.floor((j * size) / outH), y1 = Math.max(y0 + 1, Math.floor(((j + 1) * size) / outH));
    let line = '';
    for (let k = 0; k < outW; k++) {
      const x0 = Math.floor((k * size) / outW), x1 = Math.max(x0 + 1, Math.floor(((k + 1) * size) / outW));
      let sum = 0, n = 0;
      for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) { sum += lum(x, y); n++; }
      const v = sum / Math.max(1, n);
      line += RAMP[Math.min(RAMP.length - 1, Math.floor((v / 256.0) * RAMP.length))];
    }
    rows.push(line.replace(/\s+$/, ''));
  }
  return rows.join('\n');
}

export function wrapImagePrompt(text) {
  const t = (text || '').trim();
  if (!t) return 'generate the image of a white car';
  if (/^(generate|draw|show|make|create|render|paint|sketch)\b/i.test(t)) return t;
  return `generate the image of ${t}`;
}

/* Prompt builders - the JS mirror of src/tasks.py, exported so tools/check_parity.py can compare
   the two implementations token by token. */
export function imagePromptIds(tok, prompt) {
  return [IMG, ...tok.encode(wrapImagePrompt(prompt)), SEP];
}
export function imageCodeIds(layout, l1, l2) {
  // image codes -> token ids (the sampler restricts generation to these same blocks).
  // The Python mirror is img_prefix() in src/tasks.py - keep the two in step.
  const ids = [IMG, ...l1.map(c => layout.img1_base + c)];
  if (layout.detail && layout.l2) ids.push(...l2.map(c => layout.img2_base + c));
  return ids;
}
export function visionPromptIds(tok, layout, l1, l2, question) {
  return [...imageCodeIds(layout, l1, l2), SEP, ASK, ...tok.encode(question), SEP];
}
export function visionSeqIds(tok, layout, l1, l2, question, answer, tag) {
  return [...visionPromptIds(tok, layout, l1, l2, question), tag, ...tok.encode(answer), EOS];
}
export function imageSeqIds(tok, layout, instruction, l1, l2) {   // mirrors src/tasks.text2img
  return [IMG, ...tok.encode(instruction), SEP, ...imageCodeIds(layout, l1, l2).slice(1), EOS];
}
export function codeSeqIds(tok, instruction, code) {              // mirrors src/tasks.code_seq
  return [CODE, ...tok.encode(instruction), SEP, ...tok.encode(code), EOS];
}
export function lmSeqIds(tok, text) {                             // mirrors src/tasks.lm_seq
  return [BOS, ...tok.encode(text), EOS];
}
export function codePromptIds(tok, instruction) {
  return [CODE, ...tok.encode(instruction), SEP];
}

/* Combination tasks. The *SeqIds builders are the full training sequences and mirror
   src/tasks.py (ascii2code_seq / img2code_seq / code2img_seq) token for token, so
   tools/check_parity.py can compare them. The *PromptIds builders stop where generation starts. */
export function ascii2codeSeqIds(tok, art, instruction, code) {
  return [ASCII, ...tok.encode(art), SEP, CODE, ...tok.encode(instruction), SEP,
          ...tok.encode(code), EOS];
}
export function img2codeSeqIds(tok, layout, l1, l2, instruction, code) {
  return [...imageCodeIds(layout, l1, l2), SEP, CODE, ...tok.encode(instruction), SEP,
          ...tok.encode(code), EOS];
}
export function code2imgSeqIds(tok, layout, code, instruction, l1, l2) {
  return [CODE, ...tok.encode(code), SEP, IMG, ...tok.encode(instruction), SEP,
          ...imageCodeIds(layout, l1, l2).slice(1), EOS];
}
export function ascii2codePromptIds(tok, art, instruction) {
  return [ASCII, ...tok.encode(art), SEP, CODE, ...tok.encode(instruction), SEP];
}
export function img2codePromptIds(tok, layout, l1, l2, instruction) {
  return [...imageCodeIds(layout, l1, l2), SEP, CODE, ...tok.encode(instruction), SEP];
}
export function code2imgPromptIds(tok, code, instruction) {
  return [CODE, ...tok.encode(code), SEP, IMG, ...tok.encode(instruction), SEP];
}
/* "fix this, i got an error" - full sequence and prompt, mirroring src/tasks.fix_seq. */
export function fixSeqIds(tok, code, question, fixed) {
  return [CODE, ...tok.encode(code), SEP, ASK, ...tok.encode(question), SEP,
          ...tok.encode(fixed), EOS];
}
export function fixPromptIds(tok, code, question) {
  return [CODE, ...tok.encode(code), SEP, ASK, ...tok.encode(question), SEP];
}
/* Default question for the fix task: "i got this error: <message>" (src/tasks.FIX_QUESTIONS[0]). */
export function fixQuestion(error) {
  return `${INSTRUCTIONS.fix}: ${String(error || 'the program crashes').trim()}`;
}
export function textPromptIds(tok, prompt) {
  return [BOS, ...tok.encode(prompt)];
}

export function codesToRgba(engine, ids) {
  const { layout, codec } = engine;
  const l1 = [], l2 = [];
  for (const t of ids) {
    if (t >= layout.img1_base && t < layout.img1_base + layout.l1) l1.push(t - layout.img1_base);
    else if (layout.detail && t >= layout.img2_base && t < layout.img2_base + layout.l2) l2.push(t - layout.img2_base);
  }
  while (l1.length < codec.n1) l1.push(0);
  while (codec.detail && l2.length < codec.n2) l2.push(171);      // 171 == zero delta
  return codec.decode(l1, l2);
}

export async function runTask(engine, opts) {
  const { task, prompt = '', rgba = null, width = 0, maxNew = null, temp = 0.9, topK = 64,
          topP = 0.95, seed = null, onProgress = null, ascii = null, code = null,
          error = null, question = null } = opts;
  const rt = new Runtime(engine, { seed: seed ?? 1234 });
  const tok = engine.tok, layout = engine.layout, codec = engine.codec;
  // minNew: do not let sampling stop before it has produced anything useful
  const minNew = { caption: 2, ascii: 24, code: 8, text: 4, ascii2code: 12, img2code: 12, fix: 8 }[task] || 0;
  const common = { temp, topK, topP, seed, onProgress, minNew };
  if (task === 'image') {
    const ids = imagePromptIds(tok, prompt);
    const out = await rt.generate(ids, {
      ...common, maxNew: maxNew || codec.n1 + codec.n2 + 2, repPenalty: 1.05,
      region: makeRegion(layout, codec), stopIds: [],
    });
    const px = codesToRgba(engine, out);
    return { kind: 'image', rgba: px, width: codec.img_px, height: codec.img_px, ids: out };
  }
  if (task === 'caption' || task === 'ascii') {
    if (!rgba) throw new Error('this task needs an image');
    const { l1, l2 } = codec.encode(rgba, width);
    const ids = visionPromptIds(tok, layout, l1, l2, QUESTIONS[task]);
    const out = await rt.generate(ids, { ...common, maxNew: maxNew || (task === 'ascii' ? 520 : 40) });
    return { kind: 'text', text: tok.decode(out), ids: out };
  }
  if (task === 'code') {
    const ids = codePromptIds(tok, prompt || 'write html for a blue card');
    const out = await rt.generate(ids, { ...common, maxNew: maxNew || 240 });
    return { kind: 'code', text: tok.decode(out), ids: out };
  }
  if (task === 'fix') {
    const snippet = (opts.code || '').trim();
    if (!snippet) throw new Error('attach or paste the snippet that failed');
    const ids = fixPromptIds(tok, snippet, opts.question || fixQuestion(opts.error));
    const out = await rt.generate(ids, { ...common, maxNew: maxNew || 260 });
    return { kind: 'code', text: tok.decode(out), ids: out };
  }
  if (task === 'ascii2code') {
    const art = opts.ascii || (rgba ? toAscii(rgba, 32, 0.5) : '');
    if (!art) throw new Error('paste ascii art or attach an image first');
    const ids = ascii2codePromptIds(tok, art, prompt || INSTRUCTIONS.ascii2code);
    const out = await rt.generate(ids, { ...common, maxNew: maxNew || 260 });
    return { kind: 'code', text: tok.decode(out), ids: out };
  }
  if (task === 'img2code') {
    if (!rgba) throw new Error('this task needs an image');
    const { l1, l2 } = codec.encode(rgba, width);
    const ids = img2codePromptIds(tok, layout, l1, l2, prompt || INSTRUCTIONS.img2code);
    const out = await rt.generate(ids, { ...common, maxNew: maxNew || 260 });
    return { kind: 'code', text: tok.decode(out), ids: out };
  }
  if (task === 'code2img') {
    const snippet = (opts.code || prompt || '').trim() || '<div class="card">blue page</div>';
    const ids = code2imgPromptIds(tok, snippet, prompt || INSTRUCTIONS.code2img);
    const out = await rt.generate(ids, {
      ...common, maxNew: maxNew || codec.n1 + codec.n2 + 2, repPenalty: 1.05,
      region: makeRegion(layout, codec), stopIds: [],
    });
    const px = codesToRgba(engine, out);
    return { kind: 'image', rgba: px, width: codec.img_px, height: codec.img_px, ids: out };
  }
  const ids = textPromptIds(tok, prompt);
  const out = await rt.generate(ids, { ...common, maxNew: maxNew || 120 });
  return { kind: 'text', text: tok.decode(out), ids: out };
}
