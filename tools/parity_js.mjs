/* JS half of the Python<->JS parity check. Reads /tmp fixtures written by tools/check_parity.py
   and prints a JSON verdict on stdout. */
import { readFileSync } from 'node:fs';
import { deflateSync, inflateSync } from 'node:zlib';
import { Tokenizer, ImageCodec, Engine, Runtime, makeRegion, EOS as EOS_ID, BOS, CAP, ASCII } from '../web/engine.js';
import { decodePNG } from '../web/png.js';
import { imagePromptIds, visionPromptIds, visionSeqIds, imageSeqIds, codeSeqIds, lmSeqIds,
         codePromptIds, textPromptIds, toAscii, ascii2codeSeqIds, img2codeSeqIds,
         code2imgSeqIds, fixSeqIds } from '../web/tasks.js';

const dir = process.argv[2] || '/tmp';
const py = JSON.parse(readFileSync(`${dir}/parity_py.json`, 'utf8'));
const tokRef = JSON.parse(readFileSync(`${dir}/parity_tok.json`, 'utf8'));
const png = decodePNG(new Uint8Array(readFileSync(`${dir}/parity_src.png`)), b => new Uint8Array(inflateSync(b)));

const codec = new ImageCodec(py.codec);
const enc = codec.encode(png.rgba, png.width);
const dec = codec.decode(enc.l1, enc.l2);

const eqList = (a, b) => a.length === b.length && a.every((v, i) => v === b[i]);
let maxDec = 0;
for (let i = 0, j = 0; i < py.decode.length; i += 3, j += 4) {
  for (let k = 0; k < 3; k++) maxDec = Math.max(maxDec, Math.abs(dec[j + k] - py.decode[i + k]));
}

const tokenizer = new Tokenizer(JSON.parse(readFileSync(`${dir}/tokenizer.json`, 'utf8')));
const ids = tokenizer.encode(tokRef.text);

// optional: transformer forward parity against the int8-dequantised PyTorch reference
let fwd = null;
const modelDir = process.argv[3];
if (modelDir) {
  const ref = JSON.parse(readFileSync(`${dir}/parity_fwd.json`, 'utf8'));
  const meta = JSON.parse(readFileSync(`${modelDir}/model.json`, 'utf8'));
  const tok = JSON.parse(readFileSync(`${modelDir}/tokenizer.json`, 'utf8'));
  const w = new Uint8Array(readFileSync(`${modelDir}/${meta.weight_file || 'weights.int8.bin'}`));
  const engine = new Engine(meta, tok, w);
  let logits = null;
  for (let i = 0; i < ref.ids.length; i++) {
    const last = i === ref.ids.length - 1;
    logits = engine.step(ref.ids[i], i, 0, last ? engine.V : 1);
  }
  let maxFwd = 0, bestTok = 0, refBest = 0;
  for (let i = 0; i < logits.length; i++) {
    maxFwd = Math.max(maxFwd, Math.abs(logits[i] - ref.logits[i]));
    if (logits[i] > logits[bestTok]) bestTok = i;
    if (ref.logits[i] > ref.logits[refBest]) refBest = i;
  }
  fwd = { maxAbsDiff: maxFwd, jsArgmax: bestTok, pyArgmax: refBest, match: bestTok === refBest };
}

let greedy = null;
if (modelDir) {
  const g = JSON.parse(readFileSync(`${dir}/parity_greedy.json`, 'utf8'));
  const m2 = JSON.parse(readFileSync(`${modelDir}/model.json`, 'utf8'));
  const t2 = JSON.parse(readFileSync(`${modelDir}/tokenizer.json`, 'utf8'));
  const w2 = new Uint8Array(readFileSync(`${modelDir}/${m2.weight_file || 'weights.int8.bin'}`));
  const eng2 = new Engine(m2, t2, w2);
  const rt = new Runtime(eng2, { seed: 1 });
  const greedyOpts = { temp: 1, topK: 1, topP: 1, repPenalty: 1, seed: 1 };
  const textIds = await rt.generate(g.text_prompt, { ...greedyOpts, maxNew: g.text_len });
  const imgIds = await rt.generate(g.image_prompt, {
    ...greedyOpts, maxNew: g.image_len, region: makeRegion(eng2.layout, eng2.codec), stopIds: [],
  });
  greedy = {
    textMatch: eqList(textIds, g.text_ids), imageMatch: eqList(imgIds, g.image_ids),
    textLen: g.text_len, imageLen: g.image_len,
    jsText: textIds, pyText: g.text_ids, jsImage: imgIds.slice(0, 8), pyImage: g.image_ids.slice(0, 8),
  };
}

const tk = JSON.parse(readFileSync(`${dir}/parity_tasks.json`, 'utf8'));
const tox = new Tokenizer(JSON.parse(readFileSync(`${dir}/tokenizer.json`, 'utf8')));
const jsTasks = {
  img2cap: visionSeqIds(tox, tk.layout, tk.l1, tk.l2, tk.caption_question, tk.answer_cap, CAP),
  img2ascii: visionSeqIds(tox, tk.layout, tk.l1, tk.l2, tk.ascii_question, tk.answer_ascii, ASCII),
  txt2img: imageSeqIds(tox, tk.layout, tk.image_instruction, tk.l1, tk.l2),
  code: codeSeqIds(tox, tk.code_instruction, tk.code),
  text: lmSeqIds(tox, tk.text),
  ascii2code: ascii2codeSeqIds(tox, tk.combo_ascii, tk.ascii2code_instruction, tk.combo_code),
  img2code: img2codeSeqIds(tox, tk.layout, tk.l1, tk.l2, tk.img2code_instruction, tk.combo_scene_page),
  code2img: code2imgSeqIds(tox, tk.layout, tk.code, tk.code2img_instruction,
                           tk.code2img_l1, tk.code2img_l2),
  fix: fixSeqIds(tox, tk.fix_code, tk.fix_question, tk.fix_answer),
};

const jsAscii = toAscii(png.rgba, 32);
let asciiDiff = 0;
for (let i = 0; i < Math.max(jsAscii.length, tk.combo_ascii.length); i++) {
  if (jsAscii[i] !== tk.combo_ascii[i]) asciiDiff++;
}
const tasksMatch = Object.keys(jsTasks).filter(k => !eqList(jsTasks[k], tk.py[k]));

console.log(JSON.stringify({
  forward: fwd,
  greedy,
  asciiMatch: jsAscii === tk.combo_ascii,
  asciiDiffChars: asciiDiff,
  taskFormats: { mismatched: tasksMatch, checked: Object.keys(jsTasks) },
  codecL1Match: eqList(enc.l1, py.l1),
  codecL2Match: eqList(enc.l2, py.l2),
  codecDecodeMaxDiff: maxDec,
  tokenizerMatch: eqList(ids, tokRef.ids),
  tokenizerRoundTrip: tokenizer.decode(ids) === tokRef.text,
  jsL1: enc.l1.slice(0, 8), pyL1: py.l1.slice(0, 8),
  jsIds: ids.slice(0, 12), pyIds: tokRef.ids.slice(0, 12),
}, null, 1));
