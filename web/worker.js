/* Inference worker: keeps the page responsive while tokens are generated. */
import { loadBundle } from './engine.js';
import { runTask } from './tasks.js';

let engine = null;

self.onmessage = async (ev) => {
  const msg = ev.data;
  try {
    if (msg.cmd === 'load') {
      const { engine: eng, meta, ms, gz, gzBytes } = await loadBundle(msg.dir);
      engine = eng;
      self.postMessage({ type: 'loaded', meta: { parameter_count: meta.parameter_count, weight_bytes: meta.weight_bytes, architecture: meta.architecture, codec: meta.codec, layout: meta.layout, gz, gzBytes }, ms });
      return;
    }
    if (msg.cmd === 'run') {
      if (!engine) throw new Error('model not loaded');
      const t0 = performance.now();
      const res = await runTask(engine, {
        task: msg.task, prompt: msg.prompt, rgba: msg.rgba ? new Uint8ClampedArray(msg.rgba) : null,
        width: msg.width, maxNew: msg.maxNew, temp: msg.temp, topK: msg.topK, topP: msg.topP, seed: msg.seed,
        ascii: msg.ascii || null, code: msg.code || null,
        error: msg.error || null, question: msg.question || null,
        onProgress: p => self.postMessage({ type: 'progress', done: p.done, total: p.total }),
      });
      const payload = { type: 'result', kind: res.kind, text: res.text || '', tokens: res.ids.length, ms: performance.now() - t0 };
      if (res.kind === 'image') {
        payload.rgba = res.rgba.buffer;
        payload.width = res.width;
        payload.height = res.height;
        self.postMessage(payload, [payload.rgba]);
      } else {
        self.postMessage(payload);
      }
      return;
    }
  } catch (err) {
    self.postMessage({ type: 'error', message: String(err && err.message ? err.message : err) });
  }
};
