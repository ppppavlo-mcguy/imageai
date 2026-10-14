/* Inference worker: keeps the page responsive while tokens are generated.
   It owns the loaded bundle, so switching models in the UI is just another "load" command. */
import { loadBundle } from './engine.js';
import { runTask } from './tasks.js';

let engine = null;
let loaded = { dir: null, name: null, id: null };

self.onmessage = async (ev) => {
  const msg = ev.data;
  try {
    if (msg.cmd === 'load') {
      const { engine: eng, meta, ms, gz, gzBytes } = await loadBundle(msg.dir);
      engine = eng;
      loaded = { dir: msg.dir, name: msg.name || msg.dir, id: msg.id || null };
      self.postMessage({
        type: 'loaded', ms, dir: loaded.dir, name: loaded.name, id: loaded.id,
        tasks: Array.isArray(meta.tasks) ? meta.tasks : [],
        meta: {
          parameter_count: meta.parameter_count, weight_bytes: meta.weight_bytes,
          weight_bytes_gz: meta.weight_bytes_gz, architecture: meta.architecture,
          codec: meta.codec, layout: meta.layout, bits: meta.bits, quantization: meta.quantization,
          gz, gzBytes,
        },
      });
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
      const payload = { type: 'result', kind: res.kind, text: res.text || '', tokens: res.ids.length,
                        ms: performance.now() - t0, model: loaded.name };
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
    self.postMessage({ type: 'error', message: String(err && err.message ? err.message : err), dir: loaded.dir });
  }
};
