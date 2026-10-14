/* Model registry helpers: what SGM bundles exist, how they are labelled and how to reach them.
 *
 * The UI reads two sources and merges them:
 *   - web/models.json   (checked in: explicit ids/names, also lists bundles you have not trained yet)
 *   - GET /api/models   (the bundled server scans the disk, so a freshly exported bundle shows up
 *                        without editing anything - unavailable on a plain static host)
 *
 * Pure functions, so tools/check_models.mjs can test the merge without a browser or a server.
 */

/** 1073742 -> "1.07M", 10_000_000 -> "10M". */
export function formatParams(n) {
  const v = Number(n) || 0;
  if (v >= 1e9) return `${(v / 1e9).toFixed(2)}B`;
  if (v >= 1e6) return `${(v / 1e6).toFixed(v / 1e6 >= 10 ? 0 : 2)}M`;
  if (v >= 1e3) return `${Math.round(v / 1e3)}k`;
  return String(v);
}

/** Sizes a bundle is allowed to be called: 1.07M params is an "SGM-1M", 3.27M an "SGM-3M". */
const NICE_M = [0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100, 150, 200, 300, 500, 700, 1000];

/** A readable size label from a bundle: "SGM-1M", "SGM-3M", "SGM-100M". */
export function autoName(params, suffix = '') {
  const v = Number(params) || 0;
  const tail = suffix ? `-${suffix}` : '';
  if (!v) return `SGM-unknown${tail}`;
  const m = v / 1e6;
  let best = NICE_M[0];
  for (const c of NICE_M) {
    if (Math.abs(Math.log(c) - Math.log(m)) < Math.abs(Math.log(best) - Math.log(m))) best = c;
  }
  const label = best < 1 ? `${best * 1000}k` : `${best}M`;
  return `SGM-${label}${tail}`;
}

/** A file-name-safe id: "SGM-3M-code" -> "sgm-3m-code". */
export function slugify(name) {
  return String(name || '')
    .toLowerCase()
    .replace(/[^a-z0-9._-]+/g, '-')
    .replace(/^-+|-+$/g, '') || 'model';
}

/** Resolve a bundle folder to a fetchable URL, whatever sub-path the app is served from. */
export function modelUrl(dir, baseHref = (typeof location !== 'undefined' ? location.href : 'http://localhost/web/')) {
  const clean = String(dir || '').replace(/^\.?\//, '');
  const root = new URL('../', baseHref);           // the folder that holds web/ and web_model/
  return new URL(clean, root).href.replace(/\/+$/, '');
}

/** One line under a model in the picker. */
export function describeModel(model) {
  const bits = [model.name || model.id];
  if (model.params) bits.push(`${formatParams(model.params)} params`);
  if (model.bytes) bits.push(`${(model.bytes / 1048576).toFixed(2)} MiB`);
  if (model.tasks && model.tasks.length) bits.push(model.tasks.slice(0, 4).join('/'));
  return bits.join(' · ');
}

/**
 * Merge the checked-in manifest with what the server found on disk.
 * @param {object} manifest parsed web/models.json ({ models: [...] })
 * @param {Array}  found    entries from GET /api/models ({ dir, params, bytes, tasks, architecture })
 * @param {(dir: string) => string} [url] how to turn a folder into a URL
 */
export function mergeModels(manifest = {}, found = [], url = modelUrl) {
  const list = Array.isArray(found) ? found : [];
  const declared = Array.isArray(manifest.models) ? manifest.models : [];
  const byDir = new Map();
  for (const m of declared) {
    const id = m.id || slugify(m.name || m.dir);
    byDir.set(String(m.dir || '').replace(/\/+$/, ''), {
      id, name: m.name || autoName(m.params), dir: m.dir || '',
      note: m.note || '', planned: !!m.planned,
      // A declared bundle counts as usable when the manifest recorded a size for it (that only
      // happens when it was exported); "planned" entries are the ones still waiting for training.
      available: !m.planned && Number(m.params || m.bytes || 0) > 0,
      params: m.params || 0, bytes: m.bytes || 0, bits: m.bits || 8, tasks: m.tasks || [],
    });
  }
  for (const f of list) {
    const dir = String(f.dir || '').replace(/\/+$/, '');
    if (!dir) continue;
    const prev = byDir.get(dir) || {};
    const suffix = f.suffix || '';
    byDir.set(dir, {
      ...prev,
      id: prev.id || slugify(prev.name || `${dir}`),
      name: prev.name || autoName(f.params, suffix),
      dir,
      note: prev.note || (f.tasks || []).join(' · '),
      planned: false,
      available: true,
      params: f.params || prev.params || 0,
      bytes: f.bytes || 0,
      bits: f.bits || prev.bits || 8,
      tasks: f.tasks || prev.tasks || [],
      architecture: f.architecture || null,
      quantized: f.quantized || '',
      files: f.files || prev.files || [],
    });
  }
  const models = [...byDir.values()];
  for (const m of models) m.url = url(m.dir);
  const wanted = manifest.default;
  const def = models.find(m => m.available && (m.id === wanted || m.dir === wanted))
    || models.find(m => m.available) || models[0] || null;
  return { models, default: def ? def.id : null };
}
