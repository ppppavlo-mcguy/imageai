/* Unit tests for web/models.js - the model picker's rules (pure, no browser, no server).
     node tools/check_models.mjs */
import { readFileSync } from 'node:fs';
import { autoName, formatParams, slugify, mergeModels, describeModel, modelUrl } from '../web/models.js';

let failures = 0;
const check = (name, cond, detail = '') => {
  console.log(`  ${cond ? 'ok  ' : 'FAIL'} ${name}${detail ? '  ' + detail : ''}`);
  if (!cond) failures++;
};

check('parameter counts are human readable',
  formatParams(1069920) === '1.07M' && formatParams(10_000_000) === '10M' && formatParams(1234) === '1k',
  [formatParams(1069920), formatParams(10_000_000), formatParams(1234)].join(' / '));

check('bundle names follow the size',
  autoName(1_069_920) === 'SGM-1M' && autoName(10_000_000) === 'SGM-10M'
  && autoName(100_000_000) === 'SGM-100M' && autoName(3_271_520) === 'SGM-3M',
  [autoName(1_069_920), autoName(10_000_000), autoName(100_000_000), autoName(3_271_520)].join(' / '));
check('a suffix marks the quantisation', autoName(1_069_920, 'int4') === 'SGM-1M-int4');
check('ids are file-name safe', slugify('SGM-3M-code') === 'sgm-3m-code' && slugify('models/SGM 10M') === 'models-sgm-10m');

/* --- the merge: manifest names + what the server found on disk ------------------------------- */
const manifest = {
  default: 'sgm-1m',
  models: [
    { id: 'sgm-1m', name: 'SGM-1M', dir: 'web_model', note: 'demo', params: 1069920 },
    { id: 'sgm-10m', name: 'SGM-10M', dir: 'models/sgm-10m', planned: true, note: 'not exported yet' },
  ],
};
const found = [
  { dir: 'web_model', params: 1069920, bytes: 1069920, bits: 8, tasks: ['image', 'code'] },
  { dir: 'models/sgm-10m', params: 10_000_000, bytes: 10_000_000, bits: 8, tasks: ['code'] },
  { dir: 'web_model_int4', params: 1069920, bytes: 568548, bits: 4, suffix: 'int4', tasks: [] },
];
const url = dir => `/proj/${dir}`;
const merged = mergeModels(manifest, found, url);

check('every bundle on disk is selectable', merged.models.length === 3, `${merged.models.length} models`);
check('the manifest name wins over the generated one',
  merged.models.find(m => m.dir === 'web_model').name === 'SGM-1M');
check('a planned bundle that appears on disk becomes available',
  merged.models.find(m => m.dir === 'models/sgm-10m').planned !== true);
check('an unknown bundle gets a generated name and id',
  merged.models.find(m => m.dir === 'web_model_int4').name === 'SGM-1M-int4');
check('urls are resolved for every model', merged.models.every(m => m.url.startsWith('/proj/')));
check('the default is the manifest default when it exists', merged.default === 'sgm-1m');

/* --- manifest only (a plain static host: /api/models is missing) ----------------------------- */
const stat = mergeModels(manifest, [], url);
check('without discovery the manifest still lists models', stat.models.length === 2);
check('declared bundles with sizes count as ready',
  stat.models.find(m => m.dir === 'web_model').available === true);
check('declared-but-planned bundles stay disabled',
  stat.models.find(m => m.dir === 'models/sgm-10m').available === false);
check('a missing /api/models answer cannot break the default',
  stat.default === 'sgm-1m' && merged.default === 'sgm-1m');

check('the picker line mentions params, size and tasks',
  /SGM-1M · 1\.07M params · 1\.02 MiB · image\/code/.test(
    describeModel(merged.models.find(m => m.dir === 'web_model'))),
  describeModel(merged.models.find(m => m.dir === 'web_model')));

check('model folders resolve next to the page, not inside web/',
  modelUrl('web_model', 'http://x:3000/web/') === 'http://x:3000/web_model'
  && modelUrl('models/sgm-10m', 'http://x:3000/web/index.html') === 'http://x:3000/models/sgm-10m');

/* --- the checked-in manifest must match the bundles on disk --------------------------------- */
try {
  const onDisk = JSON.parse(readFileSync('web/models.json', 'utf8'));
  check('web/models.json parses and declares SGM-10M / SGM-100M',
    onDisk.models.some(m => m.id === 'sgm-10m') && onDisk.models.some(m => m.id === 'sgm-100m'));
  check('the shipped default is an exported bundle',
    onDisk.models.some(m => m.id === onDisk.default && !m.planned), onDisk.default);
} catch (e) {
  check('web/models.json is readable', false, e.message);
}

console.log(failures ? `MODELS FAILED (${failures})` : 'MODELS OK');
process.exit(failures ? 1 : 0);
