/* Round-trip test for web/zip.js: what we write must be readable by our own reader and by
   Python's zipfile (i.e. a real archive, not a private container).
     node tools/check_zip.mjs
   Writes a probe archive to samples/zip_check.zip for the Python half of the check. */
import { mkdirSync, writeFileSync } from 'node:fs';
import { buildZip, readZip, crc32, versionedName, deflateRaw, inflateRaw } from '../web/zip.js';

let failures = 0;
const check = (name, cond, detail = '') => {
  console.log(`  ${cond ? 'ok  ' : 'FAIL'} ${name}${detail ? '  ' + detail : ''}`);
  if (!cond) failures++;
};

const text = new TextEncoder().encode('SGM '.repeat(400));               // compressible
const bytes = new Uint8Array(4096);
crypto.getRandomValues(bytes);                                           // incompressible: stays stored

const { bytes: zip, entries } = await buildZip([
  { name: 'sgm-1-image.png', data: bytes },
  { name: 'notes/README.txt', data: 'hello from SGM\r\n' },
  { name: 'big.txt', data: text },
  { name: 'unicode-π.txt', data: 'π≈3.14159\n' },
], { date: new Date(Date.UTC(2026, 0, 2, 3, 4, 6)) });

check('archive has one entry per file', entries.length === 4, `${entries.length} entries`);
check('text is deflated, noise is stored',
  entries[2].method === 8 && entries[0].method === 0,
  `methods ${entries.map(e => e.method).join(',')}`);
check('deflate really shrinks the repetitive file', entries[2].compressed < entries[2].size / 4,
  `${entries[2].size} -> ${entries[2].compressed}`);
check('crc32 is stable', crc32(new TextEncoder().encode('123456789')) === 0xCBF43926,
  crc32(new TextEncoder().encode('123456789')).toString(16));

const back = await readZip(zip);
check('our reader returns every entry', back.length === 4);
check('names survive (incl. unicode + folders)',
  back.map(e => e.name).join(',') === 'sgm-1-image.png,notes/README.txt,big.txt,unicode-π.txt',
  back.map(e => e.name).join(','));
check('every payload round-trips through its crc32', back.every(e => e.crc32));
check('round-tripped bytes match the input',
  back[0].data.every((v, i) => v === bytes[i]) && back[1].data.length === 16,
  `${back[1].data.length} bytes`);

const raw = await deflateRaw(bytes);
check('deflateRaw/inflateRaw round-trip',
  raw !== null && (await inflateRaw(raw)).every((v, i) => v === bytes[i]));

check('versioned names carry the release',
  versionedName('SGM-1M', 'v1') === 'sgm-1m-v1.zip' && versionedName('sgm-desktop', '2') === 'sgm-desktop-v2.zip',
  `${versionedName('SGM-1M', 'v1')}, ${versionedName('sgm-desktop', '2')}`);

mkdirSync('samples', { recursive: true });
writeFileSync('samples/zip_check.zip', zip);
console.log(`  wrote samples/zip_check.zip (${zip.length} bytes) for the python zipfile cross-check`);

console.log(failures ? `ZIP FAILED (${failures})` : 'ZIP OK');
process.exit(failures ? 1 : 0);
