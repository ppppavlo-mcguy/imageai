/* Unit test for the chat storage rules in web/store.js. The policy functions are pure, so the
   limits can be verified without a browser or IndexedDB.

     node tools/check_store.mjs
*/
import { LIMITS, planEvictions, isExpired, chatBytes, fmtBytes, isTextFile } from '../web/store.js';

let failures = 0;
const check = (name, cond, detail = '') => {
  console.log(`  ${cond ? 'ok  ' : 'FAIL'} ${name}${detail ? '  ' + detail : ''}`);
  if (!cond) failures++;
};
const file = (name, size) => ({ id: name, name, size });

// --- the 12-file cap: the 13th evicts the oldest -------------------------------------------
{
  const files = Array.from({ length: LIMITS.fileCount }, (_, i) => file(`f${i}`, 1000));
  const plan = planEvictions(files, { name: 'new', size: 1000 });
  check('12 files + 1 evicts exactly one', plan.evict.length === 1 && plan.keep.length === 11);
  check('the evicted one is the oldest', plan.evict[0].name === 'f0');
  check('nothing rejected', !plan.rejected);
}

// --- the size cap: keep deleting the oldest until the new file fits ------------------------
{
  const per = Math.floor(LIMITS.chatBytes / 8);
  const files = Array.from({ length: 8 }, (_, i) => file(`big${i}`, per));   // exactly full
  const plan = planEvictions(files, { name: 'new', size: per });
  check('a full chat evicts oldest files until it fits',
        plan.bytes <= LIMITS.chatBytes && plan.evict.length >= 1,
        `evicted ${plan.evict.length}, ${fmtBytes(plan.bytes)} used`);
  check('eviction is oldest-first', plan.evict.map(f => f.name).join(',') === plan.evict
        .map((_, i) => `big${i}`).join(','), plan.evict.map(f => f.name).join(','));
}

// --- a single upload bigger than the whole budget is refused -------------------------------
{
  const plan = planEvictions([file('a', 10)], { name: 'huge', size: LIMITS.chatBytes + 1 });
  check('oversized upload is rejected', !!plan.rejected, plan.rejected || '');
  check('rejected upload does not evict anything', plan.evict.length === 0 && plan.keep.length === 1);
}

// --- message text counts toward the chat budget --------------------------------------------
{
  const bytes = chatBytes([file('a', 1000)], [{ text: 'x'.repeat(500) }]);
  check('chatBytes counts files + message text', bytes === 1500, `${bytes} bytes`);
  const plan = planEvictions([file('a', 1000)], { name: 'b', size: LIMITS.chatBytes }, LIMITS, 500);
  check('reserved text bytes are honoured', !!plan.rejected);
}

// --- 24 h inactivity -------------------------------------------------------------------------
{
  const now = 1_700_000_000_000;
  check('fresh chat is kept', !isExpired({ updatedAt: now - 3600_000 }, now));
  check('23 h idle is kept', !isExpired({ updatedAt: now - 23 * 3600_000 }, now));
  check('25 h idle is cleared', isExpired({ updatedAt: now - 25 * 3600_000 }, now));
  check('missing chat counts as expired', isExpired(null, now));
}

// --- attachment classification ---------------------------------------------------------------
check('code files are read as text', isTextFile('app.py', '') && isTextFile('index.html', ''));
check('images are not', !isTextFile('photo.png', 'image/png'));

console.log(failures ? `STORE FAILED (${failures})` : 'STORE OK');
process.exit(failures ? 1 : 0);
