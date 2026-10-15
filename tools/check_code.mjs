#!/usr/bin/env node
/* Unit tests for web/code.js - the rules that turn a prompt into the instruction the code task was
 * trained on, and that decide whether an answer really is the language someone asked for.
 *
 *     node tools/check_code.mjs
 */
import { requestedLang, topicOf, codeInstruction, looksLikeLang, languageNote, routeTask } from '../web/code.js';

let failures = 0;
const check = (name, cond, detail = '') => {
  console.log(`  ${cond ? 'ok  ' : 'FAIL'} ${name}${detail ? '  ' + detail : ''}`);
  if (!cond) failures++;
};

/* ------------------------------------------------------------- which language was asked for */
check('a named language is recognised',
  requestedLang('python to reverse a string') === 'python'
  && requestedLang('make a snake game in javascript') === 'javascript'
  && requestedLang('write css for a dark page') === 'css'
  && requestedLang('build a website for my shop') === 'html',
  ['python', 'javascript', 'css', 'html'].map(requestedLang).join('/'));
check('no language named -> null', requestedLang('make a snake game') === null
  && requestedLang('') === null);
check('.py counts as python', requestedLang('a .py script that reads a csv') === 'python');

/* ------------------------------------------------------------------------- the instruction */
check('the trained phrasing is produced for an explicit language',
  codeInstruction('python to reverse a string') === 'write python for reverse a string',
  codeInstruction('python to reverse a string'));
check('an unnamed language falls back to web markup',
  codeInstruction('make a snake game') === 'write html for a snake game',
  codeInstruction('make a snake game'));
check('the filler words are dropped, not the request',
  topicOf('please can you make me a small blue card for the shop') === 'a small blue card for the shop',
  topicOf('please can you make me a small blue card for the shop'));
check('a language mention inside the topic does not survive',
  !/python/.test(topicOf('make a python script to sort numbers')),
  topicOf('make a python script to sort numbers'));
check('a prompt that already reads like a training instruction is untouched',
  codeInstruction('write html for a blue card') === 'write html for a blue card'
  && codeInstruction('can you write python for a helper') === 'can you write python for a helper');
check('a blank request is left to the task default, not turned into nonsense',
  codeInstruction('   ') === '' && codeInstruction('') === '');
check('javascript survives the rewrite',
  codeInstruction('javascript function to toggle dark mode').startsWith('write javascript for '),
  codeInstruction('javascript function to toggle dark mode'));

/* ------------------------------------------------------- does the answer match the request */
check('python answers are recognised',
  looksLikeLang('def f(x):\n    return x', 'python')
  && looksLikeLang('import os\nprint(os.getcwd())', 'python')
  && !looksLikeLang('<!DOCTYPE html>\n<html lang="en">', 'python'));
check('html answers are recognised',
  looksLikeLang('<div class="card">hi</div>', 'html')
  && !looksLikeLang('def f():\n    pass', 'html'));
check('a half-written page still counts as html',
  looksLikeLang('<html>\n  pre { margin: 0; }', 'html'));
check('css and javascript are told apart',
  looksLikeLang('body { margin: 0; color: #fff; }', 'css')
  && looksLikeLang('const root = document.documentElement;', 'javascript')
  && !looksLikeLang('const root = document.body;', 'css'));
check('an unknown language never warns', looksLikeLang('anything', null) === true);

/* ---------------------------------------------------------------------------- the warning */
const warn = languageNote('<!DOCTYPE html>\n<html>', 'python', 'SGM-3M-code');
check('a mismatched answer gets an honest note',
  typeof warn === 'string' && /SGM-3M-code answered in web markup, not python/.test(warn) && /train/.test(warn),
  (warn || '').slice(0, 60));
check('a matching answer gets no note',
  languageNote('def f():\n    return 1', 'python') === null
  && languageNote('<div>hi</div>', 'html') === null
  && languageNote('<div>hi</div>', null) === null);

/* ------------------------------------------------------------------- tool routing ("auto") */
const IMG = { hasImage: true };
const cases = [
  ['python to reverse a string', {}, 'code'],          // named a language, no verb: used to go to image
  ['make a snake game in javascript', {}, 'code'],
  ['write html for a blue card', {}, 'code'],
  ['a python script that sorts numbers', {}, 'code'],
  ['turn this website into an image preview', {}, 'code2img'],   // names html, is not a code request
  ['show me a preview image of this page', {}, 'code2img'],
  ['a big white car on a blue background', {}, 'image'],
  ['draw a small red tree', {}, 'image'],
  ['turn this ascii art into a website', {}, 'ascii2code'],
  ['continue the story about a robot', {}, 'text'],
  ['i got this error: NameError: name x is not defined', { hasText: true }, 'fix'],
  ['why does this crash?', { hasText: true }, 'fix'],
  ['what is on this picture?', IMG, 'caption'],
  ['turn this into an ascii text art', IMG, 'ascii'],
  ['build a web page that looks like this', IMG, 'img2code'],
];
for (const [text, ctx, want] of cases) {
  const got = routeTask(text, ctx);
  check(`"${text.slice(0, 42)}" -> ${want}`, got === want, got === want ? '' : `got ${got}`);
}
check('a tool picked by hand is never re-routed',
  routeTask('a big white car', { task: 'code' }) === 'code'
  && routeTask('python to reverse a string', { task: 'image' }) === 'image');

console.log(failures ? `CODE FAILED (${failures})` : 'CODE OK');
process.exit(failures ? 1 : 0);
