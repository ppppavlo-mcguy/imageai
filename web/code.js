/* Turn what you typed into the instruction the model was trained on - and tell the truth when it
 * answers in another language.
 *
 * Why this exists: the code task is pure text conditioning. The training instructions all look like
 *     "write html for a blue card"        (src/synth.py)
 *     "write python for a quiz helper"
 *     "write javascript that toggles a dark theme"
 * so a prompt that never names a language ("make a snake game") lands wherever the bundle happens
 * to be densest - web markup, for every bundle trained so far. Rewriting the prompt into the
 * trained form puts the language word where the model learned to read it.
 *
 * It cannot make a bundle write Python it never learned; `languageNote()` says so instead of
 * pretending. Everything here is pure, so tools/check_code.mjs tests it without a browser.
 */

/** Languages the training set speaks, in the order they are looked for. */
const LANGS = [
  { id: 'python', re: /\bpython\b|\bpy\b|\.py\b|\bdef\b/i },
  { id: 'javascript', re: /\bjavascript\b|\bjs\b|\becmascript\b|\bnode\b/i },
  { id: 'css', re: /\bcss\b|stylesheet/i },
  { id: 'html', re: /\bhtml\b|\bweb ?page\b|\bwebsite\b|\bmarkup\b/i },
];

/** The language someone asked for, or null when they did not name one. */
export function requestedLang(text = '') {
  const t = String(text);
  for (const l of LANGS) if (l.re.test(t)) return l.id;
  return null;
}

/** "make a snake game in javascript" -> "a snake game"; the words that carry no information. */
export function topicOf(text = '') {
  let t = String(text).trim();
  // the language first: dropping "in javascript" before the verbs is what leaves "make a snake
  // game" - the other order leaves a dangling "to" behind ("python to reverse a string").
  t = t.replace(/\b(in|using|with)\s+(python|py|javascript|js|html|css|node)\b/gi, ' ');
  t = t.replace(/\b(python|javascript|html|css)\b/gi, ' ');
  t = t.replace(/\s{2,}/g, ' ').trim();           // the hole a removed language leaves is trimmed
  t = t.replace(/^(please\s+)?(can you\s+)?(could you\s+)?(help me\s+)?/i, '');
  t = t.replace(/^(make|write|build|create|generate|code|give me|show me|produce|do)\s+/i, '');
  t = t.replace(/^(me|us)\s+/i, '');
  t = t.replace(/^(to|for|that|which|so that)\s+/i, '');
  t = t.replace(/\s{2,}/g, ' ').trim();
  t = t.replace(/^[,.;:!-]+/, '').replace(/[,.;:!-]+$/, '').trim();
  return t;
}

/**
 * The instruction to send to the code task.
 *
 * A prompt that already reads like a training instruction is passed through untouched (so the CLI
 * checks and anyone writing "write html for a blue card" by hand behave exactly as before).
 */
export function codeInstruction(text = '') {
  const raw = String(text).trim();
  if (!raw) return raw;
  if (/^(write|please write|can you write|help me write)\b/i.test(raw)) return raw;
  const lang = requestedLang(raw) || 'html';            // web markup is what every bundle is best at
  const topic = topicOf(raw);
  if (!topic) return `write ${lang} for a small ${lang === 'html' ? 'page' : 'helper'}`;
  return `write ${lang} for ${topic}`;
}

/** Does this answer actually look like the language that was asked for? */
export function looksLikeLang(text = '', lang) {
  const t = String(text);
  switch (lang) {
    case 'python':
      return /(^|\n)\s*(def |class |import |from |print\(|#)/.test(t)
        || /(^|\n)\s+\S.*\n\s+\S/.test(t);               // an indented block is python too
    case 'html':
      return /<\s*\/?\s*(!doctype|html|body|div|head|script|style|button|p|h\d)\b/i.test(t);
    case 'css':
      return /[.#]?[a-z-]+\s*(,|\{)[^}]*:\s*[^;]+;/i.test(t);
    case 'javascript':
      return /\b(const|let|var|function|=>|class|document\.|window\.)\b/.test(t);
    default:
      return true;
  }
}

/** "i got an error", "it crashes" - the phrases that mean "take my snippet and fix it". */
export function alertWords(text = '') {
  return /error|traceback|exception|fails?|broken|not work|crash|bug/.test(String(text));
}

/**
 * Which tool a prompt is for, when the tool is on "auto".
 *
 * The order is the whole point, and it is the part that used to get this wrong:
 *   1. an attached image decides between describe / ascii / image->website
 *   2. "preview image" is a code->image request, even though it names a website
 *   3. naming a language is a stronger signal than the verb, so "python to reverse a string" is
 *      code even though it never says "write"
 *
 * @param {string} text            what the user typed
 * @param {{task?: string, hasText?: boolean, hasImage?: boolean}} [ctx] current tool + attachments
 */
export function routeTask(text, { task = 'auto', hasText = false, hasImage = false } = {}) {
  if (task !== 'auto') return task;
  const t = String(text).toLowerCase();
  if (alertWords(t) && (hasText || /code|function|def |import |<|>|\{|\}/.test(text))) return 'fix';
  if (hasImage) {
    return /ascii|text art/.test(t) ? 'ascii' : /website|html|page|build/.test(t) ? 'img2code' : 'caption';
  }
  if (/ascii art|ascii/.test(t) && /turn|convert|into|website|page|html/.test(t)) return 'ascii2code';
  if (/preview image|image preview|render.*image/.test(t)) return 'code2img';
  if (requestedLang(t)) return 'code';
  if (/^(make|write|build|create|code|give me|generate).*(snake|game|app|page|html|css|javascript|js|python|function|script|website|site)/.test(t)) return 'code';
  if (/continue|story|poem|paragraph|essay/.test(t)) return 'text';
  return 'image';
}

/**
 * A warning to show under an answer that came back in a different language, or null when there is
 * nothing to warn about. Kept honest: it names the bundles and the fix instead of blaming the user.
 */
export function languageNote(text, lang, name = 'this bundle') {
  if (!lang || looksLikeLang(text, lang)) return null;
  return `${name} answered in web markup, not ${lang}. The 1M and 3M bundles were trained mostly `
    + `on html/css/js, so a tag around the prompt is not enough - train a bundle with more ${lang} `
    + `in the mix (train_code.cmd already includes Python) or copy the code out and adjust it.`;
}
