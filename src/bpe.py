"""Byte-level BPE with word caching, trained from scratch (no external tokenizer library).

On-disk format is tiny and JS-friendly:

    {"base": 9, "specials": [...], "n_text": N, "merges": [[id_a, id_b], ...]}

Token bytes are rebuilt from the merge list on both sides, so Python (training) and JS
(inference) encode and decode identically. Word splitting uses one fixed regex and every byte
lands in exactly one piece, so encoding is lossless and reproducible everywhere.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from vocab import SPECIALS

WORD_RE = re.compile(r"\s+|[A-Za-z]+|[0-9]+|[^A-Za-z0-9\s]+")
BASE = len(SPECIALS)          # ids 0..8 are specials, text ids start here


def split_words(text: str):
    return WORD_RE.findall(text)


class BPE:
    """ids: 0..8 specials, BASE..BASE+n_text-1 text/code tokens."""

    def __init__(self, merges=None, cache_size: int = 100_000):
        self.merges: list[tuple[int, int]] = list(merges or [])
        self.tokens: list[bytes] = [bytes([i]) for i in range(256)]
        self.rank: dict[tuple[int, int], int] = {}
        for i, (a, b) in enumerate(self.merges):
            self.rank[(a, b)] = i
            self.tokens.append(self.tokens[a] + self.tokens[b])
        self._cache: dict[str, list[int]] = {}
        self._cache_size = cache_size

    # ------------------------------------------------------------------ props
    @property
    def n_text(self) -> int:
        return len(self.tokens)

    def __len__(self) -> int:
        return len(self.tokens)

    # ----------------------------------------------------------------- encode
    def encode_word(self, word: str) -> list[int]:
        """Greedy lowest-rank merge, identical to web/engine.js."""
        cached = self._cache.get(word)
        if cached is not None:
            return cached
        parts = list(word.encode("utf-8"))          # token index 0..255 == single byte
        while len(parts) > 1:
            best_rank, best_i = None, -1
            for i in range(len(parts) - 1):
                r = self.rank.get((parts[i], parts[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_i = r, i
            if best_i < 0:
                break
            parts[best_i : best_i + 2] = [256 + best_rank]
        ids = [BASE + p for p in parts]
        if len(self._cache) < self._cache_size:
            self._cache[word] = ids
        return ids

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids: list[int] = []
        if add_bos:
            ids.append(SPECIALS.index("<bos>"))
        for w in split_words(text):
            ids.extend(self.encode_word(w))
        if add_eos:
            ids.append(SPECIALS.index("<eos>"))
        return ids

    # ----------------------------------------------------------------- decode
    def decode(self, ids, keep_specials: bool = False) -> str:
        out = bytearray()
        for i in ids:
            i = int(i)
            if i < BASE:
                if keep_specials:
                    out.extend(SPECIALS[i].encode("utf-8"))
            elif i < BASE + len(self.tokens):
                out.extend(self.tokens[i - BASE])
        return out.decode("utf-8", errors="replace")

    # ------------------------------------------------------------------ train
    @staticmethod
    def train(texts, target_merges: int = 4096, log_every: int = 500, verbose: bool = True) -> "BPE":
        counts = Counter()
        for t in texts:
            for w in split_words(t):
                counts[w] += 1
        words = [[[b] for b in w.encode("utf-8")] for w in counts]
        weight = [counts[w] for w in counts]
        merges: list[tuple[int, int]] = []
        tokens = [bytes([i]) for i in range(256)]
        ids = {bytes([i]): i for i in range(256)}
        for step in range(target_merges):
            pair_counts = Counter()
            for parts, wt in zip(words, weight):
                for i in range(len(parts) - 1):
                    pair_counts[(parts[i][0], parts[i + 1][0])] += wt
            if not pair_counts:
                break
            (a, b), freq = pair_counts.most_common(1)[0]
            if freq < 2:
                break
            new_id = len(tokens)
            tokens.append(tokens[a] + tokens[b])
            ids[tokens[-1]] = new_id
            merges.append((a, b))
            for k, parts in enumerate(words):
                if len(parts) < 2:
                    continue
                out, i = [], 0
                while i < len(parts):
                    if i < len(parts) - 1 and parts[i][0] == a and parts[i + 1][0] == b:
                        out.append([new_id])
                        i += 2
                    else:
                        out.append(parts[i])
                        i += 1
                words[k] = out
            if verbose and log_every and (step + 1) % log_every == 0:
                print(f"  [bpe] merge {step+1}/{target_merges} freq={freq} tokens={len(tokens)}")
        return BPE(merges)

    # -------------------------------------------------------------------- i/o
    def save(self, path):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps({"base": BASE, "specials": SPECIALS, "n_text": self.n_text,
                        "merges": [[int(a), int(b)] for a, b in self.merges]}),
            encoding="utf-8",
        )

    @staticmethod
    def load(path) -> "BPE":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        if d.get("specials") != SPECIALS:
            raise ValueError("tokenizer specials do not match src/vocab.py")
        b = BPE([tuple(m) for m in d["merges"]])
        if d.get("n_text") not in (None, b.n_text):
            raise ValueError("tokenizer file is inconsistent (n_text mismatch)")
        return b
