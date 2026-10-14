"""Small decoder-only transformer that speaks text, code and image tokens with one vocabulary.

Plain PyTorch, CPU friendly: RMSNorm + RoPE + SwiGLU MLP + tied embeddings.
Parameter names are deliberately <token_embedding.weight> / blocks.N.* / norm.weight so the
checkpoint also works with the int8 exporter in src/export_web.py.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

PRESETS = {
    #            embed heads layers ctx    mlp     (ctx must fit image tokens + an answer:
    #                                            1280 image tokens + ascii art needs ~1800)
    "xs":    dict(embed_size=96,  num_heads=4, num_layers=4, context_length=768,  expansion=4),
    "tiny":  dict(embed_size=128, num_heads=4, num_layers=4, context_length=1024, expansion=4),
    "small": dict(embed_size=192, num_heads=6, num_layers=6, context_length=2048, expansion=4),
    "base":  dict(embed_size=256, num_heads=8, num_layers=8, context_length=2048, expansion=4),
}


@dataclass
class GPTConfig:
    vocab_size: int = 11520
    embed_size: int = 192
    num_heads: int = 6
    num_layers: int = 6
    context_length: int = 1536
    dropout: float = 0.0
    expansion: int = 4
    tie_embeddings: bool = True
    rope_base: float = 10000.0
    norm_eps: float = 1e-5
    loss_chunk: int = 1024          # tokens per loss chunk: bounds the B*T*vocab logits tensor

    def to_json(self) -> dict:
        return asdict(self)


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


def rope_tables(head_dim: int, length: int, base: float, device=None):
    inv = 1.0 / (base ** (torch.arange(0, head_dim // 2, dtype=torch.float32, device=device) / (head_dim // 2)))
    t = torch.arange(length, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv)                      # (L, hd/2)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x, cos, sin):
    """x: (B, H, T, hd); cos/sin: (L, hd/2) already sliced to this position range."""
    B, H, T, hd = x.shape
    x1, x2 = x[..., : hd // 2], x[..., hd // 2 :]
    c, s = cos[:T].view(1, 1, T, -1), sin[:T].view(1, 1, T, -1)
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.h = cfg.num_heads
        self.hd = cfg.embed_size // cfg.num_heads
        if self.hd * self.h == 0 or cfg.embed_size % cfg.num_heads:
            raise ValueError("embed_size must be divisible by num_heads")
        self.wq = nn.Linear(cfg.embed_size, cfg.embed_size, bias=False)
        self.wk = nn.Linear(cfg.embed_size, cfg.embed_size, bias=False)
        self.wv = nn.Linear(cfg.embed_size, cfg.embed_size, bias=False)
        self.wo = nn.Linear(cfg.embed_size, cfg.embed_size, bias=False)

    def forward(self, x, cos, sin, cache=None, pos: int = 0, want_cache: bool = False):
        B, T, D = x.shape
        q = self.wq(x).view(B, T, self.h, self.hd).transpose(1, 2)
        k = self.wk(x).view(B, T, self.h, self.hd).transpose(1, 2)
        v = self.wv(x).view(B, T, self.h, self.hd).transpose(1, 2)
        c, s = cos[pos : pos + T], sin[pos : pos + T]
        q, k = apply_rope(q, c, s), apply_rope(k, c, s)
        if cache is None:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            k_prev, v_prev = cache
            n = k_prev.size(2)
            k = torch.cat([k_prev, k], dim=2)
            v = torch.cat([v_prev, v], dim=2)
            if T == 1:
                out = F.scaled_dot_product_attention(q, k, v, is_causal=False)   # one query, all keys
            else:                                                                # prefill on top of a cache
                mask = torch.ones(T, k.size(2), dtype=torch.bool, device=q.device).tril(diagonal=n)
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        new_cache = (k, v) if (want_cache or cache is not None) else None
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.wo(out), new_cache


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        inner = cfg.expansion * cfg.embed_size
        self.w1 = nn.Linear(cfg.embed_size, inner, bias=False)
        self.w2 = nn.Linear(inner, cfg.embed_size, bias=False)
        self.w3 = nn.Linear(cfg.embed_size, inner, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.embed_size, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.embed_size, cfg.norm_eps)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin, cache=None, pos: int = 0, want_cache: bool = False):
        a, cache = self.attn(self.norm1(x), cos, sin, cache, pos, want_cache)
        x = x + a
        return x + self.mlp(self.norm2(x)), cache


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.embed_size)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.num_layers)])
        self.norm = RMSNorm(cfg.embed_size, cfg.norm_eps)
        self.head = None if cfg.tie_embeddings else nn.Linear(cfg.embed_size, cfg.vocab_size, bias=False)
        self.register_buffer("rope_cos", torch.zeros(1), persistent=False)
        self.register_buffer("rope_sin", torch.zeros(1), persistent=False)
        self._rope_len = 0
        self.apply(self._init)
        nn.init.normal_(self.token_embedding.weight, std=0.02)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)

    def ensure_rope(self, length: int):
        if self._rope_len >= length and self.rope_cos.numel() > 1:
            return
        length = max(length, self.cfg.context_length)
        cos, sin = rope_tables(self.cfg.embed_size // self.cfg.num_heads, length, self.cfg.rope_base)
        self.rope_cos, self.rope_sin = cos, sin
        self._rope_len = length

    def forward(self, idx, targets=None, cache=None, pos: int = 0, logits_last_only: bool = False,
                use_cache: bool = False):
        """use_cache=True keeps per-layer key/value tensors so generation can step one token at a
        time. The very first generation call has an empty cache, so the flag - not the cache content -
        decides whether the cache list is returned and kept."""
        B, T = idx.shape
        self.ensure_rope(pos + T + 1)
        x = self.token_embedding(idx)
        want_cache = use_cache or cache is not None
        new_caches = [] if want_cache else None
        for i, blk in enumerate(self.blocks):
            c = None if cache is None else cache[i]
            x, nc = blk(x, self.rope_cos, self.rope_sin, c, pos, want_cache)
            if want_cache:
                new_caches.append(nc)
        x = self.norm(x)
        w = self.token_embedding.weight if self.head is None else self.head.weight
        if targets is not None:
            # Chunked cross-entropy: the full (B, T, vocab) logits tensor is the memory hot spot on
            # CPU, so project and score only `loss_chunk / B` positions at a time.
            chunk = T
            if self.cfg.loss_chunk:
                chunk = max(1, min(T, -(-int(self.cfg.loss_chunk) // max(1, B))))
            total = x.new_zeros(())
            count = 0
            for start in range(0, T, chunk):
                lg = x[:, start:start + chunk] @ w.t()
                tgt = targets[:, start:start + chunk]
                total = total + F.cross_entropy(lg.reshape(-1, lg.size(-1)), tgt.reshape(-1),
                                                ignore_index=0, reduction="sum")
                count += int((tgt != 0).sum())
            return None, total / max(1, count), None
        if logits_last_only:
            x = x[:, -1:]
        logits = x @ w.t()
        return logits, None, new_caches

    # ------------------------------------------------------------------ helpers
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def state_dict_for_export(self):
        return self.state_dict()
