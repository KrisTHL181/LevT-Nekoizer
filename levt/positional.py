"""
Positional encoding strategies for the Levenshtein Transformer.

Supports three approaches:

  1. **Sinusoidal** (Vaswani et al., 2017)
     Additive absolute-position encoding injected at the embedding layer.
     This is the original LevT paper default.

  2. **RoPE** — Rotary Position Embedding (Su et al., 2021)
     Encodes relative position by rotating query / key vectors in the
     complex plane *during* attention.  No additive PE on embeddings.

  3. **ALiBi** — Attention with Linear Biases (Press et al., 2022)
     Adds a static, non-learned bias to pre-softmax attention scores:
         score[i][j] += -|i - j| * m_h
     where m_h is a head‑specific slope.  No additive PE on embeddings.

Each strategy has zero learnable parameters.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


# ═══════════════════════════════════════════════════════════════════════
# 1.  Sinusoidal Positional Embedding  (additive PE)
# ═══════════════════════════════════════════════════════════════════════

class SinusoidalPositionalEmbedding(nn.Module):
    """Classic sinusoidal PE:  PE[pos, 2i]   = sin(pos / 10000^{2i/d})
                              PE[pos, 2i+1] = cos(pos / 10000^{2i/d})

    Dynamically extends the buffer if the sequence exceeds the initial max_len.
    """

    def __init__(self, d_model: int, max_len: int = 1024, base: float = 10000.0):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.base = base
        pe = self._compute_pe(max_len)
        self.register_buffer("pe", pe, persistent=False)

    def _compute_pe(self, length: int) -> torch.Tensor:
        pe = torch.zeros(length, self.d_model)
        position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float) * (-math.log(self.base) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: self.d_model // 2])
        return pe

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (seq_len, batch, d_model).  Returns x + PE."""
        seq_len = x.size(0)
        if seq_len > self.pe.size(0):
            self.pe = self._compute_pe(seq_len).to(device=x.device, dtype=x.dtype)
        return x + self.pe[:seq_len, :].to(device=x.device, dtype=x.dtype).unsqueeze(1)


# ═══════════════════════════════════════════════════════════════════════
# 2.  Rotary Position Embedding (RoPE) — applied to Q / K in attention
# ═══════════════════════════════════════════════════════════════════════

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims:  (x1, x2) → (-x2, x1)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embedding to query and key tensors.

    Args:
        q, k:  (..., seq_len, head_dim)   — last two dims are (pos, feat)
        cos, sin: (seq_len, head_dim)      — pre-computed frequencies

    Returns:
        (q_rotated, k_rotated) — same shapes as q, k.
    """
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


class RotaryPositionalEmbedding(nn.Module):
    """Pre-compute cos / sin tables for RoPE.

    Usage::

        rope = RotaryPositionalEmbedding(head_dim)
        cos, sin = rope(seq_len, offset=0)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
    """

    def __init__(self, dim: int, max_len: int = 2048, base: float = 10000.0):
        """
        Args:
            dim:  head dimension (d_model // n_heads)
            max_len:  pre-allocated length for the cache
            base:  frequency base (default 10000 as in the original paper)
        """
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        self.base = base
        self.register_buffer("cos_cached", torch.empty(0), persistent=False)
        self.register_buffer("sin_cached", torch.empty(0), persistent=False)
        self._set_cos_sin_cache(max_len, device=torch.device("cpu"), dtype=torch.float32)

    def _set_cos_sin_cache(
        self, length: int, *, device: torch.device, dtype: torch.dtype,
    ) -> None:
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim)
        )
        t = torch.arange(length, device=device, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)                     # (length, dim//2)
        emb = torch.cat((freqs, freqs), dim=-1)              # (length, dim)
        self.cos_cached = emb.cos().to(dtype=dtype)
        self.sin_cached = emb.sin().to(dtype=dtype)

    def forward(
        self,
        seq_len: int,
        offset: int = 0,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (cos, sin) tables of shape (seq_len, head_dim)."""
        needed = offset + seq_len
        device = self.cos_cached.device if device is None else device
        dtype = self.cos_cached.dtype if dtype is None else dtype
        if (
            needed > self.cos_cached.size(0)
            or self.cos_cached.device != device
            or self.cos_cached.dtype != dtype
        ):
            length = max(needed, self.cos_cached.size(0), self.max_len)
            self._set_cos_sin_cache(length, device=device, dtype=dtype)
        return (
            self.cos_cached[offset : offset + seq_len],      # (seq_len, head_dim)
            self.sin_cached[offset : offset + seq_len],
        )


# ═══════════════════════════════════════════════════════════════════════
# 3.  ALiBi — Attention with Linear Biases
# ═══════════════════════════════════════════════════════════════════════

def get_alibi_slopes(n_heads: int) -> torch.Tensor:
    """
    ALiBi head‑specific slopes (Press et al., 2022).

    Returns (n_heads,) tensor of slopes.  For n_heads = 8:
    [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625]
    """
    start = 2 ** (-8 / n_heads)
    slopes = torch.tensor([start ** (i + 1) for i in range(n_heads)], dtype=torch.float32)
    return slopes


def get_alibi_bias(
    q_len: int,
    k_len: int,
    n_heads: int,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Build ALiBi attention bias for BIDIRECTIONAL (non-causal) attention.

    bias[h, i, j] = -|i - j| · m_h

    Returns:
        tensor of shape (1, n_heads, q_len, k_len)
        (broadcastable over batch dimension)
    """
    slopes = get_alibi_slopes(n_heads).to(device=device, dtype=dtype)            # (n_heads,)
    positions = torch.arange(q_len, device=device).unsqueeze(1) - \
                torch.arange(k_len, device=device).unsqueeze(0)                  # (q_len, k_len)
    bias = -torch.abs(positions).to(dtype)                                        # (q_len, k_len)
    bias = bias.unsqueeze(0) * slopes.view(-1, 1, 1)                             # (n_heads, q_len, k_len)
    return bias.unsqueeze(0)                                                      # (1, n_heads, q_len, k_len)




class ALiBiPositionalBias(nn.Module):
    """Convenience wrapper that caches the last-seen bias to avoid recomputation
    when sequence lengths stay the same across calls."""

    def __init__(self, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self._cached_bias: Optional[torch.Tensor] = None
        self._cached_shape: Optional[Tuple[int, int]] = None

    def forward(
        self, q_len: int, k_len: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        if self._cached_shape != (q_len, k_len) or self._cached_bias is None:
            self._cached_bias = get_alibi_bias(q_len, k_len, self.n_heads, dtype, device)
            self._cached_shape = (q_len, k_len)
        return self._cached_bias.to(device=device, dtype=dtype)
