"""
Levenshtein Transformer model architecture.

The model consists of:
  - A Transformer encoder (conditioned on source).
  - A bidirectional Transformer decoder shared by three policy heads:
      1. Deletion classifier   — binary keep/delete per position.
      2. Placeholder classifier — predict # of <PLH> tokens per adjacent slot.
      3. Token classifier       — predict the actual tokens for each <PLH>.
  - Supports *early exit*: deletion / placeholder heads can attach to an
    intermediate decoder layer instead of the final one.
  - Supports three positional-encoding strategies: sinusoidal, RoPE, ALiBi.

Reference: "Levenshtein Transformer" (Gu et al., NeurIPS 2019)
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import LevTConfig
from .positional import (
    ALiBiPositionalBias,
    RotaryPositionalEmbedding,
    SinusoidalPositionalEmbedding,
)


# ═══════════════════════════════════════════════════════════════════════
# RMSNorm  — lightweight LayerNorm variant (no centering)
# ═══════════════════════════════════════════════════════════════════════

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).

    ``RMSNorm(x) = x / sqrt(mean(x²) + eps) * weight``
    """

    def __init__(self, dim: int, eps: float = 1e-6, gated: bool = False, gate_rank: int = 32):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
        if gated:
            self.gate_proj = nn.Linear(dim, gate_rank, bias=False)
            self.up_proj = nn.Linear(gate_rank, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim)
        x = F.rms_norm(x, self.weight.shape, self.weight, self.eps)
        if hasattr(self, "gate_proj"):
            gate = self.gate_proj(x)
            x = x * F.sigmoid(self.up_proj(F.silu(gate)))
        return x


# ═══════════════════════════════════════════════════════════════════════
# Flexible multi-head attention  (sinusoidal / RoPE / ALiBi)
# ═══════════════════════════════════════════════════════════════════════

class MultiheadAttention(nn.Module):
    """
    Unified multi-head attention with optional RoPE or ALiBi.

    - ``pos_type == "sinusoidal"``: standard attention (PE already in embeddings).
    - ``pos_type == "rope"``:        rotary embeddings applied to Q and K.
    - ``pos_type == "alibi"``:       linear bias added to pre-softmax scores.

    Gated Attention (Qwen3-style):
    - ``headwise_gate``:   one learnable scalar gate per head, applied after SDPA.
    - ``elementwise_gate``: one learnable gate per head dimension, applied after SDPA.
      The gate is query-dependent: it is produced by extending the Q projection
      and passed through sigmoid before multiplying the attention output.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        pos_type: str = "sinusoidal",
        dropout: float = 0.0,
        max_len: int = 2048,
        rope_base: float = 10000.0,
        qk_norm: bool = False,
        headwise_gate: bool = False,
        elementwise_gate: bool = False,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.pos_type = pos_type
        self.dropout_p = dropout
        self.qk_norm = qk_norm
        self.headwise_gate = headwise_gate
        self.elementwise_gate = elementwise_gate

        # Q projection output dim depends on gate mode:
        #   no gate:      embed_dim
        #   headwise:     embed_dim + num_heads        (1 scalar gate per head)
        #   elementwise:  embed_dim * 2                (head_dim scalars per head)
        if headwise_gate:
            q_out_dim = embed_dim + num_heads
        elif elementwise_gate:
            q_out_dim = embed_dim * 2
        else:
            q_out_dim = embed_dim

        # Projections
        self.q_proj = nn.Linear(embed_dim, q_out_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

        # QK normalization — per-head RMSNorm (LLaMA-style)
        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        else:
            self.q_norm = None
            self.k_norm = None

        # Positional-encoding helpers
        if pos_type == "rope":
            self.rotary_emb = RotaryPositionalEmbedding(self.head_dim, max_len=max_len, base=rope_base)
        else:
            self.rotary_emb = None

        if pos_type == "alibi":
            self.alibi = ALiBiPositionalBias(num_heads)
        else:
            self.alibi = None

    # ------------------------------------------------------------------
    # Public forward
    # ------------------------------------------------------------------

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        position_offset: int = 0,
    ) -> Tuple[torch.Tensor, None]:
        """
        Args:
            query, key, value: (seq_len, batch, embed_dim)
            key_padding_mask:  (batch, seq_len)  True = pad / ignore
            attn_mask:         additional mask (reserved, not used by LevT)
            position_offset:   for RoPE, starting position index (default 0)

        Returns:
            (output, None)  — second element is a dummy to match
            ``nn.MultiheadAttention`` API.
        """
        q_len, bsz, _ = query.shape
        k_len = key.shape[0]

        # --- project and split gate from Q if gated ---
        q_raw = self.q_proj(query)  # (q_len, bsz, q_out_dim)
        gate: Optional[torch.Tensor] = None

        if self.headwise_gate:
            # Q projection outputs: embed_dim + num_heads  →  head_dim + 1 per head
            q_raw = q_raw.view(q_len, bsz, self.num_heads, self.head_dim + 1)
            gate = q_raw[..., -1:]           # (q_len, bsz, num_heads, 1)
            q = q_raw[..., :-1]              # (q_len, bsz, num_heads, head_dim)
        elif self.elementwise_gate:
            # Q projection outputs: embed_dim * 2  →  2 * head_dim per head
            q_raw = q_raw.view(q_len, bsz, self.num_heads, 2 * self.head_dim)
            q, gate = q_raw.split(self.head_dim, dim=-1)
            # q: (q_len, bsz, num_heads, head_dim)
            # gate: (q_len, bsz, num_heads, head_dim)
        else:
            q = q_raw.view(q_len, bsz, self.num_heads, self.head_dim)

        k = self.k_proj(key).view(k_len, bsz, self.num_heads, self.head_dim)
        v = self.v_proj(value).view(k_len, bsz, self.num_heads, self.head_dim)

        # --- RoPE: rotate Q and K (gate is excluded — applied raw) ---
        if self.pos_type == "rope":
            q, k = self._apply_rope(q, k, position_offset)

        # --- QK RMSNorm: normalize each head vector (stabilizes training) ---
        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # --- reshape to (batch, n_heads, seq, head_dim) for SDPA ---
        q = q.permute(1, 2, 0, 3)  # (B, H, q_len, D)
        k = k.permute(1, 2, 0, 3)  # (B, H, k_len, D)
        v = v.permute(1, 2, 0, 3)  # (B, H, k_len, D)

        # --- build combined attention mask (padding + ALiBi) ---
        combined_mask = self._build_attn_mask(
            key_padding_mask, attn_mask, q_len, k_len, bsz, query.device, query.dtype
        )

        # --- core attention ---
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=combined_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        # out: (B, H, q_len, D)

        # --- Gated Attention: apply sigmoid gate after SDPA ---
        if gate is not None:
            gate = gate.permute(1, 2, 0, 3)   # (B, H, q_len, gate_dim)
            out = out * torch.sigmoid(gate)    # broadcast: (B,H,q_len,D) * (B,H,q_len,1|D)

        # --- reshape back ---
        out = out.permute(2, 0, 1, 3).contiguous().view(q_len, bsz, self.embed_dim)
        out = self.out_proj(out)
        return out, None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        offset: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary position embeddings to Q and K independently."""
        q_len, k_len = q.shape[0], k.shape[0]
        cos_q, sin_q = self.rotary_emb(
            q_len, offset=offset, device=q.device, dtype=q.dtype,
        )
        cos_k, sin_k = self.rotary_emb(
            k_len, offset=0, device=k.device, dtype=k.dtype,
        )

        # Broadcast: (seq_len, head_dim) → (seq_len, 1, 1, head_dim)
        cos_q = cos_q.unsqueeze(1).unsqueeze(1)
        sin_q = sin_q.unsqueeze(1).unsqueeze(1)
        cos_k = cos_k.unsqueeze(1).unsqueeze(1)
        sin_k = sin_k.unsqueeze(1).unsqueeze(1)

        q_rot = (q * cos_q) + (_rotate_half(q) * sin_q)
        k_rot = (k * cos_k) + (_rotate_half(k) * sin_k)
        return q_rot, k_rot

    def _build_attn_mask(
        self,
        key_padding_mask: Optional[torch.Tensor],
        attn_mask: Optional[torch.Tensor],
        q_len: int,
        k_len: int,
        bsz: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """Combine padding mask with optional ALiBi bias into a float mask."""
        mask: Optional[torch.Tensor] = None

        # 1. Padding mask → float mask  (-inf for pad positions)
        if key_padding_mask is not None:
            # key_padding_mask: (batch, k_len), True = masked
            mask = torch.zeros(bsz, 1, 1, k_len, device=device, dtype=dtype)
            mask = mask.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        # 2. ALiBi bias
        if self.pos_type == "alibi":
            alibi_bias = self.alibi(q_len, k_len, device, dtype)  # (1, H, q_len, k_len)
            if mask is None:
                mask = alibi_bias
            else:
                mask = mask + alibi_bias

        # 3. User-supplied mask (unused in default LevT)
        if attn_mask is not None:
            mask = attn_mask if mask is None else mask + attn_mask

        return mask


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims: (x1, x2) → (-x2, x1)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# ═══════════════════════════════════════════════════════════════════════
# Feed-forward block  (standard ReLU / GELU  or  SwiGLU)
# ═══════════════════════════════════════════════════════════════════════

class FFNBlock(nn.Module):
    """
    Unified feed-forward block.

    - ``activation in {"relu", "gelu"}`` :  standard two-layer FFN.
    - ``activation == "swiglu"`` :        SwiGLU  (SiLU gate × up) → down.

    SwiGLU uses three projections instead of two; to keep the parameter
    count roughly equivalent set ``d_ff_swiglu ≈ 2/3 * d_ff``.
    """

    def __init__(
        self, d_model: int, d_ff: int, activation: str = "relu", dropout: float = 0.1,
    ):
        super().__init__()
        self.activation_name = activation
        self.dropout = nn.Dropout(dropout)

        if activation == "swiglu":
            # SwiGLU: SiLU(gate(x)) * up(x) → down
            self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
            self.up_proj   = nn.Linear(d_model, d_ff, bias=False)
            self.down_proj = nn.Linear(d_ff, d_model, bias=False)
            self.linear1 = None
            self.linear2 = None
            self.act = None
        else:
            # Standard FFN: down(act(up(x)))
            self.linear1 = nn.Linear(d_model, d_ff)
            self.linear2 = nn.Linear(d_ff, d_model)
            self.act = nn.ReLU() if activation == "relu" else nn.GELU()
            self.gate_proj = None
            self.up_proj = None
            self.down_proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation_name == "swiglu":
            gate = F.silu(self.gate_proj(x))
            up = self.up_proj(x)
            return self.dropout(self.down_proj(gate * up))
        else:
            return self.dropout(self.linear2(self.dropout(self.act(self.linear1(x)))))


# ═══════════════════════════════════════════════════════════════════════
# Encoder
# ═══════════════════════════════════════════════════════════════════════

class TransformerEncoderLayer(nn.Module):
    """One Transformer encoder layer (supports sinusoidal / RoPE / ALiBi)."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        pos_type: str = "sinusoidal",
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation: str = "relu",
        max_len: int = 2048,
        rope_base: float = 10000.0,
        qk_norm: bool = False,
        headwise_gate: bool = False,
        elementwise_gate: bool = False,
    ):
        super().__init__()
        self.self_attn = MultiheadAttention(
            d_model, n_heads, pos_type=pos_type,
            dropout=attention_dropout, max_len=max_len, rope_base=rope_base,
            qk_norm=qk_norm, headwise_gate=headwise_gate,
            elementwise_gate=elementwise_gate,
        )
        self.ffn = FFNBlock(d_model, d_ff, activation=activation, dropout=dropout)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x2, _ = self.self_attn(x, x, x, key_padding_mask=key_padding_mask)
        x = residual + self.dropout(x2)
        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)
        return x


class TransformerEncoder(nn.Module):
    """Stack of Transformer encoder layers."""

    def __init__(
        self, d_model: int, n_heads: int, d_ff: int, n_layers: int,
        pos_type: str = "sinusoidal", dropout: float = 0.1,
        attention_dropout: float = 0.1, activation: str = "relu",
        max_len: int = 2048, rope_base: float = 10000.0,
        qk_norm: bool = False,
        headwise_gate: bool = False,
        elementwise_gate: bool = False,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model, n_heads, d_ff, pos_type=pos_type,
                dropout=dropout, attention_dropout=attention_dropout,
                activation=activation, max_len=max_len, rope_base=rope_base,
                qk_norm=qk_norm, headwise_gate=headwise_gate,
                elementwise_gate=elementwise_gate,
            )
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)

    def forward(
        self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, key_padding_mask)
        return self.norm(x)


# ═══════════════════════════════════════════════════════════════════════
# Decoder  (bidirectional, returns ALL intermediate outputs)
# ═══════════════════════════════════════════════════════════════════════

class LevTDecoderLayer(nn.Module):
    """
    One decoder layer: bidirectional self-attention → cross-attention → FFN.

    Self-attention is NOT causally masked — the decoder sees the full current
    sequence simultaneously (non-autoregressive).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        pos_type: str = "sinusoidal",
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation: str = "relu",
        max_len: int = 2048,
        rope_base: float = 10000.0,
        qk_norm: bool = False,
        headwise_gate: bool = False,
        elementwise_gate: bool = False,
    ):
        super().__init__()
        attn_kw = dict(pos_type=pos_type, dropout=attention_dropout, max_len=max_len, rope_base=rope_base, qk_norm=qk_norm, headwise_gate=headwise_gate, elementwise_gate=elementwise_gate)
        self.self_attn = MultiheadAttention(d_model, n_heads, **attn_kw)
        self.cross_attn = MultiheadAttention(d_model, n_heads, **attn_kw)
        self.ffn = FFNBlock(d_model, d_ff, activation=activation, dropout=dropout)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.norm3 = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Bidirectional self-attention
        residual = tgt
        tgt = self.norm1(tgt)
        tgt2, _ = self.self_attn(tgt, tgt, tgt, key_padding_mask=tgt_key_padding_mask)
        tgt = residual + self.dropout(tgt2)
        # Cross-attention to encoder memory
        residual = tgt
        tgt = self.norm2(tgt)
        tgt2, _ = self.cross_attn(tgt, memory, memory, key_padding_mask=memory_key_padding_mask)
        tgt = residual + self.dropout(tgt2)
        # Feed-forward
        residual = tgt
        tgt = self.norm3(tgt)
        tgt = residual + self.ffn(tgt)
        return tgt


class LevTDecoder(nn.Module):
    """Stack of bidirectional decoder layers.  Returns ALL intermediate outputs."""

    def __init__(
        self, d_model: int, n_heads: int, d_ff: int, n_layers: int,
        pos_type: str = "sinusoidal", dropout: float = 0.1,
        attention_dropout: float = 0.1, activation: str = "relu",
        max_len: int = 2048, rope_base: float = 10000.0,
        qk_norm: bool = False,
        headwise_gate: bool = False,
        elementwise_gate: bool = False,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            LevTDecoderLayer(
                d_model, n_heads, d_ff, pos_type=pos_type,
                dropout=dropout, attention_dropout=attention_dropout,
                activation=activation, max_len=max_len, rope_base=rope_base,
                qk_norm=qk_norm, headwise_gate=headwise_gate,
                elementwise_gate=elementwise_gate,
            )
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        outputs: List[torch.Tensor] = []
        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_key_padding_mask, memory_key_padding_mask)
            outputs.append(self.norm(tgt))
        return outputs


# ═══════════════════════════════════════════════════════════════════════
# Levenshtein Transformer Model
# ═══════════════════════════════════════════════════════════════════════

class LevTModel(nn.Module):
    """
    Levenshtein Transformer — full model:

        encoder → decoder (N layers) → { deletion, placeholder, token } heads.

    Shape conventions (PyTorch transformer style):
      src_tokens:   (src_len, batch)
      tgt_tokens:   (tgt_len, batch)
      All padding masks: (batch, seq_len),  True = pad / ignore.
    """

    def __init__(self, config: LevTConfig):
        super().__init__()
        self.config = config
        d_model = config.d_model
        vocab_size = config.vocab_size
        pos = config.pos_encoding_type

        # ---- Shared embedding and permanent input projections ----
        embedding_dim = config.embedding_dim
        assert embedding_dim is not None
        self.shared_embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=config.pad_token_id,
        )
        self.encoder_input_projection = nn.Linear(embedding_dim, d_model, bias=False)
        self.decoder_input_projection = nn.Linear(embedding_dim, d_model, bias=False)

        # ---- Positional encoding (additive); skipped for RoPE / ALiBi ----
        if pos == "sinusoidal":
            self.src_pos: Optional[SinusoidalPositionalEmbedding] = SinusoidalPositionalEmbedding(
                d_model, config.max_source_positions,
            )
            self.tgt_pos: Optional[SinusoidalPositionalEmbedding] = SinusoidalPositionalEmbedding(
                d_model, config.max_target_positions,
            )
        else:
            self.src_pos = None
            self.tgt_pos = None

        layer_kw = dict(
            pos_type=pos, dropout=config.dropout,
            attention_dropout=config.attention_dropout, activation=config.activation,
            max_len=max(config.max_source_positions, config.max_target_positions),
            rope_base=config.rope_base, qk_norm=config.qk_norm,
            headwise_gate=config.headwise_attn_output_gate,
            elementwise_gate=config.elementwise_attn_output_gate,
        )

        # ---- Encoder / Decoder ----
        self.encoder = TransformerEncoder(
            d_model, config.n_heads, config.d_ff, config.n_encoder_layers, **layer_kw,
        )
        self.decoder = LevTDecoder(
            d_model, config.n_heads, config.d_ff, config.n_decoder_layers, **layer_kw,
        )

        # ---- Classifier heads ----
        self.deletion_head = nn.Linear(d_model, 2, bias=True)
        self.placeholder_head = nn.Linear(2 * d_model, config.max_placeholder + 1, bias=True)

        # Layer indices for early exit
        self._del_layer = (
            config.early_exit_del if config.early_exit_del is not None
            else config.n_decoder_layers - 1
        )
        self._plh_layer = (
            config.early_exit_plh if config.early_exit_plh is not None
            else config.n_decoder_layers - 1
        )

        self._init_params()

    def _init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @property
    def src_embed(self) -> nn.Embedding:
        """Compatibility alias for the shared embedding."""
        return self.shared_embedding

    @property
    def tgt_embed(self) -> nn.Embedding:
        """Compatibility alias for the shared embedding."""
        return self.shared_embedding

    def copy_embedding_weights(self, weights: torch.Tensor) -> None:
        """Copy validated external embedding weights after random initialization."""
        expected = self.shared_embedding.weight.shape
        if weights.ndim != 2 or tuple(weights.shape) != tuple(expected):
            raise ValueError(
                f"embedding weights have shape {tuple(weights.shape)}, expected {tuple(expected)}"
            )
        with torch.no_grad():
            self.shared_embedding.weight.copy_(
                weights.to(device=self.shared_embedding.weight.device,
                           dtype=self.shared_embedding.weight.dtype)
            )

    def _token_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        projected = F.linear(hidden, self.decoder_input_projection.weight.T)
        return F.linear(projected, self.shared_embedding.weight)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _add_pos(self, emb: torch.Tensor, pos_module: Optional[nn.Module]) -> torch.Tensor:
        """Apply sinusoidal PE if configured; otherwise pass through."""
        if pos_module is not None:
            return pos_module(emb * math.sqrt(self.config.d_model))
        return emb

    def forward(
        self,
        src_tokens: torch.Tensor,
        prev_output_tokens: torch.Tensor,
        src_padding_mask: Optional[torch.Tensor] = None,
        tgt_padding_mask: Optional[torch.Tensor] = None,
        return_deletion: bool = True,
        return_placeholder: bool = True,
        return_token: bool = True,
    ):
        # Encode source
        src_emb = self._add_pos(
            self.encoder_input_projection(self.shared_embedding(src_tokens)), self.src_pos,
        )
        memory = self.encoder(src_emb, src_padding_mask)

        # Decode target — get all layer outputs
        tgt_emb = self._add_pos(
            self.decoder_input_projection(self.shared_embedding(prev_output_tokens)), self.tgt_pos,
        )
        decoder_outputs = self.decoder(tgt_emb, memory, tgt_padding_mask, src_padding_mask)

        result = {}
        if return_deletion:
            del_h = decoder_outputs[self._del_layer]
            result["del_logits"] = self.deletion_head(del_h)
        if return_placeholder:
            plh_h = decoder_outputs[self._plh_layer]
            plh_pairs = torch.cat([plh_h[:-1], plh_h[1:]], dim=-1)
            result["plh_logits"] = self.placeholder_head(plh_pairs)
        if return_token:
            tok_h = decoder_outputs[-1]
            result["tok_logits"] = self._token_logits(tok_h)
        return result

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def encode(
        self, src_tokens: torch.Tensor,
        src_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode source once (reused across decoding iterations)."""
        src_emb = self._add_pos(
            self.encoder_input_projection(self.shared_embedding(src_tokens)), self.src_pos,
        )
        return self.encoder(src_emb, src_padding_mask)

    def decode_with_memory(
        self,
        memory: torch.Tensor,
        prev_output_tokens: torch.Tensor,
        src_padding_mask: Optional[torch.Tensor] = None,
        tgt_padding_mask: Optional[torch.Tensor] = None,
        return_deletion: bool = True,
        return_placeholder: bool = True,
        return_token: bool = True,
        token_positions: Optional[torch.Tensor] = None,
    ):
        """Decode memory and return requested heads (all heads by default)."""
        tgt_emb = self._add_pos(
            self.decoder_input_projection(self.shared_embedding(prev_output_tokens)), self.tgt_pos,
        )
        decoder_outputs = self.decoder(tgt_emb, memory, tgt_padding_mask, src_padding_mask)
        result = {}
        if return_deletion:
            result["del_logits"] = self.deletion_head(decoder_outputs[self._del_layer])
        if return_placeholder:
            result["plh_logits"] = self.placeholder_head(
                torch.cat([decoder_outputs[self._plh_layer][:-1],
                           decoder_outputs[self._plh_layer][1:]], dim=-1)
            )
        if return_token:
            hidden = decoder_outputs[-1]
            if token_positions is not None:
                if token_positions.shape != prev_output_tokens.shape or token_positions.dtype != torch.bool:
                    raise ValueError("token_positions must be a boolean mask matching target tokens")
                hidden = hidden[token_positions]
            result["tok_logits"] = self._token_logits(hidden)
        return result
