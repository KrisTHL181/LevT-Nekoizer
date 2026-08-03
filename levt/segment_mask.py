"""Segment-level attention masks for packed-sequence training.

Packed training concatenates ~21 unrelated examples into a single row, one
row per training sample.  Every segment is ``[BOS] ... [EOS]``, so adjacent
segments naturally form an ``[EOS][BOS]`` boundary.  Because the model is
bidirectional, self-attention (encoder and decoder) and cross-attention
(decoder -> encoder) would otherwise leak information across those
boundaries; these helpers build block-diagonal self masks and diagonal
cross masks that keep each segment attending only to its own segment.

Boundary convention:
    Position ``i`` starts a new segment iff ``i == 0`` or
    ``tokens[i-1] == eos and tokens[i] == bos``.

All token tensors are seq-first or ``(..., n)`` shaped, and padded positions
never equal ``bos``/``eos``, so padded tensors can be passed directly —
padding never introduces spurious boundaries (it inherits the segment id of
the preceding position, and any attention there is masked by the padding
mask anyway).

Only ``torch`` is imported; no other libraries.
"""

from __future__ import annotations

import torch


def segment_ids(tokens: torch.Tensor, bos: int, eos: int) -> torch.Tensor:
    """Compute 0-based segment ids for a ``(..., n)`` token tensor.

    A new segment starts at ``i == 0`` or where ``tokens[i-1] == eos`` and
    ``tokens[i] == bos``.  Implemented as a cumsum of boundary starts.

    Args:
        tokens: token ids, shape ``(..., n)``.
        bos:    BOS token id.
        eos:    EOS token id.

    Returns:
        Long tensor of the same shape with segment ids in ``[0, n_segments)``.
    """
    if tokens.shape[-1] == 0:
        return tokens.to(torch.long)
    starts = torch.zeros_like(tokens, dtype=torch.long)
    starts[..., 0] = 1
    if tokens.shape[-1] > 1:
        starts[..., 1:] = (
            (tokens[..., :-1] == eos) & (tokens[..., 1:] == bos)
        ).to(torch.long)
    return starts.cumsum(dim=-1) - 1


def self_attention_mask(segment_ids: torch.Tensor) -> torch.Tensor:
    """Build a block-diagonal self-attention mask from ``(B, n)`` segment ids.

    Returns a bool tensor of shape ``(B, n, n)`` that is ``True`` exactly where
    both positions belong to the same segment (``ids[:, :, None] == ids[:, None, :]``).
    """
    return segment_ids[:, :, None] == segment_ids[:, None, :]


def cross_attention_mask(tgt_ids: torch.Tensor, src_ids: torch.Tensor) -> torch.Tensor:
    """Build a diagonal cross-attention mask from segment id tensors.

    ``tgt_ids`` has shape ``(B, m)``, ``src_ids`` has shape ``(B, k)``; returns a
    bool tensor of shape ``(B, m, k)`` that is ``True`` exactly where a decoder
    segment attends to encoder positions of the same segment
    (``tgt_ids[:, :, None] == src_ids[:, None, :]``).
    """
    return tgt_ids[:, :, None] == src_ids[:, None, :]
