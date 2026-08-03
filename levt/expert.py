"""
Expert (oracle) policy for the Levenshtein Transformer.

Provides:
  - Levenshtein alignment (insertion + deletion only, no substitution)
  - Oracle deletion: which tokens to delete to match target
  - Oracle insertion: placeholder counts and token ids for each slot
  - Random deletion policy π^rnd for noise injection during training

Reference: "Levenshtein Transformer" (Gu et al., NeurIPS 2019), Sections 2-3.
"""

from __future__ import annotations

import random
from typing import List, Optional, Tuple

import torch

# Try to load the C++ acceleration module at import time.
_cpp_module = None
try:
    from ._levenshtein_ops import (
        levenshtein_align_cpp,
        levenshtein_deletion_cpp_batch,
        levenshtein_insertion_cpp_batch,
    )
    _cpp_module = True  # module loaded; actual function may still return None
except ImportError:
    levenshtein_align_cpp = None  # type: ignore[assignment]
    levenshtein_deletion_cpp_batch = None  # type: ignore[assignment]
    levenshtein_insertion_cpp_batch = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Levenshtein DP alignment (insert + delete only, no substitution)
# ---------------------------------------------------------------------------

def levenshtein_align(
    y: torch.Tensor,
    y_star: torch.Tensor,
    pad_idx: int = 0,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Compute optimal edit alignment between two sequences using DP.

    Operations: DELETE (cost 1), INSERT (cost 1), MATCH (cost 0 if tokens equal).
    No substitution — changing a token requires DELETE + INSERT (cost 2).

    Tries the C++ extension first; falls back to pure Python seamlessly.

    Args:
        y:      current sequence, shape (L,) — includes BOS/EOS
        y_star: target sequence, shape (M,) — includes BOS/EOS
        pad_idx: padding token id (ignored)

    Returns:
        deletions:  1-D int64 tensor — indices in y to delete
        insertions: list of 1-D int64 tensors — per-gap insertion tokens

    NOTE: Boundary tokens (first and last) are never deleted by the callers.
    """
    # ── Fast path: C++ extension (tensors in, tensors out) ────────────
    if levenshtein_align_cpp is not None:
        result = levenshtein_align_cpp(y, y_star)
        if result is not None:
            return result

    # ── Fallback: pure Python DP ────────────────────────────────────
    del_list, ins_lists = _levenshtein_align_py(y, y_star)
    # Convert to tensors for uniform interface
    del_tensor = torch.tensor(del_list, dtype=torch.long)
    ins_tensors = [torch.tensor(ins, dtype=torch.long) for ins in ins_lists]
    return del_tensor, ins_tensors


def _levenshtein_align_py(
    y: torch.Tensor,
    y_star: torch.Tensor,
) -> Tuple[List[int], List[List[int]]]:
    """Pure-Python DP implementation (kept as fallback)."""
    y_list = y.tolist()
    ys_list = y_star.tolist()
    n, m = len(y_list), len(ys_list)

    # DP table: (n+1) x (m+1)
    INF = 10 ** 9
    dp = torch.full((n + 1, m + 1), INF, dtype=torch.long)
    # back pointers: 0=match, 1=delete, 2=insert
    back = torch.zeros((n + 1, m + 1), dtype=torch.uint8)

    dp[0, 0] = 0
    for i in range(1, n + 1):
        dp[i, 0] = i
        back[i, 0] = 1  # delete
    for j in range(1, m + 1):
        dp[0, j] = j
        back[0, j] = 2  # insert

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            best_val = INF
            best_op = 0

            # Delete
            cand = dp[i - 1, j] + 1
            if cand < best_val:
                best_val, best_op = cand, 1

            # Insert
            cand = dp[i, j - 1] + 1
            if cand < best_val:
                best_val, best_op = cand, 2

            # Match (if tokens equal)
            if y_list[i - 1] == ys_list[j - 1]:
                cand = dp[i - 1, j - 1]
                if cand < best_val:
                    best_val, best_op = cand, 0

            dp[i, j] = best_val
            back[i, j] = best_op

    # Backtrack through DP
    deletions: List[int] = []
    matched_pairs: List[Tuple[int, int]] = []

    i, j = n, m
    while i > 0 or j > 0:
        op = back[i, j].item()
        if op == 0:  # match
            i -= 1
            j -= 1
            matched_pairs.append((i, j))
        elif op == 1:  # delete
            i -= 1
            deletions.append(i)
        else:  # insert
            j -= 1

    deletions.reverse()
    matched_pairs.reverse()

    # Re-backtrack to collect insertions between matched positions
    insertions_raw: List[int] = []
    insertion_after: List[int] = []

    i, j = n, m
    current_after = n
    while i > 0 or j > 0:
        op = back[i, j].item()
        if op == 0:  # match
            i -= 1
            j -= 1
            current_after = i
        elif op == 1:  # delete
            i -= 1
        else:  # insert
            j -= 1
            insertions_raw.append(ys_list[j])
            insertion_after.append(current_after)

    insertions_raw.reverse()
    insertion_after.reverse()

    del_set = set(deletions)
    surviving = [idx for idx in range(n) if idx not in del_set]
    per_gap: List[List[int]] = [[] for _ in range(max(0, len(surviving) - 1))]

    if len(surviving) >= 2:
        surv_rank = {sid: si for si, sid in enumerate(surviving)}
        for tok, after_y in zip(insertions_raw, insertion_after):
            gap_surv_idx = -1
            if after_y in surv_rank:
                gap_surv_idx = surv_rank[after_y] - 1
            else:
                for si in range(len(surviving)):
                    if surviving[si] < after_y:
                        gap_surv_idx = si
                    else:
                        break
            if 0 <= gap_surv_idx < len(per_gap):
                per_gap[gap_surv_idx].append(tok)

    return deletions, per_gap


# ---------------------------------------------------------------------------
# Oracle policies
# ---------------------------------------------------------------------------

def _segment_sequences(
    y: torch.Tensor,
    bos_idx: int,
    eos_idx: int,
) -> List[torch.Tensor]:
    """Split a packed sequence into segments at ``[EOS][BOS]`` boundaries.

    Returns *views* (no copy).  A single-segment (unpacked) sequence yields
    ``[y]`` unchanged, so segmentation is a no-op for unpacked data.  The
    segmenting keeps the oracles from ever editing across a segment boundary,
    which preserves the segment count of packed roll-ins.
    """
    if y.numel() < 2:
        return [y]
    boundary = (y[:-1] == eos_idx) & (y[1:] == bos_idx)
    if not boundary.any():
        return [y]
    cuts = (boundary.nonzero().flatten() + 1).tolist()
    starts = [0] + cuts
    ends = cuts + [len(y)]
    return [y[s:e] for s, e in zip(starts, ends)]


def _deletion_mask(y: torch.Tensor, y_star: torch.Tensor) -> torch.Tensor:
    """Core deletion oracle on one sequence (whole-sequence DP)."""
    deletions, _ = levenshtein_align(y, y_star)
    mask = torch.zeros(len(y), dtype=torch.bool)
    if deletions.numel() > 0:
        mask[deletions] = True  # fancy indexing — O(num_deletions)
    # Never delete boundaries
    mask[0] = False
    mask[-1] = False
    return mask


def oracle_deletion(
    y: torch.Tensor,
    y_star: torch.Tensor,
    bos_idx: int = 1,
    eos_idx: int = 2,
) -> torch.Tensor:
    """
    Oracle deletion policy: compute optimal tokens to delete from y.

    For packed sequences (multiple ``[BOS]...[EOS]`` segments), the DP runs
    independently on each segment so no interior segment boundary is ever
    deleted and the roll-in keeps its segment count.  Unpacked sequences are
    unchanged (one segment).

    Args:
        y:      current sequence (with BOS, EOS), shape (L,)
        y_star: target sequence (with BOS, EOS), shape (M,)

    Returns:
        mask: boolean tensor of shape (L,), True = DELETE this token.
              Boundaries (BOS/EOS) are always False (never deleted).
    """
    segs_y = _segment_sequences(y, bos_idx, eos_idx)
    segs_s = _segment_sequences(y_star, bos_idx, eos_idx)
    if len(segs_y) > 1 and len(segs_y) == len(segs_s):
        return torch.cat([
            _deletion_mask(sy, ss) for sy, ss in zip(segs_y, segs_s)
        ])
    return _deletion_mask(y, y_star)


def _insertion_oracle(
    y: torch.Tensor,
    y_star: torch.Tensor,
    max_placeholder: int,
    plh_token_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Core insertion oracle on one sequence (whole-sequence DP)."""
    deletions, insertions = levenshtein_align(y, y_star)

    # Build surviving mask (tokens NOT deleted)
    del_mask = torch.zeros(len(y), dtype=torch.bool)
    if deletions.numel() > 0:
        del_mask[deletions] = True
    surviving = torch.where(~del_mask)[0]  # (num_surviving,)

    num_gaps = max(0, len(surviving) - 1)

    # Placeholder counts for each gap in the ORIGINAL y (len(y)-1 gaps).
    # Insertion counts are assigned to the gap anchored at each surviving token.
    p_star = torch.zeros(len(y) - 1, dtype=torch.long)
    t_star_parts: List[torch.Tensor] = []

    for gi in range(num_gaps):
        left_orig = int(surviving[gi].item())
        tokens_to_insert = insertions[gi]  # 1-D tensor (may be empty)
        count = min(tokens_to_insert.numel(), max_placeholder)
        p_star[left_orig] = count
        if count > 0:
            t_star_parts.append(tokens_to_insert[:count])

    if t_star_parts:
        t_star = torch.cat(t_star_parts)
    else:
        t_star = torch.tensor([], dtype=torch.long)

    return p_star, t_star


def oracle_insertion(
    y: torch.Tensor,
    y_star: torch.Tensor,
    max_placeholder: int = 255,
    plh_token_id: int = 3,
    bos_idx: int = 1,
    eos_idx: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Oracle insertion policy: compute optimal placeholders and tokens.

    For packed sequences the DP runs per segment and a zero-insertion gap is
    forced at every segment boundary, so the edit script never crosses an
    ``[EOS][BOS]`` boundary.  Unpacked sequences are unchanged (one segment).

    Args:
        y:                current sequence (with BOS, EOS), shape (L,)
        y_star:           target sequence (with BOS, EOS), shape (M,)
        max_placeholder:  K_max — cap on placeholders per slot
        plh_token_id:     id of <PLH> token

    Returns:
        p_star: (L-1,) long tensor — number of placeholders for each gap
        t_star: (total_plh,) long tensor — flattened token ids, in gap order
    """
    segs_y = _segment_sequences(y, bos_idx, eos_idx)
    segs_s = _segment_sequences(y_star, bos_idx, eos_idx)
    if len(segs_y) > 1 and len(segs_y) == len(segs_s):
        p_parts: List[torch.Tensor] = []
        t_parts: List[torch.Tensor] = []
        for sy, ss in zip(segs_y, segs_s):
            p, t = _insertion_oracle(sy, ss, max_placeholder, plh_token_id)
            p_parts.append(p)
            t_parts.append(t)
        # Interleave a zero gap between segments (no cross-boundary insert).
        full_parts: List[torch.Tensor] = []
        for index, p in enumerate(p_parts):
            full_parts.append(p)
            if index < len(p_parts) - 1:
                full_parts.append(torch.zeros(1, dtype=p.dtype, device=p.device))
        return torch.cat(full_parts), torch.cat(t_parts)
    return _insertion_oracle(y, y_star, max_placeholder, plh_token_id)


def _slice_packed(packed: torch.Tensor, offsets: torch.Tensor) -> List[torch.Tensor]:
    """Split a packed flat tensor into per-sample tensors using [B+1] offsets."""
    offsets_list = offsets.tolist()
    return [
        packed[offsets_list[i]:offsets_list[i + 1]]
        for i in range(len(offsets_list) - 1)
    ]


def oracle_deletion_batch(
    ys: List[torch.Tensor],
    ys_stars: List[torch.Tensor],
    bos_idx: int = 1,
    eos_idx: int = 2,
) -> List[torch.Tensor]:
    """
    Batched oracle deletion policy (one C++ call for the whole batch).

    Packed sequences are split into segments first so the DP never crosses an
    ``[EOS][BOS]`` boundary; segment masks are re-concatenated per sample.
    Returns one bool mask (True = DELETE) per input pair, each of length
    ``len(y)`` with boundaries forced ``False`` — identical to
    ``[oracle_deletion(y, ys_) for y, ys_ in zip(ys, ys_stars)]``.

    Falls back to per-sample calls when the C++ extension is unavailable.
    """
    flat_ys: List[torch.Tensor] = []
    flat_stars: List[torch.Tensor] = []
    spans: List[Optional[Tuple[int, int]]] = []
    for y, ys_ in zip(ys, ys_stars):
        segs_y = _segment_sequences(y, bos_idx, eos_idx)
        segs_s = _segment_sequences(ys_, bos_idx, eos_idx)
        if len(segs_y) > 1 and len(segs_y) == len(segs_s):
            start = len(flat_ys)
            flat_ys.extend(segs_y)
            flat_stars.extend(segs_s)
            spans.append((start, len(flat_ys)))
        else:
            spans.append(None)  # single-segment or mismatched boundaries
            flat_ys.append(y)
            flat_stars.append(ys_)
    if levenshtein_deletion_cpp_batch is not None:
        result = levenshtein_deletion_cpp_batch(flat_ys, flat_stars)
        if result is not None:
            del_packed, offsets = result
            masks = [mask.bool() for mask in _slice_packed(del_packed, offsets)]
            out: List[torch.Tensor] = []
            k = 0
            for span in spans:
                if span is None:
                    out.append(masks[k])
                    k += 1
                else:
                    _, end = span
                    out.append(torch.cat(masks[k:end]))
                    k = end
            return out
    return [
        oracle_deletion(y, ys_, bos_idx=bos_idx, eos_idx=eos_idx)
        for y, ys_ in zip(ys, ys_stars)
    ]


def oracle_insertion_batch(
    ys: List[torch.Tensor],
    ys_stars: List[torch.Tensor],
    max_placeholder: int = 255,
    plh_token_id: int = 3,
    bos_idx: int = 1,
    eos_idx: int = 2,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """
    Batched oracle insertion policy (one C++ call for the whole batch).

    Packed sequences are split into segments first so the edit script never
    crosses an ``[EOS][BOS]`` boundary; a zero-insertion gap is forced at each
    segment boundary.  Returns ``(p_stars, t_stars, y_ins_plhs)`` — parallel
    per-sample lists matching ``[oracle_insertion(...) ...]`` plus the
    PLH-interleaved roll-in ``insert_placeholders(y, p_star)``. The three
    lists are aligned with ``ys`` (sample ``i`` of each list belongs to pair
    ``i``).

    Falls back to per-sample calls when the C++ extension is unavailable.
    """
    flat_ys: List[torch.Tensor] = []
    flat_stars: List[torch.Tensor] = []
    spans: List[Optional[Tuple[int, int]]] = []
    for y, ys_ in zip(ys, ys_stars):
        segs_y = _segment_sequences(y, bos_idx, eos_idx)
        segs_s = _segment_sequences(ys_, bos_idx, eos_idx)
        if len(segs_y) > 1 and len(segs_y) == len(segs_s):
            start = len(flat_ys)
            flat_ys.extend(segs_y)
            flat_stars.extend(segs_s)
            spans.append((start, len(flat_ys)))
        else:
            spans.append(None)  # single-segment or mismatched boundaries
            flat_ys.append(y)
            flat_stars.append(ys_)
    if levenshtein_insertion_cpp_batch is not None:
        result = levenshtein_insertion_cpp_batch(
            flat_ys, flat_stars, max_placeholder, plh_token_id,
        )
        if result is not None:
            (p_packed, p_offsets, t_packed, t_offsets,
             plh_packed, plh_offsets) = result
            p_segs = _slice_packed(p_packed, p_offsets)
            t_segs = _slice_packed(t_packed, t_offsets)
            plh_segs = _slice_packed(plh_packed, plh_offsets)
            p_stars: List[torch.Tensor] = []
            t_stars: List[torch.Tensor] = []
            y_ins_plhs: List[torch.Tensor] = []
            k = 0
            for span in spans:
                if span is None:
                    p_stars.append(p_segs[k])
                    t_stars.append(t_segs[k])
                    y_ins_plhs.append(plh_segs[k])
                    k += 1
                else:
                    _, end = span
                    parts: List[torch.Tensor] = []
                    for index, p in enumerate(p_segs[k:end]):
                        parts.append(p)
                        if index < (end - k) - 1:
                            parts.append(torch.zeros(
                                1, dtype=p.dtype, device=p.device,
                            ))
                    p_stars.append(torch.cat(parts))
                    t_stars.append(torch.cat(t_segs[k:end]))
                    y_ins_plhs.append(torch.cat(plh_segs[k:end]))
                    k = end
            return p_stars, t_stars, y_ins_plhs
    p_stars = []
    t_stars = []
    y_ins_plhs = []
    for y, ys_ in zip(ys, ys_stars):
        p_star, t_star = oracle_insertion(
            y, ys_,
            max_placeholder=max_placeholder, plh_token_id=plh_token_id,
            bos_idx=bos_idx, eos_idx=eos_idx,
        )
        p_stars.append(p_star)
        t_stars.append(t_star)
        y_ins_plhs.append(insert_placeholders(y, p_star, plh_token_id=plh_token_id))
    return p_stars, t_stars, y_ins_plhs


# ---------------------------------------------------------------------------
# Apply oracle actions to sequences (environment E)
# ---------------------------------------------------------------------------

def apply_deletion(y: torch.Tensor, deletion_mask: torch.Tensor) -> torch.Tensor:
    """Remove tokens marked for deletion. Returns new sequence."""
    keep_mask = ~deletion_mask
    return y[keep_mask]


def insert_placeholders(
    y: torch.Tensor,
    p_counts: torch.Tensor,
    plh_token_id: int = 3,
) -> torch.Tensor:
    """
    Insert <PLH> placeholders into y according to p_counts.

    Args:
        y:        sequence, shape (L,)
        p_counts: placeholder counts per gap, shape (L-1,)

    Returns:
        new sequence with <PLH> tokens inserted, shape (L + sum(p_counts),)
    """
    y_list = y.tolist()
    result: List[int] = []
    for i in range(len(y_list)):
        result.append(y_list[i])
        if i < len(p_counts):
            for _ in range(int(p_counts[i].item())):
                result.append(plh_token_id)
    return torch.tensor(result, dtype=y.dtype, device=y.device)


def fill_placeholders(
    y_with_plh: torch.Tensor,
    tokens: torch.Tensor,
    plh_token_id: int = 3,
) -> torch.Tensor:
    """
    Replace <PLH> tokens in y_with_plh with actual tokens (in order).

    Args:
        y_with_plh: sequence with <PLH> tokens
        tokens:     token ids, one per <PLH> (in left-to-right order)

    Returns:
        sequence with <PLH> replaced by tokens
    """
    result = y_with_plh.clone()
    token_iter = iter(tokens.tolist())
    for i in range(len(result)):
        if result[i].item() == plh_token_id:
            try:
                result[i] = next(token_iter)
            except StopIteration:
                break
    return result


# ---------------------------------------------------------------------------
# Random deletion policy π^rnd (noise injection for insertion training)
# ---------------------------------------------------------------------------

def random_deletion(
    y_star: torch.Tensor,
    drop_prob: float = 0.3,
    bos_idx: int = 1,
    eos_idx: int = 2,
    pad_idx: int = 0,
) -> torch.Tensor:
    """
    Randomly delete tokens from y* (excluding boundaries).
    Implements π^rnd from the paper: sample k ~ Uniform[0, |non-boundary|]
    and randomly delete k tokens.

    Args:
        y_star:    target sequence, shape (L,)
        drop_prob: probability of dropping each non-boundary token

    Returns:
        sequence with randomly deleted tokens removed
    """
    y_list = y_star.tolist()
    # Keep BOS, EOS, and drop others with probability drop_prob.  Every BOS/EOS
    # is kept (not just the first/last) so packed rows never lose an interior
    # segment boundary.
    keep: List[int] = []
    for idx, tok in enumerate(y_list):
        if tok == bos_idx or tok == eos_idx:
            keep.append(tok)
        elif tok == pad_idx:
            continue
        else:
            if random.random() > drop_prob:
                keep.append(tok)
    return torch.tensor(keep, dtype=y_star.dtype, device=y_star.device)
