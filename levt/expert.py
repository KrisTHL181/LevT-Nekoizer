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
from typing import List, Tuple

import torch

# Try to load the C++ acceleration module at import time.
_cpp_module = None
try:
    from ._levenshtein_ops import levenshtein_align_cpp
    _cpp_module = True  # module loaded; actual function may still return None
except ImportError:
    levenshtein_align_cpp = None  # type: ignore[assignment]


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

def oracle_deletion(
    y: torch.Tensor,
    y_star: torch.Tensor,
    bos_idx: int = 1,
    eos_idx: int = 2,
) -> torch.Tensor:
    """
    Oracle deletion policy: compute optimal tokens to delete from y.

    Args:
        y:      current sequence (with BOS, EOS), shape (L,)
        y_star: target sequence (with BOS, EOS), shape (M,)

    Returns:
        mask: boolean tensor of shape (L,), True = DELETE this token.
              Boundaries (BOS/EOS) are always False (never deleted).
    """
    deletions, _ = levenshtein_align(y, y_star)
    mask = torch.zeros(len(y), dtype=torch.bool)
    if deletions.numel() > 0:
        mask[deletions] = True  # fancy indexing — O(num_deletions)
    # Never delete boundaries
    mask[0] = False
    mask[-1] = False
    return mask


def oracle_insertion(
    y: torch.Tensor,
    y_star: torch.Tensor,
    max_placeholder: int = 255,
    plh_token_id: int = 3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Oracle insertion policy: compute optimal placeholders and tokens.

    Args:
        y:                current sequence (with BOS, EOS), shape (L,)
        y_star:           target sequence (with BOS, EOS), shape (M,)
        max_placeholder:  K_max — cap on placeholders per slot
        plh_token_id:     id of <PLH> token

    Returns:
        p_star: (L-1,) long tensor — number of placeholders for each gap
        t_star: (total_plh,) long tensor — flattened token ids, in gap order
    """
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
    # Keep BOS, EOS, and drop others with probability drop_prob
    keep: List[int] = []
    for idx, tok in enumerate(y_list):
        if idx == 0:  # BOS
            keep.append(tok)
        elif idx == len(y_list) - 1:  # EOS
            keep.append(tok)
        elif tok == pad_idx:
            continue
        else:
            if random.random() > drop_prob:
                keep.append(tok)
    return torch.tensor(keep, dtype=y_star.dtype, device=y_star.device)
