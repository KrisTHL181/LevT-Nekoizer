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


# ---------------------------------------------------------------------------
# Levenshtein DP alignment (insert + delete only, no substitution)
# ---------------------------------------------------------------------------

def levenshtein_align(
    y: torch.Tensor,
    y_star: torch.Tensor,
    pad_idx: int = 0,
) -> Tuple[List[int], List[List[int]]]:
    """
    Compute optimal edit alignment between two sequences using DP.

    Operations: DELETE (cost 1), INSERT (cost 1), MATCH (cost 0 if tokens equal).
    No substitution — changing a token requires DELETE + INSERT (cost 2).

    Args:
        y:      current sequence, shape (L,) — includes BOS/EOS
        y_star: target sequence, shape (M,) — includes BOS/EOS
        pad_idx: padding token id (ignored)

    Returns:
        deletions:  list of indices in y to delete (0-indexed, excluding boundaries)
        insertions: list-of-lists, insertions[i] = tokens to insert between
                    y_pos[i] and y_pos[i+1] in the *surviving* sequence
                    (i.e., after deletions are removed)

    NOTE: Boundary tokens (first and last) are never deleted.
    """
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
    # matched_pairs: list of (idx_in_y, idx_in_y_star) for matched positions
    matched_pairs: List[Tuple[int, int]] = []

    i, j = n, m
    inserted_segments: List[Tuple[int, int, int]] = []  # (insert_before_y_idx, start_j, end_j)

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
            # We'll reconstruct insertion groups after backtrack

    # Reverse to get forward order
    deletions.reverse()
    matched_pairs.reverse()

    # Reconstruct insertion groups from the DP alignment
    # Re-backtrack to collect insertions between matched positions
    insertions_raw: List[int] = []  # tokens from y* that are inserted, in order
    insertion_after: List[int] = []  # y-index (in matched positions) after which they appear

    i, j = n, m
    current_after = n  # tracks where in y we are (before any match)
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

    # Build per-slot insertion lists
    # The slots are gaps in the ORIGINAL y (before deletion), indexed by
    # the position BEFORE the gap (0..n-1 for pairs 0|1, 1|2, ..., n-2|n-1)
    # But tokens at deleted positions are removed, so we need to map to
    # gaps between SURVIVING positions.

    # Strategy: for each gap i in the original y (between pos i and i+1),
    # collect inserted tokens. If pos i+1 is deleted, the gap spans further.

    # Simplified: group inserted tokens by the gap AFTER the preceding
    # non-deleted position.
    del_set = set(deletions)
    surviving = [idx for idx in range(n) if idx not in del_set]

    # Build insertion lists: insertions_by_gap[i] for gap after surviving[i]
    per_gap: List[List[int]] = [[] for _ in range(max(0, len(surviving) - 1))]

    if len(surviving) >= 2:
        # Map from y-index to surviving-index
        surv_rank = {sid: si for si, sid in enumerate(surviving)}

        for tok, after_y in zip(insertions_raw, insertion_after):
            # Backward-pass semantics: after_y is the *next* matched y-index
            # in the forward direction. The inserted token goes in the gap
            # BEFORE position after_y, i.e. between the previous surviving
            # token and after_y.  So we find the surviving index where
            # surviving[si] == after_y, and assign the token to gap si-1.
            gap_surv_idx = -1
            if after_y in surv_rank:
                gap_surv_idx = surv_rank[after_y] - 1  # gap before this position
            else:
                # after_y points to a deleted position — find the gap after
                # the last surviving token before after_y.
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
    for d in deletions:
        mask[d] = True
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

    # Build surviving sequence (tokens NOT deleted)
    del_set = set(deletions)
    surviving = [idx for idx in range(len(y)) if idx not in del_set]

    num_gaps = max(0, len(surviving) - 1)

    # placeholder counts for EACH gap in the ORIGINAL y (len(y)-1 gaps)
    # But we need them for the surviving sequence gaps.
    # The paper uses y_ins which already has deletions applied (from oracle
    # or random), so the gaps are between adjacent surviving tokens.
    p_star = torch.zeros(len(y) - 1, dtype=torch.long)
    t_star_parts: List[torch.Tensor] = []

    # Map gap index in surviving sequence to gap index in original y
    for gi in range(num_gaps):
        left_orig = surviving[gi]
        right_orig = surviving[gi + 1]
        # The gap in original y spans from left_orig to right_orig-1
        # We assign the insertion count to the gap at position left_orig
        tokens_to_insert = insertions[gi] if gi < len(insertions) else []
        count = min(len(tokens_to_insert), max_placeholder)
        p_star[left_orig] = count
        if count > 0:
            t_star_parts.append(torch.tensor(tokens_to_insert[:count], dtype=torch.long))

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
