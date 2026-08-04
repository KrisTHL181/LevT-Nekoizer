#!/usr/bin/env python3
"""Pack (src, target, initial) examples into capacity-bounded rows via bin packing.

Examples are grouped so that a packed row's concatenated ``src`` stays within
``--src-capacity`` tokens and its concatenated ``target`` within
``--target-capacity``.  Both sequences are packed *together* (a 2-D bin): the
same examples always land in the same row, so the segment correspondence
src_i -> target_i is preserved.  Because every example's src/target starts with
BOS and ends with EOS, concatenation produces a natural ``[EOS][BOS]`` boundary
between segments.

Packing sorts examples by ``max(src_len, target_len)`` (then total length)
descending and uses Best-Fit-Decreasing (default) or First-Fit-Decreasing over
the open bins.  A bin is a valid fit when *both* remaining capacities
accommodate the example; BFD picks the fitting bin that leaves the smallest
``max(remaining_src, remaining_tgt)`` residual (the packed row's effective cost
is the longer of its two sequences).

Output rows (one per bin):
    {"src": [...], "target": [...], "initial": [...], "n_segments": k}

Packed rows contain interior BOS/EOS boundary tokens.  The output file begins
with a ``{"__meta__": {"format": "levt-jsonl", "version": 1, "packed": true}}``
header line so training auto-detects the packed format.  ``n_segments`` is
metadata for stats; the training pipeline ignores it.

When an input row omits ``initial`` it defaults to that row's ``src``
(edit-task semantics), so the packed ``initial`` is the concatenation of the
source segments.

The packing core is vectorised with numpy because a naive Python first-fit
scan is O(examples x bins) — intractable for ~10^6 examples.

Usage:
    python scripts/pack_dataset.py --input zhihu_train.jsonl \
        --output zhihu_train_packed.jsonl \
        --src-capacity 1024 --tgt-capacity 1024 --algorithm bfd
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np

# Make the repo root importable when run as ``python scripts/pack_dataset.py``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from levt.dataset_meta import META_KEY, dataset_header


def _load_sizes(
    path: Path, src_cap: int, tgt_cap: int, bos: int, eos: int, limit: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Pass 1: stream the file, return per-row (src_len, tgt_len) arrays.

    Also verifies the BOS-start / EOS-end invariant that concatenation relies
    on for clean ``[EOS][BOS]`` segment boundaries.  ``limit`` caps how many
    rows are read (0 = all) for quick experiments on a slice.
    """
    src_len: list[int] = []
    tgt_len: list[int] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_no}: blank line")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc.msg}") from exc
            if line_no == 1 and isinstance(record, dict) and META_KEY in record:
                continue  # skip the dataset metadata header
            if not isinstance(record, dict) or "src" not in record or "target" not in record:
                raise ValueError(f"{path}:{line_no}: missing src/target")
            src = record["src"]
            target = record["target"]
            if not src or not target:
                raise ValueError(f"{path}:{line_no}: empty src/target")
            if len(src) > src_cap or len(target) > tgt_cap:
                raise ValueError(
                    f"{path}:{line_no}: src/target length exceeds capacity "
                    f"({len(src)}/{len(target)} > {src_cap}/{tgt_cap})"
                )
            if src[0] != bos or src[-1] != eos:
                raise ValueError(f"{path}:{line_no}: src must start with BOS and end with EOS")
            if target[0] != bos or target[-1] != eos:
                raise ValueError(f"{path}:{line_no}: target must start with BOS and end with EOS")
            src_len.append(len(src))
            tgt_len.append(len(target))
            if limit and len(src_len) >= limit:
                break
    if not src_len:
        raise ValueError(f"{path}: dataset is empty")
    return np.asarray(src_len, dtype=np.int64), np.asarray(tgt_len, dtype=np.int64)


def _pack(
    src_len: np.ndarray,
    tgt_len: np.ndarray,
    src_cap: int,
    tgt_cap: int,
    algorithm: str,
) -> tuple[np.ndarray, int, np.ndarray, np.ndarray]:
    """Return (bin_of_item, n_bins, bin_src_used, bin_tgt_used)."""
    n = len(src_len)
    # Descending order: primary = total length, secondary = max length.
    order = np.lexsort((-np.maximum(src_len, tgt_len), -(src_len + tgt_len)))

    rem_src = np.full(n, src_cap, dtype=np.int64)   # upper bound on bins
    rem_tgt = np.full(n, tgt_cap, dtype=np.int64)
    bin_of_item = np.full(n, -1, dtype=np.int32)
    n_bins = 0
    worst = np.iinfo(np.int64).max

    for item in order:
        s = int(src_len[item])
        t = int(tgt_len[item])
        # Candidate bins (both dims must fit), as a view over active bins.
        fit = (rem_src[:n_bins] >= s) & (rem_tgt[:n_bins] >= t)
        if fit.any():
            if algorithm == "bfd":
                resid = np.maximum(rem_src[:n_bins] - s, rem_tgt[:n_bins] - t)
                resid = np.where(fit, resid, worst)
                j = int(np.argmin(resid))
            else:
                j = int(np.argmax(fit))
            rem_src[j] -= s
            rem_tgt[j] -= t
            bin_of_item[item] = j
        else:
            j = n_bins
            n_bins += 1
            rem_src[j] = src_cap - s
            rem_tgt[j] = tgt_cap - t
            bin_of_item[item] = j

    return bin_of_item, n_bins, src_cap - rem_src[:n_bins], tgt_cap - rem_tgt[:n_bins]


def _rebuild_and_write(
    path: Path,
    output: Path,
    bin_of_item: np.ndarray,
    n_bins: int,
    bos: int,
    eos: int,
    tgt_cap: int,
    limit: int = 0,
) -> tuple[list[int], int, int]:
    """Pass 2: stream the file again, assemble per-bin rows, write output."""
    # bins_data[b] = list of (src, target, initial) tuples
    bins_data: list[list[tuple[list[int], list[int], list[int]]]] = [
        [] for _ in range(n_bins)
    ]
    total_tokens = 0
    data_index = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if line_no == 1 and META_KEY in record:
                continue  # skip the dataset metadata header
            src: list[int] = record["src"]
            target: list[int] = record["target"]
            initial: list[int] = record.get("initial", src)
            if initial[0] != bos or initial[-1] != eos:
                raise ValueError(
                    f"{path}:{line_no}: initial must start with BOS and end with EOS"
                )
            if len(initial) > tgt_cap:
                raise ValueError(f"{path}:{line_no}: initial exceeds target capacity")
            bins_data[int(bin_of_item[data_index])].append((src, target, initial))
            total_tokens += len(src) + len(target)
            data_index += 1
            if limit and data_index >= limit:
                break

    n_segments: list[int] = []
    written = 0
    with output.open("w", encoding="utf-8") as out:
        out.write(json.dumps(dataset_header(True), ensure_ascii=False) + "\n")
        for b in range(n_bins):
            items = bins_data[b]
            n_segments.append(len(items))
            src_packed: list[int] = []
            tgt_packed: list[int] = []
            initial_packed: list[int] = []
            for src, target, initial in items:
                src_packed.extend(src)
                tgt_packed.extend(target)
                initial_packed.extend(initial)
            if len(initial_packed) > tgt_cap:
                raise ValueError(f"bin {b}: packed initial length {len(initial_packed)} exceeds {tgt_cap}")
            row = {
                "src": src_packed,
                "target": tgt_packed,
                "initial": initial_packed,
                "n_segments": len(items),
            }
            out.write(json.dumps(row, ensure_ascii=False))
            out.write("\n")
            written += 1
    return n_segments, written, total_tokens


def _pack_stats(
    src_len: np.ndarray,
    tgt_len: np.ndarray,
    bin_of_item: np.ndarray,
    n_bins: int,
    src_cap: int,
    tgt_cap: int,
    n_segments: list[int],
    total_tokens: int,
) -> None:
    per_bin_src = np.zeros(n_bins, dtype=np.int64)
    per_bin_tgt = np.zeros(n_bins, dtype=np.int64)
    np.add.at(per_bin_src, bin_of_item, src_len)
    np.add.at(per_bin_tgt, bin_of_item, tgt_len)

    src_fill = per_bin_src / src_cap
    tgt_fill = per_bin_tgt / tgt_cap
    eff_fill = np.maximum(src_fill, tgt_fill)

    print(f"input rows        : {len(src_len)}")
    print(f"packed rows       : {n_bins}")
    print(f"avg segments/row  : {len(src_len) / n_bins:.2f}")
    print(f"segments/row hist : min={min(n_segments)} med={statistics.median(n_segments):.0f} "
          f"p90={sorted(n_segments)[int(0.9 * len(n_segments))]} max={max(n_segments)}")
    print(f"singleton rows    : {sum(1 for s in n_segments if s == 1)}")
    print(f"src fill ratio    : mean={src_fill.mean():.3f} median={np.median(src_fill):.3f}")
    print(f"tgt fill ratio    : mean={tgt_fill.mean():.3f} median={np.median(tgt_fill):.3f}")
    print(f"effective fill    : mean={eff_fill.mean():.3f} (max dim; higher = less padding waste)")
    print(f"total tokens      : {total_tokens} (src+tgt)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="source JSONL with src/target/initial")
    parser.add_argument("--output", required=True, help="packed JSONL output")
    parser.add_argument("--src-capacity", type=int, default=1024)
    parser.add_argument("--tgt-capacity", type=int, default=1024)
    parser.add_argument("--algorithm", choices=["bfd", "ffd"], default="bfd")
    parser.add_argument("--bos-id", type=int, default=1)
    parser.add_argument("--eos-id", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0, help="only pack the first N rows (0 = all)")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    print(f"pass 1: scanning lengths of {input_path} ...", flush=True)
    src_len, tgt_len = _load_sizes(
        input_path, args.src_capacity, args.tgt_capacity, args.bos_id, args.eos_id,
        limit=args.limit,
    )
    print(f"  {len(src_len)} rows", flush=True)

    print(f"packing ({args.algorithm}) ...", flush=True)
    bin_of_item, n_bins, _, _ = _pack(
        src_len, tgt_len, args.src_capacity, args.tgt_capacity, args.algorithm,
    )
    print(f"  {n_bins} bins", flush=True)

    print(f"pass 2: rebuilding rows -> {output_path} ...", flush=True)
    n_segments, written, total_tokens = _rebuild_and_write(
        input_path, output_path, bin_of_item, n_bins, args.bos_id, args.eos_id,
        args.tgt_capacity, limit=args.limit,
    )
    assert written == n_bins, f"wrote {written} rows but packed {n_bins} bins"

    _pack_stats(
        src_len, tgt_len, bin_of_item, n_bins,
        args.src_capacity, args.tgt_capacity, n_segments, total_tokens,
    )


if __name__ == "__main__":
    main()
