#!/usr/bin/env python3
"""Benchmark / profile harness for the C++ Levenshtein alignment op.

Measures the *real* per-call cost of ``levenshtein_align`` as used in
training: pybind boundary + ``y.contiguous().to(int64).cpu()`` conversion
overhead + tensor allocation + the DP itself.

Two modes:
  --tiny        fixed n=m=4 sequences  -> isolates the per-call fixed overhead
  --real        lengths sampled from the training distribution (default)

Run under perf for hardware counters:

  echo kris | sudo -S perf record -F 2000 -g \
      python scripts/bench_levt_ops.py --real --iters 200000
  echo kris | sudo -S perf report --stdio
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time

import torch


def _load_cpp() -> object:
    from levt._levenshtein_ops import _get_module, verify_cpp_extension
    st = verify_cpp_extension()
    if not st.available:
        sys.exit(f"CPP extension unavailable: {st.error}")
    return _get_module()


def _real_pairs(n_pairs: int, rng: random.Random):
    """Pairs with lengths sampled like real training data (median ~36, up to 256)."""
    pairs = []
    for _ in range(n_pairs):
        # lognormal-ish; median ~36, mean ~70, tail to a few hundred
        ln = max(4, int(rng.lognormvariate(3.6, 0.55)))
        m = max(4, ln + rng.randint(-10, 10))
        ln = min(ln, 256)
        m = min(max(m, 4), 256)
        y = torch.randint(1, 30000, (ln,), dtype=torch.long)
        y_star = torch.randint(1, 30000, (m,), dtype=torch.long)
        pairs.append((y, y_star))
    return pairs


def _tiny_pairs(n_pairs: int):
    return [(torch.tensor([1, 2, 3, 4], dtype=torch.long),
             torch.tensor([1, 3, 4, 5, 2], dtype=torch.long))] * n_pairs


def _slice_oracle(mod, y, ys, max_placeholder=255, plh_token_id=3):
    """Single-pair packed insertion oracle -> (p_star, t_star, y_ins_plh)."""
    p_star, _, t_star, _, plh, _ = mod.levenshtein_insertion_batch(
        [y], [ys], max_placeholder, plh_token_id)
    return p_star, t_star, plh


def _run_oracle_batch(mod, pairs, chunk, max_placeholder=255, plh_token_id=3):
    """One call of the packed deletion+insertion oracle per chunk."""
    acc = 0
    n = len(pairs)
    for i in range(0, n, chunk):
        ys = [p[0] for p in pairs[i:i + chunk]]
        ys_stars = [p[1] for p in pairs[i:i + chunk]]
        del_mask, _ = mod.levenshtein_deletion_batch(ys, ys_stars)
        acc += int(del_mask.sum())
        p_star, _, t_star, _, _, _ = mod.levenshtein_insertion_batch(
            ys, ys_stars, max_placeholder, plh_token_id)
        acc += int(p_star.sum()) + int(t_star.sum())
    return acc


def _run_oracle_single(pairs, max_placeholder=255, plh_token_id=3):
    """Per-sample Python oracle path (deletion + insertion + PLH roll-in)."""
    from levt.expert import (
        insert_placeholders,
        oracle_deletion,
        oracle_insertion,
    )
    acc = 0
    for y, y_star in pairs:
        acc += int(oracle_deletion(y, y_star).sum())
        p_star, t_star = oracle_insertion(
            y, y_star, max_placeholder=max_placeholder, plh_token_id=plh_token_id)
        insert_placeholders(y, p_star, plh_token_id=plh_token_id)
        acc += int(p_star.sum()) + int(t_star.numel())
    return acc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=100000)
    ap.add_argument("--mode", choices=["real", "tiny"], default="real")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--batch", type=int, default=1,
                    help="packed oracle batch size; >1 benchmarks the batched "
                         "deletion+insertion oracle path (default 1 = single "
                         "levenshtein_align)")
    ap.add_argument("--json", type=str, default=None,
                    help="write results as JSON to this path")
    ap.add_argument("--no-self-check", action="store_true",
                    help="skip correctness spot-check (perf runs)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    mod = _load_cpp()
    fn = mod.levenshtein_align

    pairs = (_tiny_pairs(args.iters) if args.mode == "tiny"
             else _real_pairs(args.iters, rng))

    # correctness spot-check vs pure-Python oracle on a few pairs
    if not args.no_self_check:
        from levt.expert import (
            _levenshtein_align_py,
            insert_placeholders,
            oracle_deletion,
            oracle_insertion,
        )
        for y, ys in _tiny_pairs(3) + _real_pairs(2, rng):
            del_cpp, ins_cpp = fn(y, ys)
            del_py, ins_py = _levenshtein_align_py(y, ys)
            assert list(del_cpp.tolist()) == del_py, (y, ys)
            assert [t.tolist() for t in ins_cpp] == ins_py, (y, ys)
            # packed oracles must equal the per-sample single oracles
            dm, do = mod.levenshtein_deletion_batch([y], [ys])
            assert torch.equal(dm.bool(), oracle_deletion(y, ys)), (y, ys)
            p_star, t_star, plh = _slice_oracle(mod, y, ys)
            py_ps, py_ts = oracle_insertion(y, ys, max_placeholder=255, plh_token_id=3)
            assert torch.equal(p_star, py_ps) and torch.equal(t_star, py_ts), (y, ys)
            assert torch.equal(plh, insert_placeholders(y, py_ps, plh_token_id=3)), (y, ys)
        print("[ok] C++ matches pure-Python oracle (single + packed)")

    n = args.iters

    if args.batch > 1:
        # ── Batched packed oracle path vs per-sample Python oracle path ──
        chunk = args.batch
        # warmup
        for _ in range(200):
            _run_oracle_batch(mod, pairs[:chunk], chunk)

        t0 = time.perf_counter()
        acc = _run_oracle_batch(mod, pairs, chunk)
        t_batch = time.perf_counter() - t0

        t0 = time.perf_counter()
        acc_s = _run_oracle_single(pairs)
        t_single = time.perf_counter() - t0

        per_pair_batch = t_batch / n * 1e6
        per_pair_single = t_single / n * 1e6
        n_cells = sum((y.numel() + 1) * (ys.numel() + 1) for y, ys in pairs)
        print(f"mode={args.mode} iters={n} batch={chunk}")
        print(f"  per-sample oracle path : {per_pair_single:9.2f} us/pair")
        print(f"  packed batch path      : {per_pair_batch:9.2f} us/pair")
        print(f"  speedup                : {t_single / t_batch:6.1f}x")
        print(f"  avg (n+1)*(m+1)        : {n_cells / n:,.0f} DP cells/pair")
        print(f"  acc check              : {acc} vs {acc_s}")

        res = {
            "mode": args.mode, "iters": n, "batch": chunk,
            "per_sample_us": per_pair_single, "per_pair_batch_us": per_pair_batch,
            "speedup": t_single / t_batch,
            "avg_dp_cells": n_cells / n,
            "acc": acc, "acc_single": acc_s,
        }
    else:
        # ── Single-pair levenshtein_align (original benchmark) ──────────
        # warmup
        for _ in range(2000):
            fn(*pairs[rng.randrange(len(pairs))])

        t0 = time.perf_counter()
        acc_del = 0
        for i in range(n):
            y, ys = pairs[i]
            delt, ins = fn(y, ys)
            acc_del += int(delt.numel())
        elapsed = time.perf_counter() - t0

        per_call_us = elapsed / n * 1e6
        n_cells = sum((y.numel() + 1) * (ys.numel() + 1) for y, ys in pairs)
        print(f"mode={args.mode} iters={n}")
        print(f"  total wall       : {elapsed:.3f} s")
        print(f"  avg per-call     : {per_call_us:,.2f} us")
        print(f"  calls/sec        : {n / elapsed:,.0f}")
        print(f"  avg (n+1)*(m+1)  : {n_cells / n:,.0f} DP cells/call")
        print(f"  DP cells/sec     : {n_cells / elapsed:,.0f}")

        res = {
            "mode": args.mode, "iters": n,
            "total_wall_s": elapsed, "per_call_us": per_call_us,
            "calls_per_sec": n / elapsed,
            "avg_dp_cells": n_cells / n, "dp_cells_per_sec": n_cells / elapsed,
            "acc_del_numel": acc_del,
        }
    if args.json:
        with open(args.json, "w") as f:
            json.dump(res, f, indent=2)
        print(f"[json] wrote {args.json}")


if __name__ == "__main__":
    main()
