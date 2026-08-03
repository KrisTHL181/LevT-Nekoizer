#!/usr/bin/env python3
"""
Pre-compute insertion oracles for Levenshtein Transformer training.

Reads already-tokenized JSONL with ``src``, ``target``, and optional ``initial``
and pre-computes both the oracle-deletion and random-deletion insertion paths.
Outputs the original fields plus 8 new pre-computed fields.

An optional ``{"__meta__": {...}}`` header on the input is skipped and
forwarded to the output, so a packed dataset keeps its packed marker.

Usage::

    python scripts/precompute_oracles.py config.json policy_config.json input.jsonl output.jsonl
    python scripts/precompute_oracles.py config.json policy_config.json input.jsonl output.jsonl --dry-run
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

# Make the repo root importable when run as ``python scripts/precompute_oracles.py``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from levt.dataset_meta import META_KEY, dataset_header

import torch

from levt.config import LevTConfig
from levt.expert import (
    apply_deletion,
    insert_placeholders,
    oracle_deletion,
    oracle_insertion,
    random_deletion,
)


def load_policy_config(path: str) -> Dict[str, Any]:
    """Load the policy config JSON (beta, random_delete_prob)."""
    path_obj = Path(path)
    if not path_obj.exists():
        print(f"Error: policy config not found: {path}", file=sys.stderr)
        sys.exit(1)
    with open(path_obj, "r", encoding="utf-8") as f:
        config = json.load(f)
    if not isinstance(config, dict):
        print("Error: policy config must be a JSON object", file=sys.stderr)
        sys.exit(1)
    config.setdefault("beta", 0.5)
    config.setdefault("random_delete_prob", 0.3)
    return config


def process_row(
    row: Dict[str, Any],
    config: LevTConfig,
    drop_prob: float,
    line_number: int,
) -> Dict[str, Any]:
    """Compute oracle and random insertion paths for one row."""
    initial_list = row.get("initial", [config.bos_token_id, config.eos_token_id])
    target_list = row["target"]

    initial = torch.tensor(initial_list, dtype=torch.long)
    target = torch.tensor(target_list, dtype=torch.long)

    # --- Oracle path ---
    deletion = oracle_deletion(
        initial, target,
        bos_idx=config.bos_token_id,
        eos_idx=config.eos_token_id,
    )
    y_ins = apply_deletion(initial, deletion)
    p_star, t_star = oracle_insertion(
        y_ins, target,
        max_placeholder=config.max_placeholder,
        plh_token_id=config.plh_token_id,
    )
    y_ins_plh = insert_placeholders(y_ins, p_star, config.plh_token_id)

    # --- Random path ---
    y_ins_rnd = random_deletion(
        target,
        drop_prob=drop_prob,
        bos_idx=config.bos_token_id,
        eos_idx=config.eos_token_id,
        pad_idx=config.pad_token_id,
    )
    p_star_rnd, t_star_rnd = oracle_insertion(
        y_ins_rnd, target,
        max_placeholder=config.max_placeholder,
        plh_token_id=config.plh_token_id,
    )
    y_ins_plh_rnd = insert_placeholders(y_ins_rnd, p_star_rnd, config.plh_token_id)

    # Build output row (preserves all original fields)
    out = dict(row)
    out["y_ins"] = y_ins.tolist()
    out["p_star"] = p_star.tolist()
    out["t_star"] = t_star.tolist()
    out["y_ins_plh"] = y_ins_plh.tolist()
    out["y_ins_rnd"] = y_ins_rnd.tolist()
    out["p_star_rnd"] = p_star_rnd.tolist()
    out["t_star_rnd"] = t_star_rnd.tolist()
    out["y_ins_plh_rnd"] = y_ins_plh_rnd.tolist()
    return out


def show_stats(rows: List[Dict[str, Any]], config: LevTConfig) -> None:
    """Print statistics for the first 3 rows."""
    for i, row in enumerate(rows[:3]):
        init_len = len(row.get("initial", [config.bos_token_id, config.eos_token_id]))
        print(f"--- Row {i + 1} ---")
        print(f"  src:            {len(row['src'])} tokens")
        print(f"  target:         {len(row['target'])} tokens")
        print(f"  initial:        {init_len} tokens")
        print(f"  y_ins:          {len(row['y_ins'])} tokens  p_star={row['p_star']}  "
              f"t_star: {len(row['t_star'])} tokens")
        print(f"  y_ins_plh:      {len(row['y_ins_plh'])} tokens")
        print(f"  y_ins_rnd:      {len(row['y_ins_rnd'])} tokens  "
              f"p_star_rnd={row['p_star_rnd']}  "
              f"t_star_rnd: {len(row['t_star_rnd'])} tokens")
        print(f"  y_ins_plh_rnd:  {len(row['y_ins_plh_rnd'])} tokens")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-compute insertion oracles for LevT training.",
    )
    parser.add_argument("model_config", help="Path to LevT model config.json")
    parser.add_argument("policy_config", help="Path to policy config JSON")
    parser.add_argument("input", help="Path to input tokenized JSONL")
    parser.add_argument("output", help="Path to output augmented JSONL")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show first 3 rows' stats instead of writing output",
    )
    args = parser.parse_args()

    # 1. Load model config
    try:
        config = LevTConfig.from_json(args.model_config)
    except ValueError as exc:
        print(f"Error loading model config: {exc}", file=sys.stderr)
        sys.exit(1)

    # 2. Load policy config
    policy = load_policy_config(args.policy_config)
    drop_prob = policy["random_delete_prob"]
    print(f"Policy: beta={policy['beta']}, random_delete_prob={drop_prob}",
          file=sys.stderr)

    # 3. Set deterministic seed for random_deletion
    random.seed(0)

    # 4. Read input JSONL
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    input_header: Dict[str, Any] | None = None
    rows: List[Dict[str, Any]] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            stripped = line.strip()
            if not stripped:
                print(f"Error: blank line at {input_path}:{line_number}",
                      file=sys.stderr)
                sys.exit(1)
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                print(f"Error: invalid JSON at {input_path}:{line_number}: {exc}",
                      file=sys.stderr)
                sys.exit(1)
            if not isinstance(row, dict):
                print(f"Error: non-object at {input_path}:{line_number}",
                      file=sys.stderr)
                sys.exit(1)
            if line_number == 1 and META_KEY in row:
                input_header = row  # forward to output below
                continue
            if "src" not in row or "target" not in row:
                print(f"Error: missing src/target at {input_path}:{line_number}",
                      file=sys.stderr)
                sys.exit(1)
            rows.append(row)

    if not rows:
        print("Error: input file is empty", file=sys.stderr)
        sys.exit(1)

    print(f"Read {len(rows)} rows from {args.input}", file=sys.stderr)

    # 5. Dry-run mode
    if args.dry_run:
        sample = rows[:3]
        processed = [process_row(r, config, drop_prob, i + 1)
                     for i, r in enumerate(sample)]
        show_stats(processed, config)
        return

    # 6. Process all rows and write output JSONL
    output_path = Path(args.output)
    written = 0
    with open(output_path, "w", encoding="utf-8") as out_f:
        # Forward the input's metadata header (keeps packed datasets marked
        # packed); legacy inputs without a header become regular.
        header = input_header if input_header is not None else dataset_header(False)
        out_f.write(json.dumps(header, ensure_ascii=False) + "\n")
        for i, row in enumerate(rows):
            try:
                out_row = process_row(row, config, drop_prob, i + 1)
            except Exception as exc:
                print(f"Error processing line {i + 1}: {exc}",
                      file=sys.stderr)
                sys.exit(1)
            out_f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            written += 1
            if written % 1000 == 0:
                print(f"Processed {written} rows...", file=sys.stderr)

    print(f"Done. Wrote {written} rows to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
