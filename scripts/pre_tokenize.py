"""
Pre-tokenization script for Levenshtein Transformer training pipeline.

Converts raw text JSONL files into tokenized integer ID JSONL files
using a HuggingFace tokenizer. The output is ready for consumption
by the LevT training pipeline (which expects ``src``, ``target``,
and optionally ``initial`` as integer token ID lists).

Usage::

    python scripts/pre_tokenize.py tokenizer_config.json input.jsonl output.jsonl
    python scripts/pre_tokenize.py tokenizer_config.json input.jsonl output.jsonl --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO


def load_config(config_path: str) -> Dict[str, Any]:
    """Load and return the tokenizer configuration JSON."""
    path = Path(config_path)
    if not path.exists():
        print(f"Error: config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config


def load_tokenizer(config: Dict[str, Any]):
    """Load a HuggingFace AutoTokenizer from the config.

    Handles missing ``pad_token`` by setting it from the config.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name_or_path"],
        trust_remote_code=config.get("trust_remote_code", False),
        local_files_only=config.get("local_files_only", False),
    )
    # Ensure pad_token is set
    if tokenizer.pad_token is None:
        pad_token_id = config["pad_token_id"]
        tokenizer.pad_token = tokenizer.convert_ids_to_tokens(pad_token_id)
        if tokenizer.pad_token is None:
            # Tokenizer can't resolve the ID; add a new pad token manually
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            print(
                f"Warning: tokenizer had no pad_token; added '[PAD]' "
                f"with id {tokenizer.pad_token_id}",
                file=sys.stderr,
            )

    # Warn if vocab sizes differ
    tokenizer_vocab_size = getattr(tokenizer, "vocab_size", None)
    config_vocab_size = config["vocab_size"]
    if tokenizer_vocab_size is not None and tokenizer_vocab_size != config_vocab_size:
        print(
            f"Warning: tokenizer vocab_size ({tokenizer_vocab_size}) "
            f"differs from config vocab_size ({config_vocab_size})",
            file=sys.stderr,
        )

    return tokenizer


def _should_prepend_bos(tokenizer, token_ids: List[int], bos_token_id: int) -> bool:
    """Return True if BOS should be prepended (avoids duplicate)."""
    if bos_token_id is None:
        return False
    # If the tokenizer already handles BOS and the first token matches, skip.
    if tokenizer.bos_token_id is not None and tokenizer.bos_token_id == bos_token_id:
        if token_ids and token_ids[0] == bos_token_id:
            return False
    return True


def _should_append_eos(tokenizer, token_ids: List[int], eos_token_id: int) -> bool:
    """Return True if EOS should be appended (avoids duplicate)."""
    if eos_token_id is None:
        return False
    # If the tokenizer already handles EOS and the last token matches, skip.
    if tokenizer.eos_token_id is not None and tokenizer.eos_token_id == eos_token_id:
        if token_ids and token_ids[-1] == eos_token_id:
            return False
    return True


def tokenize_text(
    tokenizer,
    text: str,
    max_length: int,
    config: Dict[str, Any],
) -> List[int]:
    """Tokenize a single text string and return an integer ID list.

    Handles BOS/EOS addition with duplicate detection, truncation, and
    vocab-range validation.
    """
    bos_token_id = config.get("bos_token_id")
    eos_token_id = config.get("eos_token_id")
    add_bos = config.get("add_bos", False)
    add_eos = config.get("add_eos", False)

    # Tokenize (truncation is handled manually after BOS/EOS addition)
    ids: List[int] = tokenizer.encode(text, add_special_tokens=False)

    # Truncate to max_length, leaving room for optional BOS/EOS
    # We apply truncation here, before adding BOS/EOS, so the content
    # fits within max_length after boundary tokens are prepended/appended.
    budget = max_length
    if add_bos and _should_prepend_bos(tokenizer, ids, bos_token_id):
        budget -= 1
    if add_eos and _should_append_eos(tokenizer, ids, eos_token_id):
        budget -= 1
    if budget < 0:
        budget = 0
    if len(ids) > budget:
        ids = ids[:budget]

    # Prepend BOS
    if add_bos and bos_token_id is not None:
        if _should_prepend_bos(tokenizer, ids, bos_token_id):
            ids.insert(0, bos_token_id)

    # Append EOS
    if add_eos and eos_token_id is not None:
        if _should_append_eos(tokenizer, ids, eos_token_id):
            ids.append(eos_token_id)

    return ids


def validate_ids(ids: List[int], vocab_size: int, line_num: int, field: str) -> None:
    """Check all token IDs are in ``[0, vocab_size)``; raise on violation."""
    for tid in ids:
        if not (0 <= tid < vocab_size):
            raise ValueError(
                f"Line {line_num}: {field} token id {tid} is out of range "
                f"[0, {vocab_size})"
            )


def warn_interior_boundary_tokens(
    ids: List[int],
    bos_token_id: Optional[int],
    eos_token_id: Optional[int],
    line_num: int,
    field: str,
) -> None:
    """Warn if BOS/EOS appear inside a sequence; only the boundaries are valid.

    LevT training data rejects interior BOS/EOS (the deletion oracle never
    touches them and the decoder masks them from fill predictions), but they
    slip in when the raw text literally contains a string that tokenizes to
    the BOS/EOS id (e.g. ``<s>`` in a code string). This is a warning, not a
    hard error: the training-side validator in ``levt.data`` still catches any
    rows that slip through.
    """
    if len(ids) < 2:
        return
    for index, tid in enumerate(ids):
        if index in (0, len(ids) - 1):
            continue  # boundary BOS/EOS is expected
        if tid == bos_token_id or tid == eos_token_id:
            label = "<bos>" if tid == bos_token_id else "<eos>"
            print(
                f"Warning: line {line_num} '{field}' has interior {label} "
                f"(id {tid}) at position {index}; LevT training data rejects this",
                file=sys.stderr,
            )
            return


def process_dry_run(
    rows: List[Dict[str, Any]],
    tokenizer,
    config: Dict[str, Any],
    src_field: str,
    target_field: str,
    initial_field: Optional[str],
) -> None:
    """Print tokenization statistics for the first 3 rows and exit."""
    vocab_size = config["vocab_size"]
    max_src = config["max_src_length"]
    max_tgt = config["max_tgt_length"]

    for i, row in enumerate(rows[:3]):
        src_text = row.get(src_field, "")
        tgt_text = row.get(target_field, "")

        src_ids = tokenize_text(tokenizer, src_text, max_src, config)
        tgt_ids = tokenize_text(tokenizer, tgt_text, max_tgt, config)

        # Unique tokens per sequence
        src_unique = len(set(src_ids))
        tgt_unique = len(set(tgt_ids))

        # Vocab coverage (sequence-level)
        src_coverage = f"{src_unique}/{vocab_size}" if vocab_size else "N/A"
        tgt_coverage = f"{tgt_unique}/{vocab_size}" if vocab_size else "N/A"

        print(f"--- Row {i + 1} ---")
        print(f"  src:     {len(src_ids)} tokens, {src_unique} unique, coverage {src_coverage}")
        print(f"  target:  {len(tgt_ids)} tokens, {tgt_unique} unique, coverage {tgt_coverage}")

        if initial_field and initial_field in row:
            init_text = row[initial_field]
            init_ids = tokenize_text(tokenizer, init_text, max_tgt, config)
            init_unique = len(set(init_ids))
            init_coverage = f"{init_unique}/{vocab_size}" if vocab_size else "N/A"
            print(f"  initial: {len(init_ids)} tokens, {init_unique} unique, coverage {init_coverage}")

        # Text preview
        print(f"  src_text[:120]:    {src_text[:120]!r}")
        print(f"  target_text[:120]: {tgt_text[:120]!r}")
        print()

    # Full vocab coverage across first 3 rows
    print("--- Aggregate vocab coverage (first 3 rows) ---")
    all_src_ids: List[int] = []
    all_tgt_ids: List[int] = []
    for row in rows[:3]:
        all_src_ids.extend(
            tokenize_text(tokenizer, row.get(src_field, ""), max_src, config)
        )
        all_tgt_ids.extend(
            tokenize_text(tokenizer, row.get(target_field, ""), max_tgt, config)
        )

    for name, ids in [("src", all_src_ids), ("target", all_tgt_ids)]:
        total = len(ids)
        unique = len(set(ids))
        cov = f"{unique}/{vocab_size}" if vocab_size else "N/A"
        print(f"  {name}: {total} total tokens, {unique} unique, coverage {cov}")


def read_jsonl(infile: TextIO) -> List[Dict[str, Any]]:
    """Read all non-blank JSON lines from *infile*.

    Returns a list of parsed row dicts and tracks original line numbers
    for error reporting.
    """
    rows: List[Dict[str, Any]] = []
    line_numbers: List[int] = []
    for raw_line_num, line in enumerate(infile, start=1):
        stripped = line.strip()
        if not stripped:
            continue  # skip blank lines silently
        try:
            row = json.loads(stripped)
        except json.JSONDecodeError as exc:
            print(
                f"Error: malformed JSON on line {raw_line_num}: {exc}",
                file=sys.stderr,
            )
            sys.exit(1)
        if not isinstance(row, dict):
            print(
                f"Error: line {raw_line_num} is not a JSON object: {stripped!r}",
                file=sys.stderr,
            )
            sys.exit(1)
        rows.append(row)
        line_numbers.append(raw_line_num)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tokenize text JSONL into integer-ID JSONL for LevT training.",
    )
    parser.add_argument("config", help="Path to tokenizer_config.json")
    parser.add_argument("input", help="Path to input JSONL file (text)")
    parser.add_argument("output", help="Path to output JSONL file (integer IDs)")
    parser.add_argument(
        "--src-field",
        default="src",
        help="Field name for source text (default: 'src')",
    )
    parser.add_argument(
        "--target-field",
        default="target",
        help="Field name for target text (default: 'target')",
    )
    parser.add_argument(
        "--initial-field",
        default=None,
        help="Optional field name for initial text (default: None)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print tokenization stats for first 3 rows instead of writing output",
    )
    args = parser.parse_args()

    # ---------------------------------------------------------------
    # 1. Load config
    # ---------------------------------------------------------------
    config = load_config(args.config)
    vocab_size = config["vocab_size"]
    max_src_len = config["max_src_length"]
    max_tgt_len = config["max_tgt_length"]

    # ---------------------------------------------------------------
    # 2. Load tokenizer
    # ---------------------------------------------------------------
    print("Loading tokenizer...", file=sys.stderr)
    tokenizer = load_tokenizer(config)
    print(f"  tokenizer: {config['model_name_or_path']}", file=sys.stderr)
    print(f"  vocab_size (config): {vocab_size}", file=sys.stderr)
    print(f"  max_src_length:      {max_src_len}", file=sys.stderr)
    print(f"  max_tgt_length:      {max_tgt_len}", file=sys.stderr)
    if args.initial_field:
        print(f"  initial_field:       {args.initial_field}", file=sys.stderr)
    print(file=sys.stderr)

    # ---------------------------------------------------------------
    # 3. Read input JSONL
    # ---------------------------------------------------------------
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    with open(input_path, "r", encoding="utf-8") as f:
        rows = read_jsonl(f)

    if not rows:
        print("Error: input file contains no valid data rows", file=sys.stderr)
        sys.exit(1)

    print(f"Read {len(rows)} rows from {args.input}", file=sys.stderr)
    print(file=sys.stderr)

    # ---------------------------------------------------------------
    # 4. Dry-run mode
    # ---------------------------------------------------------------
    if args.dry_run:
        process_dry_run(
            rows, tokenizer, config,
            args.src_field, args.target_field, args.initial_field,
        )
        return

    # ---------------------------------------------------------------
    # 5. Process all rows and write output JSONL
    # ---------------------------------------------------------------
    output_path = Path(args.output)
    written = 0
    with open(output_path, "w", encoding="utf-8") as out_f:
        for i, row in enumerate(rows):
            line_display = i + 1  # 1-based for error messages

            # Check required fields exist
            if args.src_field not in row:
                print(
                    f"Error: line ~{line_display} missing required field "
                    f"'{args.src_field}'",
                    file=sys.stderr,
                )
                sys.exit(1)
            if args.target_field not in row:
                print(
                    f"Error: line ~{line_display} missing required field "
                    f"'{args.target_field}'",
                    file=sys.stderr,
                )
                sys.exit(1)

            src_text = row[args.src_field]
            tgt_text = row[args.target_field]

            # Ensure fields are strings
            if not isinstance(src_text, str):
                print(
                    f"Error: line ~{line_display} '{args.src_field}' "
                    f"is not a string: {type(src_text).__name__}",
                    file=sys.stderr,
                )
                sys.exit(1)
            if not isinstance(tgt_text, str):
                print(
                    f"Error: line ~{line_display} '{args.target_field}' "
                    f"is not a string: {type(tgt_text).__name__}",
                    file=sys.stderr,
                )
                sys.exit(1)

            # Tokenize
            src_ids = tokenize_text(tokenizer, src_text, max_src_len, config)
            tgt_ids = tokenize_text(tokenizer, tgt_text, max_tgt_len, config)

            # Validate vocab range
            validate_ids(src_ids, vocab_size, line_display, args.src_field)
            validate_ids(tgt_ids, vocab_size, line_display, args.target_field)

            # Warn about interior BOS/EOS (LevT training data rejects these rows)
            warn_interior_boundary_tokens(
                src_ids, config.get("bos_token_id"), config.get("eos_token_id"),
                line_display, args.src_field,
            )
            warn_interior_boundary_tokens(
                tgt_ids, config.get("bos_token_id"), config.get("eos_token_id"),
                line_display, args.target_field,
            )

            # Build output row (preserving all original keys, replacing tokenized ones)
            out_row = dict(row)  # shallow copy preserves extra keys
            out_row[args.src_field] = src_ids
            out_row[args.target_field] = tgt_ids

            # Optional initial field
            if args.initial_field and args.initial_field in row:
                init_text = row[args.initial_field]
                if not isinstance(init_text, str):
                    print(
                        f"Error: line ~{line_display} '{args.initial_field}' "
                        f"is not a string: {type(init_text).__name__}",
                        file=sys.stderr,
                    )
                    sys.exit(1)
                init_ids = tokenize_text(tokenizer, init_text, max_tgt_len, config)
                validate_ids(init_ids, vocab_size, line_display, args.initial_field)
                warn_interior_boundary_tokens(
                    init_ids, config.get("bos_token_id"), config.get("eos_token_id"),
                    line_display, args.initial_field,
                )
                out_row[args.initial_field] = init_ids

            # Write
            out_f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            written += 1

            # Progress
            if written % 1000 == 0:
                print(f"Processed {written} rows...", file=sys.stderr)

    print(f"Done. Wrote {written} rows to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
