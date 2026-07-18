#!/usr/bin/env python3
"""Preview LevT model editing effects on random samples from train & validation sets.

Usage::

    python preview.py --checkpoint checkpoints/step_00001000.pt
    python preview.py --checkpoint checkpoints/latest.pt --samples 10
    python preview.py --checkpoint checkpoints/latest.pt --device cpu --no-tokenizer
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import List, Tuple

import torch

from levt.checkpoint import load_checkpoint
from levt.config import LevTConfig
from levt.data import JsonlDataset
from levt.decoder import GreedyDecoder
from levt.model import LevTModel


# ═══════════════════════════════════════════════════════════════════════════
# Tokenizer helper
# ═══════════════════════════════════════════════════════════════════════════

def _load_tokenizer(model_name: str, *, local_files_only: bool = False,
                    trust_remote_code: bool = False):
    """Try to load the Hugging Face tokenizer for the given model.

    Returns the tokenizer or ``None`` if unavailable.
    """
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return None
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )
        return tokenizer
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Display helpers
# ═══════════════════════════════════════════════════════════════════════════

def _tokens_to_text(token_ids: List[int], tokenizer,
                    config: LevTConfig) -> str:
    """Convert a list of token IDs to human-readable text.

    Strips BOS/EOS and replaces PLH with a visible marker.
    """
    if tokenizer is not None:
        # Remove special tokens for clean display
        filtered = [t for t in token_ids
                    if t not in {config.pad_token_id, config.bos_token_id,
                                 config.eos_token_id, config.plh_token_id}]
        if not filtered:
            return "(empty)"
        text = tokenizer.decode(filtered, skip_special_tokens=True)
        return text if text.strip() else "(empty)"
    else:
        # Numeric display: group tokens compactly
        parts: List[str] = []
        for t in token_ids:
            if t == config.bos_token_id:
                parts.append("<BOS>")
            elif t == config.eos_token_id:
                parts.append("<EOS>")
            elif t == config.pad_token_id:
                parts.append("<PAD>")
            elif t == config.plh_token_id:
                parts.append("<PLH>")
            else:
                parts.append(str(t))
        return " ".join(parts)


def _divider(char: str = "─", width: int = 80) -> str:
    return char * width


def _print_sample(index: int, source_text: str, target_text: str,
                  predicted_text: str, iterations: int) -> None:
    """Pretty-print one sample with source, target, and prediction."""
    print(_divider())
    print(f"  Sample #{index + 1}  (decoding iterations: {iterations})")
    print(_divider())
    print(f"  Source:     {source_text}")
    print(f"  Target:     {target_text}")
    print(f"  Predicted:  {predicted_text}")
    print()


# ═══════════════════════════════════════════════════════════════════════════
# Core logic
# ═══════════════════════════════════════════════════════════════════════════

def preview_sample(
    decoder: GreedyDecoder,
    config: LevTConfig,
    row: dict,
    device: torch.device,
    tokenizer,
) -> Tuple[str, str, str, int]:
    """Run decoding on one sample and return (source, target, predicted, iters)."""
    src_tokens = torch.tensor(row["src"], dtype=torch.long, device=device)
    initial = torch.tensor(row["initial"], dtype=torch.long, device=device)

    output, iterations = decoder.decode(src_tokens, initial)

    source_text = _tokens_to_text(row["src"], tokenizer, config)
    target_text = _tokens_to_text(row["target"], tokenizer, config)
    predicted_text = _tokens_to_text(output.tolist(), tokenizer, config)

    return source_text, target_text, predicted_text, iterations


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preview LevT model editing on random samples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python preview.py --checkpoint checkpoints/step_00001000.pt
  python preview.py --checkpoint checkpoints/latest.pt --samples 10
  python preview.py --checkpoint checkpoints/latest.pt --no-tokenizer
        """,
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to a training checkpoint (.pt file)",
    )
    parser.add_argument(
        "--samples", type=int, default=5,
        help="Number of random samples per dataset (default: 5)",
    )
    parser.add_argument(
        "--device", default="auto",
        help="Device to run inference on (default: auto)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="Sampling temperature (0 = greedy argmax, >0 = sample from softmax). "
             "Higher values produce more diverse outputs. (default: 0)",
    )
    parser.add_argument(
        "--no-tokenizer", action="store_true",
        help="Show token IDs instead of decoded text",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducible sample selection",
    )
    args = parser.parse_args()

    # --- Resolve device ---------------------------------------------------
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # --- Load checkpoint --------------------------------------------------
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")

    # --- Reconstruct model config -----------------------------------------
    saved_model_cfg = checkpoint.get("model_config")
    if saved_model_cfg is None:
        print("error: checkpoint does not contain model_config", file=sys.stderr)
        sys.exit(1)
    model_cfg = LevTConfig.from_dict(saved_model_cfg, strict_model=True)
    train_cfg_dict = checkpoint.get("train_config", {})

    print(f"  Model:  d_model={model_cfg.d_model}, "
          f"n_enc={model_cfg.n_encoder_layers}, "
          f"n_dec={model_cfg.n_decoder_layers}, "
          f"pos={model_cfg.pos_encoding_type}")
    print(f"  Step:   {checkpoint.get('global_step', 'N/A')}")

    # --- Build model and load weights -------------------------------------
    print("Building model ...")
    model = LevTModel(model_cfg)

    # Strip torch.compile _orig_mod. prefix if present (checkpoints saved
    # under compile have mangled state-dict keys).
    state_dict = checkpoint["model"]
    sample_key = next(iter(state_dict))
    if sample_key.startswith("_orig_mod."):
        print("  (stripping _orig_mod. prefix from compiled checkpoint)")
        state_dict = {
            k.removeprefix("_orig_mod.") if k.startswith("_orig_mod.") else k: v
            for k, v in state_dict.items()
        }

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # --- Create decoder ---------------------------------------------------
    decoder = GreedyDecoder(model, model_cfg, temperature=args.temperature)
    if args.temperature > 0:
        print(f"  Temperature: {args.temperature} (sampling mode)")

    # --- Load tokenizer ---------------------------------------------------
    tokenizer = None
    if not args.no_tokenizer:
        hf_name = train_cfg_dict.get("hf_model_name_or_path", "")
        if hf_name:
            print(f"Loading tokenizer: {hf_name}")
            tokenizer = _load_tokenizer(
                hf_name,
                local_files_only=train_cfg_dict.get("local_files_only", False),
                trust_remote_code=train_cfg_dict.get("trust_remote_code", False),
            )
            if tokenizer is None:
                print("  (tokenizer unavailable — showing token IDs)")
        else:
            print("  (no HF model name in checkpoint — showing token IDs)")

    # --- Pick random samples from each dataset ----------------------------
    if args.seed is not None:
        random.seed(args.seed)

    train_data_path = train_cfg_dict.get("train_data", "")
    valid_data_path = train_cfg_dict.get("validation_data", "")

    datasets: List[Tuple[str, str]] = []
    for label, path in [("Training set", train_data_path),
                        ("Validation set", valid_data_path)]:
        if not path:
            print(f"  ({label} path not found in checkpoint — skipping)")
            continue
        if not Path(path).exists():
            print(f"  ({label} file '{path}' not found — skipping)")
            continue
        datasets.append((label, path))

    if not datasets:
        print("error: no datasets available for sampling", file=sys.stderr)
        sys.exit(1)

    max_src_len = train_cfg_dict.get("max_source_length", 1024)
    max_tgt_len = train_cfg_dict.get("max_target_length", 1024)

    for label, path in datasets:
        print(f"\n{'=' * 80}")
        print(f"  {label}: {path}")
        print(f"{'=' * 80}")

        dataset = JsonlDataset(
            path, model_cfg,
            max_source_length=max_src_len,
            max_target_length=max_tgt_len,
        )

        n_total = len(dataset)
        n_samples = min(args.samples, n_total)
        indices = random.sample(range(n_total), n_samples)

        print(f"  ({n_samples} random samples from {n_total} total)\n")

        for i, idx in enumerate(indices):
            row = dataset[idx]
            try:
                src_text, tgt_text, pred_text, iters = preview_sample(
                    decoder, model_cfg, row, device, tokenizer,
                )
            except Exception as exc:
                print(f"  !! Sample #{i + 1} (row {idx}) failed: {exc}")
                continue
            _print_sample(i, src_text, tgt_text, pred_text, iters)

    print(_divider("═"))
    print("  Done.")
    print(_divider("═"))


if __name__ == "__main__":
    main()
