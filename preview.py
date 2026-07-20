#!/usr/bin/env python3
"""Preview LevT model editing effects on random samples from train & validation sets.

Usage::

    python preview.py --checkpoint checkpoints/step_00001000.pt
    python preview.py --checkpoint checkpoints/latest.pt --samples 10
    python preview.py --checkpoint checkpoints/latest.pt --device cpu --no-tokenizer
    python preview.py --checkpoint checkpoints/latest.pt --interactive
"""

from __future__ import annotations

import argparse
import difflib
import random
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import torch

from levt.checkpoint import load_checkpoint
from levt.config import LevTConfig
from levt.data import JsonlDataset
from levt.decoder import GreedyDecoder
from levt.model import LevTModel


# ═══════════════════════════════════════════════════════════════════════════════
# ANSI colour / style helpers
# ═══════════════════════════════════════════════════════════════════════════════

_RESET   = "\033[0m"
_BOLD    = "\033[1m"
_DIM     = "\033[2m"

_RED     = "\033[31m"
_GREEN   = "\033[32m"
_YELLOW  = "\033[33m"
_BLUE    = "\033[34m"
_MAGENTA = "\033[35m"
_CYAN    = "\033[36m"
_WHITE   = "\033[37m"

_BRIGHT_RED     = "\033[91m"
_BRIGHT_GREEN   = "\033[92m"
_BRIGHT_YELLOW  = "\033[93m"
_BRIGHT_BLUE    = "\033[94m"
_BRIGHT_MAGENTA = "\033[95m"
_BRIGHT_CYAN    = "\033[96m"

_STRIKE = "\033[9m"

ICON_DEL = "✗"
ICON_ADD = "+"
ICON_FILL = "→"
ICON_SAME = "="


# ═══════════════════════════════════════════════════════════════════════════════
# Tokenizer helper
# ═══════════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════════
# Token ↔ text helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _token_to_str(tok: int, config: LevTConfig) -> str:
    """Single token ID → human-readable label (no tokenizer)."""
    _map = {
        config.bos_token_id: "«BOS»",
        config.eos_token_id: "«EOS»",
        config.pad_token_id: "«PAD»",
        config.plh_token_id: "«PLH»",
    }
    if tok in _map:
        return _map[tok]
    return str(tok)


def _tokens_to_label_list(token_ids: Sequence[int], tokenizer,
                          config: LevTConfig) -> List[str]:
    """Convert a sequence of token IDs into a list of human-readable labels.

    When a tokenizer is available, each label is the decoded text of a single
    token (with special characters escaped for display).  Otherwise numeric IDs
    and special-token markers are used.
    """
    specials = {config.pad_token_id, config.bos_token_id,
                config.eos_token_id, config.plh_token_id}
    if tokenizer is not None:
        labels: List[str] = []
        for t in token_ids:
            if t in specials:
                labels.append(_token_to_str(t, config))
            else:
                text = tokenizer.decode([t], skip_special_tokens=True)
                # Replace newlines / tabs so output stays single-line.
                text = text.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
                labels.append(text if text else _token_to_str(t, config))
        return labels
    else:
        return [_token_to_str(t, config) for t in token_ids]


def _tokens_to_text(token_ids: List[int], tokenizer,
                    config: LevTConfig) -> str:
    """Convert a list of token IDs to human-readable text.

    Strips BOS/EOS/PAD/PLH and replaces PLH with a visible marker.
    """
    if tokenizer is not None:
        filtered = [t for t in token_ids
                    if t not in {config.pad_token_id, config.bos_token_id,
                                 config.eos_token_id, config.plh_token_id}]
        if not filtered:
            return "(empty)"
        text = tokenizer.decode(filtered, skip_special_tokens=True)
        return text if text.strip() else "(empty)"
    else:
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


# ═══════════════════════════════════════════════════════════════════════════════
# Decoration helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _divider(char: str = "─", width: int = 80) -> str:
    return char * width


# ═══════════════════════════════════════════════════════════════════════════════
# Trace rendering for interactive mode
# ═══════════════════════════════════════════════════════════════════════════════

def _render_trace(
    trace: List[Tuple[str, torch.Tensor, float | None]],
    tokenizer,
    config: LevTConfig,
) -> str:
    """Render a decoding trace to a colourised string.

    Parameters
    ----------
    trace:
        List of ``(phase, token_ids_tensor)`` tuples as returned by
        :meth:`GreedyDecoder.decode_with_trace`.
    tokenizer:
        HF tokenizer or ``None``.
    config:
        Model configuration (for special-token IDs).

    Returns
    -------
    str
        ANSI-coloured rendering of the full editing process.
    """
    lines: List[str] = []

    # We need labels for display.  Build a lookup lazily.
    def _labels(tensor: torch.Tensor) -> List[str]:
        return _tokens_to_label_list(tensor.tolist(), tokenizer, config)

    # Group into iterations: start is iteration 0; then del,plh,fill repeat.
    # trace[0]  = ("start", ...)
    # trace[1]  = ("del",   ...)   } iter 1
    # trace[2]  = ("plh",   ...)   }
    # trace[3]  = ("fill",  ...)   }
    # trace[4]  = ("del",   ...)   } iter 2
    # ...
    start_tokens = trace[0][1]
    start_labels = _labels(start_tokens)
    lines.append(f"  {_DIM}start:{_RESET}  "
                 f"{' '.join(_coloured_labels(start_tokens, start_labels, config, 'start'))}")

    phases = trace[1:]  # del, plh, fill, del, plh, fill, ...
    iteration = 0
    i = 0
    while i < len(phases):
        iteration += 1
        lines.append(f"  {_BRIGHT_CYAN}── iter {iteration} ──{_RESET}")

        for _ in range(3):  # up to 3 phases per iteration
            if i >= len(phases):
                break
            phase_name, tokens, entropy = phases[i]
            i += 1

            labels = _labels(tokens)
            phase_icon = {"del": f"{_RED}✗{_RESET}", "plh": f"{_YELLOW}+{_RESET}", "fill": f"{_GREEN}→{_RESET}"}.get(phase_name, "?")
            phase_colour = {"del": _RED, "plh": _YELLOW, "fill": _GREEN}.get(phase_name, _DIM)

            # Find the previous token tensor for diffing.
            if i == 1:
                prev = start_tokens
            else:
                prev = phases[i - 2][1]  # the previous phase's tensor

            # Build coloured token line + annotations.
            token_line, annotations = _build_phase_line(
                prev.tolist(), tokens.tolist(), labels, config, phase_name,
            )

            entropy_str = f"  {_DIM}[H={entropy:.3f}]{_RESET}" if entropy is not None else ""
            lines.append(f"    {phase_colour}{phase_name:>4}{_RESET}  {token_line}{entropy_str}")
            if annotations:
                lines.append(f"          {', '.join(annotations)}")

        # After fill, check if next phase starts a new iteration.
        # If i < len(phases) and phases[i][0] == "del", we loop again.

    # Final result
    if phases:
        final_tokens = phases[-1][1]
        final_text = _tokens_to_text(final_tokens.tolist(), tokenizer, config)
        lines.append(f"  {_BRIGHT_GREEN}{_BOLD}final:{_RESET}  {final_text}")

    return "\n".join(lines)


def _coloured_labels(
    tokens: torch.Tensor,
    labels: List[str],
    config: LevTConfig,
    _phase: str,  # unused — kept for future colour variation
) -> str:
    """Render token labels in dim white (uniform, for start)."""
    parts: List[str] = []
    for t in tokens.tolist():
        parts.append(f"{_DIM}{_token_to_str(t, config)}{_RESET}")
    return " ".join(parts)


def _build_phase_line(
    before_ids: List[int],
    after_ids: List[int],
    after_labels: List[str],
    config: LevTConfig,
    phase: str,
) -> Tuple[str, List[str]]:
    """Build a colourised token line for one phase plus annotation snippets.

    Returns ``(token_line, annotations)``.
    """
    plh = config.plh_token_id
    matcher = difflib.SequenceMatcher(None, before_ids, after_ids)

    coloured_parts: List[str] = []
    annotations: List[str] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for k in range(j1, j2):
                lbl = after_labels[k]
                coloured_parts.append(f"{_DIM}{_safe_strip(lbl)}{_RESET}")
        elif tag == "delete":
            for t in before_ids[i1:i2]:
                annotations.append(f"{_RED}{_STRIKE}{ICON_DEL} {_token_to_str(t, config)}{_RESET}")
        elif tag == "insert":
            for k in range(j1, j2):
                t = after_ids[k]
                lbl = after_labels[k]
                if t == plh:
                    coloured_parts.append(f"{_BRIGHT_YELLOW}{lbl}{_RESET}")
                    annotations.append(f"{_BRIGHT_YELLOW}{ICON_ADD} «PLH»{_RESET}")
                else:
                    coloured_parts.append(f"{_BRIGHT_GREEN}{lbl}{_RESET}")
                    annotations.append(f"{_BRIGHT_GREEN}{ICON_ADD} {_token_to_str(t, config)}{_RESET}")
        elif tag == "replace":
            if phase == "fill":
                # Fill phase: PLH → token is a *fill*, not a delete+insert.
                for k in range(j1, j2):
                    t = after_ids[k]
                    lbl = after_labels[k]
                    if t == plh:
                        coloured_parts.append(f"{_BRIGHT_YELLOW}{lbl}{_RESET}")
                    else:
                        coloured_parts.append(f"{_BRIGHT_GREEN}{lbl}{_RESET}")
                        annotations.append(f"{_BRIGHT_GREEN}{ICON_FILL} {_token_to_str(t, config)}{_RESET}")
            else:
                # Del / plh phases: genuine replace = delete old + insert new.
                for t in before_ids[i1:i2]:
                    annotations.append(f"{_RED}{_STRIKE}{ICON_DEL} {_token_to_str(t, config)}{_RESET}")
                for k in range(j1, j2):
                    t = after_ids[k]
                    lbl = after_labels[k]
                    if t == plh:
                        coloured_parts.append(f"{_BRIGHT_YELLOW}{lbl}{_RESET}")
                        annotations.append(f"{_BRIGHT_YELLOW}{ICON_ADD} «PLH»{_RESET}")
                    else:
                        coloured_parts.append(f"{_BRIGHT_GREEN}{lbl}{_RESET}")
                        annotations.append(f"{_BRIGHT_GREEN}{ICON_ADD} {_token_to_str(t, config)}{_RESET}")

    return " ".join(coloured_parts), annotations


def _safe_strip(s: str) -> str:
    """Return *s* or a visible placeholder for whitespace-only tokens."""
    stripped = s.strip()
    if not stripped:
        # Show a visible dot for whitespace tokens.
        return f"·{s}·" if s else "··"
    return stripped


# ═══════════════════════════════════════════════════════════════════════════════
# Non-interactive: sample preview
# ═══════════════════════════════════════════════════════════════════════════════

def _print_sample(index: int, source_text: str, target_text: str,
                  predicted_text: str, iterations: int,
                  avg_entropy: float | None = None) -> None:
    """Pretty-print one sample with source, target, and prediction."""
    print(_divider())
    header = f"  Sample #{index + 1}  (decoding iterations: {iterations})"
    if avg_entropy is not None:
        header += f"  avg H = {avg_entropy:.3f} nats"
    print(header)
    print(_divider())
    print(f"  Source:     {source_text}")
    print(f"  Target:     {target_text}")
    print(f"  Predicted:  {predicted_text}")
    print()


def preview_sample(
    decoder: GreedyDecoder,
    config: LevTConfig,
    row: dict,
    device: torch.device,
    tokenizer,
) -> Tuple[str, str, str, int, float | None]:
    """Run decoding on one sample and return (source, target, predicted, iters, avg_entropy)."""
    src_tokens = torch.tensor(row["src"], dtype=torch.long, device=device)
    initial = torch.tensor(row["initial"], dtype=torch.long, device=device)

    output, iterations, trace = decoder.decode_with_trace(src_tokens, initial)

    source_text = _tokens_to_text(row["src"], tokenizer, config)
    target_text = _tokens_to_text(row["target"], tokenizer, config)
    predicted_text = _tokens_to_text(output.tolist(), tokenizer, config)

    # Average entropy across all phases that have entropy values.
    entropies = [ent for _, _, ent in trace if ent is not None]
    avg_entropy = sum(entropies) / len(entropies) if entropies else None

    return source_text, target_text, predicted_text, iterations, avg_entropy


# ═══════════════════════════════════════════════════════════════════════════════
# Interactive REPL
# ═══════════════════════════════════════════════════════════════════════════════

def _interactive_loop(
    decoder: GreedyDecoder,
    config: LevTConfig,
    device: torch.device,
    tokenizer,
) -> None:
    """Run an interactive REPL showing model editing step-by-step."""
    print()
    print(f"  {_BOLD}Interactive LevT Preview{_RESET}")
    print(f"  {_DIM}Type a sentence to see how the model edits it.{_RESET}")
    print(f"  {_DIM}Type {_BRIGHT_CYAN}:quit{_DIM} or {_BRIGHT_CYAN}:q{_DIM} to exit, "
          f"{_BRIGHT_CYAN}:help{_DIM} for commands.{_RESET}")
    print()

    tokenizer_available = tokenizer is not None

    while True:
        try:
            raw = input(f"  {_BOLD}{_BRIGHT_CYAN}>>>{_RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not raw:
            continue

        # --- Meta commands ---------------------------------------------------
        if raw in (":quit", ":q", ":exit"):
            print(f"  {_DIM}Bye!{_RESET}")
            break
        if raw == ":help":
            print(f"  {_DIM}Commands:{_RESET}")
            print(f"    {_BRIGHT_CYAN}:quit, :q{_RESET}  {_DIM}Exit{_RESET}")
            print(f"    {_BRIGHT_CYAN}:help{_RESET}      {_DIM}Show this message{_RESET}")
            print(f"    {_BRIGHT_CYAN}:notrace{_RESET}   {_DIM}Toggle trace display off{_RESET}")
            print(f"    {_BRIGHT_CYAN}:trace{_RESET}     {_DIM}Toggle trace display on{_RESET}")
            continue
        if raw == ":notrace":
            _interactive_loop._show_trace = False
            print(f"  {_DIM}Trace display {_RED}OFF{_DIM} — showing final result only.{_RESET}")
            continue
        if raw == ":trace":
            _interactive_loop._show_trace = True
            print(f"  {_DIM}Trace display {_GREEN}ON{_DIM} — showing step-by-step edits.{_RESET}")
            continue

        show_trace = getattr(_interactive_loop, "_show_trace", True)

        # --- Tokenize input --------------------------------------------------
        if tokenizer_available:
            encoded = tokenizer(raw, add_special_tokens=False, return_tensors="pt")
            input_ids = encoded["input_ids"][0].tolist()
            if not input_ids:
                print(f"  {_RED}(empty tokenization — nothing to do){_RESET}")
                continue
            # Wrap with BOS/EOS.
            src_list = [config.bos_token_id] + input_ids + [config.eos_token_id]
        else:
            print(f"  {_RED}(no tokenizer available — cannot parse text){_RESET}")
            continue

        src_tensor = torch.tensor(src_list, dtype=torch.long, device=device)

        # --- Decode ----------------------------------------------------------
        try:
            if show_trace:
                output, iterations, trace = decoder.decode_with_trace(src_tensor)
                rendered = _render_trace(trace, tokenizer, config)
                entropies = [ent for _, _, ent in trace if ent is not None]
                avg_h = sum(entropies) / len(entropies) if entropies else None
                print(f"\n{rendered}\n")
                parts = [f"{_DIM}{iterations} iteration{'s' if iterations != 1 else ''}{_RESET}"]
                if avg_h is not None:
                    parts.append(f"{_DIM}avg H = {avg_h:.3f} nats{_RESET}")
                print(f"  {', '.join(parts)}")
            else:
                output, iterations = decoder.decode(src_tensor)
                final_text = _tokens_to_text(output.tolist(), tokenizer, config)
                print(f"  {_BRIGHT_GREEN}{_BOLD}→{_RESET} {final_text}")
                print(f"  {_DIM}({iterations} iteration{'s' if iterations != 1 else ''}){_RESET}")
        except Exception as exc:
            print(f"  {_RED}Error: {exc}{_RESET}")

        print()


# Attach mutable state as a function attribute (simple, no global).
_interactive_loop._show_trace = True


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preview LevT model editing on random samples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python preview.py --checkpoint checkpoints/step_00001000.pt
  python preview.py --checkpoint checkpoints/latest.pt --samples 10
  python preview.py --checkpoint checkpoints/latest.pt --no-tokenizer
  python preview.py --checkpoint checkpoints/latest.pt --interactive
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
    parser.add_argument(
        "--interactive", action="store_true",
        help="Enter interactive REPL mode: type a sentence and see step-by-step "
             "model edits with colour highlighting",
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

    # --- Interactive mode -------------------------------------------------
    if args.interactive:
        if tokenizer is None:
            print("error: --interactive requires a tokenizer. "
                  "Remove --no-tokenizer and ensure the checkpoint has "
                  "an hf_model_name_or_path.", file=sys.stderr)
            sys.exit(1)
        _interactive_loop(decoder, model_cfg, device, tokenizer)
        return

    # --- Non-interactive: pick random samples from each dataset -----------
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
                src_text, tgt_text, pred_text, iters, avg_ent = preview_sample(
                    decoder, model_cfg, row, device, tokenizer,
                )
            except Exception as exc:
                print(f"  !! Sample #{i + 1} (row {idx}) failed: {exc}")
                continue
            _print_sample(i, src_text, tgt_text, pred_text, iters, avg_ent)

    print(_divider("═"))
    print("  Done.")
    print(_divider("═"))


if __name__ == "__main__":
    main()
