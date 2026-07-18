"""Versioned training checkpoints and random-state helpers."""

from __future__ import annotations

import os
import random
import re
from pathlib import Path
from typing import Any, Dict, Optional

import torch


CHECKPOINT_VERSION = 1

_STEP_PATTERN = re.compile(r"^step_(\d+)\.pt$")


def capture_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Dict[str, Any]) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path: str | Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    final_payload = {"version": CHECKPOINT_VERSION, **payload}
    temporary = path.with_name(path.name + ".tmp")
    torch.save(final_payload, temporary)
    os.replace(temporary, path)


def cleanup_checkpoints(
    directory: Path,
    keep_last: int,
    checkpoint_val_loss: Dict[int, float],
) -> None:
    """Remove old checkpoints, keeping the best (by eval loss) and last *N*.

    Args:
        directory: Checkpoint directory to scan.
        keep_last: Number of most-recent checkpoints (by step number) to retain.
            Values ≤ 0 are a no-op.
        checkpoint_val_loss: Mapping of ``step → eval_loss`` for every saved
            checkpoint.  The step with the smallest loss is always preserved.
            Checkpoint steps not present in this mapping are not eligible for
            best-checkpoint protection (but may still be kept as part of the
            last-*N* window).
    """
    if keep_last <= 0:
        return

    pattern = "step_*.pt"
    checkpoint_files = sorted(directory.glob(pattern))
    if not checkpoint_files:
        return

    step_files: list[tuple[int, Path]] = []
    for fp in checkpoint_files:
        m = _STEP_PATTERN.match(fp.name)
        if m is None:
            continue
        step_files.append((int(m.group(1)), fp))

    if not step_files:
        return

    step_files.sort(key=lambda x: x[0])

    valid_steps = {s for s, _ in step_files}
    valid_val_loss = {s: v for s, v in checkpoint_val_loss.items() if s in valid_steps}

    keep_steps: set[int] = set()

    if valid_val_loss:
        best_step = min(valid_val_loss, key=valid_val_loss.__getitem__)
        keep_steps.add(best_step)

    for step, _ in step_files[-keep_last:]:
        keep_steps.add(step)

    for step, filepath in step_files:
        if step not in keep_steps:
            filepath.unlink()


def load_checkpoint(path: str | Path, *, map_location: Any = "cpu") -> Dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"failed to load checkpoint {path}: {exc}") from exc
    if not isinstance(checkpoint, dict) or checkpoint.get("version") != CHECKPOINT_VERSION:
        raise ValueError(f"unsupported checkpoint version in {path}")
    return checkpoint
