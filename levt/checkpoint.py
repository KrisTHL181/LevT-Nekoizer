"""Versioned training checkpoints and random-state helpers."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict

import torch


CHECKPOINT_VERSION = 1


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


def load_checkpoint(path: str | Path, *, map_location: Any = "cpu") -> Dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"failed to load checkpoint {path}: {exc}") from exc
    if not isinstance(checkpoint, dict) or checkpoint.get("version") != CHECKPOINT_VERSION:
        raise ValueError(f"unsupported checkpoint version in {path}")
    return checkpoint
