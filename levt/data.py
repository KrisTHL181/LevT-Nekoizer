"""Strict JSONL dataset and seq-first collation for LevT training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch.utils.data import Dataset

from .config import LevTConfig


def _validate_token_list(
    value: Any,
    name: str,
    *,
    vocab_size: int,
    max_length: int,
    source: str,
) -> List[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{source}: {name} must be a nonempty list of integers")
    if len(value) > max_length:
        raise ValueError(f"{source}: {name} length {len(value)} exceeds {max_length}")
    for index, token in enumerate(value):
        if isinstance(token, bool) or not isinstance(token, int):
            raise ValueError(f"{source}: {name}[{index}] must be an integer (bool is invalid)")
        if not 0 <= token < vocab_size:
            raise ValueError(f"{source}: {name}[{index}]={token} is outside [0, {vocab_size})")
    return list(value)


def _validate_precomputed_token_list(
    value: Any,
    name: str,
    *,
    vocab_size: int,
    source: str,
) -> List[int]:
    """Validate a pre-computed oracle token list (may be empty, no BOS/EOS checks)."""
    if not isinstance(value, list):
        raise ValueError(f"{source}: {name} must be a list of integers")
    for index, token in enumerate(value):
        if isinstance(token, bool) or not isinstance(token, int):
            raise ValueError(f"{source}: {name}[{index}] must be an integer (bool is invalid)")
        if not 0 <= token < vocab_size:
            raise ValueError(f"{source}: {name}[{index}]={token} is outside [0, {vocab_size})")
    return list(value)


def _validate_count_list(
    value: Any,
    name: str,
    *,
    max_count: int,
    source: str,
) -> List[int]:
    """Validate a pre-computed placeholder-count list (values in [0, max_count])."""
    if not isinstance(value, list):
        raise ValueError(f"{source}: {name} must be a list of integers")
    for index, token in enumerate(value):
        if isinstance(token, bool) or not isinstance(token, int):
            raise ValueError(f"{source}: {name}[{index}] must be an integer (bool is invalid)")
        if not 0 <= token <= max_count:
            raise ValueError(f"{source}: {name}[{index}]={token} is outside [0, {max_count}]")
    return list(value)


def validate_record(
    record: Any,
    config: LevTConfig,
    *,
    max_source_length: int,
    max_target_length: int,
    max_placeholder: int,
    source: str = "record",
) -> Dict[str, List[int]]:
    if not isinstance(record, dict):
        raise ValueError(f"{source}: row must be a JSON object")
    _ALLOWED_KEYS = {
        "src", "target", "initial",
        "y_ins", "p_star", "t_star", "y_ins_plh",
        "y_ins_rnd", "p_star_rnd", "t_star_rnd", "y_ins_plh_rnd",
    }
    # Silently ignore extra keys — training only uses the fields it needs
    missing = sorted({"src", "target"} - set(record))
    if missing:
        raise ValueError(f"{source}: missing required keys: {', '.join(missing)}")

    src = _validate_token_list(
        record["src"], "src", vocab_size=config.vocab_size,
        max_length=max_source_length, source=source,
    )
    target = _validate_token_list(
        record["target"], "target", vocab_size=config.vocab_size,
        max_length=max_target_length, source=source,
    )
    initial_value = record.get("initial", [config.bos_token_id, config.eos_token_id])
    initial = _validate_token_list(
        initial_value, "initial", vocab_size=config.vocab_size,
        max_length=max_target_length, source=source,
    )

    if config.pad_token_id in src:
        raise ValueError(f"{source}: src must not contain the padding token")
    if config.plh_token_id in src:
        raise ValueError(f"{source}: src must not contain the placeholder token")
    for name, sequence in (("target", target), ("initial", initial)):
        if len(sequence) < 2:
            raise ValueError(f"{source}: {name} must contain at least BOS and EOS")
        if sequence[0] != config.bos_token_id or sequence[-1] != config.eos_token_id:
            raise ValueError(f"{source}: {name} must start with BOS and end with EOS")
        forbidden = {config.pad_token_id, config.plh_token_id}
        for index, token in enumerate(sequence):
            if token in forbidden:
                raise ValueError(f"{source}: {name}[{index}] contains a reserved training token")
            if index not in (0, len(sequence) - 1) and token in {config.bos_token_id, config.eos_token_id}:
                raise ValueError(f"{source}: {name} contains an interior BOS/EOS token")

    result = {"src": src, "target": target, "initial": initial}

    # Pre-computed oracle fields (optional — skip BOS/EOS/PAD/PLH semantic checks)
    _PRECOMPUTED_TOKEN = [
        "y_ins", "y_ins_rnd",
        "t_star", "t_star_rnd",
        "y_ins_plh", "y_ins_plh_rnd",
    ]
    _PRECOMPUTED_COUNT = ["p_star", "p_star_rnd"]
    for name in _PRECOMPUTED_TOKEN:
        if name in record:
            result[name] = _validate_precomputed_token_list(
                record[name], name, vocab_size=config.vocab_size, source=source,
            )
    for name in _PRECOMPUTED_COUNT:
        if name in record:
            result[name] = _validate_count_list(
                record[name], name, max_count=max_placeholder, source=source,
            )
    return result


class JsonlDataset(Dataset):
    """In-memory validated JSONL rows with ``src``, ``target``, and ``initial``."""

    def __init__(
        self,
        path: str | Path,
        config: LevTConfig,
        *,
        max_source_length: int,
        max_target_length: int,
    ) -> None:
        self.path = Path(path)
        self.config = config
        self.rows: List[Dict[str, List[int]]] = []
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        raise ValueError(f"{self.path}:{line_number}: blank lines are not allowed")
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"{self.path}:{line_number}: invalid JSON: {exc.msg}") from exc
                    self.rows.append(validate_record(
                        row, config,
                        max_source_length=max_source_length,
                        max_target_length=max_target_length,
                        max_placeholder=config.max_placeholder,
                        source=f"{self.path}:{line_number}",
                    ))
        except OSError as exc:
            raise ValueError(f"failed to read {self.path}: {exc}") from exc
        if not self.rows:
            raise ValueError(f"{self.path}: dataset is empty")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        return self.rows[index]


def _pad_seq_first(sequences: Sequence[Sequence[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = len(sequences)
    max_length = max(len(sequence) for sequence in sequences)
    tokens = torch.full((max_length, batch_size), pad_id, dtype=torch.long)
    for batch_index, sequence in enumerate(sequences):
        tokens[:len(sequence), batch_index] = torch.tensor(sequence, dtype=torch.long)
    return tokens, tokens.eq(pad_id).transpose(0, 1)


class LevTCollator:
    def __init__(
        self,
        config: LevTConfig,
        *,
        max_source_length: int,
        max_target_length: int,
    ) -> None:
        self.config = config
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length

    def __call__(self, rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        if not rows:
            raise ValueError("cannot collate an empty batch")
        validated = [validate_record(
            row, self.config,
            max_source_length=self.max_source_length,
            max_target_length=self.max_target_length,
            max_placeholder=self.config.max_placeholder,
            source=f"batch row {index}",
        ) for index, row in enumerate(rows)]
        src, src_mask = _pad_seq_first([row["src"] for row in validated], self.config.pad_token_id)
        result: Dict[str, Any] = {
            "src_tokens": src,
            "src_padding_mask": src_mask,
            "initial": [torch.tensor(row["initial"], dtype=torch.long) for row in validated],
            "targets": [torch.tensor(row["target"], dtype=torch.long) for row in validated],
        }
        # Pass through pre-computed oracle fields if present in the data
        _PRECOMPUTED_FIELDS = [
            "y_ins", "p_star", "t_star", "y_ins_plh",
            "y_ins_rnd", "p_star_rnd", "t_star_rnd", "y_ins_plh_rnd",
        ]
        if validated and "y_ins" in validated[0]:
            for name in _PRECOMPUTED_FIELDS:
                result[name] = [
                    torch.tensor(row[name], dtype=torch.long) for row in validated
                ]
        return result
