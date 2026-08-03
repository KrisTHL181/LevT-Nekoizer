"""Metadata header spec for LevT JSONL dataset files.

A dataset file may begin with a single metadata line that declares the
file's format and whether rows are ``packed`` (concatenated segments) or
regular.  This lets training auto-detect the dataset type instead of
relying on a manually-set ``packed`` flag in ``train_config.json``.

Header line (must be the first line of the file)::

    {"__meta__": {"format": "levt-jsonl", "version": 1, "packed": true}}

The header line is *not* a data row: loaders skip it and never validate
it as ``src``/``target`` data.  Files without a header are treated as
legacy regular datasets (packed=false).

This module has no third-party imports so it can be imported cheaply by
both the ``levt`` package and the standalone ``scripts/`` tools.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

META_KEY = "__meta__"
DATASET_FORMAT = "levt-jsonl"
DATASET_FORMAT_VERSION = 1


@dataclass(frozen=True)
class DatasetMetadata:
    """Declared properties of a dataset file."""

    packed: bool


def parse_metadata_line(record: Any, source: str = "dataset") -> Optional[DatasetMetadata]:
    """Return parsed metadata if *record* is a metadata header, else ``None``.

    A header is a JSON object carrying the ``META_KEY`` key.  A malformed
    header (wrong format/version, non-boolean ``packed``, extra keys)
    raises ``ValueError`` with *source* context; a record without the key
    is a normal data row and yields ``None``.
    """
    if not isinstance(record, dict) or META_KEY not in record:
        return None
    meta = record[META_KEY]
    if not isinstance(meta, dict):
        raise ValueError(f"{source}: {META_KEY} must be a JSON object")
    allowed = {"format", "version", "packed"}
    unknown = sorted(set(meta) - allowed)
    if unknown:
        raise ValueError(f"{source}: unknown {META_KEY} keys: {', '.join(unknown)}")
    fmt = meta.get("format")
    if fmt != DATASET_FORMAT:
        raise ValueError(
            f"{source}: unsupported dataset format {fmt!r} (expected {DATASET_FORMAT!r})"
        )
    version = meta.get("version")
    if version != DATASET_FORMAT_VERSION:
        raise ValueError(
            f"{source}: unsupported dataset format version {version!r} "
            f"(expected {DATASET_FORMAT_VERSION})"
        )
    packed = meta.get("packed")
    if not isinstance(packed, bool):
        raise ValueError(f"{source}: {META_KEY} 'packed' must be a boolean")
    return DatasetMetadata(packed=packed)


def dataset_header(packed: bool) -> Dict[str, Any]:
    """Build the metadata header dict to emit as the first dataset line."""
    return {
        META_KEY: {
            "format": DATASET_FORMAT,
            "version": DATASET_FORMAT_VERSION,
            "packed": packed,
        }
    }


def read_dataset_metadata(path: str | Path) -> Optional[DatasetMetadata]:
    """Parse just the first line of *path* and return its metadata header.

    Returns ``None`` for legacy files that have no header (including an
    empty file or a first line that is a normal data row).  Raises
    ``ValueError`` if the file cannot be read or the header is malformed.
    """
    path = Path(path)
    try:
        with path.open("r", encoding="utf-8") as handle:
            first = handle.readline()
    except OSError as exc:
        raise ValueError(f"failed to read {path}: {exc}") from exc
    if not first.strip():
        return None
    try:
        record = json.loads(first)
    except json.JSONDecodeError:
        return None  # first line is a data row, not a header
    return parse_metadata_line(record, source=str(path))
