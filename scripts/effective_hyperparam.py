#!/usr/bin/env python3
"""Report effective training hyperparameters under sequence packing.

Sequence packing concatenates ~K unrelated examples into a single dataset row.
Training still uses ``batch_size`` *rows* per batch, so one batch actually sees
``batch_size * K`` real examples — the effective batch size — and one epoch
spans fewer optimizer steps.  This script reads the real dataset files and
reports the concrete numbers: examples per batch, examples per optimizer step,
optimizer steps per epoch, and how many epochs the step-indexed LR schedule
now covers, plus an unpacked baseline for comparison.

Because this repo normalizes each loss head by its valid-label count (not by
row count) and packs with segment-diagonal attention masks, the packed
gradient is equivalent (in distribution) to raising the batch size to the
total number of segments per step — see CLAUDE.md for the conditions.  This
script quantifies that equivalence with actual numbers.

Usage:
    python scripts/effective_hyperparam.py \
        --train-config train_config.json --model-config config.json

Optional:
    --data PATH     override the train dataset path (defaults to train_config)
    --batch-size N  override batch_size
    --limit N       only scan the first N rows (epoch/schedule numbers are
                    then for the scanned prefix, not the whole file)
    --json          emit machine-readable JSON on stdout
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import TypedDict, cast

# Make the repo root importable when run as ``python scripts/...``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from levt.config import LevTConfig, TrainConfig
from levt.dataset_meta import parse_metadata_line


class SegmentDist(TypedDict):
    min: int
    p50: int
    p90: int
    max: int


class DatasetInfo(TypedDict):
    train_path: str
    packed: bool | None
    rows: int
    total_segments: int
    src_tokens: int
    avg_segments_per_row: float
    std_segments_per_row: float
    segments_per_row: SegmentDist
    scanned: str


class BatchInfo(TypedDict):
    batch_size_rows: int
    gradient_accumulation_steps: int
    microbatches_per_epoch: int
    examples_per_batch_mean: float
    examples_per_batch_std: float
    examples_per_batch_min: int
    examples_per_batch_max: int


class PerEpochInfo(TypedDict):
    examples_per_epoch: int
    optimizer_steps_per_epoch: int
    examples_per_opt_step: float
    configured_epochs: int


class ScheduleInfo(TypedDict):
    warmup_steps: int
    max_training_steps: int
    data_in_schedule: float
    schedule_epochs: float
    warmup_data: float
    warmup_epochs: float


class UnpackedBaseline(TypedDict):
    examples_per_batch: int
    microbatches_per_epoch: int
    optimizer_steps_per_epoch: int
    examples_per_opt_step: int
    schedule_epochs: float
    packing_multiplier: float


class ValidationInfo(TypedDict):
    path: str
    rows: int
    segments: int
    segments_per_row: float
    packed: bool | None
    steps_per_epoch: int
    examples_per_step: float


class Report(TypedDict):
    dataset: DatasetInfo
    batch: BatchInfo
    per_epoch: PerEpochInfo
    schedule: ScheduleInfo
    unpacked_baseline: UnpackedBaseline
    validation: ValidationInfo | None


def _count_segments(tokens: list[int], eos: int, bos: int) -> int:
    """Number of segments in one row, via ``[EOS][BOS]`` boundaries.

    Matches ``levt.segment_mask.segment_ids``: a new segment starts at index 0
    or where ``tokens[i-1] == eos and tokens[i] == bos``.  Regular rows always
    have exactly one segment.
    """
    n = len(tokens)
    if n == 0:
        return 0
    count = 1
    for i in range(1, n):
        if tokens[i - 1] == eos and tokens[i] == bos:
            count += 1
    return count


class DatasetStats:
    path: Path
    rows: int
    segs: list[int]
    total_segments: int
    src_tokens: int
    packed: bool | None

    def __init__(self, path: Path, rows: int, segs: list[int], src_tokens: int,
                 packed: bool | None) -> None:
        self.path = path
        self.rows = rows
        self.segs = segs
        self.total_segments = sum(segs)
        self.src_tokens = src_tokens
        self.packed = packed  # None = legacy file with no header

    @property
    def avg_segments(self) -> float:
        return self.total_segments / self.rows if self.rows else 0.0

    @property
    def std_segments(self) -> float:
        return statistics.pstdev(self.segs) if self.rows > 1 else 0.0

    @property
    def seg_dist(self) -> SegmentDist:
        if not self.segs:
            return {"min": 0, "p50": 0, "p90": 0, "max": 0}
        s = sorted(self.segs)
        return {
            "min": s[0],
            "p50": s[len(s) // 2],
            "p90": s[int(0.9 * (len(s) - 1))],
            "max": s[-1],
        }


def scan_dataset(path: Path, bos: int, eos: int, limit: int = 0) -> DatasetStats:
    """Stream one dataset file, returning per-row segment counts.

    The optional metadata header is skipped (and its ``packed`` flag recorded).
    ``limit`` caps the number of data rows scanned (0 = all).
    """
    segs: list[int] = []
    src_tokens = 0
    packed: bool | None = None
    with path.open("r", encoding="utf-8") as handle:
        first = handle.readline()
        if not first.strip():
            raise ValueError(f"{path}: dataset is empty")
        try:
            first_row = cast(dict[str, object], json.loads(first))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:1: invalid JSON: {exc.msg}") from exc
        meta = parse_metadata_line(first_row, source=str(path))
        if meta is not None:
            packed = meta.packed
            first = None  # header consumed
        if first is not None:
            # Legacy file: first line is a data row.
            src = cast(list[int], first_row["src"])
            segs.append(_count_segments(src, eos, bos))
            src_tokens += len(src)
            if limit and len(segs) >= limit:
                return DatasetStats(path, len(segs), segs, src_tokens, packed)
        for line_no, line in enumerate(handle, 2):
            if not line.strip():
                raise ValueError(f"{path}:{line_no}: blank line")
            try:
                row = cast(dict[str, object], json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc.msg}") from exc
            src = cast(list[int], row["src"])
            segs.append(_count_segments(src, eos, bos))
            src_tokens += len(src)
            if limit and len(segs) >= limit:
                break
    if not segs:
        raise ValueError(f"{path}: dataset has no data rows")
    return DatasetStats(path, len(segs), segs, src_tokens, packed)


def _fmt(n: float, nd: int = 1) -> str:
    """Format a number with thousands separators and ``nd`` decimals."""
    return f"{n:,.{nd}f}"


def _fmt_int(n: float) -> str:
    return f"{int(round(n)):,}"


def compute(train: DatasetStats, cfg: TrainConfig, *, val: DatasetStats | None,
            batch_size_override: int | None = None) -> Report:
    B = batch_size_override if batch_size_override is not None else cfg.batch_size
    accum = cfg.gradient_accumulation_steps
    K = train.avg_segments
    sigK = train.std_segments

    # Effective batch (per microbatch / per optimizer step), assuming rows in
    # one batch are drawn roughly i.i.d.:  mean = B*K, std = sqrt(B)*sigma_K.
    eff_mean = B * K
    eff_std = math.sqrt(B) * sigK if B else 0.0

    microbatches = math.ceil(train.rows / B)
    steps_per_epoch = math.ceil(microbatches / accum)
    examples_per_step = eff_mean * accum

    # Schedule coverage in epoch units: the step-indexed schedule (warmup +
    # linear decay over max_steps) measured against optimizer steps per epoch.
    data_in_schedule = cfg.max_training_steps * examples_per_step
    warmup_data = cfg.warmup_steps * examples_per_step
    schedule_epochs = cfg.max_training_steps / steps_per_epoch if steps_per_epoch else 0.0
    warmup_epochs = cfg.warmup_steps / steps_per_epoch if steps_per_epoch else 0.0

    # Unpacked baseline: same real examples as individual rows.
    up_microbatches = math.ceil(train.total_segments / B)
    up_steps_per_epoch = math.ceil(up_microbatches / accum)
    up_schedule_epochs = (
        cfg.max_training_steps / up_steps_per_epoch if up_steps_per_epoch else 0.0
    )

    val_stats: ValidationInfo | None = None
    if val is not None:
        val_stats = {
            "path": str(val.path),
            "rows": val.rows,
            "segments": val.total_segments,
            "segments_per_row": val.avg_segments,
            "packed": val.packed,
            "steps_per_epoch": math.ceil(val.rows / B),  # validation has no accumulation
            "examples_per_step": B * val.avg_segments,
        }

    return {
        "dataset": {
            "train_path": str(train.path),
            "packed": train.packed,
            "rows": train.rows,
            "total_segments": train.total_segments,
            "src_tokens": train.src_tokens,
            "avg_segments_per_row": K,
            "std_segments_per_row": sigK,
            "segments_per_row": train.seg_dist,
            "scanned": "all",
        },
        "batch": {
            "batch_size_rows": B,
            "gradient_accumulation_steps": accum,
            "microbatches_per_epoch": microbatches,
            "examples_per_batch_mean": eff_mean,
            "examples_per_batch_std": eff_std,
            "examples_per_batch_min": B * train.seg_dist["min"],
            "examples_per_batch_max": B * train.seg_dist["max"],
        },
        "per_epoch": {
            "examples_per_epoch": train.total_segments,
            "optimizer_steps_per_epoch": steps_per_epoch,
            "examples_per_opt_step": examples_per_step,
            "configured_epochs": cfg.epochs,
        },
        "schedule": {
            "warmup_steps": cfg.warmup_steps,
            "max_training_steps": cfg.max_training_steps,
            "data_in_schedule": data_in_schedule,
            "schedule_epochs": schedule_epochs,
            "warmup_data": warmup_data,
            "warmup_epochs": warmup_epochs,
        },
        "unpacked_baseline": {
            "examples_per_batch": B,
            "microbatches_per_epoch": up_microbatches,
            "optimizer_steps_per_epoch": up_steps_per_epoch,
            "examples_per_opt_step": B * accum,
            "schedule_epochs": up_schedule_epochs,
            "packing_multiplier": K,
        },
        "validation": val_stats,
    }


def render(report: Report) -> str:
    """Render the report as a readable aligned table."""
    d = report["dataset"]
    b = report["batch"]
    pe = report["per_epoch"]
    sc = report["schedule"]
    ub = report["unpacked_baseline"]
    dist = d["segments_per_row"]

    L = 46  # label column width
    lines: list[str] = []
    a = lines.append
    a("effective training hyperparameters (packed vs regular)")
    a("=" * (L + 32))
    a("")
    a("-- dataset".ljust(L) + "--")
    a(f"{'train_data':<{L}}{d['train_path']}")
    a(f"{'packed':<{L}}{'True' if d['packed'] else 'False'}"
      + f"{' (dataset header)' if d['packed'] is not None else ' (no header)'}")
    a(f"{'rows':<{L}}{_fmt_int(d['rows'])}")
    a(f"{'segments (real examples)':<{L}}{_fmt_int(d['total_segments'])}")
    a(f"{'avg segments / row':<{L}}{_fmt(d['avg_segments_per_row'])}  "
      + f"(sd {_fmt(d['std_segments_per_row'])})")
    a(f"{'segments/row dist':<{L}}min={dist['min']}  p50={dist['p50']}  "
      + f"p90={dist['p90']}  max={dist['max']}")
    a(f"{'src tokens':<{L}}{_fmt_int(d['src_tokens'])}")
    a("")
    a("-- per batch / per step --")
    a(f"{'batch_size (rows)':<{L}}{b['batch_size_rows']}")
    a(f"{'gradient accumulation':<{L}}{b['gradient_accumulation_steps']}")
    a(f"{'microbatches / epoch':<{L}}{_fmt_int(b['microbatches_per_epoch'])}"
      + f"  = ceil({_fmt_int(d['rows'])} / {b['batch_size_rows']})")
    a(f"{'examples / batch':<{L}}{_fmt(b['examples_per_batch_mean'])}"
      + f" ± {_fmt(b['examples_per_batch_std'])}"
      + f"  (≈ effective batch size, {b['batch_size_rows']} × {_fmt(d['avg_segments_per_row'])})")
    a(f"{'examples / batch range':<{L}}[{_fmt_int(b['examples_per_batch_min'])}, "
      + f"{_fmt_int(b['examples_per_batch_max'])}]"
      + f"  (whole batch of all-min / all-max rows)")
    a(f"{'examples / optimizer step':<{L}}{_fmt(pe['examples_per_opt_step'])}"
      + f"  = examples/batch × {b['gradient_accumulation_steps']}")
    a("")
    a("-- per epoch --")
    a(f"{'examples / epoch':<{L}}{_fmt_int(pe['examples_per_epoch'])}")
    a(f"{'optimizer steps / epoch':<{L}}{_fmt_int(pe['optimizer_steps_per_epoch'])}"
      + f"  = ceil({_fmt_int(b['microbatches_per_epoch'])} / {b['gradient_accumulation_steps']})")
    a(f"{'configured epochs':<{L}}{pe['configured_epochs']}")
    a("")
    a("-- LR schedule (step-indexed) --")
    a(f"{'warmup_steps':<{L}}{_fmt_int(sc['warmup_steps'])}")
    a(f"{'max_training_steps':<{L}}{_fmt_int(sc['max_training_steps'])}")
    a(f"{'data seen by schedule':<{L}}{_fmt(sc['data_in_schedule'])} examples"
      + f"  = {_fmt_int(sc['max_training_steps'])} × {_fmt(pe['examples_per_opt_step'])}")
    a(f"{'schedule coverage':<{L}}{_fmt(sc['schedule_epochs'])} epochs"
      + f"  = max_steps ÷ optimizer steps/epoch")
    a(f"{'warmup coverage':<{L}}{_fmt(sc['warmup_epochs'])} epochs")
    a("")
    a("-- unpacked baseline (same examples, no packing) --")
    a(f"{'examples / batch':<{L}}{ub['examples_per_batch']}")
    a(f"{'optimizer steps / epoch':<{L}}{_fmt_int(ub['optimizer_steps_per_epoch'])}"
      + f"  = ceil({_fmt_int(d['total_segments'])} / {ub['examples_per_batch']})")
    a(f"{'schedule coverage':<{L}}{_fmt(ub['schedule_epochs'])} epochs")
    a(f"{'packing multiplier':<{L}}{_fmt(ub['packing_multiplier'])}×"
      + f"  (= avg segments / row: samples per step multiply by this)")
    a("")
    a("-- schedule per epoch (packed data) --")
    a(f"{'1 epoch':<{L}}{_fmt_int(pe['optimizer_steps_per_epoch'])} optimizer steps")
    a(f"{'to run N epochs':<{L}}max_training_steps = round(N × "
      + f"{_fmt_int(pe['optimizer_steps_per_epoch'])})")

    if report["validation"]:
        v = report["validation"]
        a("")
        a("-- validation --")
        a(f"{'validation_data':<{L}}{v['path']}")
        a(f"{'rows':<{L}}{_fmt_int(v['rows'])}")
        a(f"{'segments':<{L}}{_fmt_int(v['segments'])}"
          + f"  ({'packed' if v['packed'] else 'regular'})")
        a(f"{'steps / epoch':<{L}}{_fmt_int(v['steps_per_epoch'])}"
          + f"  (no gradient accumulation)")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-config", default="train_config.json",
                        help="train_config.json path (default: train_config.json)")
    parser.add_argument("--model-config", default="config.json",
                        help="config.json path (default: config.json)")
    parser.add_argument("--data", default=None,
                        help="override the train dataset path")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="override batch_size")
    parser.add_argument("--limit", type=int, default=0,
                        help="scan only the first N rows (0 = all)")
    parser.add_argument("--json", action="store_true",
                        help="emit machine-readable JSON on stdout")
    args = parser.parse_args()

    model_cfg = LevTConfig.from_json(args.model_config)
    train_cfg = TrainConfig.from_json(args.train_config)
    train_path = Path(args.data or train_cfg.train_data)

    print(f"[scan] {train_path} ...", file=sys.stderr, flush=True)
    train = scan_dataset(
        train_path, model_cfg.bos_token_id, model_cfg.eos_token_id, limit=args.limit,
    )
    val: DatasetStats | None = None
    if train_cfg.validation_data:
        val_path = Path(train_cfg.validation_data)
        print(f"[scan] {val_path} ...", file=sys.stderr, flush=True)
        val = scan_dataset(
            val_path, model_cfg.bos_token_id, model_cfg.eos_token_id, limit=args.limit,
        )

    report = compute(train, train_cfg, val=val, batch_size_override=args.batch_size)

    if args.limit:
        report["dataset"]["scanned"] = "prefix"
        print(f"[warn] --limit {args.limit}: epoch/schedule numbers are for the "
              f"scanned prefix only", file=sys.stderr, flush=True)

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(render(report))


if __name__ == "__main__":
    main()
