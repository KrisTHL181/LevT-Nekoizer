#!/usr/bin/env python3
"""Training profiler — diagnoses GPU under-utilisation with torch.profiler.

Run on the remote machine:
    python profile_train.py --model-config config.json --train-config train_config.json

It warms up (torch.compile + cuDNN autotune), then profiles a fixed window of
training steps.  Outputs:

    profile_logs/
    ├── trace.json           chrome://tracing  (open in chrome://tracing or Perfetto)
    ├── trace.json.gz         compressed copy for download
    ├── memory_timeline.csv   per-step VRAM
    ├── kernel_summary.csv    top-30 CUDA kernels by total time
    └── profile_report.txt    human-readable summary
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gzip
import json
import math
import os
import random
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.profiler import (
    ProfilerActivity,
    profile,
    record_function,
    schedule,
    tensorboard_trace_handler,
)
from torch.utils.data import DataLoader

# -- project imports ----------------------------------------------------------
from levt._levenshtein_ops import verify_cpp_extension
from levt.checkpoint import capture_rng_state, restore_rng_state
from levt.config import LevTConfig, TrainConfig
from levt.data import JsonlDataset, LevTCollator
from levt.embeddings import import_hf_embeddings
from levt.fast_muon import FastMuon
from levt.model import LevTModel
from levt.perf_optims import accelerated, enable_all, prealloc_model_grads
from levt.trainer import DualPolicyTrainer


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-config", default="config.json")
    p.add_argument("--train-config", default="train_config.json")
    p.add_argument("--warmup-steps", type=int, default=15,
                   help="training steps before profiling starts (default: 15)")
    p.add_argument("--profile-steps", type=int, default=30,
                   help="training steps to profile (default: 30)")
    p.add_argument("--output-dir", default="profile_logs")
    p.add_argument("--skip-cuda-profile", action="store_true",
                   help="skip CUDA kernel trace (faster, less detail)")
    p.add_argument("--record-shapes", action="store_true",
                   help="record tensor shapes in trace (large files!)")
    p.add_argument("--record-stack", action="store_true",
                   help="record Python call stacks (even larger files!)")
    p.add_argument("--dump-model-graph", action="store_true",
                   help="export compiled FX graph as SVG before profiling")
    p.add_argument("--num-threads", type=int, default=8,
                   help="limit PyTorch CPU parallelism (default: 8)")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers (mirror train.py)
# ═══════════════════════════════════════════════════════════════════════════════

def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(
    path: str, model_cfg: LevTConfig, train_cfg: TrainConfig,
    *, shuffle: bool, shuffle_seed: Optional[int] = None,
) -> DataLoader:
    dataset = JsonlDataset(
        path, model_cfg,
        max_source_length=train_cfg.max_source_length,
        max_target_length=train_cfg.max_target_length,
    )
    collator = LevTCollator(
        model_cfg,
        max_source_length=train_cfg.max_source_length,
        max_target_length=train_cfg.max_target_length,
    )
    generator = None
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(train_cfg.seed if shuffle_seed is None else shuffle_seed)
    return DataLoader(
        dataset, batch_size=train_cfg.batch_size, shuffle=shuffle,
        generator=generator, num_workers=train_cfg.num_workers,
        collate_fn=collator, pin_memory=True,
    )


def autocast_context(device: torch.device, amp_dtype: str):
    if amp_dtype == "none":
        return contextlib.nullcontext()
    if device.type == "cuda":
        dtype = torch.float16 if amp_dtype == "float16" else torch.bfloat16
        return torch.autocast(device_type="cuda", dtype=dtype)
    return torch.autocast(device_type="cpu", dtype=torch.bfloat16)


def make_scaler(device: torch.device, amp_dtype: str):
    enabled = device.type == "cuda" and amp_dtype == "float16"
    return torch.amp.GradScaler("cuda", enabled=enabled)


def scheduler_factor(step: int, warmup: int, total: int) -> float:
    if warmup and step < warmup:
        return float(step + 1) / float(warmup)
    remaining = max(0, total - step)
    decay_steps = max(1, total - warmup)
    return float(remaining) / float(decay_steps)


def _linear_weight_ids(model: nn.Module) -> set[int]:
    ids: set[int] = set()
    for module in model.modules():
        if isinstance(module, nn.Linear):
            ids.add(id(module.weight))
    return ids


def build_optimizers(model, train_cfg):
    linear_ids = _linear_weight_ids(model)
    muon_params, adamw_params = [], []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        (muon_params if id(param) in linear_ids else adamw_params).append(param)
    adamw = accelerated.AdamW(
        adamw_params, lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay, betas=train_cfg.betas, eps=train_cfg.eps,
    )
    muon = FastMuon(
        muon_params, lr=train_cfg.muon_lr,
        weight_decay=train_cfg.muon_weight_decay, momentum=train_cfg.muon_momentum,
        nesterov=train_cfg.muon_nesterov, ns_steps=train_cfg.muon_ns_steps,
    )
    return adamw, muon


# ═══════════════════════════════════════════════════════════════════════════════
# Wall-clock micro-benchmark helper
# ═══════════════════════════════════════════════════════════════════════════════

class StepTimer:
    """Record per-phase wall-clock times for every profiled step."""

    def __init__(self):
        self.phases: Dict[str, List[float]] = defaultdict(list)
        self._marks: Dict[str, float] = {}

    def mark(self, name: str):
        self._marks[name] = time.perf_counter()

    def record(self, phase: str, start_mark: str, end_mark: str = "now"):
        end = time.perf_counter() if end_mark == "now" else self._marks[end_mark]
        self.phases[phase].append(end - self._marks[start_mark])

    def summarise(self) -> Dict[str, Dict[str, float]]:
        out = {}
        for phase, times in self.phases.items():
            s = sorted(times)
            n = len(s)
            out[phase] = {
                "count": n,
                "total": sum(s),
                "mean": sum(s) / n,
                "median": s[n // 2],
                "min": s[0],
                "max": s[-1],
                "p90": s[int(n * 0.9)],
                "p95": s[int(n * 0.95)],
                "p99": s[int(n * 0.99)],
            }
        return out


# ═══════════════════════════════════════════════════════════════════════════════
# Report generation
# ═══════════════════════════════════════════════════════════════════════════════

def _ms(v: float) -> str:
    return f"{v * 1000:7.1f} ms"


def generate_report(
    timer: StepTimer,
    gpu_metrics: List[Dict],
    memory_mb: List[float],
    out_dir: Path,
    prof: Optional[profile] = None,
) -> str:
    lines: List[str] = []
    sep = "─" * 70
    W = 60

    lines.append("=" * W)
    lines.append("  TRAINING PROFILE REPORT")
    lines.append("=" * W)

    # ── per-phase wall-clock ──────────────────────────────────────────────
    lines.append(f"\n{sep}")
    lines.append("  Per-phase wall-clock timing")
    lines.append(sep)

    total_mean = 0.0
    stats = timer.summarise()
    for phase in ("data", "prepare", "forward", "backward", "adamw", "muon", "step"):
        if phase not in stats:
            continue
        s = stats[phase]
        lines.append(f"\n  [{phase}]  ({s['count']} steps)")
        lines.append(f"    Mean:   {_ms(s['mean'])}")
        lines.append(f"    Median: {_ms(s['median'])}")
        lines.append(f"    Min:    {_ms(s['min'])}")
        lines.append(f"    Max:    {_ms(s['max'])}")
        lines.append(f"    P90:    {_ms(s['p90'])}")
        lines.append(f"    P95:    {_ms(s['p95'])}")
        lines.append(f"    P99:    {_ms(s['p99'])}")
        if phase not in ("data", "step"):
            total_mean += s["mean"]

    # ── GPU utilisation ───────────────────────────────────────────────────
    if gpu_metrics:
        lines.append(f"\n{sep}")
        lines.append("  GPU utilisation (nvidia-smi samples)")
        lines.append(sep)
        utils = [m["gpu_util"] for m in gpu_metrics]
        mems = [m["mem_used"] / 1024 for m in gpu_metrics]  # MiB → GiB
        temps = [m["temp"] for m in gpu_metrics]
        powers = [m["power"] for m in gpu_metrics]

        lines.append(f"  GPU util:  mean={sum(utils)/len(utils):.1f}%, "
                     f"max={max(utils):.1f}%, min={min(utils):.1f}%")
        lines.append(f"  GPU mem:   mean={sum(mems)/len(mems):.1f} GiB, "
                     f"max={max(mems):.1f} GiB")
        lines.append(f"  GPU temp:  mean={sum(temps)/len(temps):.0f}°C, "
                     f"max={max(temps):.0f}°C")
        lines.append(f"  GPU power: mean={sum(powers)/len(powers):.0f}W, "
                     f"max={max(powers):.0f}W")

    # ── VRAM ──────────────────────────────────────────────────────────────
    if memory_mb:
        lines.append(f"\n{sep}")
        lines.append("  VRAM (torch.cuda)")
        lines.append(sep)
        lines.append(f"  Peak allocated:  {max(memory_mb) / 1024:.2f} GiB")
        lines.append(f"  Peak reserved:   {max(memory_mb) / 1024 * 1.05:.2f} GiB (est.)")

    # ── throughput ────────────────────────────────────────────────────────
    if "step" in stats:
        s = stats["step"]
        steps_per_sec = 1.0 / s["mean"]
        samples_per_sec = steps_per_sec * 8  # batch_size=8
        lines.append(f"\n{sep}")
        lines.append("  Throughput")
        lines.append(sep)
        lines.append(f"  {steps_per_sec:.1f} steps/s  →  {samples_per_sec:.0f} samples/s")

    # ── proportion breakdown ──────────────────────────────────────────────
    if total_mean > 0 and "step" in stats:
        step_mean = stats["step"]["mean"]
        lines.append(f"\n{sep}")
        lines.append(f"  Time breakdown (of {_ms(step_mean)} total)")
        lines.append(sep)
        accounted = 0.0
        for phase in ("forward", "backward", "adamw", "muon", "prepare"):
            if phase in stats:
                pct = stats[phase]["mean"] / step_mean * 100
                lines.append(f"    {phase:<12s} {_ms(stats[phase]['mean'])}  ({pct:.1f}%)")
                accounted += pct
        other = 100.0 - accounted
        if other > 0.5:
            lines.append(f"    {'other':<12s} {'—':>9s}  ({other:.1f}%)")

    # ── CUDA kernel summary ───────────────────────────────────────────────
    if prof is not None:
        lines.append(f"\n{sep}")
        lines.append("  Top CUDA kernels by total time")
        lines.append(sep)
        kernel_rows = _extract_kernel_table(prof)
        for i, row in enumerate(kernel_rows[:25], 1):
            lines.append(
                f"  {i:2d}. {row['name'][:55]:55s} "
                f"{row['total']*1000:8.1f}ms  ({row['pct']:5.1f}%)  "
                f"×{row['count']}"
            )

        # ── CPU-GPU idle gap ────────────────────────────────────────────
        cpu_total, gpu_total, gap = _cpu_gpu_gap(prof)
        if gap is not None:
            lines.append(f"\n{sep}")
            lines.append("  CPU / GPU idle gap")
            lines.append(sep)
            lines.append(f"  CPU time in step window:  {cpu_total * 1000:.0f} ms")
            lines.append(f"  GPU time in step window:  {gpu_total * 1000:.0f} ms")
            lines.append(f"  Gap (GPU idle waiting):  {gap * 1000:.0f} ms  ({gap/cpu_total*100:.1f}% of CPU time)")

    lines.append(f"\n{sep}")
    lines.append("  Output files")
    lines.append(sep)
    lines.append(f"  {out_dir / 'profile_report.txt'}")
    lines.append(f"  {out_dir / 'trace.json.gz'}        ← download & open in chrome://tracing")
    lines.append(f"  {out_dir / 'kernel_summary.csv'}")
    lines.append(f"  {out_dir / 'memory_timeline.csv'}")

    return "\n".join(lines)


def _extract_kernel_table(prof: profile) -> List[Dict]:
    """Pull CUDA kernel events from the profiler and aggregate by name.

    Uses ``device_time_total`` (PyTorch ≥2.10, µs) — the unified
    CPU/CUDA/XPU attribute that replaced the deprecated ``cuda_time_total``.
    """
    rows: Dict[str, Dict] = {}
    events = prof.key_averages(group_by_input_shape=False)
    for evt in events:
        if evt.count == 0:
            continue
        dev_us = getattr(evt, "device_time_total", 0.0) or 0.0
        rows[evt.key] = {
            "name": evt.key,
            "count": evt.count,
            "total": dev_us / 1e6,  # µs → s
            "pct": 0.0,
            "mean": dev_us / 1e6 / evt.count if evt.count else 0.0,
        }
    total_s = sum(r["total"] for r in rows.values())
    for r in rows.values():
        r["pct"] = r["total"] / total_s * 100 if total_s else 0
    return sorted(rows.values(), key=lambda r: r["total"], reverse=True)


def _cpu_gpu_gap(prof: profile) -> Tuple[float, float, Optional[float]]:
    """Estimate how much wall-clock time the GPU sits idle inside steps."""
    try:
        overall = prof.key_averages(group_by_input_shape=False)
        cpu_total = sum(e.cpu_time_total for e in overall) / 1e6  # µs → s
        gpu_total = sum(getattr(e, "device_time_total", 0.0) or 0.0 for e in overall) / 1e6
        if cpu_total <= 0:
            return cpu_total, gpu_total, None
        gap = cpu_total - gpu_total
        return cpu_total, gpu_total, max(0, gap)
    except Exception:
        return 0, 0, None


# ═══════════════════════════════════════════════════════════════════════════════
# GPU metric sampler (background, minimal overhead)
# ═══════════════════════════════════════════════════════════════════════════════

def sample_gpu_metrics() -> Optional[Dict]:
    """Query nvidia-smi for instant GPU stats.  Returns None on failure."""
    import subprocess
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            timeout=2,
        ).decode().strip()
        util, mem, temp, power = out.split(", ")
        return {
            "timestamp": time.time(),
            "gpu_util": int(util),
            "mem_used": int(mem),  # MiB
            "temp": int(temp),
            "power": float(power),
        }
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# Main profile loop
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    model_cfg = LevTConfig.from_json(args.model_config)
    train_cfg = TrainConfig.from_json(args.train_config)
    device = resolve_device(train_cfg.device)
    seed_everything(train_cfg.seed)

    # ── limit CPU parallelism ───────────────────────────────────────────
    # Must set env vars BEFORE enable_all() — it calls set_num_interop_threads
    # which can only be invoked once.  We also override set_num_threads after.
    os.environ["OMP_NUM_THREADS"] = str(args.num_threads)
    os.environ["MKL_NUM_THREADS"] = str(args.num_threads)
    enable_all()
    torch.set_num_threads(args.num_threads)
    print(f"CPU threads limited to {args.num_threads} (detected {os.cpu_count()} cores)")

    # Check C++ extension
    status = verify_cpp_extension()
    if not status.available:
        print(f"WARNING: C++ extension unavailable — {status.error}", flush=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir.resolve()}")
    print(f"Device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'N/A'}")

    # ── build model ──────────────────────────────────────────────────────
    print("Building model ...", flush=True)
    model = LevTModel(model_cfg)
    import_hf_embeddings(
        model, train_cfg.hf_model_name_or_path,
        local_files_only=train_cfg.local_files_only,
        trust_remote_code=train_cfg.trust_remote_code,
        dtype=train_cfg.hf_dtype,
    )
    model.to(device)
    model.shared_embedding.weight.requires_grad_(not train_cfg.freeze_embeddings)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params:,} total, {trainable_params:,} trainable")

    prealloc_model_grads(model)
    model = torch.compile(model, fullgraph=True)
    trainer = DualPolicyTrainer(model, model_cfg, train_cfg.policy)
    adamw, muon = build_optimizers(model, train_cfg)
    adamw_scheduler = LambdaLR(adamw, lambda s: scheduler_factor(s, train_cfg.warmup_steps, train_cfg.max_training_steps))
    muon_scheduler = LambdaLR(muon, lambda s: scheduler_factor(s, train_cfg.warmup_steps, train_cfg.max_training_steps))
    scaler = make_scaler(device, train_cfg.amp_dtype)

    # ── data ─────────────────────────────────────────────────────────────
    train_loader = make_loader(train_cfg.train_data, model_cfg, train_cfg, shuffle=True, shuffle_seed=train_cfg.seed)
    print(f"Train batches per epoch: {len(train_loader)}")
    train_iter = iter(train_loader)

    # ── dump compiled graph (optional) ────────────────────────────────────
    if args.dump_model_graph and hasattr(torch, "export"):
        try:
            print("Exporting compiled FX graph ...", flush=True)
            batch = next(train_iter)
            prepared = trainer.prepare_batch(batch)
            ep = torch.export.export(model, (prepared,))
            graph_path = out_dir / "model_graph.svg"
            with open(graph_path, "w") as f:
                f.write(ep.graph)
            print(f"  Graph saved to {graph_path}")
        except Exception as e:
            print(f"  Graph export failed (non-critical): {e}")

    # ── warmup (torch.compile + cuDNN autotune) ─────────────────────────
    total_warmup = args.warmup_steps
    print(f"\nWarming up ({total_warmup} steps) ...", flush=True)
    warmup_start = time.perf_counter()
    for w in range(total_warmup):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        prepared = trainer.prepare_batch(batch)
        # Real forward/backward/step
        if train_cfg.amp_dtype != "none":
            with autocast_context(device, train_cfg.amp_dtype):
                sums, counts = trainer.loss_sums_and_counts(prepared)
        else:
            sums, counts = trainer.loss_sums_and_counts(prepared)
        loss = sum(sums[n] / counts[n] if counts[n] else 0.0 for n in sums)
        scaler.scale(loss).backward()
        scaler.unscale_(adamw)
        scaler.unscale_(muon)
        if train_cfg.max_grad_norm > 0:
            accelerated.clip_grad_norm_(model.parameters(), train_cfg.max_grad_norm)
        scaler.step(adamw)
        scaler.step(muon)
        scaler.update()
        adamw.zero_grad(set_to_none=True)
        muon.zero_grad(set_to_none=True)
        adamw_scheduler.step()
        muon_scheduler.step()

        if (w + 1) % 5 == 0:
            print(f"  warmup {w + 1}/{total_warmup}", flush=True)
    warmup_elapsed = time.perf_counter() - warmup_start
    print(f"Warmup complete in {warmup_elapsed:.1f}s", flush=True)

    # ── profiled steps ───────────────────────────────────────────────────
    total_profile = args.profile_steps
    print(f"\nProfiling {total_profile} steps ...", flush=True)

    # Build profiler schedule: skip 0 (already warmed up), wait 0, then
    # capture every step in the profiling window.
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda" and not args.skip_cuda_profile:
        activities.append(ProfilerActivity.CUDA)

    profiler_schedule = schedule(
        skip_first=0, wait=0, warmup=0, active=total_profile, repeat=1,
    )

    # Decide trace handler
    trace_dir = out_dir / "detailed_trace"
    trace_dir.mkdir(exist_ok=True)

    prof = profile(
        activities=activities,
        schedule=profiler_schedule,
        on_trace_ready=tensorboard_trace_handler(str(trace_dir), use_gzip=True),
        record_shapes=args.record_shapes,
        with_stack=args.record_stack,
    )

    timer = StepTimer()
    gpu_metrics: List[Dict] = []
    memory_mb: List[float] = []

    # Start continuous GPU sampling in the background (via subprocess)
    gpu_sample_proc = None
    if device.type == "cuda":
        import subprocess
        gpu_csv = out_dir / "gpu_metrics.csv"
        try:
            gpu_sample_proc = subprocess.Popen(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw",
                 "--format=csv,noheader,nounits", "--loop-ms=200"],
                stdout=open(gpu_csv, "w"),
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            pass

    prof.start()
    prof_start_wall = time.perf_counter()

    for p_idx in range(total_profile):
        timer.mark("step_start")

        # ── data ────────────────────────────────────────────────────
        timer.mark("data_start")
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        timer.record("data", "data_start")

        # ── prepare ─────────────────────────────────────────────────
        timer.mark("prepare_start")
        with record_function("prepare_batch"):
            prepared = trainer.prepare_batch(batch)
        timer.record("prepare", "prepare_start")

        # ── forward ─────────────────────────────────────────────────
        timer.mark("fwd_start")
        with record_function("forward"):
            if train_cfg.amp_dtype != "none":
                with autocast_context(device, train_cfg.amp_dtype):
                    sums, counts = trainer.loss_sums_and_counts(prepared)
            else:
                sums, counts = trainer.loss_sums_and_counts(prepared)
        timer.record("forward", "fwd_start")

        # ── backward ────────────────────────────────────────────────
        timer.mark("bwd_start")
        with record_function("backward"):
            loss = sum(sums[n] / counts[n] if counts[n] else 0.0 for n in sums)
            scaler.scale(loss).backward()
        timer.record("backward", "bwd_start")

        # ── grad ops ────────────────────────────────────────────────
        timer.mark("grad_start")
        with record_function("unscale+clip"):
            scaler.unscale_(adamw)
            scaler.unscale_(muon)
            if train_cfg.max_grad_norm > 0:
                accelerated.clip_grad_norm_(model.parameters(), train_cfg.max_grad_norm)
        timer.record("grad_ops", "grad_start")

        # ── AdamW step ──────────────────────────────────────────────
        timer.mark("adamw_start")
        with record_function("adamw_step"):
            scaler.step(adamw)
            scaler.update()
        timer.record("adamw", "adamw_start")

        # ── Muon step ───────────────────────────────────────────────
        timer.mark("muon_start")
        with record_function("muon_step"):
            scaler.step(muon)
        timer.record("muon", "muon_start")

        # ── zero grad + scheduler ───────────────────────────────────
        timer.mark("zero_start")
        with record_function("zero_grad"):
            adamw.zero_grad(set_to_none=True)
            muon.zero_grad(set_to_none=True)
        adamw_scheduler.step()
        muon_scheduler.step()
        timer.record("zero_grad", "zero_start")

        timer.record("step", "step_start")
        prof.step()

        # ── sample VRAM ─────────────────────────────────────────────
        if device.type == "cuda":
            memory_mb.append(torch.cuda.max_memory_allocated() / 1024 / 1024)
            torch.cuda.reset_peak_memory_stats()

        if (p_idx + 1) % 10 == 0:
            elapsed = time.perf_counter() - prof_start_wall
            sps = (p_idx + 1) / elapsed
            print(f"  profiled {p_idx + 1}/{total_profile} steps  ({sps:.1f} steps/s)", flush=True)

    prof.stop()
    prof_wall = time.perf_counter() - prof_start_wall

    # Stop background GPU sampling
    if gpu_sample_proc is not None:
        gpu_sample_proc.terminate()
        gpu_sample_proc.wait(timeout=5)

    # Load GPU metrics from CSV
    gpu_csv_path = out_dir / "gpu_metrics.csv"
    if gpu_csv_path.exists():
        with open(gpu_csv_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 4:
                    try:
                        gpu_metrics.append({
                            "gpu_util": int(parts[0]),
                            "mem_used": int(parts[1]),
                            "temp": int(parts[2]),
                            "power": float(parts[3]) if parts[3] else 0.0,
                        })
                    except (ValueError, IndexError):
                        continue

    print(f"\nProfiling complete: {total_profile} steps in {prof_wall:.1f}s "
          f"({total_profile / prof_wall:.1f} steps/s)", flush=True)

    # ── generate report ──────────────────────────────────────────────────
    report = generate_report(timer, gpu_metrics, memory_mb, out_dir, prof)
    report_path = out_dir / "profile_report.txt"
    with open(report_path, "w") as f:
        f.write(report)

    # ── kernel summary CSV ───────────────────────────────────────────────
    kernel_rows = _extract_kernel_table(prof) if prof is not None else []
    kernel_csv = out_dir / "kernel_summary.csv"
    with open(kernel_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["rank", "name", "calls", "total_ms", "mean_ms", "pct"])
        w.writeheader()
        for i, row in enumerate(kernel_rows[:50], 1):
            w.writerow({
                "rank": i, "name": row["name"],
                "calls": row["count"],
                "total_ms": f"{row['total'] * 1000:.3f}",
                "mean_ms": f"{row['mean'] * 1000:.4f}",
                "pct": f"{row['pct']:.1f}",
            })

    # ── memory timeline CSV ──────────────────────────────────────────────
    mem_csv = out_dir / "memory_timeline.csv"
    with open(mem_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "peak_allocated_mb"])
        for i, mb in enumerate(memory_mb):
            w.writerow([i, f"{mb:.1f}"])

    # ── consolidate chrome trace ─────────────────────────────────────────
    # torch.profiler writes one JSON per step to detailed_trace/.
    # Concatenate them into a single trace.json for easy download.
    trace_files = sorted(trace_dir.glob("*.json"))
    if trace_files:
        merged = out_dir / "trace.json"
        _merge_traces(trace_files, merged)
        gz_path = out_dir / "trace.json.gz"
        with open(merged, "rb") as src, gzip.open(gz_path, "wb", compresslevel=6) as dst:
            shutil.copyfileobj(src, dst)

    print(f"profile_data_written to {out_dir.resolve()}", flush=True)


def _merge_traces(json_files: List[Path], output: Path):
    """Merge per-step chrome-trace JSON files into a single array."""
    merged_events: List[Dict] = []
    for path in json_files:
        try:
            with open(path) as f:
                data = json.load(f)
            if isinstance(data, list):
                merged_events.extend(data)
            elif isinstance(data, dict) and "traceEvents" in data:
                merged_events.extend(data["traceEvents"])
        except (json.JSONDecodeError, OSError):
            continue
    with open(output, "w") as f:
        json.dump({"traceEvents": merged_events}, f)


if __name__ == "__main__":
    main()
