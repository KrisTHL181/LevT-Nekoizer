#!/usr/bin/env python3
"""Single-machine training entry point for the Levenshtein Transformer."""

from __future__ import annotations

import argparse
import contextlib
import csv
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from levt._levenshtein_ops import verify_cpp_extension
from levt.checkpoint import (
    capture_rng_state,
    cleanup_checkpoints,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
)
from levt.config import LevTConfig, TrainConfig
from levt.data import JsonlDataset, LevTCollator
from levt.embeddings import import_hf_embeddings
from levt.model import LevTModel
from levt.perf_optims import accelerated, enable_all, prealloc_model_grads
from levt.progress import HAS_RICH, TrainingDisplay
from levt.trainer import DualPolicyTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", default="config.json")
    parser.add_argument("--train-config", default="train_config.json")
    parser.add_argument("--resume", default=None, help="checkpoint path; overrides train_config")
    parser.add_argument(
        "--resume-csv", default=None,
        help="append to this CSV progress file instead of overwriting it",
    )
    parser.add_argument(
        "--bypass-config-check", action="store_true",
        help="skip config mismatch check when resuming from checkpoint",
    )
    return parser.parse_args()


def _check_cpp_extension() -> None:
    """Verify the C++ Levenshtein extension works before training starts.

    A silent fallback to pure Python costs ~3.5× throughput and wastes GPU
    resources.  This check runs a real smoke test and exits with a clear
    diagnostic if the extension is unavailable.
    """
    import os
    import sys

    status = verify_cpp_extension()
    if status.available:
        return  # all good

    separator = "=" * 72
    print(f"\n{separator}", file=sys.stderr)
    print("  FATAL — C++ Levenshtein extension is NOT available", file=sys.stderr)
    print(separator, file=sys.stderr)
    print(file=sys.stderr)
    print(f"  Reason: {status.error}", file=sys.stderr)
    if status.fix_hint:
        print(f"  Fix:    {status.fix_hint}", file=sys.stderr)
    print(file=sys.stderr)
    print("  Training will be ~3.5× slower with pure-Python DP.", file=sys.stderr)
    print("  GPU utilisation will stay at ~20 % instead of >80 %.", file=sys.stderr)
    print(file=sys.stderr)
    print("  If you MUST proceed with pure Python (e.g. no compiler", file=sys.stderr)
    print("  available), set the environment variable:", file=sys.stderr)
    print("      LEVT_ALLOW_PYTHON_DP=1", file=sys.stderr)
    print(file=sys.stderr)
    print(separator, file=sys.stderr)

    if os.environ.get("LEVT_ALLOW_PYTHON_DP") == "1":
        print("  LEVT_ALLOW_PYTHON_DP=1 is set — proceeding anyway.", file=sys.stderr)
        print(separator, file=sys.stderr)
        return

    sys.exit(1)


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
    path: str,
    model_cfg: LevTConfig,
    train_cfg: TrainConfig,
    *,
    shuffle: bool,
    shuffle_seed: Optional[int] = None,
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
    loader = DataLoader(
        dataset,
        batch_size=train_cfg.batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=train_cfg.num_workers,
        collate_fn=collator,
    )
    if len(loader) == 0:
        raise ValueError(f"data loader for {path} has zero batches")
    return loader


def scheduler_factor(step: int, warmup: int, total: int) -> float:
    if warmup and step < warmup:
        return float(step + 1) / float(warmup)
    remaining = max(0, total - step)
    decay_steps = max(1, total - warmup)
    return float(remaining) / float(decay_steps)


def autocast_context(device: torch.device, amp_dtype: str):
    if amp_dtype == "none":
        return contextlib.nullcontext()
    if device.type != "cuda":
        if amp_dtype == "float16":
            raise ValueError("float16 AMP is only supported on CUDA")
        return torch.autocast(device_type="cpu", dtype=torch.bfloat16)
    dtype = torch.float16 if amp_dtype == "float16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def make_scaler(device: torch.device, amp_dtype: str):
    enabled = device.type == "cuda" and amp_dtype == "float16"
    return torch.amp.GradScaler("cuda", enabled=enabled)


def _linear_weight_ids(model: nn.Module) -> set[int]:
    """Collect parameter ids of every ``nn.Linear.weight`` in the model.

    These are the 2-D matrix parameters that Muon will optimize; everything
    else (biases, norms, embeddings) stays with AdamW.
    """
    ids: set[int] = set()
    for module in model.modules():
        if isinstance(module, nn.Linear):
            ids.add(id(module.weight))
    return ids


def build_optimizers(
    model: nn.Module,
    train_cfg: TrainConfig,
) -> tuple[AdamW, Muon]:
    """Return ``(adamw, muon)`` with parameters routed by type."""
    linear_ids = _linear_weight_ids(model)
    muon_params: list[nn.Parameter] = []
    adamw_params: list[nn.Parameter] = []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        if id(param) in linear_ids:
            muon_params.append(param)
        else:
            adamw_params.append(param)

    if not muon_params:
        raise ValueError("no Linear.weight parameters found for Muon optimizer")
    if not adamw_params:
        raise ValueError("no non-Linear parameters found for AdamW optimizer")

    adamw = accelerated.AdamW(
        adamw_params,
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
        betas=train_cfg.betas,
        eps=train_cfg.eps,
    )
    muon = accelerated.Muon(
        muon_params,
        lr=train_cfg.muon_lr,
        weight_decay=train_cfg.muon_weight_decay,
        momentum=train_cfg.muon_momentum,
        nesterov=train_cfg.muon_nesterov,
        ns_steps=train_cfg.muon_ns_steps,
    )
    return adamw, muon


def checkpoint_payload(
    model: LevTModel,
    adamw: AdamW,
    muon: Muon,
    adamw_scheduler: LambdaLR,
    muon_scheduler: LambdaLR,
    scaler: Any,
    model_cfg: LevTConfig,
    train_cfg: TrainConfig,
    *,
    global_step: int,
    epoch: int,
    next_batch_index: int,
) -> Dict[str, Any]:
    return {
        "model": model.state_dict(),
        "optimizer": {
            "adamw": adamw.state_dict(),
            "muon": muon.state_dict(),
        },
        "scheduler": {
            "adamw": adamw_scheduler.state_dict(),
            "muon": muon_scheduler.state_dict(),
        },
        "scaler": scaler.state_dict(),
        "model_config": model_cfg.to_dict(),
        "train_config": train_cfg.to_dict(),
        "global_step": global_step,
        "epoch": epoch,
        "next_batch_index": next_batch_index,
        "rng_state": capture_rng_state(),
    }


def write_checkpoints(directory: Path, step: int, payload: Dict[str, Any]) -> None:
    save_checkpoint(directory / f"step_{step:08d}.pt", payload)
    save_checkpoint(directory / "latest.pt", payload)


def evaluate(
    model: LevTModel,
    trainer: DualPolicyTrainer,
    loader: DataLoader,
    device: torch.device,
    amp_dtype: str,
) -> float:
    was_training = model.training
    model.eval()
    totals = {"plh": 0.0, "tok": 0.0, "del": 0.0}
    counts = {"plh": 0, "tok": 0, "del": 0}
    rng_state = capture_rng_state()
    seed_everything(0)
    try:
        with torch.no_grad():
            for batch in loader:
                prepared = trainer.prepare_batch(batch)
                with autocast_context(device, amp_dtype):
                    sums, batch_counts = trainer.loss_sums_and_counts(prepared)
                for name in totals:
                    totals[name] += float(sums[name])
                    counts[name] += batch_counts[name]
    finally:
        restore_rng_state(rng_state)
        model.train(was_training)
    if not any(counts.values()):
        raise ValueError("validation loader has no valid labels")
    return sum(totals[name] / counts[name] for name in totals if counts[name])


def main() -> None:
    args = parse_args()
    model_cfg = LevTConfig.from_json(args.model_config)
    train_cfg = TrainConfig.from_json(args.train_config)
    device = resolve_device(train_cfg.device)
    seed_everything(train_cfg.seed)
    enable_all()

    # --- Pre-flight: verify C++ Levenshtein extension -------------------
    _check_cpp_extension()

    validation_loader = (
        make_loader(train_cfg.validation_data, model_cfg, train_cfg, shuffle=False)
        if train_cfg.validation_data else None
    )

    model = LevTModel(model_cfg)
    resume_path = args.resume or train_cfg.resume_from
    checkpoint: Optional[Dict[str, Any]] = None
    if resume_path:
        checkpoint = load_checkpoint(resume_path, map_location="cpu")
        if not args.bypass_config_check and checkpoint["model_config"] != model_cfg.to_dict():
            raise ValueError("checkpoint model configuration does not match config.json")
        saved_train_config = checkpoint.get("train_config")
        current_train_config = train_cfg.to_dict()
        if isinstance(saved_train_config, dict):
            saved_train_config = {**saved_train_config, "resume_from": None}
            current_train_config = {**current_train_config, "resume_from": None}
        if not args.bypass_config_check and saved_train_config != current_train_config:
            diffs: list[str] = []
            if isinstance(saved_train_config, dict) and isinstance(current_train_config, dict):
                all_keys = set(saved_train_config.keys()) | set(current_train_config.keys())
                for key in sorted(all_keys):
                    sv = saved_train_config.get(key)
                    cv = current_train_config.get(key)
                    if sv != cv:
                        diffs.append(f"  {key}: checkpoint={sv!r}, train_config.json={cv!r}")
            else:
                diffs.append(f"  checkpoint={saved_train_config!r}, train_config.json={current_train_config!r}")
            raise ValueError(
                "checkpoint training configuration does not match train_config.json:\n"
                + "\n".join(diffs)
            )
    else:
        import_hf_embeddings(
            model,
            train_cfg.hf_model_name_or_path,
            local_files_only=train_cfg.local_files_only,
            trust_remote_code=train_cfg.trust_remote_code,
            dtype=train_cfg.hf_dtype,
        )
    model.to(device)
    model.shared_embedding.weight.requires_grad_(not train_cfg.freeze_embeddings)

    prealloc_model_grads(model)
    model = torch.compile(model, fullgraph=True)

    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])

    trainer = DualPolicyTrainer(model, model_cfg, train_cfg.policy)
    adamw, muon = build_optimizers(model, train_cfg)
    adamw_scheduler = LambdaLR(
        adamw,
        lambda step: scheduler_factor(step, train_cfg.warmup_steps, train_cfg.max_training_steps),
    )
    muon_scheduler = LambdaLR(
        muon,
        lambda step: scheduler_factor(step, train_cfg.warmup_steps, train_cfg.max_training_steps),
    )
    scaler = make_scaler(device, train_cfg.amp_dtype)
    global_step = 0
    start_epoch = 0
    next_batch_index = 0
    if checkpoint is not None:
        adamw.load_state_dict(checkpoint["optimizer"]["adamw"])
        muon.load_state_dict(checkpoint["optimizer"]["muon"])
        adamw_scheduler.load_state_dict(checkpoint["scheduler"]["adamw"])
        muon_scheduler.load_state_dict(checkpoint["scheduler"]["muon"])
        scaler.load_state_dict(checkpoint.get("scaler", {}))
        global_step = int(checkpoint["global_step"])
        start_epoch = int(checkpoint["epoch"])
        next_batch_index = int(checkpoint.get("next_batch_index", 0))
        restore_rng_state(checkpoint["rng_state"])

    checkpoint_dir = Path(train_cfg.checkpoint_dir)
    adamw.zero_grad(set_to_none=True)
    muon.zero_grad(set_to_none=True)
    resume_batch_index = next_batch_index
    final_epoch = start_epoch
    final_batch_index = resume_batch_index
    best_val_loss = float("inf")
    best_val_step: Optional[int] = None
    steps_since_improvement = 0
    current_val_loss: Optional[float] = None
    checkpoint_val_loss: Dict[int, float] = {}

    # --- Rich progress display -------------------------------------------
    display = None
    _display_total_set = False
    if HAS_RICH:
        display = TrainingDisplay(train_cfg.max_training_steps)

    # --- CSV progress log -------------------------------------------------
    csv_file = None
    csv_writer = None
    csv_append = args.resume_csv is not None
    csv_path_str = args.resume_csv or train_cfg.log_csv_path
    if csv_path_str:
        csv_path = Path(csv_path_str)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_mode = "a" if csv_append else "w"
        csv_file = csv_path.open(csv_mode, newline="")
        csv_writer = csv.writer(csv_file)
        if not csv_append:
            csv_writer.writerow([
                "step", "epoch", "batch",
                "loss_total", "loss_ins_plh", "loss_ins_tok", "loss_del",
                "lr_adamw", "lr_muon", "grad_norm", "val_loss",
            ])
        csv_file.flush()

    for epoch in range(start_epoch, train_cfg.epochs):
        train_loader = make_loader(
            train_cfg.train_data,
            model_cfg,
            train_cfg,
            shuffle=True,
            shuffle_seed=train_cfg.seed + epoch,
        )
        if resume_batch_index >= len(train_loader):
            # batch_size (or the dataset) may have changed since the
            # checkpoint was saved; the stored batch position no longer
            # fits within this epoch.  Advance to the next epoch instead
            # of raising an error — the data those batches covered has
            # already been processed.
            print(
                f"checkpoint batch position {resume_batch_index} >= epoch "
                f"length {len(train_loader)} — batch_size or dataset may "
                f"have changed since the checkpoint; advancing to next epoch",
                flush=True,
            )
            resume_batch_index = 0
            continue
        if display is not None and not _display_total_set:
            # Compute the real training ceiling: max_training_steps _or_
            # epoch exhaustion — whichever comes first.  The progress bar
            # then shows "how far through what we'll actually run."
            steps_per_epoch = math.ceil(
                len(train_loader) / train_cfg.gradient_accumulation_steps
            )
            remaining_epochs = train_cfg.epochs - epoch
            actual_total = min(
                train_cfg.max_training_steps,
                remaining_epochs * steps_per_epoch,
            )
            display.set_total(actual_total)
            _display_total_set = True
        train_iterator = iter(enumerate(train_loader))
        for batch_index, batch in train_iterator:
            if global_step >= train_cfg.max_training_steps:
                break
            if batch_index < resume_batch_index:
                continue
            window = [(batch_index, trainer.prepare_batch(batch))]
            for _ in range(1, train_cfg.gradient_accumulation_steps):
                try:
                    next_index_in_window, next_batch = next(train_iterator)
                except StopIteration:
                    break
                window.append((next_index_in_window, trainer.prepare_batch(next_batch)))

            # First pass: compute all losses and collect counts
            results = []
            metric_sums = {"plh": 0.0, "tok": 0.0, "del": 0.0}
            for _, prepared in window:
                with autocast_context(device, train_cfg.amp_dtype):
                    sums, batch_counts = trainer.loss_sums_and_counts(prepared)
                results.append((sums, batch_counts))
                for name in metric_sums:
                    metric_sums[name] += float(sums[name].detach())

            # Aggregate total counts across the window
            window_counts = {
                name: sum(r[1][name] for r in results)
                for name in ("plh", "tok", "del")
            }

            # Second pass: backward with normalized losses
            for sums, _ in results:
                loss = sum(
                    sums[name] / window_counts[name]
                    if window_counts[name] else sums[name] * 0.0
                    for name in sums
                )
                scaler.scale(loss).backward()
            metrics = {
                "loss_ins_plh": metric_sums["plh"] / window_counts["plh"] if window_counts["plh"] else 0.0,
                "loss_ins_tok": metric_sums["tok"] / window_counts["tok"] if window_counts["tok"] else 0.0,
                "loss_del": metric_sums["del"] / window_counts["del"] if window_counts["del"] else 0.0,
            }
            metrics["loss_total"] = sum(metrics.values())
            last_batch_index = window[-1][0]
            scaler.unscale_(adamw)
            scaler.unscale_(muon)
            grad_norm = None
            if train_cfg.max_grad_norm > 0:
                grad_norm = float(
                    accelerated.clip_grad_norm_(model.parameters(), train_cfg.max_grad_norm)
                )
            scaler.step(adamw)
            scaler.step(muon)
            scaler.update()
            adamw.zero_grad(set_to_none=True)
            muon.zero_grad(set_to_none=True)
            adamw_scheduler.step()
            muon_scheduler.step()
            global_step += 1

            next_epoch = epoch
            next_index = last_batch_index + 1
            if next_index == len(train_loader):
                next_epoch = epoch + 1
                next_index = 0
            final_epoch = next_epoch
            final_batch_index = next_index

            if display is not None:
                display.update(
                    step=global_step,
                    epoch=epoch,
                    batch=last_batch_index + 1,
                    batches_per_epoch=len(train_loader),
                    loss_total=metrics["loss_total"],
                    loss_plh=metrics["loss_ins_plh"],
                    loss_tok=metrics["loss_ins_tok"],
                    loss_del=metrics["loss_del"],
                    lr_adamw=float(adamw_scheduler.get_last_lr()[0]),
                    lr_muon=float(muon_scheduler.get_last_lr()[0]),
                    grad_norm=grad_norm,
                )
            elif global_step % train_cfg.log_every_steps == 0:
                print(
                    f"step={global_step} epoch={epoch} "
                    f"loss={metrics['loss_total']:.6f} lr_adamw={adamw_scheduler.get_last_lr()[0]:.8g} lr_muon={muon_scheduler.get_last_lr()[0]:.8g}",
                    flush=True,
                )
            val_loss_this_step = None
            if validation_loader is not None and global_step % train_cfg.validate_every_steps == 0:
                val_loss_this_step = evaluate(model, trainer, validation_loader, device, train_cfg.amp_dtype)
                current_val_loss = val_loss_this_step
                if val_loss_this_step < best_val_loss:
                    best_val_loss = val_loss_this_step
                    best_val_step = global_step
                    steps_since_improvement = 0
                else:
                    steps_since_improvement += 1
                if display is not None:
                    display.set_validation_loss(global_step, val_loss_this_step)
                    display.set_early_stopping(
                        steps_since_improvement, train_cfg.early_stopping_patience,
                    )
                else:
                    print(f"step={global_step} validation_loss={val_loss_this_step:.6f}", flush=True)
                if train_cfg.early_stopping_patience > 0 and steps_since_improvement >= train_cfg.early_stopping_patience:
                    print(
                        f"early stopping at step {global_step}: "
                        f"no improvement for {steps_since_improvement} validations "
                        f"(best val_loss={best_val_loss:.6f} at step {best_val_step})",
                        flush=True,
                    )
                    break
            if csv_file is not None:
                assert csv_writer is not None
                csv_writer.writerow([
                    global_step, epoch, last_batch_index + 1,
                    f"{metrics['loss_total']:.6f}",
                    f"{metrics['loss_ins_plh']:.6f}",
                    f"{metrics['loss_ins_tok']:.6f}",
                    f"{metrics['loss_del']:.6f}",
                    f"{adamw_scheduler.get_last_lr()[0]:.8g}",
                    f"{muon_scheduler.get_last_lr()[0]:.8g}",
                    f"{grad_norm:.6f}" if grad_norm is not None else "",
                    f"{val_loss_this_step:.6f}" if val_loss_this_step is not None else "",
                ])
                csv_file.flush()
            if global_step % train_cfg.checkpoint_every_steps == 0:
                write_checkpoints(
                    checkpoint_dir, global_step,
                    checkpoint_payload(
                        model, adamw, muon, adamw_scheduler, muon_scheduler,
                        scaler, model_cfg, train_cfg,
                        global_step=global_step,
                        epoch=next_epoch,
                        next_batch_index=next_index,
                    ),
                )
                if current_val_loss is not None:
                    checkpoint_val_loss[global_step] = current_val_loss
                if train_cfg.keep_last_checkpoints > 0:
                    cleanup_checkpoints(
                        checkpoint_dir, train_cfg.keep_last_checkpoints,
                        checkpoint_val_loss,
                    )
            if global_step >= train_cfg.max_training_steps:
                break
        resume_batch_index = 0
        if global_step >= train_cfg.max_training_steps:
            break

    payload = checkpoint_payload(
        model, adamw, muon, adamw_scheduler, muon_scheduler,
        scaler, model_cfg, train_cfg,
        global_step=global_step,
        epoch=final_epoch,
        next_batch_index=final_batch_index,
    )
    if display is not None:
        display.close()
    if csv_file is not None:
        csv_file.close()
    if current_val_loss is not None:
        checkpoint_val_loss[global_step] = current_val_loss
    write_checkpoints(checkpoint_dir, global_step, payload)
    if train_cfg.keep_last_checkpoints > 0:
        cleanup_checkpoints(
            checkpoint_dir, train_cfg.keep_last_checkpoints,
            checkpoint_val_loss,
        )
    print(f"training complete: step={global_step}, checkpoint={checkpoint_dir / 'latest.pt'}")


if __name__ == "__main__":
    main()
