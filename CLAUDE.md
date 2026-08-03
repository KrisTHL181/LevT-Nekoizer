# CLAUDE.md

## Overview

Pure-Python PyTorch implementation of the Levenshtein Transformer (Gu et al.,
NeurIPS 2019). It includes strict JSON configuration/data loading, batched
dual-policy training, Hugging Face input-embedding import, checkpoints, and
iterative inference. There is no packaging or dependency file.

## Architecture

All token tensors are seq-first `(length, batch)`. Padding masks are
`(batch, length)` with `True` meaning ignored.

`LevTModel` has one `shared_embedding` of shape
`(vocab_size, embedding_dim)`. `src_embed` and `tgt_embed` are compatibility
properties returning that same module. Independent, bias-free
`encoder_input_projection` and `decoder_input_projection` layers always map
`embedding_dim -> d_model`, including when the dimensions are equal.

The encoder uses sinusoidal, RoPE, or ALiBi position handling. Sinusoidal tables
are nonpersistent runtime buffers and support odd `d_model`; RoPE cos/sin
caches are nonpersistent and refresh for the active device/dtype. The decoder is
bidirectional and returns every layer output for deletion/placeholder early
exit. RMSNorm is used throughout; optional QK normalization and headwise or
elementwise attention output gates are supported.

Packed training (multiple unrelated examples concatenated per row) uses
segment-level attention masks so bidirectional attention never leaks across
segment boundaries: encoder/decoder self-attention get a block-diagonal mask
and decoder→encoder cross-attention gets a diagonal mask. Segment boundaries
are inferred from `[EOS][BOS]` token pairs (`levt/segment_mask.py`); unpacked
training passes `None` masks and is behaviorally identical.

### Prediction heads

- `deletion_head`: `d_model -> 2`, classes keep/delete.
- `placeholder_head`: concatenated adjacent decoder states to
  `max_placeholder + 1` counts.
- Token prediction has no `nn.Linear` output module or output parameter.
  Decoder states map back through the transpose of
  `decoder_input_projection.weight`, then logits use
  `F.linear(projected, shared_embedding.weight)`.

## Configuration ownership

`config.json` is loaded by `LevTConfig.from_json` and contains model
architecture plus special IDs only. Unknown keys are rejected. Boolean switches
require actual booleans, numeric settings must be finite, and `rope_base` must
be positive. Special IDs must be distinct and in range. `embedding_dim` defaults
to `d_model` for old Python callers. Legacy training constructor attributes
remain optional only so older code can instantiate `DualPolicyTrainer`; strict
model JSON rejects them.

`train_config.json` is loaded by `TrainConfig.from_json`. It is a flat strict
schema owning:

- train/validation JSONL paths and DataLoader batch size;
- Hugging Face model source, `local_files_only`, `trust_remote_code`, import
  dtype, and embedding freezing;
- dual-policy alpha/beta, random deletion, and label smoothing;
- AdamW and Muon optimizers with warmup/linear-decay schedule;
- device, AMP, accumulation, clipping, logging, validation, checkpoints,
  resume path, early stopping, and checkpoint retention.

## Data

JSONL rows use required `src` and `target`, plus optional `initial`. Missing
`initial` becomes `[BOS, EOS]`; there is no `id` field. Lists must be nonempty
integers (bool invalid), in vocabulary range and configured length limits.
Raw rows cannot contain pad/placeholder tokens. Target and initial must start
with BOS, end with EOS, and have no interior boundary tokens. Unknown keys,
blank lines, and malformed data fail with file/line context.

A dataset file may begin with an optional metadata header
`{"__meta__": {"format": "levt-jsonl", "version": 1, "packed": true}}` that
declares whether rows are packed (concatenated segments, interior BOS/EOS
boundaries allowed) or regular. The header line is skipped, not validated as a
data row. Training auto-detects packed vs regular from the header; a
header-less (legacy) file falls back to the `packed` flag in `train_config.json`
(default regular). `scripts/pack_dataset.py` (packed) and
`scripts/pre_tokenize.py` (regular) write the header on their outputs;
`scripts/precompute_oracles.py` forwards an existing header.

`LevTCollator` returns seq-first padded `src_tokens`, batch-first
`src_padding_mask`, and unpadded CPU `initial`/`targets` lists.

## Hugging Face embeddings

`levt/embeddings.py` imports Transformers lazily and calls only
`AutoModel.from_pretrained(..., torch_dtype=...).get_input_embeddings()`. No tokenizer is
loaded. Vocabulary and embedding dimensions must match exactly. Construct and
randomly initialize LevT first, then call `copy_embedding_weights`, ensuring
external weights are not reinitialized. The core model has no Transformers
dependency.

## Batched dual-policy training

`DualPolicyTrainer` accepts a `PolicyConfig`, with fallback to legacy
`LevTConfig` training attributes/defaults. For each collated batch:

1. Move each unpadded initial/target to CPU and construct insertion oracle
   roll-ins per sample.
2. Pad `y_ins` for placeholder loss and `y_ins_plh` for token loss.
3. Generate model-filled deletion roll-ins under `no_grad`, preserving source
   memory/mask device correctness; compute deletion oracles on CPU.
4. Pad `y_del` separately and compute deletion loss.

`PreparedBatch` fixes all stochastic roll-ins once. `prepare_batch()` builds it
without retaining a graph; `loss_sums_and_counts()` returns differentiable
per-head sums plus valid-label counts. Accumulation normalizes each head by its
exact total count across the complete optimizer window while backpropagating
one prepared microbatch at a time. Validation aggregates the same sums/counts
over the full loader, making results independent of batch partitioning. Loss
targets use `-100` for padding. Placeholder/token losses use smoothing;
deletion does not. Shape/count mismatches raise errors; targets and logits are
never silently trimmed. The legacy single-example
`train_step(src, initial, target)` remains supported.

## Training CLI and checkpoints

Run:

```bash
python train.py --model-config config.json --train-config train_config.json
```

This is single-machine, single-GPU or CPU training. CUDA FP16 alone uses
GradScaler; BF16 uses autocast without scaling. It supports accumulation,
clipping, validation, warmup plus linear decay, periodic logging, and optional
CSV progress logging (`log_csv_path`). `--resume <checkpoint>` restores model,
optimizer, scheduler, scaler, and RNG state from a checkpoint. `--resume-csv
<path>` opens an existing CSV in append mode instead of overwriting, preserving
prior progress rows.

**Optimizers**: `nn.Linear.weight` parameters use Muon; all other parameters
(biases, RMSNorm weights, embedding weights) use AdamW. `build_optimizers()`
routes parameters by collecting `id(module.weight)` for every `nn.Linear`
submodule. Both optimizers share the same warmup+linear-decay LR schedule but
with independently configurable base learning rates (`learning_rate` for AdamW,
`muon_lr` for Muon). `TrainConfig` holds separate hyperparameters for each:
AdamW uses `learning_rate`, `weight_decay`, `betas`, `eps`; Muon uses
`muon_lr`, `muon_weight_decay`, `muon_momentum`, `muon_nesterov`,
`muon_ns_steps`.

Checkpoints store both optimizer/scheduler state dicts under nested
`"optimizer"` and `"scheduler"` keys (`"adamw"` / `"muon"` sub-keys).
Checkpoints are atomic and versioned. `latest.pt` and numbered step files hold
model, optimizers, schedulers, scaler, step, epoch plus next-batch resume
cursor, both configs, and Python, Torch, and CUDA RNG state where available.
Training shuffle order is deterministic per epoch, so an optimizer-boundary
checkpoint resumes without replaying prior batches. Resume verifies exact model
config before restoring state. Validation uses a fixed temporary RNG state and
restores the training RNG afterward.

### Early stopping

`early_stopping_patience` (default 0, disabled) sets how many consecutive
validation checks without improvement trigger training termination. The
best validation loss and its step are tracked throughout training; a lower
loss resets the counter. When the patience threshold is reached, training
stops and a message is printed with the best-loss details.

### Checkpoint cleanup

`keep_last_checkpoints` (default 0, keep all) limits the number of numbered
`step_*.pt` files in the checkpoint directory. The checkpoint with the lowest
recorded validation loss is **always preserved** regardless of age; on top of
that, the *N* most-recent checkpoints (by highest step number) are retained.
Cleanup runs after every checkpoint save (including the final one). When both
are zero (the defaults), behaviour is unchanged from earlier versions.

The Rich display shows a patience counter (`X/Y`) when early stopping is
active, coloured yellow while below threshold and red when triggered.

## Expert and inference

`expert.py` uses insertion+deletion Levenshtein DP; substitution costs two.
Oracles return CPU tensors and training deliberately applies them before moving
padded tasks to the model device.

For packed rows the oracles are segment-aware: sequences are split at
`[EOS][BOS]` boundaries and the DP runs per segment, so an edit script never
crosses a segment boundary and a roll-in keeps its segment count (which keeps
the packed-segment attention masks aligned). The batched C++ path stays fast by
flattening the segments into one batch call and re-concatenating. `random_deletion`
keeps every BOS/EOS token, not just the first/last.

`GreedyDecoder` computes encoder memory once, then iterates delete, insert, and
fill until convergence/loop detection or `max_iterations`. Each phase requests
only its required head, and token logits are computed only at PLH positions.
PAD/BOS/EOS/PLH IDs are masked from fill predictions. Decoding temporarily
switches to evaluation mode and restores the previous mode. BOS/EOS positions
are never deleted.

## Verification

```bash
PYTHONPATH=. pytest -q tests
python -m py_compile train.py levt/*.py tests/*.py
```
