# Levenshtein Transformer (LevT)

A PyTorch implementation of the Levenshtein Transformer (Gu, Wang, Zhao,
NeurIPS 2019). The model refines a sequence with deletion, placeholder
insertion, and placeholder filling operations.

## Architecture

The encoder and bidirectional decoder share one vocabulary embedding. Both
sides always have their own bias-free `embedding_dim -> d_model` projection,
even when the dimensions are equal. Token logits have no standalone output
parameter:

```python
projected = F.linear(hidden, model.decoder_input_projection.weight.T)
logits = F.linear(projected, model.shared_embedding.weight)
```

Deletion and placeholder prediction use ordinary classifier heads. Decoder
states and masks are seq-first: token tensors are `(length, batch)` and padding
masks are `(batch, length)`, where `True` means padding.

## Data

Training and validation data are JSONL. Every row has exactly these keys:

```json
{"src": [4, 5, 6], "target": [1, 9, 10, 2], "initial": [1, 2]}
```

- `src` and `target` are required.
- `initial` is optional and defaults to `[bos_token_id, eos_token_id]`.
- Values must be nonempty lists of integer token IDs. Boolean values are not
  integers for validation purposes.
- IDs must be in range. Padding and placeholder IDs are not accepted in raw
  rows. Target and initial sequences must start with BOS, end with EOS, and
  contain no interior BOS/EOS.
- Unknown row keys, blank lines, malformed JSON, and configured length
  overflows are errors. No row is truncated.

No tokenizer is loaded. Input JSONL must already contain token IDs compatible
with the configured vocabulary and imported embedding table.

## Configuration

`config.json` owns model architecture and special IDs only. Unknown keys are
rejected. Boolean switches require JSON booleans, and numeric settings must be
finite; `rope_base` must additionally be positive. `embedding_dim` defaults to
`d_model` for older Python callers.

```json
{
  "vocab_size": 32000,
  "embedding_dim": 768,
  "d_model": 512,
  "n_heads": 8,
  "d_ff": 2048,
  "n_encoder_layers": 6,
  "n_decoder_layers": 6,
  "pad_token_id": 0,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "plh_token_id": 3
}
```

`train_config.json` owns data, Hugging Face embedding import, policy,
optimizer, scheduler, runtime, and checkpoint settings. It is also strict.
See the root example for all supported keys.

Hugging Face integration calls only:

```python
transformers.AutoModel.from_pretrained(...).get_input_embeddings()
```

It never loads `AutoTokenizer`. `torch_dtype` controls the temporary Hugging
Face model load. The imported table must exactly match
`(vocab_size, embedding_dim)`. The LevT model is initialized first, then the
external weights are copied, so copied weights are not reinitialized.

## Training

Install PyTorch, Transformers, and pytest in the environment, then run:

```bash
python train.py --model-config config.json --train-config train_config.json
```

Resume from a checkpoint with:

```bash
python train.py --model-config config.json --train-config train_config.json \
  --resume checkpoints/latest.pt
```

Training is single-process on one CPU or one GPU. DataLoader batching is by
example count. For each batch, CPU oracle construction produces three independently padded
target batches (`y_ins`, `y_ins_plh`, `y_del`). `PreparedBatch` fixes these
stochastic roll-ins once. The public `prepare_batch()` and
`loss_sums_and_counts()` APIs expose differentiable per-head loss sums plus
valid-label counts. Gradient accumulation normalizes each head by its exact
count over the whole optimizer window, backpropagating one prepared microbatch
at a time without retaining graphs. Validation uses the same per-head
sum/count reduction over the complete loader, so its result is independent of
batch partitioning. Placeholder and token losses use label smoothing; deletion
does not. `-100` marks padded loss targets.

CUDA FP16 uses GradScaler. CUDA BF16 and CPU BF16 use autocast without scaling.
Set `amp_dtype` to `none` for full precision. Checkpoints include model,
optimizer, scheduler, scaler, global step, epoch plus the next-batch resume
cursor, strict configs, and Python, Torch, and CUDA RNG state where available.
Shuffle order is deterministic per epoch, so optimizer-boundary checkpoints
resume without replaying prior batches. Both `latest.pt` and numbered step
checkpoints are written atomically.

The Python API retains single-example use:

```python
loss, metrics = DualPolicyTrainer(model, model_config).train_step(src, initial, target)
loss.backward()
```

For real training, pass the batch dictionary produced by `LevTCollator`.

## Inference

```python
import torch
from levt import GreedyDecoder, LevTConfig, LevTModel

config = LevTConfig(vocab_size=32000)
model = LevTModel(config)
decoder = GreedyDecoder(model, config)
output, iterations = decoder.decode(torch.tensor([4, 5, 6]))
```

The encoder memory is computed once and reused across refinement iterations.
Each phase requests only its required classifier head; token logits are computed
only at placeholder positions. PAD, BOS, EOS, and PLH IDs are masked from fill
predictions. `decode()` temporarily uses evaluation mode and restores the
model's prior training state. Boundary tokens are never deleted. Decoding stops
on convergence, a direct loop, or `max_iterations`.

Sinusoidal and RoPE tables are runtime caches and are omitted from checkpoints.
Sinusoidal encoding supports odd model dimensions; RoPE caches refresh for the
active device and dtype.

## Verification

```bash
PYTHONPATH=. pytest -q tests
python -m py_compile train.py levt/*.py tests/*.py
```
