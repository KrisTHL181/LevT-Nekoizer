# LevT-Nekoizer

A Levenshtein Transformer that rewrites any Chinese text into catgirl-speak.
Trained on 207k Zhihu answer pairs — takes boring, serious prose and injects
the playful, mischievous tone of a Neko.

📝 **[Read the full backstory (Chinese)](https://zhuanlan.zhihu.com/p/2062297940572500896)**

> **Input:** 我们厌倦了知乎上千篇一律的严肃长文和各种爹味说教。为了净化眼球，我们开发了这个模型。
>
> **Output:** 人家厌倦了知乎上上千篇一律的严肃长文和各种爹味说教喵。为了净化眼球，人家就开发出了这个模型喵。

Instead of generating left-to-right, the model iteratively refines text through
three operations: **delete** what doesn't belong, **insert placeholders** where
new tokens are needed, and **fill** those placeholders with actual words. Each
pass improves the sequence — rinse and repeat until the text is sufficiently
nyaa.

## Quick Start

### Get the model

Pretrained weights are on Hugging Face:

🔗 **[KrisTHL181/LevT-Nekoizer](https://huggingface.co/KrisTHL181/LevT-Nekoizer)**

### Get the code

```bash
git clone https://github.com/KrisTHL181/LevT-Nekoizer.git
cd LevT-Nekoizer
```

### Tokenizer

The model uses the tokenizer from **MiniCPM4-0.5B** (`openbmb/MiniCPM4-0.5B`).
The vocabulary has 73,448 Chinese + English tokens. The model's embedding table
was initialized from MiniCPM4-0.5B's pretrained embeddings and projected down
from 1024 → 512 dimensions.

You'll need the tokenizer to encode input text and decode model output:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    'openbmb/MiniCPM4-0.5B', trust_remote_code=True
)
```

### Install

```bash
pip install torch transformers huggingface_hub
```

### Inference

```python
import torch
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
from levt import LevTConfig, LevTModel, GreedyDecoder

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    'openbmb/MiniCPM4-0.5B', trust_remote_code=True
)

# Load model
config = LevTConfig.from_json(
    hf_hub_download('KrisTHL181/LevT-Nekoizer', 'config.json')
)
model = LevTModel(config)
state_dict = torch.load(
    hf_hub_download('KrisTHL181/LevT-Nekoizer', 'pytorch_model.bin'),
    map_location='cpu', weights_only=True
)
model.load_state_dict(state_dict)
model.eval()

# Encode → refine → decode
decoder = GreedyDecoder(model, config)
src_ids = tokenizer.encode('把全知乎都变成猫娘！', add_special_tokens=False)
output, iterations = decoder.decode(torch.tensor(src_ids))
print(tokenizer.decode(output.tolist(), skip_special_tokens=True))
```

An interactive REPL is also available:

```bash
python preview.py \
  --config config.json \
  --checkpoint pytorch_model.bin \
  --tokenizer openbmb/MiniCPM4-0.5B \
  --interactive
```

## How It Works

LevT refines text in iterative edit passes. Each iteration runs three
classifier heads:

| Phase | What it does |
|---|---|
| **Delete** | Marks every token as keep or delete |
| **Placeholder** | Predicts how many new tokens to insert between each pair |
| **Fill** | Replaces each placeholder with an actual vocabulary token |

The model repeats until the sequence stabilizes (or hits `max_iterations`).
Because insertions and deletions can change sequence length, LevT isn't bound
by a fixed output length — it can freely restructure sentences.

### Architecture

This implementation modernizes the original 2019 LevT with techniques from
contemporary decoder-only LLMs:

| Component | Original Paper | This Implementation |
|---|---|---|
| Position encoding | Learned absolute / sinusoidal | **ALiBi** — linear bias, strong length extrapolation |
| Normalization | Post-LayerNorm | **Pre-RMSNorm** — stable gradient flow, cheaper compute |
| FFN activation | ReLU | **GELU** — smooth, non-monotonic, no dead neurons |
| Optimizer | Adam | **Muon** for linear layers, **AdamW** for everything else |
| Attention | Standard SDPA | **SDPA + QK Norm** — prevents entropy collapse, stabilizes training |
| Embeddings | Random init | **MiniCPM4-0.5B** pretrained embeddings, projected 1024 → 512 |

The token head shares weights with the input embedding matrix (no separate
output projection), and the 512↔1024 projection layer is reused via transpose
for the output logits.

### Training

Trained with **dual-policy learning**: the insertion and deletion policies are
trained alternately, each learning to fix the other's imperfect output. Oracle
edit paths are computed via Levenshtein DP (insertion + deletion only, no
substitution) using longest-common-subsequence alignment.

| Stat | Value |
|---|---|
| Training samples | 207,151 (from 7,726 Zhihu answers) |
| Vocabulary | 73,448 (Chinese + English tokens) |
| Embedding dim | 1024 (projected → 512) |
| Model dim | 512 |
| Attention heads | 8 |
| Encoder / Decoder layers | 6 / 6 |
| Training steps | 300,000 |
| Optimizer | Muon (lr 0.02) + AdamW (lr 0.001), warmup + linear decay |

### Why not an LLM?

A browser-hosted LLM (~6 GB resident, per-tab) is untenable for rewriting every
Zhihu answer on page load. A 60–220M parameter seq2seq model is the right
weight class — and among those, LevT's iterative-edit approach is far more
sample-efficient than one-shot generation for tasks where most of the input
stays intact.

## Train Your Own

```bash
python train.py --model-config config.json --train-config train_config.json
```

Data format (JSONL):

```json
{"src": [4, 5, 6], "target": [1, 9, 10, 2], "initial": [1, 2]}
```

- `src` and `target` are required; `initial` defaults per `config.json`'s `initial_strategy` (`"src"` = the full source sequence, `"bos_eos"` = `[BOS, EOS]`)
- Values are pre-tokenized integer ID lists — no tokenizer is bundled

See `config.json` and `train_config.json` for all model and training knobs.

## Tests

```bash
PYTHONPATH=. pytest -q tests
python -m py_compile train.py levt/*.py tests/*.py
```

## Citation

```bibtex
@inproceedings{gu2019levenshtein,
  title={Levenshtein Transformer},
  author={Gu, Jiatao and Wang, Changhan and Zhao, Junbo},
  booktitle={Advances in Neural Information Processing Systems},
  year={2019}
}
```

## License

MIT
