"""Hugging Face input-embedding import without a tokenizer dependency."""

from __future__ import annotations

from typing import Any

import torch

from .model import LevTModel


_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def load_hf_embedding_weights(
    source: str,
    *,
    local_files_only: bool = False,
    trust_remote_code: bool = False,
    dtype: str = "float32",
) -> torch.Tensor:
    """Load only ``AutoModel.get_input_embeddings()`` and return a CPU copy."""
    if dtype not in _DTYPES:
        raise ValueError("dtype must be float32, float16, or bfloat16")
    try:
        from transformers import AutoModel
    except ImportError as exc:
        raise RuntimeError("transformers is required to import Hugging Face embeddings") from exc

    pretrained: Any = AutoModel.from_pretrained(
        source,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
        torch_dtype=_DTYPES[dtype],
    )
    embedding = pretrained.get_input_embeddings()
    if embedding is None or not hasattr(embedding, "weight"):
        raise ValueError(f"{source} does not expose input embeddings")
    weight = embedding.weight.detach().to(device="cpu", dtype=torch.float32).clone()
    del pretrained
    return weight


def import_hf_embeddings(
    model: LevTModel,
    source: str,
    *,
    local_files_only: bool = False,
    trust_remote_code: bool = False,
    dtype: str = "float32",
) -> None:
    weights = load_hf_embedding_weights(
        source,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
        dtype=dtype,
    )
    expected = (model.config.vocab_size, model.config.embedding_dim)
    if tuple(weights.shape) != expected:
        raise ValueError(
            f"Hugging Face embedding shape {tuple(weights.shape)} does not match "
            f"configured (vocab_size, embedding_dim) {expected}"
        )
    model.copy_embedding_weights(weights)
