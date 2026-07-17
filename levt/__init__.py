"""
Levenshtein Transformer (LevT) — a neural sequence generation model based on
insertion and deletion operations.

LevT iteratively refines a sequence by alternating between:
  1. Deletion  — remove tokens predicted as incorrect
  2. Insertion — add <PLH> tokens then fill them with predicted words

This unified framework handles both generation (from empty sequence) and
refinement (e.g. automatic post-editing) with the same model.

Reference: "Levenshtein Transformer" (Gu, Wang, Zhao, NeurIPS 2019)
"""

from .config import LevTConfig, PolicyConfig, TrainConfig
from .data import JsonlDataset, LevTCollator, validate_record
from .embeddings import import_hf_embeddings, load_hf_embedding_weights
from .checkpoint import load_checkpoint, save_checkpoint
from .positional import (
    ALiBiPositionalBias,
    RotaryPositionalEmbedding,
    SinusoidalPositionalEmbedding,
    apply_rotary_pos_emb,
    get_alibi_bias,
    get_alibi_slopes,
)
from .model import (
    LevTModel,
    LevTDecoder,
    LevTDecoderLayer,
    MultiheadAttention,
    TransformerEncoder,
    RMSNorm,
)
from .expert import (
    levenshtein_align,
    oracle_deletion,
    oracle_insertion,
    apply_deletion,
    insert_placeholders,
    fill_placeholders,
    random_deletion,
)
from .trainer import DualPolicyTrainer, PreparedBatch
from .decoder import GreedyDecoder


__all__ = [
    # Config
    "LevTConfig",
    "PolicyConfig",
    "TrainConfig",
    "JsonlDataset",
    "LevTCollator",
    "validate_record",
    "import_hf_embeddings",
    "load_hf_embedding_weights",
    "save_checkpoint",
    "load_checkpoint",
    # Positional encoding
    "SinusoidalPositionalEmbedding",
    "RotaryPositionalEmbedding",
    "ALiBiPositionalBias",
    "apply_rotary_pos_emb",
    "get_alibi_slopes",
    "get_alibi_bias",
    # Model
    "LevTModel",
    "LevTDecoder",
    "LevTDecoderLayer",
    "MultiheadAttention",
    "TransformerEncoder",
    "RMSNorm",
    # Expert / Oracle
    "levenshtein_align",
    "oracle_deletion",
    "oracle_insertion",
    "apply_deletion",
    "insert_placeholders",
    "fill_placeholders",
    "random_deletion",
    # Training
    "DualPolicyTrainer",
    "PreparedBatch",
    # Inference
    "GreedyDecoder",
]
