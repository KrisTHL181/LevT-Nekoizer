"""Strict configuration schemas for the Levenshtein Transformer."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar


T = TypeVar("T")


def _strict_dataclass(cls: Type[T], data: Dict[str, Any], source: str) -> T:
    if not isinstance(data, dict):
        raise ValueError(f"{source} must contain a JSON object")
    allowed = {field.name for field in fields(cls)}
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"unknown keys in {source}: {', '.join(unknown)}")
    try:
        return cls(**data)
    except TypeError as exc:
        raise ValueError(f"invalid {source}: {exc}") from exc


def _load_json(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"failed to load {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _positive_int(name: str, value: Any, *, allow_zero: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    lower = 0 if allow_zero else 1
    if value < lower:
        op = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {op}")


def _probability(name: str, value: Any, *, upper_inclusive: bool = True) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    if not math.isfinite(value) or value < 0 or value > 1 or (not upper_inclusive and value == 1):
        suffix = "[0, 1]" if upper_inclusive else "[0, 1)"
        raise ValueError(f"{name} must be in {suffix}")


@dataclass
class LevTConfig:
    """Model architecture and special-token configuration only.

    ``embedding_dim`` defaults to ``d_model`` for compatibility with older
    callers. Legacy training fields are accepted as optional attributes so old
    code can still construct a trainer, but they are not accepted by the strict
    root ``config.json`` loader and are not model-owned settings.
    """

    vocab_size: int
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    plh_token_id: int = 3
    embedding_dim: Optional[int] = None
    d_model: int = 512
    n_heads: int = 8
    d_ff: int = 2048
    n_encoder_layers: int = 6
    n_decoder_layers: int = 6
    dropout: float = 0.3
    attention_dropout: float = 0.1
    activation: str = "relu"
    pos_encoding_type: str = "sinusoidal"
    rope_base: float = 10000.0
    max_source_positions: int = 1024
    max_target_positions: int = 1024
    qk_norm: bool = False
    headwise_attn_output_gate: bool = False
    elementwise_attn_output_gate: bool = False
    max_placeholder: int = 255
    early_exit_del: Optional[int] = None
    early_exit_plh: Optional[int] = None
    max_iterations: int = 10
    placeholder_penalty: float = 0.0

    # Constructor-only compatibility fields. Strict model JSON rejects these.
    alpha: Optional[float] = None
    beta: Optional[float] = None
    random_delete_prob: Optional[float] = None
    label_smoothing: Optional[float] = None
    lr: Optional[float] = None
    warmup_steps: Optional[int] = None
    max_training_steps: Optional[int] = None
    batch_tokens: Optional[int] = None

    MODEL_JSON_FIELDS = {
        "vocab_size", "pad_token_id", "bos_token_id", "eos_token_id",
        "plh_token_id", "embedding_dim", "d_model", "n_heads", "d_ff",
        "n_encoder_layers", "n_decoder_layers", "dropout",
        "attention_dropout", "activation", "pos_encoding_type", "rope_base",
        "max_source_positions", "max_target_positions", "qk_norm",
        "headwise_attn_output_gate", "elementwise_attn_output_gate",
        "max_placeholder", "early_exit_del", "early_exit_plh",
        "max_iterations", "placeholder_penalty",
    }

    def __post_init__(self) -> None:
        if self.embedding_dim is None:
            self.embedding_dim = self.d_model
        for name in (
            "qk_norm", "headwise_attn_output_gate", "elementwise_attn_output_gate",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")
        for name in (
            "vocab_size", "embedding_dim", "d_model", "n_heads", "d_ff",
            "n_encoder_layers", "n_decoder_layers", "max_source_positions",
            "max_target_positions", "max_placeholder", "max_iterations",
        ):
            _positive_int(name, getattr(self, name))
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.pos_encoding_type == "rope" and (self.d_model // self.n_heads) % 2:
            raise ValueError("RoPE requires an even attention head dimension")
        if self.activation not in {"relu", "gelu", "swiglu"}:
            raise ValueError("activation must be one of: relu, gelu, swiglu")
        if self.pos_encoding_type not in {"sinusoidal", "rope", "alibi"}:
            raise ValueError("pos_encoding_type must be one of: sinusoidal, rope, alibi")
        _probability("dropout", self.dropout)
        _probability("attention_dropout", self.attention_dropout)
        for name in ("alpha", "beta", "random_delete_prob"):
            value = getattr(self, name)
            if value is not None:
                _probability(name, value)
        if self.label_smoothing is not None:
            _probability("label_smoothing", self.label_smoothing, upper_inclusive=False)
        if self.lr is not None:
            if (
                isinstance(self.lr, bool)
                or not isinstance(self.lr, (int, float))
                or not math.isfinite(self.lr)
                or self.lr <= 0
            ):
                raise ValueError("lr must be finite and positive")
        for name, allow_zero in (
            ("warmup_steps", True),
            ("max_training_steps", False),
            ("batch_tokens", False),
        ):
            value = getattr(self, name)
            if value is not None:
                _positive_int(name, value, allow_zero=allow_zero)
        if isinstance(self.rope_base, bool) or not isinstance(self.rope_base, (int, float)):
            raise ValueError("rope_base must be numeric")
        if not math.isfinite(self.rope_base) or self.rope_base <= 0:
            raise ValueError("rope_base must be finite and positive")
        if (
            isinstance(self.placeholder_penalty, bool)
            or not isinstance(self.placeholder_penalty, (int, float))
            or not math.isfinite(self.placeholder_penalty)
        ):
            raise ValueError("placeholder_penalty must be finite numeric")
        special = {
            "pad_token_id": self.pad_token_id,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "plh_token_id": self.plh_token_id,
        }
        for name, value in special.items():
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < self.vocab_size:
                raise ValueError(f"{name} must be an integer in [0, vocab_size)")
        if len(set(special.values())) != len(special):
            raise ValueError("pad, bos, eos, and placeholder token IDs must be distinct")
        if self.headwise_attn_output_gate and self.elementwise_attn_output_gate:
            raise ValueError("attention output gate modes are mutually exclusive")
        for name in ("early_exit_del", "early_exit_plh"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < self.n_decoder_layers):
                raise ValueError(f"{name} must be a valid decoder layer index")

    @classmethod
    def from_dict(cls, data: Dict[str, Any], *, strict_model: bool = False) -> "LevTConfig":
        if strict_model:
            unknown = sorted(set(data) - cls.MODEL_JSON_FIELDS)
            if unknown:
                raise ValueError(f"unknown keys in model config: {', '.join(unknown)}")
        return _strict_dataclass(cls, data, "model config")

    @classmethod
    def from_json(cls, path: str | Path) -> "LevTConfig":
        return cls.from_dict(_load_json(path), strict_model=True)

    def to_dict(self, *, model_only: bool = True) -> Dict[str, Any]:
        data = asdict(self)
        if model_only:
            return {key: data[key] for key in self.MODEL_JSON_FIELDS}
        return data


@dataclass
class PolicyConfig:
    alpha: float = 0.5
    beta: float = 0.5
    random_delete_prob: float = 0.3
    label_smoothing: float = 0.1

    def __post_init__(self) -> None:
        for name in ("alpha", "beta", "random_delete_prob"):
            _probability(name, getattr(self, name))
        _probability("label_smoothing", self.label_smoothing, upper_inclusive=False)


@dataclass
class TrainConfig:
    train_data: str
    hf_model_name_or_path: str
    validation_data: Optional[str] = None
    batch_size: int = 8
    oracle_batch_size: int = 0
    num_workers: int = 0
    prefetch_maxsize: int = 5
    max_source_length: int = 1024
    max_target_length: int = 1024
    local_files_only: bool = False
    trust_remote_code: bool = False
    hf_dtype: str = "float32"
    freeze_embeddings: bool = False
    alpha: float = 0.5
    beta: float = 0.5
    random_delete_prob: float = 0.3
    label_smoothing: float = 0.1
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    muon_lr: float = 0.02
    muon_weight_decay: float = 0.01
    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    muon_ns_steps: int = 5
    warmup_steps: int = 10000
    max_training_steps: int = 300000
    epochs: int = 1
    seed: int = 1
    device: str = "auto"
    amp_dtype: str = "none"
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    log_every_steps: int = 100
    validate_every_steps: int = 1000
    checkpoint_every_steps: int = 1000
    checkpoint_dir: str = "checkpoints"
    log_csv_path: str = ""
    resume_from: Optional[str] = None
    early_stopping_patience: int = 0
    keep_last_checkpoints: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.train_data, str) or not self.train_data:
            raise ValueError("train_data must be a nonempty path")
        if not isinstance(self.hf_model_name_or_path, str) or not self.hf_model_name_or_path:
            raise ValueError("hf_model_name_or_path must be nonempty")
        for name in ("local_files_only", "trust_remote_code", "freeze_embeddings"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")
        for name in (
            "batch_size", "max_source_length", "max_target_length",
            "max_training_steps", "epochs", "gradient_accumulation_steps",
            "log_every_steps", "validate_every_steps", "checkpoint_every_steps",
        ):
            _positive_int(name, getattr(self, name))
        _positive_int("num_workers", self.num_workers, allow_zero=True)
        # 0 = process the whole training batch in one C++ oracle call; a
        # positive value chunks the batch into groups of that size.
        _positive_int("oracle_batch_size", self.oracle_batch_size, allow_zero=True)
        _positive_int("prefetch_maxsize", self.prefetch_maxsize)
        _positive_int("warmup_steps", self.warmup_steps, allow_zero=True)
        PolicyConfig(self.alpha, self.beta, self.random_delete_prob, self.label_smoothing)
        for name in ("learning_rate", "weight_decay", "eps", "max_grad_norm"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite numeric")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.eps <= 0:
            raise ValueError("optimizer values must be positive (weight_decay may be zero)")
        if not isinstance(self.betas, (list, tuple)) or len(self.betas) != 2:
            raise ValueError("betas must contain exactly two values")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in self.betas):
            raise ValueError("AdamW betas must be numeric")
        self.betas = (float(self.betas[0]), float(self.betas[1]))
        if not all(math.isfinite(value) for value in self.betas):
            raise ValueError("AdamW betas must be finite")
        if not 0 <= self.betas[0] < 1 or not 0 <= self.betas[1] < 1:
            raise ValueError("AdamW betas must be in [0, 1)")
        if self.hf_dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("hf_dtype must be float32, float16, or bfloat16")
        if self.amp_dtype not in {"none", "float16", "bfloat16"}:
            raise ValueError("amp_dtype must be none, float16, or bfloat16")
        if self.max_grad_norm < 0:
            raise ValueError("max_grad_norm must be non-negative")
        for name in ("muon_lr", "muon_weight_decay", "muon_momentum"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite numeric")
        if self.muon_lr <= 0 or self.muon_weight_decay < 0 or self.muon_momentum < 0:
            raise ValueError("Muon values must be positive (weight_decay/momentum may be zero)")
        if not 0 <= self.muon_momentum < 1:
            raise ValueError("muon_momentum must be in [0, 1)")
        if not isinstance(self.muon_nesterov, bool):
            raise ValueError("muon_nesterov must be a boolean")
        _positive_int("muon_ns_steps", self.muon_ns_steps)
        _positive_int("early_stopping_patience", self.early_stopping_patience, allow_zero=True)
        _positive_int("keep_last_checkpoints", self.keep_last_checkpoints, allow_zero=True)

    @property
    def policy(self) -> PolicyConfig:
        return PolicyConfig(self.alpha, self.beta, self.random_delete_prob, self.label_smoothing)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainConfig":
        return _strict_dataclass(cls, data, "train config")

    @classmethod
    def from_json(cls, path: str | Path) -> "TrainConfig":
        return cls.from_dict(_load_json(path))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
