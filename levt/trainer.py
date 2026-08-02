"""Batched dual-policy training for the Levenshtein Transformer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .config import LevTConfig, PolicyConfig
from .expert import (
    apply_deletion,
    oracle_deletion_batch,
    oracle_insertion_batch,
    random_deletion,
)
from .model import LevTModel


@dataclass
class PreparedBatch:
    """A batch whose stochastic roll-ins and CPU oracle labels are fixed."""

    src_tokens: torch.Tensor
    src_padding_mask: torch.Tensor
    y_ins: List[torch.Tensor]
    p_star: List[torch.Tensor]
    y_ins_plh: List[torch.Tensor]
    t_star: List[torch.Tensor]
    y_del: List[torch.Tensor]
    d_star: List[torch.Tensor]
    initial: List[torch.Tensor]
    targets: List[torch.Tensor]

    @property
    def counts(self) -> Dict[str, int]:
        return {
            "plh": sum(oracle.numel() for oracle in self.p_star),
            "tok": sum(oracle.numel() for oracle in self.t_star),
            "del": 0,
        }


class DualPolicyTrainer:
    """Build CPU oracle roll-ins per sample and compute three batched losses."""

    def __init__(
        self,
        model: LevTModel,
        config: LevTConfig,
        policy_config: Optional[PolicyConfig] = None,
        oracle_batch_size: int = 0,
    ) -> None:
        self.model = model
        self.cfg = config
        # Pairs per C++ oracle call. 0 = whole training batch in one call;
        # a positive value chunks the batch (trades fixed per-call overhead
        # against per-call latency / memory).
        self._oracle_batch_size = int(oracle_batch_size)
        self.policy = policy_config or PolicyConfig(
            alpha=0.5 if config.alpha is None else config.alpha,
            beta=0.5 if config.beta is None else config.beta,
            random_delete_prob=(
                0.3 if config.random_delete_prob is None else config.random_delete_prob
            ),
            label_smoothing=(
                0.1 if config.label_smoothing is None else config.label_smoothing
            ),
        )

    def _oracle_chunks(self, n: int):
        """Yield (start, end) ranges for chunking ``n`` pairs into oracle calls.

        ``oracle_batch_size == 0`` yields a single whole-batch range (one C++
        call); a positive value yields ranges of that size.
        """
        bs = self._oracle_batch_size
        if bs <= 0 or bs >= n:
            yield 0, n
            return
        for start in range(0, n, bs):
            yield start, min(start + bs, n)

    def prepare_batch(
        self,
        batch_or_src: Dict[str, Any] | torch.Tensor,
        y0: Optional[torch.Tensor] = None,
        y_star: Optional[torch.Tensor] = None,
        src_padding_mask: Optional[torch.Tensor] = None,
    ) -> PreparedBatch:
        """Sample roll-ins once and materialize all oracle labels without a graph."""
        batch = self._normalize_batch(batch_or_src, y0, y_star, src_padding_mask)
        device = next(self.model.parameters()).device
        src = batch["src_tokens"].to(device)
        src_mask = batch["src_padding_mask"].to(device)
        initial = [sequence.detach().cpu() for sequence in batch["initial"]]
        targets = [sequence.detach().cpu() for sequence in batch["targets"]]
        if len(initial) != src.size(1) or len(targets) != src.size(1):
            raise ValueError("batch lists must match src_tokens batch dimension")

        if "y_ins" in batch:
            # Use pre-computed oracles with beta mixing
            y_ins: List[torch.Tensor] = []
            p_star: List[torch.Tensor] = []
            t_star: List[torch.Tensor] = []
            y_ins_plh: List[torch.Tensor] = []
            for i in range(len(initial)):
                if torch.rand(()).item() < self.policy.beta:
                    y_ins.append(batch["y_ins"][i])
                    p_star.append(batch["p_star"][i])
                    t_star.append(batch["t_star"][i])
                    y_ins_plh.append(batch["y_ins_plh"][i])
                else:
                    y_ins.append(batch["y_ins_rnd"][i])
                    p_star.append(batch["p_star_rnd"][i])
                    t_star.append(batch["t_star_rnd"][i])
                    y_ins_plh.append(batch["y_ins_plh_rnd"][i])
        else:
            # Batched oracle computation (backward compatible; falls back to
            # per-sample calls when the C++ extension is unavailable).
            # Beta / random branch decisions are drawn in index order to keep
            # the torch.rand + random.random RNG streams identical to the
            # original per-sample loop.
            use_oracle = [
                torch.rand(()).item() < self.policy.beta for _ in initial
            ]
            oracle_idx = [i for i, u in enumerate(use_oracle) if u]
            rnd_idx = [i for i, u in enumerate(use_oracle) if not u]

            ins_by_idx: Dict[int, torch.Tensor] = {}
            if oracle_idx:
                sub_y0 = [initial[i] for i in oracle_idx]
                sub_targets = [targets[i] for i in oracle_idx]
                masks: List[torch.Tensor] = []
                for s, e in self._oracle_chunks(len(sub_y0)):
                    masks.extend(oracle_deletion_batch(sub_y0[s:e], sub_targets[s:e]))
                for k, i in enumerate(oracle_idx):
                    ins_by_idx[i] = apply_deletion(initial[i], masks[k])

            for k, i in enumerate(rnd_idx):
                ins_by_idx[i] = random_deletion(
                    targets[i],
                    drop_prob=self.policy.random_delete_prob,
                    bos_idx=self.cfg.bos_token_id,
                    eos_idx=self.cfg.eos_token_id,
                    pad_idx=self.cfg.pad_token_id,
                )

            y_ins = [ins_by_idx[i] for i in range(len(initial))]

            p_star_parts: List[torch.Tensor] = []
            t_star_parts: List[torch.Tensor] = []
            plh_parts: List[torch.Tensor] = []
            for s, e in self._oracle_chunks(len(y_ins)):
                p_part, t_part, plh_part = oracle_insertion_batch(
                    y_ins[s:e], targets[s:e],
                    max_placeholder=self.cfg.max_placeholder,
                    plh_token_id=self.cfg.plh_token_id,
                )
                p_star_parts.extend(p_part)
                t_star_parts.extend(t_part)
                plh_parts.extend(plh_part)
            p_star, t_star, y_ins_plh = p_star_parts, t_star_parts, plh_parts

        return PreparedBatch(
            src, src_mask, y_ins, p_star, y_ins_plh, t_star, [], [], initial, targets,
        )

    def loss_sums_and_counts(
        self, prepared: PreparedBatch,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, int]]:
        """Return differentiable per-head loss sums and valid-label counts."""
        device = next(self.model.parameters()).device
        memory = self.model.encode(prepared.src_tokens, prepared.src_padding_mask)

        # --- Deferred fill + deletion oracle (uses detached memory) ---
        with torch.no_grad():
            model_filled = self._model_fill_batch(
                memory.detach(), prepared.src_padding_mask, prepared.y_ins_plh,
            )

        y_del: List[torch.Tensor] = []
        roll_ins: List[torch.Tensor] = []
        for seed, filled, target in zip(prepared.initial, model_filled, prepared.targets):
            roll_in = seed if torch.rand((), device="cpu").item() < self.policy.alpha else filled
            roll_in = roll_in.detach().cpu()
            y_del.append(roll_in)
            roll_ins.append(roll_in)

        # Batched C++ alignments (per-sample fallback if the extension is
        # unavailable). bos/eos are vestigial — the mask always keeps
        # positions 0 and n-1.
        d_star: List[torch.Tensor] = []
        for s, e in self._oracle_chunks(len(roll_ins)):
            d_star.extend(oracle_deletion_batch(
                roll_ins[s:e], prepared.targets[s:e],
            ))

        prepared.y_del = y_del
        prepared.d_star = d_star

        # --- Combined batched decoder pass for all three heads ---
        bs = len(prepared.y_ins)

        # Pad each group separately (kept for target creation)
        plh_tokens, _ = self._pad(prepared.y_ins, device)
        tok_tokens, _ = self._pad(prepared.y_ins_plh, device)
        del_tokens, _ = self._pad(prepared.y_del, device)

        # Determine max length across all three groups
        max_len = max(plh_tokens.size(0), tok_tokens.size(0), del_tokens.size(0))

        # Build combined tokens: (max_len, 3*bs)
        combined = torch.full(
            (max_len, 3 * bs), self.cfg.pad_token_id,
            dtype=torch.long, device=device,
        )
        combined[:plh_tokens.size(0), :bs] = plh_tokens
        combined[:tok_tokens.size(0), bs:2 * bs] = tok_tokens
        combined[:del_tokens.size(0), 2 * bs:] = del_tokens

        # Combined target padding mask: (3*bs, max_len)
        combined_mask = combined.eq(self.cfg.pad_token_id).transpose(0, 1)

        # Replicate source mask 3× along batch dim
        src_mask_3x = prepared.src_padding_mask.repeat(3, 1)

        # Replicate memory 3× so cross-attention batch dim matches the combined decoder
        memory_3x = memory.repeat(1, 3, 1)

        # Build token_positions mask for the tok slice only (columns bs:2*bs)
        tok_pos_combined = torch.zeros((max_len, 3 * bs), dtype=torch.bool, device=device)
        for batch_index, sequence in enumerate(prepared.y_ins_plh):
            positions = sequence.eq(self.cfg.plh_token_id)
            tok_pos_combined[:sequence.numel(), bs + batch_index] = positions.to(device)

        # Single decoder call — return all three heads
        out = self.model.decode_with_memory(
            memory_3x, combined, src_mask_3x, combined_mask,
            return_deletion=True, return_placeholder=True, return_token=True,
            token_positions=tok_pos_combined,
        )

        # Split outputs back
        plh_out = out["plh_logits"][:, :bs, :]       # (max_len-1, bs, plh_classes)
        del_out = out["del_logits"][:, 2 * bs:, :]    # (max_len, bs, 2)
        tok_out = out["tok_logits"]                   # already flattened by token_positions

        # --- Target creation (unchanged logic) ---
        plh_targets = torch.full(plh_out.shape[:2], -100, dtype=torch.long, device=device)
        for batch_index, oracle in enumerate(prepared.p_star):
            expected = len(prepared.y_ins[batch_index]) - 1
            if oracle.numel() != expected:
                raise RuntimeError("placeholder oracle length does not match valid gap count")
            plh_targets[:expected, batch_index] = oracle.to(device)

        tok_positions = tok_tokens.eq(self.cfg.plh_token_id)
        tok_targets_full = torch.full(
            tok_tokens.shape, -100, dtype=torch.long, device=device,
        )
        for batch_index, (sequence, oracle) in enumerate(
            zip(prepared.y_ins_plh, prepared.t_star)
        ):
            positions = sequence.eq(self.cfg.plh_token_id).nonzero(as_tuple=True)[0]
            if positions.numel() != oracle.numel():
                raise RuntimeError("token oracle count does not match placeholder count")
            tok_targets_full[positions.to(device), batch_index] = oracle.to(device)
        tok_targets = tok_targets_full[tok_positions]

        del_targets = torch.full(del_out.shape[:2], -100, dtype=torch.long, device=device)
        for batch_index, (sequence, oracle) in enumerate(zip(prepared.y_del, prepared.d_star)):
            if oracle.numel() != sequence.numel():
                raise RuntimeError("deletion oracle length does not match roll-in length")
            if sequence.numel() > 2:
                del_targets[1:sequence.numel() - 1, batch_index] = oracle[1:-1].long().to(device)

        sums = {
            "plh": self._cross_entropy_sum(plh_out, plh_targets, self.policy.label_smoothing),
            "tok": self._cross_entropy_sum(tok_out, tok_targets, self.policy.label_smoothing),
            "del": self._cross_entropy_sum(del_out, del_targets, 0.0),
        }
        counts = {
            "plh": sum(oracle.numel() for oracle in prepared.p_star),
            "tok": sum(oracle.numel() for oracle in prepared.t_star),
            "del": sum(max(0, sequence.numel() - 2) for sequence in y_del),
        }
        return sums, counts

    @staticmethod
    def normalized_loss(
        sums: Dict[str, torch.Tensor], counts: Dict[str, int],
    ) -> torch.Tensor:
        """Sum per-head means, treating a head with no labels as zero."""
        return sum(
            value / counts[name] if counts[name] else value * 0.0
            for name, value in sums.items()
        )

    def train_step(
        self,
        batch_or_src: Dict[str, Any] | torch.Tensor,
        y0: Optional[torch.Tensor] = None,
        y_star: Optional[torch.Tensor] = None,
        src_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compatibility API returning per-head means for one prepared batch."""
        prepared = self.prepare_batch(batch_or_src, y0, y_star, src_padding_mask)
        sums, counts = self.loss_sums_and_counts(prepared)
        losses = {
            name: value / counts[name] if counts[name] else value * 0.0
            for name, value in sums.items()
        }
        total = sum(losses.values())
        metrics = {
            "loss_ins_plh": float(losses["plh"].detach()),
            "loss_ins_tok": float(losses["tok"].detach()),
            "loss_del": float(losses["del"].detach()),
            "loss_total": float(total.detach()),
        }
        return total, metrics

    def _normalize_batch(
        self,
        batch_or_src: Dict[str, Any] | torch.Tensor,
        y0: Optional[torch.Tensor],
        y_star: Optional[torch.Tensor],
        src_padding_mask: Optional[torch.Tensor],
    ) -> Dict[str, Any]:
        if isinstance(batch_or_src, dict):
            required = {"src_tokens", "src_padding_mask", "initial", "targets"}
            missing = sorted(required - set(batch_or_src))
            if missing:
                raise ValueError(f"batch missing keys: {', '.join(missing)}")
            return batch_or_src
        if y0 is None or y_star is None:
            raise ValueError("legacy train_step requires y0 and y_star")
        src = batch_or_src
        if src.ndim != 1 or y0.ndim != 1 or y_star.ndim != 1:
            raise ValueError("legacy train_step tensors must be one-dimensional")
        src_2d = src.unsqueeze(1)
        if src_padding_mask is None:
            mask = src_2d.eq(self.cfg.pad_token_id).transpose(0, 1)
        else:
            mask = src_padding_mask
            if mask.ndim == 1:
                mask = mask.unsqueeze(0)
        return {
            "src_tokens": src_2d,
            "src_padding_mask": mask,
            "initial": [y0],
            "targets": [y_star],
        }

    def _model_fill_batch(
        self,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        sequences: Sequence[torch.Tensor],
    ) -> List[torch.Tensor]:
        device = memory.device
        tokens, padding_mask = self._pad(sequences, device)
        positions = tokens.eq(self.cfg.plh_token_id)
        logits = self.model.decode_with_memory(
            memory, tokens, src_mask, padding_mask,
            return_deletion=False, return_placeholder=False, return_token=True,
            token_positions=positions,
        )["tok_logits"]
        predictions = logits.argmax(dim=-1)
        predicted_tokens = tokens.clone()
        predicted_tokens[positions] = predictions
        predicted_tokens = predicted_tokens.cpu()
        results: List[torch.Tensor] = []
        for batch_index, sequence in enumerate(sequences):
            result = sequence.clone()
            sequence_positions = result.eq(self.cfg.plh_token_id)
            result[sequence_positions] = predicted_tokens[
                :sequence.numel(), batch_index
            ][sequence_positions]
            results.append(result)
        return results

    def _pad(
        self, sequences: Sequence[torch.Tensor], device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not sequences:
            raise ValueError("cannot pad an empty sequence collection")
        max_length = max(sequence.numel() for sequence in sequences)
        tokens = torch.full(
            (max_length, len(sequences)), self.cfg.pad_token_id,
            dtype=torch.long, device=device,
        )
        for batch_index, sequence in enumerate(sequences):
            tokens[:sequence.numel(), batch_index] = sequence.to(device)
        return tokens, tokens.eq(self.cfg.pad_token_id).transpose(0, 1)

    @staticmethod
    def _cross_entropy_sum(
        logits: torch.Tensor,
        targets: torch.Tensor,
        label_smoothing: float,
    ) -> torch.Tensor:
        valid = targets.ne(-100)
        if not valid.any():
            return logits.sum() * 0.0
        return F.cross_entropy(
            logits[valid], targets[valid],
            label_smoothing=label_smoothing, reduction="sum",
        )
