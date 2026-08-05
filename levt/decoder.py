"""
Greedy decoder for the Levenshtein Transformer (Algorithm 2 from the paper).

The decoding iterates:
  1. Delete tokens   (with the default ``"src"`` initial strategy y0 is the full
                      source sequence, so the deletion phase runs on iteration 1;
                      an explicitly short initial sequence may still skip it)
  2. Insert placeholders
  3. Fill placeholders with predicted tokens
  4. Repeat until loop is detected or max iterations reached.

Reference: "Levenshtein Transformer" (Gu et al., NeurIPS 2019), Algorithm 2.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from .config import LevTConfig
from .model import LevTModel


class GreedyDecoder:
    """
    Greedy iterative decoding for LevT.

    Usage::

        decoder = GreedyDecoder(model, config)
        output = decoder.decode(src_tokens, y0_tokens)
        # or for batched decoding:
        outputs = decoder.decode_batch(src_batch, y0_batch)
    """

    def __init__(self, model: LevTModel, config: LevTConfig,
                 temperature: float = 0.0):
        self.model = model
        self.cfg = config
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        self.temperature = float(temperature)

    # ------------------------------------------------------------------
    # Single-sequence decoding
    # ------------------------------------------------------------------

    @torch.no_grad()
    def decode(
        self,
        src_tokens: torch.Tensor,
        y0: Optional[torch.Tensor] = None,
        src_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, int]:
        was_training = self.model.training
        self.model.eval()
        try:
            return self._decode(src_tokens, y0, src_padding_mask)
        finally:
            self.model.train(was_training)

    @torch.no_grad()
    def decode_with_trace(
        self,
        src_tokens: torch.Tensor,
        y0: Optional[torch.Tensor] = None,
        src_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, int, list]:
        """Like :meth:`decode` but also returns a step-by-step trace.

        Returns:
            output, num_iterations, trace

        ``trace`` is a list of ``(phase_label, token_tensor, entropy)`` tuples
        where ``phase_label`` is one of ``"start"``, ``"del"``, ``"plh"``,
        ``"fill"`` and ``entropy`` is the average per-decision entropy in nats
        (``None`` for ``"start"`` and skipped phases).
        """
        was_training = self.model.training
        self.model.eval()
        try:
            return self._decode(src_tokens, y0, src_padding_mask, return_trace=True)
        finally:
            self.model.train(was_training)

    def _decode(
        self,
        src_tokens: torch.Tensor,
        y0: Optional[torch.Tensor] = None,
        src_padding_mask: Optional[torch.Tensor] = None,
        return_trace: bool = False,
    ) -> Tuple[torch.Tensor, int] | Tuple[torch.Tensor, int, list]:
        """
        Decode one sequence.

        Args:
            src_tokens:       (src_len,) source tokens
            y0:               (L0,) initial target tokens. Default: per
                              ``config.initial_strategy`` — the full source
                              sequence (``"src"``, edit task) or ``[BOS, EOS]``
                              (``"bos_eos"``). Pass an explicit y0 to override.
            src_padding_mask: (src_len,) or None
            return_trace:     if True, also return a step-by-step trace with
                              per-phase average entropies

        Returns:
            output:           final token sequence
            num_iterations:   number of refinement iterations taken
            trace:            (only if ``return_trace=True``) list of
                              ``(phase, token_tensor, entropy)`` tuples
        """
        if y0 is None:
            # Default y0 follows config.initial_strategy: the full source
            # sequence (edit task) or a bare [BOS, EOS] (generation from scratch).
            y0 = torch.tensor(
                self.cfg.default_initial(src_tokens.tolist()),
                dtype=src_tokens.dtype,
                device=src_tokens.device,
            )

        src = src_tokens.unsqueeze(1)   # (S, 1)
        src_mask = src_padding_mask.unsqueeze(0) if src_padding_mask is not None else None

        # Encode source once (reused across iterations)
        memory = self.model.encode(src, src_mask)

        y = y0
        prev_y = None
        iteration = 0
        trace: list = []
        if return_trace:
            trace.append(("start", y.clone(), None))

        while iteration < self.cfg.max_iterations:
            # ---- Phase 1: Deletion ----
            is_empty = len(y) <= 2 and y[0].item() == self.cfg.bos_token_id and y[-1].item() == self.cfg.eos_token_id
            del_ent = None
            if not is_empty:
                if return_trace:
                    y_after_del, del_ent = self._delete_tokens(memory, y, src_mask, return_entropy=True)
                else:
                    y_after_del = self._delete_tokens(memory, y, src_mask)
            else:
                y_after_del = y  # skip deletion for empty sequence

            if return_trace:
                trace.append(("del", y_after_del.clone(), del_ent))

            # ---- Termination: loop detection ----
            if iteration > 0 and prev_y is not None:
                if torch.equal(y_after_del, prev_y):
                    break  # direct loop

            prev_y = y_after_del.clone()

            # ---- Phase 2: Insert placeholders ----
            plh_ent = None
            if return_trace:
                y_with_plh, plh_ent = self._insert_placeholders(memory, y_after_del, src_mask, return_entropy=True)
            else:
                y_with_plh = self._insert_placeholders(memory, y_after_del, src_mask)

            if return_trace:
                trace.append(("plh", y_with_plh.clone(), plh_ent))

            # ---- Termination: nothing changed ----
            if torch.equal(y_with_plh, y_after_del) and torch.equal(y_with_plh, y):
                break

            # ---- Phase 3: Fill placeholders ----
            fill_ent = None
            if torch.equal(y_with_plh, y_after_del):
                y = y_with_plh  # nothing inserted, skip fill
            else:
                if return_trace:
                    y, fill_ent = self._fill_tokens(memory, y_with_plh, src_mask, return_entropy=True)
                else:
                    y = self._fill_tokens(memory, y_with_plh, src_mask)

            if return_trace:
                trace.append(("fill", y.clone(), fill_ent))

            iteration += 1

        if return_trace:
            return y, iteration, trace
        return y, iteration

    # ------------------------------------------------------------------
    # Batched decoding
    # ------------------------------------------------------------------

    @torch.no_grad()
    def decode_batch(
        self,
        src_tokens: torch.Tensor,
        y0_batch: Optional[torch.Tensor] = None,
        src_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], List[int]]:
        """
        Decode a batch by iterating each example independently.

        Args:
            src_tokens:       (src_len, batch) source tokens
            y0_batch:         list of initial sequences, or None to use each
                              example's config-driven default y0
            src_padding_mask: (batch, src_len) or None

        Returns:
            outputs:        list of output tensors
            iterations:     list of iteration counts
        """
        outputs: List[torch.Tensor] = []
        iterations: List[int] = []

        batch_size = src_tokens.size(1)
        for i in range(batch_size):
            src_i = src_tokens[:, i]
            mask_i = src_padding_mask[i] if src_padding_mask is not None else None
            y0_i = y0_batch[i] if y0_batch is not None else None
            out_i, iters_i = self.decode(src_i, y0_i, mask_i)
            outputs.append(out_i)
            iterations.append(iters_i)

        return outputs, iterations

    # ------------------------------------------------------------------
    # Sampling helper
    # ------------------------------------------------------------------

    def _sample(self, logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Sample from logits with temperature scaling.

        When ``temperature == 0`` this is equivalent to argmax.
        """
        if self.temperature == 0.0:
            return logits.argmax(dim=dim)
        probs = torch.softmax(logits / self.temperature, dim=dim)
        return torch.multinomial(probs, num_samples=1).squeeze(dim)

    # ------------------------------------------------------------------
    # Entropy helper
    # ------------------------------------------------------------------

    @staticmethod
    def _entropy_from_logits(logits: torch.Tensor) -> float:
        """Average per-decision entropy (nats) over the last dimension.

        Returns 0.0 for empty tensors.
        """
        if logits.numel() == 0:
            return 0.0
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        entropy = -(probs * log_probs).sum(dim=-1)
        return entropy.mean().item()

    # ------------------------------------------------------------------
    # Phase implementations
    # ------------------------------------------------------------------

    def _delete_tokens(
        self,
        memory: torch.Tensor,
        y: torch.Tensor,
        src_mask: Optional[torch.Tensor],
        return_entropy: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, float]:
        """Apply deletion policy: sample / argmax π^del → keep only tokens predicted as 'keep'."""
        y_t = y.unsqueeze(1)  # (T, 1)
        tgt_mask = torch.zeros(1, len(y), dtype=torch.bool, device=y.device)

        out = self.model.decode_with_memory(
            memory, y_t, src_mask, tgt_mask,
            return_deletion=True, return_placeholder=False, return_token=False,
        )
        del_logits = out["del_logits"]  # (T, 1, 2)

        # Class 0 = keep, Class 1 = delete
        del_preds = self._sample(del_logits.squeeze(1), dim=-1)  # (T,)
        del_preds = del_preds.bool()  # True = delete

        # Never delete boundaries
        del_preds[0] = False
        del_preds[-1] = False

        result = y[~del_preds]
        if return_entropy:
            return result, self._entropy_from_logits(del_logits)
        return result

    def _insert_placeholders(
        self,
        memory: torch.Tensor,
        y: torch.Tensor,
        src_mask: Optional[torch.Tensor],
        return_entropy: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, float]:
        """Apply placeholder policy: sample / argmax π^plh → insert <PLH> tokens."""
        y_t = y.unsqueeze(1)
        tgt_mask = torch.zeros(1, len(y), dtype=torch.bool, device=y.device)

        out = self.model.decode_with_memory(
            memory, y_t, src_mask, tgt_mask,
            return_deletion=False, return_placeholder=True, return_token=False,
        )
        plh_logits = out["plh_logits"]  # (T-1, 1, K_max+1)

        # Apply empty-placeholder penalty
        if self.cfg.placeholder_penalty != 0.0:
            plh_logits = plh_logits.clone()
            plh_logits[:, :, 0] = plh_logits[:, :, 0] - self.cfg.placeholder_penalty

        p_preds = self._sample(plh_logits.squeeze(1), dim=-1)  # (T-1,)

        # Build sequence with placeholders
        y_list = y.tolist()
        result: List[int] = []
        for i, tok in enumerate(y_list):
            result.append(tok)
            if i < len(p_preds):
                count = int(p_preds[i].item())
                for _ in range(min(count, self.cfg.max_placeholder)):
                    result.append(self.cfg.plh_token_id)

        result_tensor = torch.tensor(result, dtype=y.dtype, device=y.device)
        if return_entropy:
            return result_tensor, self._entropy_from_logits(plh_logits)
        return result_tensor

    def _fill_tokens(
        self,
        memory: torch.Tensor,
        y_with_plh: torch.Tensor,
        src_mask: Optional[torch.Tensor],
        return_entropy: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, float]:
        """Apply token policy: sample / argmax π^tok → replace <PLH> with predicted tokens."""
        y_t = y_with_plh.unsqueeze(1)
        tgt_mask = torch.zeros(1, len(y_with_plh), dtype=torch.bool, device=y_with_plh.device)

        placeholder_positions = y_t.eq(self.cfg.plh_token_id)
        out = self.model.decode_with_memory(
            memory, y_t, src_mask, tgt_mask,
            return_deletion=False, return_placeholder=False, return_token=True,
            token_positions=placeholder_positions,
        )
        tok_logits = out["tok_logits"]  # (num_placeholders, V)

        # Compute entropy before masking reserved tokens — masking produces
        # -inf logits which cause NaN in the entropy calculation.
        if return_entropy:
            fill_entropy = self._entropy_from_logits(tok_logits)

        reserved = {
            self.cfg.pad_token_id, self.cfg.bos_token_id,
            self.cfg.eos_token_id, self.cfg.plh_token_id,
        }
        tok_logits[:, list(reserved)] = float("-inf")
        tok_preds = self._sample(tok_logits, dim=-1)

        result = y_with_plh.clone()
        result[result.eq(self.cfg.plh_token_id)] = tok_preds
        if return_entropy:
            return result, fill_entropy
        return result
