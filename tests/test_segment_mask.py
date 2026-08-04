"""Tests for packed-segment attention masks and packed training."""

import torch

from levt import (
    DualPolicyTrainer,
    LevTCollator,
    LevTConfig,
    LevTModel,
    PolicyConfig,
)
from levt.segment_mask import (
    cross_attention_mask,
    segment_ids,
    self_attention_mask,
)


def tiny_config(**overrides):
    """Same model config as tests/test_training_pipeline.py::tiny_config."""
    values = dict(
        vocab_size=24,
        embedding_dim=6,
        d_model=8,
        n_heads=2,
        d_ff=16,
        n_encoder_layers=1,
        n_decoder_layers=1,
        dropout=0.0,
        attention_dropout=0.0,
        max_placeholder=4,
    )
    values.update(overrides)
    return LevTConfig(**values)


def test_segment_ids_detects_boundaries():
    cfg = tiny_config()  # bos=1, eos=2, pad=0
    # Two segments: [BOS]4[EOS][BOS]5 6[EOS]
    tokens = torch.tensor([1, 4, 2, 1, 5, 6, 2])
    ids = segment_ids(tokens, cfg.bos_token_id, cfg.eos_token_id)
    assert ids.tolist() == [0, 0, 0, 1, 1, 1, 1]

    # Single segment: [BOS]4[EOS]
    single = segment_ids(torch.tensor([1, 4, 2]), cfg.bos_token_id, cfg.eos_token_id)
    assert single.tolist() == [0, 0, 0]


def test_self_mask_is_block_diagonal():
    # (1, 7) batch-first ids for the two-segment example above.
    ids = torch.tensor([[0, 0, 0, 1, 1, 1, 1]])
    mask = self_attention_mask(ids)
    assert mask.shape == (1, 7, 7)
    assert mask.dtype == torch.bool
    m = mask[0]
    # Blocks along the diagonal are fully True...
    assert m[:3, :3].all()
    assert m[3:, 3:].all()
    assert m.diagonal().all()
    # ...and off-diagonal blocks are fully False.
    assert not m[:3, 3:].any()
    assert not m[3:, :3].any()


def test_cross_mask_aligns_segments():
    tgt_ids = torch.tensor([[0, 0, 1, 1]])          # (1, 4)
    src_ids = torch.tensor([[0, 0, 0, 1, 1]])       # (1, 5)
    mask = cross_attention_mask(tgt_ids, src_ids)
    assert mask.shape == (1, 4, 5)
    assert mask.dtype == torch.bool
    m = mask[0]
    # Segment 0 (tgt rows 0..1) attends only to src positions 0..2.
    assert m[:2, :3].all()
    assert not m[:2, 3:].any()
    # Segment 1 (tgt rows 2..3) attends only to src positions 3..4.
    assert m[2:, 3:].all()
    assert not m[2:, :3].any()


def test_segment_ids_handles_empty_or_all_pad():
    cfg = tiny_config()
    empty = segment_ids(
        torch.zeros(0, dtype=torch.long), cfg.bos_token_id, cfg.eos_token_id,
    )
    assert empty.shape == (0,)
    # All-pad column must not start any new segment (pad != bos/eos).
    ids = segment_ids(torch.tensor([[0, 0, 0]]), cfg.bos_token_id, cfg.eos_token_id)
    assert ids.tolist() == [[0, 0, 0]]


def test_packed_batch_trains_without_error():
    cfg = tiny_config()
    model = LevTModel(cfg)
    # alpha=0.0 keeps the deletion roll-in on the model-filled sequence
    # (torch.rand() < 0.0 is never True), so the deletion count is
    # deterministically > 0; the packed-segment mask paths are exercised
    # identically regardless of the alpha choice.
    trainer = DualPolicyTrainer(
        model, cfg,
        PolicyConfig(alpha=0.0, beta=0.5, random_delete_prob=0.3, label_smoothing=0.1),
    )
    rows = [
        {"src": [4, 5, 2, 1, 6, 7], "target": [1, 7, 2, 1, 8, 2], "initial": [1, 2]},
        {"src": [8, 9, 2, 1, 5, 6], "target": [1, 9, 2, 1, 6, 7, 2], "initial": [1, 2]},
    ]
    collator = LevTCollator(
        cfg, max_source_length=10, max_target_length=10,
        allow_interior_boundaries=True,
    )
    batch = collator(rows)
    torch.manual_seed(0)
    prepared = trainer.prepare_batch(batch)
    sums, counts = trainer.loss_sums_and_counts(prepared)
    for name in ("plh", "tok", "del"):
        assert torch.isfinite(sums[name]), f"loss_{name} not finite"
        assert counts[name] > 0, f"count_{name} should be > 0"


def test_unpacked_training_unchanged():
    cfg = tiny_config()
    model = LevTModel(cfg)
    trainer = DualPolicyTrainer(model, cfg, PolicyConfig(alpha=1.0, beta=1.0))
    rows = [
        {"src": [4, 5], "target": [1, 7, 2], "initial": [1, 2]},
        {"src": [6], "target": [1, 8, 9, 2], "initial": [1, 10, 2]},
    ]
    batch = LevTCollator(cfg, max_source_length=10, max_target_length=10)(rows)
    torch.manual_seed(3)
    prepared = trainer.prepare_batch(batch)
    sums, counts = trainer.loss_sums_and_counts(prepared)
    for name in ("plh", "tok", "del"):
        assert torch.isfinite(sums[name]), f"loss_{name} not finite"
    assert counts["del"] > 0
