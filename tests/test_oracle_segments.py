"""Segment-boundary preservation in the oracles for packed training.

Packed rows concatenate unrelated examples; the oracles must never edit
across an ``[EOS][BOS]`` segment boundary, otherwise the roll-in loses
segment boundaries and the packed-segment attention masks misalign.
"""

import random

import torch

from levt import (
    DualPolicyTrainer,
    LevTCollator,
    LevTConfig,
    LevTModel,
    PolicyConfig,
    apply_deletion,
    insert_placeholders,
    oracle_deletion,
    oracle_insertion,
    random_deletion,
)
from levt.expert import oracle_deletion_batch, oracle_insertion_batch


def tiny_config(**overrides):
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


def count_segments(tokens, bos=1, eos=2):
    """Number of ``[BOS]...[EOS]`` segments in a packed sequence."""
    toks = tokens.tolist()
    return 1 + sum(
        1 for i in range(len(toks) - 1) if toks[i] == eos and toks[i + 1] == bos
    )


BOS, EOS = 1, 2
# Two segments: [BOS]4[EOS] | [BOS]5[EOS]
Y = torch.tensor([BOS, 4, EOS, BOS, 5, EOS])
Y_STAR = torch.tensor([BOS, 7, EOS, BOS, 8, 9, EOS])
BOUNDARY_POSITIONS = (0, 2, 3, 5)  # every BOS/EOS of both segments


def test_deletion_mask_preserves_interior_boundaries():
    mask = oracle_deletion(Y, Y_STAR, bos_idx=BOS, eos_idx=EOS)
    assert mask.tolist() == [False, True, False, False, True, False]
    for pos in BOUNDARY_POSITIONS:
        assert mask[pos].item() is False, f"boundary {pos} marked for deletion"
    kept = apply_deletion(Y, mask)
    assert count_segments(kept) == 2, "roll-in lost a segment boundary"


def test_insertion_forces_zero_gap_at_boundary():
    p_star, t_star = oracle_insertion(Y, Y_STAR, max_placeholder=4, plh_token_id=3,
                                       bos_idx=BOS, eos_idx=EOS)
    assert p_star.numel() == Y.numel() - 1
    # The gap right after the segment-ending EOS (index 2) must insert nothing.
    assert p_star[2].item() == 0
    plh = insert_placeholders(Y, p_star, plh_token_id=3)
    assert count_segments(plh) == 2, "insertion crossed a segment boundary"
    # Interior placeholders are populated from t_star in gap order.
    assert t_star.tolist() == [7, 8, 9]


def test_random_deletion_preserves_boundaries():
    random.seed(0)
    dropped = random_deletion(
        Y_STAR, drop_prob=1.0, bos_idx=BOS, eos_idx=EOS, pad_idx=0,
    )
    assert dropped.tolist() == [BOS, EOS, BOS, EOS]
    assert count_segments(dropped) == 2


def test_single_segment_is_unchanged():
    y = torch.tensor([BOS, 4, EOS])
    y_star = torch.tensor([BOS, 7, EOS])
    mask = oracle_deletion(y, y_star, bos_idx=BOS, eos_idx=EOS)
    assert mask.tolist() == [False, True, False]
    p_star, t_star = oracle_insertion(y, y_star, max_placeholder=4, plh_token_id=3,
                                       bos_idx=BOS, eos_idx=EOS)
    assert p_star.tolist() == [1, 0]
    assert t_star.tolist() == [7]


def test_batch_equals_single_on_packed():
    d_batch = oracle_deletion_batch([Y], [Y_STAR], bos_idx=BOS, eos_idx=EOS)
    assert torch.equal(d_batch[0], oracle_deletion(Y, Y_STAR, bos_idx=BOS, eos_idx=EOS))

    p_new, t_new, plh_new = oracle_insertion_batch(
        [Y], [Y_STAR], max_placeholder=4, plh_token_id=3, bos_idx=BOS, eos_idx=EOS,
    )
    p_single, t_single = oracle_insertion(
        Y, Y_STAR, max_placeholder=4, plh_token_id=3, bos_idx=BOS, eos_idx=EOS,
    )
    assert torch.equal(p_new[0], p_single)
    assert torch.equal(t_new[0], t_single)
    assert torch.equal(plh_new[0], insert_placeholders(Y, p_single, plh_token_id=3))


def test_trainer_roll_ins_preserve_segment_count():
    cfg = tiny_config()
    model = LevTModel(cfg)
    trainer = DualPolicyTrainer(
        model, cfg,
        PolicyConfig(alpha=0.5, beta=0.5, random_delete_prob=0.3, label_smoothing=0.1),
    )
    # Real packed rows carry a packed ``initial`` (one [BOS,EOS] per segment).
    packed_initial = [1, 2, 1, 2]
    rows = [
        {"src": [4, 5, 2, 1, 6, 7], "target": [1, 7, 2, 1, 8, 2],
         "initial": packed_initial},
        {"src": [8, 9, 2, 1, 5, 6], "target": [1, 9, 2, 1, 6, 7, 2],
         "initial": packed_initial},
        {"src": [4, 2, 1, 6], "target": [1, 4, 2, 1, 6, 9, 2],
         "initial": packed_initial},
    ]
    batch = LevTCollator(
        cfg, max_source_length=10, max_target_length=10,
        allow_interior_boundaries=True,
    )(rows)
    prepared = trainer.prepare_batch(batch)
    for index, target in enumerate(prepared.targets):
        expected = count_segments(target)
        assert count_segments(prepared.y_ins[index]) == expected, (
            f"sample {index}: y_ins lost a segment boundary"
        )
        assert count_segments(prepared.y_ins_plh[index]) == expected, (
            f"sample {index}: y_ins_plh lost a segment boundary"
        )
