"""Tests for the batched/packed C++ Levenshtein oracle ops.

Validates that the packed batch entry points (levenshtein_deletion_batch,
levenshtein_insertion_batch) produce exactly the same oracles as the
per-sample single-pair path and as an independent pure-Python oracle, plus
edge cases (empty batch, length mismatch, max_placeholder cap) and the
trainer integration (RNG-order-preserving batched path).
"""

from __future__ import annotations

import random

import pytest
import torch

from levt.config import LevTConfig, PolicyConfig
from levt.expert import (
    _levenshtein_align_py,
    apply_deletion,
    insert_placeholders,
    oracle_deletion,
    oracle_deletion_batch,
    oracle_insertion,
    oracle_insertion_batch,
    random_deletion,
)
from levt.model import LevTModel
from levt.trainer import DualPolicyTrainer


def _mod():
    from levt._levenshtein_ops import _get_module
    return _get_module()


CPP = _mod() is not None


# ── Edge-case pairs (BOS=1, EOS=2, interior tokens ≥ 4) ──────────────────
def _edge_pairs():
    raw = [
        ([1, 2], [1, 3, 4, 2]),                    # single gap, no deletion
        ([1, 2, 3, 4], [1, 2, 3, 4]),              # full keep
        ([1, 9, 2], [1, 2]),                       # delete interior
        ([1, 2], [1, 9, 8, 2]),                    # insert into one gap
        ([1, 2, 3, 4, 5], [1, 2]),                 # delete trailing
        ([1, 2], [1, 2, 3, 4, 5]),                 # trailing insertions
        ([1, 2, 3, 4], [1, 9, 2, 8, 3, 4]),        # multi-gap insertions
        ([1, 2, 3, 4], [9, 9, 9]),                 # DP deletes boundaries; masks force keep
        ([1, 2], [9, 1, 2]),                       # leading insertion is dropped
        ([1, 2], [1]),                             # y_star missing EOS (boundary deleted by DP)
        ([1, 2], [1, 2]),                          # identical
    ]
    return [(torch.tensor(y, dtype=torch.long), torch.tensor(ys, dtype=torch.long))
            for y, ys in raw]


def _random_pairs(seed: int, n: int, max_len: int = 24):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        ln = rng.randint(2, max_len)
        m = rng.randint(2, max_len)
        y = [1] + [rng.randint(4, 40) for _ in range(ln - 2)] + [2]
        ys = [1] + [rng.randint(4, 40) for _ in range(m - 2)] + [2]
        out.append((torch.tensor(y, dtype=torch.long),
                    torch.tensor(ys, dtype=torch.long)))
    return out


# ── Independent pure-Python oracles (built on _levenshtein_align_py) ─────
def _del_mask_py(y: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
    del_list, _ = _levenshtein_align_py(y, ys)
    mask = torch.zeros(len(y), dtype=torch.bool)
    for d in del_list:
        mask[d] = True
    mask[0] = False
    mask[-1] = False
    return mask


def _ins_py(y, ys, max_placeholder=255, plh_token_id=3):
    del_list, ins_lists = _levenshtein_align_py(y, ys)
    del_set = set(del_list)
    surviving = [i for i in range(len(y)) if i not in del_set]
    p_star = torch.zeros(len(y) - 1, dtype=torch.long)
    t_parts = []
    for gi, tokens in enumerate(ins_lists):
        left_orig = surviving[gi]
        count = min(len(tokens), max_placeholder)
        p_star[left_orig] = count
        if count:
            t_parts.append(torch.tensor(tokens[:count], dtype=torch.long))
    t_star = torch.cat(t_parts) if t_parts else torch.tensor([], dtype=torch.long)
    plh = insert_placeholders(y, p_star, plh_token_id=plh_token_id)
    return p_star, t_star, plh


# ── Correctness: batch vs single vs pure-Python ──────────────────────────
@pytest.mark.parametrize("y,ys", _edge_pairs() + _random_pairs(1, 40, 16))
def test_deletion_batch_matches_single(y, ys):
    got = oracle_deletion_batch([y], [ys])[0]
    exp = oracle_deletion(y, ys)
    assert torch.equal(got, exp)


@pytest.mark.parametrize("y,ys", _edge_pairs() + _random_pairs(2, 40, 16))
def test_insertion_batch_matches_single(y, ys):
    p_new, t_new, plh_new = oracle_insertion_batch([y], [ys], max_placeholder=4, plh_token_id=3)
    p_single, t_single = oracle_insertion(y, ys, max_placeholder=4, plh_token_id=3)
    plh_single = insert_placeholders(y, p_single, plh_token_id=3)
    assert torch.equal(p_new[0], p_single)
    assert torch.equal(t_new[0], t_single)
    assert torch.equal(plh_new[0], plh_single)


@pytest.mark.parametrize("y,ys", _edge_pairs() + _random_pairs(3, 40, 16))
def test_deletion_batch_matches_pure_python(y, ys):
    got = oracle_deletion_batch([y], [ys])[0]
    assert torch.equal(got, _del_mask_py(y, ys))


@pytest.mark.parametrize("y,ys", _edge_pairs() + _random_pairs(4, 40, 16))
def test_insertion_batch_matches_pure_python(y, ys):
    p_new, t_new, plh_new = oracle_insertion_batch([y], [ys], max_placeholder=4, plh_token_id=3)
    p_py, t_py, plh_py = _ins_py(y, ys, max_placeholder=4, plh_token_id=3)
    assert torch.equal(p_new[0], p_py)
    assert torch.equal(t_new[0], t_py)
    assert torch.equal(plh_new[0], plh_py)


def test_multi_sample_batch_is_per_sample():
    pairs = [(torch.tensor(y, dtype=torch.long), torch.tensor(ys, dtype=torch.long))
             for y, ys in _edge_pairs()[:5]]
    ys_l = [p[0] for p in pairs]
    ys_s = [p[1] for p in pairs]
    dms = oracle_deletion_batch(ys_l, ys_s)
    p_all, t_all, plh_all = oracle_insertion_batch(ys_l, ys_s, max_placeholder=4, plh_token_id=3)
    for i, (y, y_star) in enumerate(pairs):
        assert torch.equal(dms[i], oracle_deletion(y, y_star))
        p_s, t_s = oracle_insertion(y, y_star, max_placeholder=4, plh_token_id=3)
        assert torch.equal(p_all[i], p_s)
        assert torch.equal(t_all[i], t_s)
        assert torch.equal(plh_all[i], insert_placeholders(y, p_s, plh_token_id=3))


def test_max_placeholder_cap():
    y = torch.tensor([1, 2], dtype=torch.long)
    ys = torch.tensor([1, 9, 8, 7, 6, 2], dtype=torch.long)  # one gap with 4 tokens
    p_new, t_new, plh_new = oracle_insertion_batch([y], [ys], max_placeholder=2, plh_token_id=3)
    assert p_new[0].tolist() == [2]
    assert t_new[0].tolist() == [9, 8]
    assert plh_new[0].tolist() == [1, 3, 3, 2]
    # pure-Python reference agrees
    p_py, t_py, plh_py = _ins_py(y, ys, max_placeholder=2, plh_token_id=3)
    assert torch.equal(p_new[0], p_py) and torch.equal(t_new[0], t_py) and torch.equal(plh_new[0], plh_py)


# ── C++-specific: empty batch + length mismatch ──────────────────────────
@pytest.mark.skipif(not CPP, reason="C++ extension unavailable")
def test_empty_batch():
    mod = _mod()
    dm, do = mod.levenshtein_deletion_batch([], [])
    assert dm.numel() == 0
    assert do.tolist() == [0]
    pp, po, tt, to, ip, io = mod.levenshtein_insertion_batch([], [], 255, 3)
    for t in (pp, tt, ip):
        assert t.numel() == 0
    for off in (po, to, io):
        assert off.tolist() == [0]


@pytest.mark.skipif(not CPP, reason="C++ extension unavailable")
def test_length_mismatch_raises():
    mod = _mod()
    y = torch.tensor([1, 2], dtype=torch.long)
    with pytest.raises(ValueError):
        mod.levenshtein_deletion_batch([y], [])
    with pytest.raises(ValueError):
        mod.levenshtein_insertion_batch([y], [], 255, 3)


# ── Fallback path (extension disabled) produces identical results ────────
@pytest.mark.parametrize("y,ys", _edge_pairs()[:6])
def test_batch_fallback_matches(y, ys, monkeypatch):
    import levt.expert as expert

    monkeypatch.setattr(expert, "levenshtein_deletion_cpp_batch", None)
    monkeypatch.setattr(expert, "levenshtein_insertion_cpp_batch", None)
    got = oracle_deletion_batch([y], [ys])[0]
    assert torch.equal(got, _del_mask_py(y, ys))
    p, t, plh = oracle_insertion_batch([y], [ys], max_placeholder=4, plh_token_id=3)
    p_py, t_py, plh_py = _ins_py(y, ys, max_placeholder=4, plh_token_id=3)
    assert torch.equal(p[0], p_py)
    assert torch.equal(t[0], t_py)
    assert torch.equal(plh[0], plh_py)


# ── Trainer integration: batched path is RNG-order-preserving ────────────
def test_trainer_batched_path_equals_old_per_sample():
    cfg = LevTConfig(
        vocab_size=32, embedding_dim=16, d_model=16, n_heads=2, d_ff=32,
        n_encoder_layers=1, n_decoder_layers=1, dropout=0.0, attention_dropout=0.0,
        max_source_positions=128, max_target_positions=128,
        pad_token_id=0, bos_token_id=1, eos_token_id=2, plh_token_id=3,
        max_placeholder=5,
    )
    model = LevTModel(cfg)
    policy = PolicyConfig(alpha=0.5, beta=0.5, random_delete_prob=0.3, label_smoothing=0.1)
    trainer = DualPolicyTrainer(model, cfg, policy)

    def rand_seq():
        n = random.randint(4, 12)
        return torch.tensor([1] + [random.randint(4, 31) for _ in range(n - 2)] + [2],
                            dtype=torch.long)

    random.seed(1234)
    torch.manual_seed(1234)
    initial = [rand_seq() for _ in range(8)]
    targets = [rand_seq() for _ in range(8)]
    maxlen = max(t.numel() for t in initial)
    src = torch.stack([torch.nn.functional.pad(t, (0, maxlen - t.numel())) for t in initial]).transpose(0, 1)
    src_mask = src.eq(cfg.pad_token_id).transpose(0, 1)
    batch = {"src_tokens": src, "src_padding_mask": src_mask, "initial": initial, "targets": targets}

    # NEW batched path
    torch.manual_seed(1234)
    random.seed(1234)
    pb = trainer.prepare_batch(batch)
    new = {
        "y_ins": [t.clone() for t in pb.y_ins],
        "p_star": [t.clone() for t in pb.p_star],
        "t_star": [t.clone() for t in pb.t_star],
        "y_ins_plh": [t.clone() for t in pb.y_ins_plh],
    }

    # OLD per-sample path (reimplementation, identical RNG order)
    torch.manual_seed(1234)
    random.seed(1234)
    old = {"y_ins": [], "p_star": [], "t_star": [], "y_ins_plh": []}
    for y0, target in zip(initial, targets):
        if torch.rand(()).item() < policy.beta:
            deletion = oracle_deletion(y0, target, cfg.bos_token_id, cfg.eos_token_id)
            y_ins = apply_deletion(y0, deletion)
        else:
            y_ins = random_deletion(
                target, drop_prob=policy.random_delete_prob,
                bos_idx=cfg.bos_token_id, eos_idx=cfg.eos_token_id, pad_idx=cfg.pad_token_id,
            )
        p_star, t_star = oracle_insertion(
            y_ins, target, max_placeholder=cfg.max_placeholder, plh_token_id=cfg.plh_token_id,
        )
        old["y_ins"].append(y_ins)
        old["p_star"].append(p_star)
        old["t_star"].append(t_star)
        old["y_ins_plh"].append(insert_placeholders(y_ins, p_star, cfg.plh_token_id))

    for key in new:
        for a, b in zip(new[key], old[key]):
            assert torch.equal(a, b), f"{key} differs"

    # deletion oracle: batched == per-sample, on the roll-in selection
    roll_ins = [t.detach().cpu() for t in initial]
    d_new = oracle_deletion_batch(roll_ins, targets)
    d_old = [oracle_deletion(r, t) for r, t in zip(roll_ins, targets)]
    for a, b in zip(d_new, d_old):
        assert torch.equal(a, b)
