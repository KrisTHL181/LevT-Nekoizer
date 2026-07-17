import copy
import json
import sys
import types

import pytest
import torch
import torch.nn.functional as F

from train import evaluate

from levt import (
    DualPolicyTrainer,
    GreedyDecoder,
    JsonlDataset,
    LevTCollator,
    LevTConfig,
    LevTModel,
    PolicyConfig,
    PreparedBatch,
    RotaryPositionalEmbedding,
    SinusoidalPositionalEmbedding,
    TrainConfig,
    import_hf_embeddings,
    load_checkpoint,
    save_checkpoint,
)


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


def test_positional_caches_are_nonpersistent_and_dtype_aware():
    sinusoidal = SinusoidalPositionalEmbedding(5, max_len=2)
    output = sinusoidal(torch.zeros(4, 1, 5, dtype=torch.float64))
    assert output.shape == (4, 1, 5)
    assert output.dtype == torch.float64
    state = sinusoidal.state_dict()
    assert "pe" not in state
    restored = SinusoidalPositionalEmbedding(5, max_len=1)
    restored.load_state_dict(state)

    rope = RotaryPositionalEmbedding(4, max_len=2)
    cos, sin = rope(4, dtype=torch.float64, device=torch.device("cpu"))
    assert cos.shape == sin.shape == (4, 4)
    assert cos.dtype == sin.dtype == torch.float64
    assert not rope.state_dict()


def test_configs_reject_bool_and_nonfinite_numeric_values():
    with pytest.raises(ValueError, match="qk_norm must be a boolean"):
        tiny_config(qk_norm=1)
    with pytest.raises(ValueError, match="rope_base"):
        tiny_config(rope_base=float("inf"))
    with pytest.raises(ValueError, match="dropout"):
        tiny_config(dropout=float("nan"))
    with pytest.raises(ValueError, match="local_files_only must be a boolean"):
        TrainConfig(
            train_data="train.jsonl", hf_model_name_or_path="local",
            local_files_only=1,
        )
    with pytest.raises(ValueError, match="freeze_embeddings must be a boolean"):
        TrainConfig(
            train_data="train.jsonl", hf_model_name_or_path="local",
            freeze_embeddings="false",
        )
    with pytest.raises(ValueError, match="max_grad_norm must be finite numeric"):
        TrainConfig(
            train_data="train.jsonl", hf_model_name_or_path="local",
            max_grad_norm=float("nan"),
        )
    with pytest.raises(ValueError, match="alpha"):
        tiny_config(alpha=float("nan"))


def test_accumulation_window_matches_merged_batch_gradients():
    cfg = tiny_config()
    base = LevTModel(cfg)
    accumulated = copy.deepcopy(base)
    merged = copy.deepcopy(base)
    policy = PolicyConfig(alpha=1.0, beta=1.0, label_smoothing=0.0)
    rows = [
        {"src": [4], "target": [1, 7, 2]},
        {"src": [5, 6, 7], "target": [1, 8, 9, 10, 2], "initial": [1, 11, 2]},
    ]
    collator = LevTCollator(cfg, max_source_length=10, max_target_length=10)
    micro_batches = [collator([row]) for row in rows]
    merged_batch = collator(rows)

    accumulated_trainer = DualPolicyTrainer(accumulated, cfg, policy)
    torch.manual_seed(17)
    prepared = [accumulated_trainer.prepare_batch(batch) for batch in micro_batches]
    window_counts = {
        name: sum(item.counts[name] for item in prepared)
        for name in ("plh", "tok", "del")
    }
    for item in prepared:
        sums, _ = accumulated_trainer.loss_sums_and_counts(item)
        loss = sum(
            sums[name] / window_counts[name] if window_counts[name] else sums[name] * 0.0
            for name in sums
        )
        loss.backward()

    merged_trainer = DualPolicyTrainer(merged, cfg, policy)
    torch.manual_seed(17)
    merged_prepared = merged_trainer.prepare_batch(merged_batch)
    merged_sums, merged_counts = merged_trainer.loss_sums_and_counts(merged_prepared)
    merged_trainer.normalized_loss(merged_sums, merged_counts).backward()

    assert window_counts == merged_counts
    for accumulated_parameter, merged_parameter in zip(
        accumulated.parameters(), merged.parameters()
    ):
        if accumulated_parameter.grad is None or merged_parameter.grad is None:
            assert accumulated_parameter.grad is merged_parameter.grad is None
        else:
            torch.testing.assert_close(
                accumulated_parameter.grad, merged_parameter.grad, rtol=1e-5, atol=1e-6,
            )


def test_selective_token_head_and_decoder_masks_reserved_ids(monkeypatch):
    cfg = tiny_config()
    model = LevTModel(cfg)
    memory = model.encode(torch.tensor([[4], [5]]))
    target = torch.tensor([[cfg.bos_token_id], [cfg.plh_token_id], [cfg.eos_token_id]])
    positions = target.eq(cfg.plh_token_id)
    out = model.decode_with_memory(
        memory, target,
        return_deletion=False, return_placeholder=False, return_token=True,
        token_positions=positions,
    )
    assert set(out) == {"tok_logits"}
    assert out["tok_logits"].shape == (1, cfg.vocab_size)

    decoder = GreedyDecoder(model, cfg)
    model.train()

    def fake_logits(hidden):
        logits = torch.zeros(hidden.shape[0], cfg.vocab_size, device=hidden.device)
        logits[:, cfg.pad_token_id] = 100
        logits[:, 7] = 10
        return logits

    monkeypatch.setattr(model, "_token_logits", fake_logits)
    filled = decoder._fill_tokens(memory, target[:, 0], None)
    assert filled.tolist() == [cfg.bos_token_id, 7, cfg.eos_token_id]

    monkeypatch.setattr(decoder, "_decode", lambda *args: (torch.tensor([1, 2]), 0))
    decoder.decode(torch.tensor([4, 5]))
    assert model.training
    model.eval()
    decoder.decode(torch.tensor([4, 5]))
    assert not model.training


def test_prepared_batch_loss_sums_match_train_step_means():
    cfg = tiny_config()
    model = LevTModel(cfg)
    trainer = DualPolicyTrainer(model, cfg, PolicyConfig(alpha=1.0, beta=1.0))
    batch = LevTCollator(cfg, max_source_length=10, max_target_length=10)([
        {"src": [4, 5], "target": [1, 7, 2]},
        {"src": [6], "target": [1, 8, 9, 2], "initial": [1, 10, 2]},
    ])
    torch.manual_seed(3)
    prepared = trainer.prepare_batch(batch)
    assert isinstance(prepared, PreparedBatch)
    sums, counts = trainer.loss_sums_and_counts(prepared)
    loss = trainer.normalized_loss(sums, counts)
    expected = sum(sums[name] / counts[name] for name in sums if counts[name])
    torch.testing.assert_close(loss, expected)
    assert counts == prepared.counts


def test_validation_is_partition_invariant(monkeypatch):
    model = torch.nn.Linear(1, 1)

    class FakeTrainer:
        def prepare_batch(self, batch):
            return batch

        def loss_sums_and_counts(self, prepared):
            value, count = prepared
            zero = model.weight.sum() * 0
            return ({"plh": zero + value, "tok": zero, "del": zero},
                    {"plh": count, "tok": 0, "del": 0})

    monkeypatch.setattr("train.capture_rng_state", lambda: {})
    monkeypatch.setattr("train.restore_rng_state", lambda state: None)
    first = evaluate(model, FakeTrainer(), [(2.0, 1), (8.0, 3)], torch.device("cpu"), "none")
    second = evaluate(model, FakeTrainer(), [(10.0, 4)], torch.device("cpu"), "none")
    assert first == second == 2.5


def test_shared_embedding_and_exact_tied_formula():
    model = LevTModel(tiny_config())
    assert model.src_embed is model.tgt_embed is model.shared_embedding
    assert model.encoder_input_projection is not model.decoder_input_projection
    assert model.encoder_input_projection.bias is None
    assert model.decoder_input_projection.bias is None
    assert "token_head" not in dict(model.named_modules())

    hidden = torch.randn(3, 2, 8)
    actual = model._token_logits(hidden)
    expected = F.linear(
        F.linear(hidden, model.decoder_input_projection.weight.T),
        model.shared_embedding.weight,
    )
    torch.testing.assert_close(actual, expected)


def test_equal_dimensions_still_have_independent_projections():
    model = LevTModel(tiny_config(embedding_dim=8))
    assert isinstance(model.encoder_input_projection, torch.nn.Linear)
    assert isinstance(model.decoder_input_projection, torch.nn.Linear)
    assert model.encoder_input_projection.weight.data_ptr() != model.decoder_input_projection.weight.data_ptr()


def test_strict_configs_reject_unknown_keys(tmp_path):
    model_path = tmp_path / "config.json"
    model_path.write_text(json.dumps({"vocab_size": 10, "alpha": 0.5}), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown keys"):
        LevTConfig.from_json(model_path)

    model_path.write_text(json.dumps({
        "vocab_size": 10,
        "max_iterations": 20,
        "placeholder_penalty": 0.5,
    }), encoding="utf-8")
    loaded = LevTConfig.from_json(model_path)
    assert loaded.max_iterations == 20
    assert loaded.placeholder_penalty == 0.5
    assert loaded.to_dict()["max_iterations"] == 20
    assert loaded.to_dict()["placeholder_penalty"] == 0.5

    with pytest.raises(ValueError, match="unknown keys"):
        TrainConfig.from_dict({
            "train_data": "train.jsonl",
            "validation_data": None,
            "hf_model_name_or_path": "local",
            "typo": True,
        })


def test_jsonl_defaults_initial_and_collates_seq_first(tmp_path):
    cfg = tiny_config()
    path = tmp_path / "train.jsonl"
    path.write_text(
        json.dumps({"src": [4, 5], "target": [1, 7, 2]}) + "\n" +
        json.dumps({"src": [6], "target": [1, 8, 9, 2], "initial": [1, 10, 2]}) + "\n",
        encoding="utf-8",
    )
    dataset = JsonlDataset(path, cfg, max_source_length=10, max_target_length=10)
    assert dataset[0]["initial"] == [cfg.bos_token_id, cfg.eos_token_id]
    batch = LevTCollator(cfg, max_source_length=10, max_target_length=10)([dataset[0], dataset[1]])
    assert batch["src_tokens"].shape == (2, 2)
    assert batch["src_padding_mask"].shape == (2, 2)
    assert batch["src_padding_mask"].tolist() == [[False, False], [False, True]]


def test_data_rejects_bool_and_bad_boundaries(tmp_path):
    cfg = tiny_config()
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"src": [True], "target": [1, 2]}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="bool is invalid"):
        JsonlDataset(path, cfg, max_source_length=10, max_target_length=10)

    path.write_text(json.dumps({"src": [4], "target": [5, 2]}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="start with BOS"):
        JsonlDataset(path, cfg, max_source_length=10, max_target_length=10)


def test_batch_loss_backward():
    cfg = tiny_config()
    model = LevTModel(cfg)
    trainer = DualPolicyTrainer(model, cfg, PolicyConfig(alpha=1.0, beta=1.0))
    rows = [
        {"src": [4, 5], "target": [1, 7, 2]},
        {"src": [6], "target": [1, 8, 9, 2], "initial": [1, 10, 2]},
    ]
    batch = LevTCollator(cfg, max_source_length=10, max_target_length=10)(rows)
    loss, metrics = trainer.train_step(batch)
    assert torch.isfinite(loss)
    assert set(metrics) == {"loss_ins_plh", "loss_ins_tok", "loss_del", "loss_total"}
    loss.backward()
    assert model.shared_embedding.weight.grad is not None
    assert model.encoder_input_projection.weight.grad is not None
    assert model.decoder_input_projection.weight.grad is not None


def test_hf_import_uses_automodel_and_validates_shape(monkeypatch):
    cfg = tiny_config()
    model = LevTModel(cfg)
    weights = torch.arange(24 * 6, dtype=torch.float32).view(24, 6)
    calls = {}

    class FakeModel:
        def get_input_embeddings(self):
            return types.SimpleNamespace(weight=weights)

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(source, **kwargs):
            calls["source"] = source
            calls.update(kwargs)
            return FakeModel()

    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoModel=FakeAutoModel))
    import_hf_embeddings(model, "offline-model", local_files_only=True, dtype="float32")
    assert calls["source"] == "offline-model"
    assert calls["local_files_only"] is True
    assert calls["torch_dtype"] is torch.float32
    torch.testing.assert_close(model.shared_embedding.weight, weights)


def test_checkpoint_round_trip(tmp_path):
    path = tmp_path / "latest.pt"
    tensor = torch.randn(2, 3)
    save_checkpoint(path, {"model": {"weight": tensor}, "global_step": 4})
    loaded = load_checkpoint(path)
    assert loaded["version"] == 1
    assert loaded["global_step"] == 4
    torch.testing.assert_close(loaded["model"]["weight"], tensor)
