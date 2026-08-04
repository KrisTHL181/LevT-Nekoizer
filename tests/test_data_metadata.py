"""Tests for the dataset metadata header and packed/regular auto-detection."""

import json

import pytest

from levt import (
    DatasetMetadata,
    JsonlDataset,
    LevTCollator,
    LevTConfig,
    dataset_header,
    parse_metadata_line,
    read_dataset_metadata,
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


PACKED_HEADER = {"__meta__": {"format": "levt-jsonl", "version": 1, "packed": True}}
REGULAR_HEADER = {"__meta__": {"format": "levt-jsonl", "version": 1, "packed": False}}

# A packed target holds two concatenated segments: [BOS] 7 [EOS] [BOS] 8 [EOS].
# These rows are not about the missing-initial default, so they pass an
# explicit [BOS, EOS] initial to keep src outside the BOS/EOS invariant.
PACKED_ROW = {"src": [4, 5], "target": [1, 7, 2, 1, 8, 2], "initial": [1, 2]}
REGULAR_ROW = {"src": [4, 5], "target": [1, 7, 2], "initial": [1, 2]}


def _write(path, *lines):
    path.write_text(
        "".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines),
        encoding="utf-8",
    )


def test_legacy_file_without_header_is_regular(tmp_path):
    path = tmp_path / "legacy.jsonl"
    _write(path, REGULAR_ROW, {"src": [6], "target": [1, 8, 9, 2], "initial": [1, 2]})
    dataset = JsonlDataset(path, tiny_config(), max_source_length=10, max_target_length=10)
    assert dataset.packed is False
    assert dataset.has_header is False
    assert len(dataset) == 2


def test_packed_header_autodetects_and_skips_header(tmp_path):
    cfg = tiny_config()
    path = tmp_path / "packed.jsonl"
    _write(path, PACKED_HEADER, PACKED_ROW,
           {"src": [6, 7], "target": [1, 9, 2, 1, 10, 2], "initial": [1, 2]})
    dataset = JsonlDataset(path, cfg, max_source_length=10, max_target_length=10)
    assert dataset.packed is True
    assert dataset.has_header is True
    assert len(dataset) == 2  # header is not counted as a row
    assert dataset[0]["target"] == PACKED_ROW["target"]
    # A collator wired with the dataset's detected mode accepts packed rows.
    collator = LevTCollator(
        cfg, max_source_length=10, max_target_length=10,
        allow_interior_boundaries=dataset.packed,
    )
    batch = collator([dataset[0]])
    assert batch["targets"][0].tolist() == PACKED_ROW["target"]


def test_regular_header_enforces_strict_boundaries(tmp_path):
    cfg = tiny_config()
    path = tmp_path / "regular.jsonl"
    _write(path, REGULAR_HEADER, PACKED_ROW)
    with pytest.raises(ValueError, match="interior BOS/EOS"):
        JsonlDataset(path, cfg, max_source_length=10, max_target_length=10)

    path.write_text(
        json.dumps(REGULAR_HEADER) + "\n" +
        json.dumps(REGULAR_ROW) + "\n",
        encoding="utf-8",
    )
    dataset = JsonlDataset(path, cfg, max_source_length=10, max_target_length=10)
    assert dataset.packed is False


def test_header_is_not_validated_as_data_row(tmp_path):
    # The header has no src/target, so it must be skipped rather than validated.
    path = tmp_path / "header_only_data.jsonl"
    _write(path, PACKED_HEADER, REGULAR_ROW)
    dataset = JsonlDataset(path, tiny_config(), max_source_length=10, max_target_length=10)
    assert len(dataset) == 1


def test_explicit_flag_is_fallback_for_legacy_files(tmp_path):
    path = tmp_path / "legacy_packed.jsonl"
    _write(path, PACKED_ROW)
    dataset = JsonlDataset(
        path, tiny_config(), max_source_length=10, max_target_length=10,
        allow_interior_boundaries=True,
    )
    assert dataset.packed is True
    assert dataset.has_header is False


def test_metadata_beats_explicit_flag(tmp_path):
    path = tmp_path / "packed.jsonl"
    _write(path, PACKED_HEADER, PACKED_ROW)
    dataset = JsonlDataset(
        path, tiny_config(), max_source_length=10, max_target_length=10,
        allow_interior_boundaries=False,
    )
    assert dataset.packed is True  # header wins over the explicit/config flag
    assert dataset[0]["target"] == PACKED_ROW["target"]


@pytest.mark.parametrize(
    ("header", "match"),
    [
        ({"__meta__": {"format": "levt-jsonl", "version": 2, "packed": True}},
         "unsupported dataset format version"),
        ({"__meta__": {"format": "levt-jsonl", "version": 1}},
         "packed' must be a boolean"),
        ({"__meta__": {"format": "levt-jsonl", "version": 1, "packed": 1}},
         "packed' must be a boolean"),
        ({"__meta__": {"format": "levt-jsonl", "version": 1, "packed": True, "extra": 1}},
         "unknown __meta__ keys"),
        ({"__meta__": {"format": "other", "version": 1, "packed": True}},
         "unsupported dataset format"),
        ({"__meta__": 5},
         "__meta__ must be a JSON object"),
    ],
)
def test_malformed_header_raises(tmp_path, header, match):
    path = tmp_path / "bad_header.jsonl"
    _write(path, header, REGULAR_ROW)
    with pytest.raises(ValueError, match=match):
        JsonlDataset(path, tiny_config(), max_source_length=10, max_target_length=10)


def test_read_dataset_metadata_helper(tmp_path):
    packed = tmp_path / "packed.jsonl"
    _write(packed, PACKED_HEADER, REGULAR_ROW)
    assert read_dataset_metadata(packed) == DatasetMetadata(packed=True)

    legacy = tmp_path / "legacy.jsonl"
    _write(legacy, REGULAR_ROW)
    assert read_dataset_metadata(legacy) is None


def test_dataset_header_round_trip():
    assert parse_metadata_line(dataset_header(True)) == DatasetMetadata(packed=True)
    assert parse_metadata_line(dataset_header(False)) == DatasetMetadata(packed=False)
    assert parse_metadata_line({"src": [4]}) is None
