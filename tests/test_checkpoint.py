"""Tests for checkpoint I/O and cleanup."""

from __future__ import annotations

from pathlib import Path

from levt.checkpoint import cleanup_checkpoints


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


class TestCleanupCheckpoints:
    def test_keep_last_zero_is_noop(self, tmp_path: Path) -> None:
        _touch(tmp_path / "step_00001000.pt")
        _touch(tmp_path / "step_00002000.pt")
        cleanup_checkpoints(tmp_path, 0, {})
        assert sorted(tmp_path.glob("step_*.pt")) == sorted([
            tmp_path / "step_00001000.pt",
            tmp_path / "step_00002000.pt",
        ])

    def test_keep_last_n(self, tmp_path: Path) -> None:
        for s in (1000, 2000, 3000, 4000, 5000):
            _touch(tmp_path / f"step_{s:08d}.pt")
        cleanup_checkpoints(tmp_path, 2, {})
        remaining = sorted(tmp_path.glob("step_*.pt"))
        assert remaining == [
            tmp_path / "step_00004000.pt",
            tmp_path / "step_00005000.pt",
        ]

    def test_keep_best(self, tmp_path: Path) -> None:
        for s in (1000, 2000, 3000, 4000, 5000):
            _touch(tmp_path / f"step_{s:08d}.pt")
        # step 2000 has the lowest eval loss
        val_loss = {1000: 0.5, 2000: 0.2, 3000: 0.6, 4000: 0.4, 5000: 0.8}
        cleanup_checkpoints(tmp_path, 2, val_loss)
        remaining = sorted(tmp_path.glob("step_*.pt"))
        # last 2 (4000, 5000) + best (2000)
        assert remaining == [
            tmp_path / "step_00002000.pt",
            tmp_path / "step_00004000.pt",
            tmp_path / "step_00005000.pt",
        ]

    def test_best_within_last_n(self, tmp_path: Path) -> None:
        for s in (1000, 2000, 3000, 4000, 5000):
            _touch(tmp_path / f"step_{s:08d}.pt")
        # step 5000 is both the best and in the last N
        val_loss = {1000: 1.0, 2000: 0.9, 3000: 0.8, 4000: 0.7, 5000: 0.1}
        cleanup_checkpoints(tmp_path, 2, val_loss)
        remaining = sorted(tmp_path.glob("step_*.pt"))
        # last 2 (4000, 5000) — 5000 is best but already included
        assert remaining == [
            tmp_path / "step_00004000.pt",
            tmp_path / "step_00005000.pt",
        ]

    def test_ignores_non_matching_files(self, tmp_path: Path) -> None:
        _touch(tmp_path / "step_00001000.pt")
        _touch(tmp_path / "latest.pt")
        _touch(tmp_path / "other_file.txt")
        cleanup_checkpoints(tmp_path, 1, {})
        remaining = sorted(tmp_path.glob("*"))
        assert remaining == [
            tmp_path / "latest.pt",
            tmp_path / "other_file.txt",
            tmp_path / "step_00001000.pt",
        ]

    def test_empty_dir(self, tmp_path: Path) -> None:
        cleanup_checkpoints(tmp_path, 3, {1000: 0.5})
        # should not raise

    def test_stale_val_loss_entries(self, tmp_path: Path) -> None:
        """val_loss references a checkpoint that was already deleted."""
        for s in (2000, 3000, 4000):
            _touch(tmp_path / f"step_{s:08d}.pt")
        # best is step 1000 but that file doesn't exist
        val_loss = {1000: 0.01, 2000: 0.5, 3000: 0.4, 4000: 0.3}
        cleanup_checkpoints(tmp_path, 2, val_loss)
        remaining = sorted(tmp_path.glob("step_*.pt"))
        # last 2 + best-existing (3000 has lowest among existing: 0.4)
        assert remaining == [
            tmp_path / "step_00003000.pt",
            tmp_path / "step_00004000.pt",
        ]
