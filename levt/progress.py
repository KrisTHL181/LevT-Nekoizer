"""Rich-based live training progress display with all key metrics."""

from __future__ import annotations

import time
from datetime import timedelta
from typing import Optional

Console: object = None
Live: object = None
Panel: object = None
Table: object = None
HAS_RICH = False

try:
    from rich.console import Console  # type: ignore[no-redef]
    from rich.live import Live        # type: ignore[no-redef]
    from rich.panel import Panel      # type: ignore[no-redef]
    from rich.table import Table      # type: ignore[no-redef]

    HAS_RICH = True
except ImportError:
    pass


class TrainingDisplay:
    """Live Rich panel showing a progress bar and every key training metric.

    Updates are applied at ``log_every_steps`` intervals; the ``Live`` context
    re-renders automatically at *refresh_per_second* Hz in between.
    """

    def __init__(
        self,
        total_steps: int,
        console: Console | None = None,
        refresh_per_second: float = 4.0,
    ) -> None:
        if not HAS_RICH:
            raise ImportError(
                "rich is not installed — install with: pip install rich"
            )
        self.total_steps = max(total_steps, 1)
        self.start_time = time.monotonic()
        self.console = console or Console()
        self.live = Live(
            console=self.console,
            refresh_per_second=refresh_per_second,
            transient=False,
        )
        self.live.start()
        self._step: int = 0
        self._last: dict = {}
        self._val_loss: Optional[tuple[int, float]] = None   # (step, loss)
        self._best_val: Optional[tuple[int, float]] = None   # (step, loss)
        self._patience_used: int = 0
        self._patience_total: int = 0

    # -- public API --------------------------------------------------------

    def update(
        self,
        *,
        step: int,
        epoch: int,
        batch: int = 0,
        batches_per_epoch: int = 0,
        loss_total: float = 0.0,
        loss_plh: float = 0.0,
        loss_tok: float = 0.0,
        loss_del: float = 0.0,
        lr_adamw: float = 0.0,
        lr_muon: float = 0.0,
        grad_norm: Optional[float] = None,
    ) -> None:
        """Push the latest training metrics and re-render immediately."""
        self._step = step
        self._last = dict(
            step=step,
            epoch=epoch,
            batch=batch,
            batches_per_epoch=batches_per_epoch,
            loss_total=loss_total,
            loss_plh=loss_plh,
            loss_tok=loss_tok,
            loss_del=loss_del,
            lr_adamw=lr_adamw,
            lr_muon=lr_muon,
            grad_norm=grad_norm,
        )
        self._render()

    def set_validation_loss(self, step: int, loss: float) -> None:
        """Record a validation loss (auto-tracks best)."""
        self._val_loss = (step, loss)
        if self._best_val is None or loss < self._best_val[1]:
            self._best_val = (step, loss)
        if self._last:
            self._render()

    def set_early_stopping(self, steps_since: int, patience: int) -> None:
        """Update the early-stopping patience counter (0 = disabled)."""
        self._patience_used = steps_since
        self._patience_total = patience

    def close(self) -> None:
        """Stop the live display so further output is unadorned."""
        self.live.stop()

    # -- internals ---------------------------------------------------------

    def _render(self) -> None:
        d = self._last
        if not d:
            self.live.update(
                Panel(
                    "Initialising …",
                    title="[bold]Training Progress",
                    title_align="left",
                    border_style="bright_blue",
                    padding=(1, 2),
                )
            )
            return

        step = d["step"]
        elapsed = time.monotonic() - self.start_time
        progress = min(step / self.total_steps, 1.0)
        if step > 0:
            eta = (elapsed / step) * (self.total_steps - step)
        else:
            eta = 0.0

        # Unicode progress bar ─────────────────────────────────────
        bar_width = 30
        filled = int(bar_width * progress)
        bar = "█" * filled + "░" * (bar_width - filled)

        # Build compact key-value table ────────────────────────────
        table = Table(box=None, expand=True, show_header=False, padding=(0, 1))
        table.add_column("key", style="bold cyan", width=10, no_wrap=True)
        table.add_column("value", style="white", overflow="fold")

        table.add_row(
            "Progress",
            f"{bar}  {step:,}/{self.total_steps:,}  ({progress * 100:.1f}%)",
        )

        epoch_info = f"Epoch {d['epoch']}"
        if d.get("batches_per_epoch", 0) > 0:
            epoch_info += f"  ·  Batch {d['batch']}/{d['batches_per_epoch']}"
        table.add_row("Epoch", epoch_info)

        table.add_row(
            "Loss",
            f"total={d['loss_total']:.6f}  "
            f"[PLH={d['loss_plh']:.4f}, TOK={d['loss_tok']:.4f}, DEL={d['loss_del']:.4f}]",
        )

        table.add_row("LR", f"adamw={d['lr_adamw']:.2e}  muon={d['lr_muon']:.2e}")

        if d.get("grad_norm") is not None:
            table.add_row("Grad Norm", f"{d['grad_norm']:.4f}")

        if self._val_loss is not None:
            vs, vl = self._val_loss
            val_text = f"{vl:.6f}  (step {vs})"
            if self._best_val is not None and abs(vl - self._best_val[1]) < 1e-9:
                val_text += "  [green]★ best[/green]"
            table.add_row("Val Loss", val_text)

        if self._patience_total > 0:
            colour = "red" if self._patience_used >= self._patience_total else "yellow"
            table.add_row(
                "Patience",
                f"[{colour}]{self._patience_used}/{self._patience_total}[/{colour}]",
            )

        table.add_row(
            "Time",
            f"elapsed {timedelta(seconds=int(elapsed))}  ·  "
            f"ETA {timedelta(seconds=int(eta))}",
        )

        self.live.update(
            Panel(
                table,
                title="[bold]Training Progress",
                title_align="left",
                border_style="bright_blue",
                padding=(1, 2),
            )
        )
