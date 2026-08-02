"""
Loader for the C++ Levenshtein alignment extension.

On first import, compiles ``_levenshtein_ops.cpp`` via
``torch.utils.cpp_extension.load`` and caches the result.
If compilation fails (no compiler, missing headers, …), the module
gracefully degrades to ``None`` so callers can fall back to pure Python.

Callers that need a hard guarantee should use ``verify_cpp_extension()``
*before* training — it returns a structured diagnostic result and does not
rely on the one-shot ``warnings.warn`` path.
"""

from __future__ import annotations

import os
import shutil
import sys
import warnings
from dataclasses import dataclass
from typing import Any

import torch


_module: Any = None
_load_attempted = False
_last_error: str | None = None

_NINJA_HINTS = (
    # Directory containing the active interpreter (conda envs ship ninja here).
    "bin",
    # Common standalone conda roots.
    os.path.expanduser("~/miniconda3/bin"),
    os.path.expanduser("~/anaconda3/bin"),
    "/usr/local/bin",
    "/usr/bin",
)


def _ensure_ninja_on_path() -> None:
    """Put ``ninja`` on ``PATH`` if it is installed but not findable.

    ``torch.utils.cpp_extension.load`` requires ``ninja`` via ``shutil.which``,
    even to load a cached build.  When training is launched from a bare shell
    (no conda env activated), ninja is silently missing and the extension falls
    back to the ~3.5x slower pure-Python DP.  If ninja exists next to the
    interpreter or in a common conda root, prepend that directory so JIT
    compilation works regardless of how the process was started.
    """
    if shutil.which("ninja") is not None:
        return
    candidates = (
        os.path.join(os.path.dirname(sys.executable), "bin"),
    ) + _NINJA_HINTS
    for directory in candidates:
        if os.path.isfile(os.path.join(directory, "ninja")):
            os.environ["PATH"] = f"{directory}:{os.environ.get('PATH', '')}"
            return


def _compile_and_load() -> Any:
    """Try to JIT-compile and load the C++ extension.  Returns module or None."""
    global _module, _load_attempted, _last_error
    if _load_attempted:
        return _module
    _load_attempted = True

    try:
        from torch.utils.cpp_extension import load  # type: ignore[import-untyped]

        _ensure_ninja_on_path()
        source_dir = os.path.dirname(os.path.abspath(__file__))
        source_path = os.path.join(source_dir, "_levenshtein_ops.cpp")

        if not os.path.exists(source_path):
            _last_error = f"C++ source not found at {source_path!r}"
            warnings.warn(
                f"C++ Levenshtein extension source not found at "
                f"{source_path!r}; falling back to pure Python. "
                "Training will be significantly slower.",
                RuntimeWarning,
            )
            return None

        _module = load(
            name="_levt_levenshtein_ops",
            sources=[source_path],
            extra_cflags=["-O3", "-march=native"],
            verbose=False,
        )
        _last_error = None
        return _module
    except Exception as exc:
        _last_error = str(exc)
        warnings.warn(
            "Failed to compile C++ Levenshtein extension; "
            "falling back to pure Python. "
            "Training will be significantly slower. "
            f"Error: {exc}",
            RuntimeWarning,
        )
        return None


def _get_module() -> Any:
    return _compile_and_load()


def levenshtein_align_cpp(
    y: torch.Tensor, y_star: torch.Tensor,
) -> tuple[torch.Tensor, list[torch.Tensor]] | None:
    """
    C++ accelerated Levenshtein alignment.

    Accepts PyTorch tensors directly — no Python list serialization.

    Returns ``None`` if the extension is unavailable (caller should fall back
    to the pure-Python implementation).
    """
    mod = _get_module()
    if mod is None:
        return None
    return mod.levenshtein_align(y, y_star)


def levenshtein_deletion_cpp_batch(
    ys: list[torch.Tensor], ys_stars: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """
    C++ batched deletion oracle.

    Returns ``(mask_packed, offsets)`` — a uint8 flat tensor of per-sample
    concatenated delete flags plus a ``[B+1]`` int64 offsets tensor — or
    ``None`` if the extension is unavailable.
    """
    mod = _get_module()
    if mod is None:
        return None
    return mod.levenshtein_deletion_batch(ys, ys_stars)


def levenshtein_insertion_cpp_batch(
    ys: list[torch.Tensor],
    ys_stars: list[torch.Tensor],
    max_placeholder: int,
    plh_token_id: int,
) -> tuple[torch.Tensor, ...] | None:
    """
    C++ batched insertion oracle.

    Returns ``(p_star_packed, p_star_offsets, t_star_packed, t_star_offsets,
    y_ins_plh_packed, y_ins_plh_offsets)`` or ``None`` if the extension is
    unavailable.
    """
    mod = _get_module()
    if mod is None:
        return None
    return mod.levenshtein_insertion_batch(ys, ys_stars, max_placeholder, plh_token_id)


# ---------------------------------------------------------------------------
# Pre-flight verification (called before training starts)
# ---------------------------------------------------------------------------


@dataclass
class CppExtensionStatus:
    """Result of ``verify_cpp_extension()`` — independent pre-flight check."""

    available: bool
    """True when the C++ extension loads AND produces correct results."""

    error: str | None = None
    """Human-readable diagnosis when ``available`` is False."""

    fix_hint: str | None = None
    """Concrete remediation steps (e.g. 'export PATH=…')."""


def verify_cpp_extension() -> CppExtensionStatus:
    """Independently check that the C++ extension loads and works correctly.

    Unlike the lazy ``warnings.warn`` path, this function forces a fresh
    load attempt, runs a smoke test with real data, and returns a structured
    result.  Call this **before** model creation in ``train.py`` so the user
    sees a clear error immediately instead of discovering the slowdown hours
    later via low GPU utilisation.
    """
    # ── 1. Reset cached state so we get a fresh attempt ─────────────────
    global _module, _load_attempted, _last_error
    _module = None
    _load_attempted = False
    _last_error = None

    mod = _get_module()

    # ── 2. Module load failed ──────────────────────────────────────────
    if mod is None:
        return _diagnose_failure()

    # ── 3. Smoke test: run a real alignment (tensors, not lists) ───────
    try:
        y = torch.tensor([1, 2], dtype=torch.long)
        y_star = torch.tensor([1, 3, 4, 2], dtype=torch.long)
        result = mod.levenshtein_align(y, y_star)
    except Exception as exc:
        return CppExtensionStatus(
            available=False,
            error=f"C++ extension loaded but raised during smoke test: {exc}",
            fix_hint="The cached .so may be stale — try removing "
                     "~/.cache/torch_extensions and re-running.",
        )

    if result is None:
        return CppExtensionStatus(
            available=False,
            error="C++ extension function returned None unexpectedly.",
            fix_hint="Reinstall the PyTorch JIT cache: "
                     "rm -rf ~/.cache/torch_extensions",
        )

    deletions, per_gap = result
    expected_deletions = torch.tensor([], dtype=torch.long)
    expected_per_gap = [torch.tensor([3, 4], dtype=torch.long)]

    if not torch.equal(deletions, expected_deletions) or \
       len(per_gap) != len(expected_per_gap) or \
       not torch.equal(per_gap[0], expected_per_gap[0]):
        return CppExtensionStatus(
            available=False,
            error=f"Smoke test produced wrong result: {result} != "
                  f"({expected_deletions}, {expected_per_gap})",
            fix_hint="The cached .so may be from a different source version. "
                     "Remove ~/.cache/torch_extensions and re-run.",
        )

    # ── 4. Batched packed oracles (same inputs) ────────────────────────
    try:
        del_packed, del_offsets = mod.levenshtein_deletion_batch([y], [y_star])
        (p_packed, p_offsets, t_packed, t_offsets,
         plh_packed, plh_offsets) = mod.levenshtein_insertion_batch(
            [y], [y_star], 255, 3)
    except Exception as exc:
        return CppExtensionStatus(
            available=False,
            error=f"C++ extension loaded but batched smoke test raised: {exc}",
            fix_hint="The cached .so may be stale — try removing "
                     "~/.cache/torch_extensions and re-running.",
        )

    # y=[1,2], y_star=[1,3,4,2] → deletions none; 1 gap [3,4]:
    #   deletion mask [0,0]; p_star [2]; t_star [3,4]; plh roll-in [1,3,3,2].
    if not torch.equal(del_packed, torch.tensor([0, 0], dtype=torch.uint8)) or \
       not torch.equal(del_offsets, torch.tensor([0, 2], dtype=torch.long)) or \
       not torch.equal(p_packed, torch.tensor([2], dtype=torch.long)) or \
       not torch.equal(t_packed, torch.tensor([3, 4], dtype=torch.long)) or \
       not torch.equal(plh_packed, torch.tensor([1, 3, 3, 2], dtype=torch.long)) or \
       not torch.equal(p_offsets, torch.tensor([0, 1], dtype=torch.long)) or \
       not torch.equal(t_offsets, torch.tensor([0, 2], dtype=torch.long)) or \
       not torch.equal(plh_offsets, torch.tensor([0, 4], dtype=torch.long)):
        return CppExtensionStatus(
            available=False,
            error=f"Smoke test produced wrong batched result: "
                  f"({del_packed}, {del_offsets}, {p_packed}, {t_packed}, "
                  f"{plh_packed})",
            fix_hint="The cached .so may be from a different source version. "
                     "Remove ~/.cache/torch_extensions and re-run.",
        )

    return CppExtensionStatus(available=True)


def _diagnose_failure() -> CppExtensionStatus:
    """Build a detailed diagnostic when the C++ extension won't load."""
    parts: list[str] = []

    # What went wrong
    if _last_error:
        parts.append(f"JIT compilation failed: {_last_error}")
    else:
        parts.append("C++ extension module is None (unknown reason)")

    # Check common causes
    ninja_path = shutil.which("ninja")
    cc_path = shutil.which("g++") or shutil.which("gcc") or shutil.which("cc")

    hints: list[str] = []

    if ninja_path is None:
        # ninja not found at all
        conda_ninja = os.path.expanduser("~/miniconda3/bin/ninja")
        if os.path.exists(conda_ninja):
            hints.append(
                f"ninja is installed at {conda_ninja} but not on PATH. "
                f"Add 'export PATH=\"{os.path.dirname(conda_ninja)}:$PATH\"' "
                f"before running training."
            )
        else:
            hints.append(
                "ninja is not installed. Run: pip install ninja"
            )
    elif _last_error and "ninja" in _last_error.lower():
        hints.append(
            f"ninja found at {ninja_path} but the build still failed. "
            "Check that the C++ compiler works: `ninja --version` and `g++ --version`"
        )

    if cc_path is None:
        hints.append(
            "No C++ compiler found on PATH. Install build-essential (apt) "
            "or gcc-toolset (yum)."
        )

    if not hints:
        hints.append(
            "Try: pip install ninja && rm -rf ~/.cache/torch_extensions"
        )

    return CppExtensionStatus(
        available=False,
        error="; ".join(parts),
        fix_hint=" | ".join(hints),
    )
