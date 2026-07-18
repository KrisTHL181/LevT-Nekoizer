"""
Loader for the C++ Levenshtein alignment extension.

On first import, compiles ``_levenshtein_ops.cpp`` via
``torch.utils.cpp_extension.load`` and caches the result.
If compilation fails (no compiler, missing headers, …), the module
gracefully degrades to ``None`` so callers can fall back to pure Python.
"""

from __future__ import annotations

import os
import warnings
from typing import Any

_module: Any = None
_load_attempted = False


def _compile_and_load() -> Any:
    """Try to JIT-compile and load the C++ extension.  Returns module or None."""
    global _module, _load_attempted
    if _load_attempted:
        return _module
    _load_attempted = True

    try:
        from torch.utils.cpp_extension import load  # type: ignore[import-untyped]

        source_dir = os.path.dirname(os.path.abspath(__file__))
        source_path = os.path.join(source_dir, "_levenshtein_ops.cpp")

        if not os.path.exists(source_path):
            warnings.warn(
                "C++ Levenshtein extension source not found at "
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
        return _module
    except Exception as exc:
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
    y: list[int], y_star: list[int],
) -> tuple[list[int], list[list[int]]] | None:
    """
    C++ accelerated Levenshtein alignment.

    Returns ``None`` if the extension is unavailable (caller should fall back
    to the pure-Python implementation).
    """
    mod = _get_module()
    if mod is None:
        return None
    return mod.levenshtein_align(y, y_star)
