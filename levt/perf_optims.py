import inspect
import os
import warnings
from collections.abc import Callable
from functools import partial

import torch

IS_CUDA_AVAILABLE = torch.cuda.is_available()


class AttributeContainer:
    """Generic container supporting both attribute and item access."""

    __slots__ = ("_storage",)  # Prevent arbitrary attribute creation

    def __init__(self):
        object.__setattr__(self, "_storage", {})

    def __getattr__(self, name: str):
        try:
            return self._storage[name]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def __setattr__(self, name: str, value):
        self._storage[name] = value

    def __getitem__(self, name: str):
        return self._storage[name]

    def __setitem__(self, name: str, value):
        self._storage[name] = value

    def __contains__(self, name: str):
        return name in self._storage


def _get_cgroup_v1_cpu_limit() -> int | None:
    """Get CPU limit from cgroup v1 files if available."""
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as quota_file:
            quota = int(quota_file.read().strip())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as period_file:
            period = int(period_file.read().strip())
        if quota > 0 and period > 0:
            return quota // period
    except (FileNotFoundError, ValueError, OSError):
        return None


def _get_cgroup_v2_cpu_limit() -> int | None:
    """Get CPU limit from cgroup v2 files if available."""
    try:
        with open("/sys/fs/cgroup/cpu.max") as cpu_max_file:
            content = cpu_max_file.read().strip()
            if content == "max":
                return None
            quota_str, period_str = content.split()
            quota, period = int(quota_str), int(period_str)
            if quota > 0 and period > 0:
                return quota // period
    except (FileNotFoundError, ValueError, OSError):
        return None


def get_cpu_limit() -> int:
    """Determine available CPU cores considering cgroup limits."""
    # Check cgroup limits first (container environments)
    cpu_limit = _get_cgroup_v1_cpu_limit() or _get_cgroup_v2_cpu_limit()
    if cpu_limit:
        return cpu_limit

    # Fallback to OS-reported cores
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1  # Fallback to 1 if all else fails


# Determine available CPU cores considering environment constraints
AVAILABLE_CPU_CORES = get_cpu_limit()

# Main acceleration configuration container
accelerated = AttributeContainer()


def optimizer_has_parameter(optimizer_class, param_name: str) -> bool:
    """Check if optimizer's __init__ accepts a specific parameter."""
    init_signature = inspect.signature(optimizer_class.__init__)
    return param_name in init_signature.parameters


def register_optimizers(condition_fn: Callable, parameter_name: str) -> None:
    """
    Register optimizers with special acceleration parameters.

    For each optimizer in torch.optim:
    - If it meets the condition, register a partial with acceleration parameter
    - Otherwise, register the original class
    """
    parameter_check = partial(condition_fn, param_name=parameter_name)

    for optimizer_name in dir(torch.optim):
        # Skip private attributes
        if optimizer_name.startswith("_"):
            continue

        optimizer_class = getattr(torch.optim, optimizer_name)

        # Only process actual optimizer classes
        if not inspect.isclass(optimizer_class):
            continue

        # Register accelerated version if condition met
        if parameter_check(optimizer_class):
            accelerated[optimizer_name] = partial(optimizer_class, **{parameter_name: True})
        else:
            accelerated[optimizer_name] = optimizer_class


# Configure optimizers based on hardware
if IS_CUDA_AVAILABLE:
    register_optimizers(optimizer_has_parameter, "fused")
else:
    register_optimizers(optimizer_has_parameter, "foreach")

# Register accelerated gradient clipping operations
accelerated["clip_grad_norm"] = partial(torch.nn.utils.clip_grad_norm_, foreach=True)
accelerated["clip_grad_norm_"] = partial(torch.nn.utils.clip_grad_norm_, foreach=True)
accelerated["clip_grad_value_"] = partial(torch.nn.utils.clip_grad_value_, foreach=True)
accelerated["clip_grads_with_norm_"] = partial(torch.nn.utils.clip_grads_with_norm_, foreach=True)


def _sanitize_thread_env() -> None:
    """Unset invalid OMP_NUM_THREADS etc. to prevent forked worker crashes."""
    _VARS = ["OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"]
    for var in _VARS:
        v = os.environ.get(var)
        if v is None:
            continue
        try:
            n = int(v.strip())
            if n <= 0 or n > 2_147_483_647:
                raise ValueError
        except (ValueError, TypeError):
            warnings.warn(f"{var}={v!r} invalid — unsetting to prevent worker crashes")
            del os.environ[var]


def set_hardware_optimizations():
    """Set hardware-specific optimizations."""
    _sanitize_thread_env()
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    torch.set_float32_matmul_precision("high")  # Enable TensorFloat-32
    # Configure CPU thread count
    torch.set_num_threads(AVAILABLE_CPU_CORES)
    torch.set_num_interop_threads(AVAILABLE_CPU_CORES)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def enable_nondeterministic_speedups():
    """Enable non-deterministic but faster CUDA operations."""
    if not IS_CUDA_AVAILABLE:
        warnings.warn("cuDNN non-deterministic speedups require CUDA")
        return

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.use_deterministic_algorithms(False, warn_only=True)


# Enable attention optimizations if available
def enable_flash_attention():
    """Enable FlashAttention optimizations if available."""
    if IS_CUDA_AVAILABLE and hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        torch.backends.cuda.enable_flash_sdp(True)  # Best performance
        torch.backends.cuda.enable_mem_efficient_sdp(True)  # Memory efficient
        torch.backends.cuda.enable_math_sdp(False)  # Disable fallback


def make_model_contiguous(model: torch.nn.Module, memory_format: torch.memory_format = torch.channels_last):
    """Ensure all model parameters and gradients are contiguous in memory."""
    if IS_CUDA_AVAILABLE:
        torch.cuda.synchronize()

    model.to(memory_format=memory_format)

    for buffer in model.buffers():
        if buffer.is_floating_point():
            buffer.data = buffer.data.contiguous()

    for parameter in model.parameters():
        parameter.data = parameter.data.contiguous()
        if parameter.grad is not None:
            parameter.grad = parameter.grad.contiguous()


def prealloc_model_grads(model: torch.nn.Module, keep_existing: bool = False):
    """Preallocate model gradients to optimize memory usage."""
    if IS_CUDA_AVAILABLE:
        torch.cuda.synchronize()

    for parameter in model.parameters():
        if keep_existing and parameter.grad is not None:
            continue
        parameter.grad = torch.zeros_like(parameter.data).contiguous()


def finalize_memory_layout(model: torch.nn.Module):
    """Finalize memory layout for model parameters and gradients."""
    if IS_CUDA_AVAILABLE:
        torch.cuda.synchronize()

    prealloc_model_grads(model, keep_existing=True)
    make_model_contiguous(model)

    return model


# Expose core configuration values
accelerated["cpu_cores"] = AVAILABLE_CPU_CORES


def enable_all(deterministic: bool = False) -> None:
    """
    Apply all acceleration configurations using existing utilities.

    Args:
        deterministic: If True, disables non-deterministic speedups.
    """
    set_hardware_optimizations()

    enable_flash_attention()

    # Reuse CUDA speedup logic when appropriate
    if IS_CUDA_AVAILABLE and not deterministic:
        enable_nondeterministic_speedups()
