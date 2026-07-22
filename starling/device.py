"""Hardware detection, memory budgeting and chunk planning.

starling auto-selects the fastest available torch device (cuda > mps > cpu).
Override with the ``device=`` keyword on any public function or the
``STARLING_DEVICE`` environment variable (e.g. ``STARLING_DEVICE=cpu``).
"""

import os

import psutil
import torch

# Tensor-core TF32 matmuls on Ampere+ GPUs: 2-4x on the batched J^T J
# GEMMs inside the fits at fit-irrelevant precision cost (fp32 accumulate,
# parameters converge to the same xtol band). The analysis notebooks set
# this too; setting it here covers batch/CLI runs. Opt out with
# STARLING_TF32=0.
if os.environ.get("STARLING_TF32", "1") not in ("0", "false", "False"):
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def get_device(prefer=None):
    """Resolve the compute device.

    Args:
        prefer (str or torch.device, optional): explicit device request,
            e.g. "cuda", "mps", "cpu". Defaults to None (auto-detect).

    Returns:
        torch.device
    """
    if isinstance(prefer, torch.device):
        return prefer
    name = prefer or os.environ.get("STARLING_DEVICE")
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def memory_budget(device):
    """Bytes of memory usable for tensors on the given device."""
    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        free, _total = torch.cuda.mem_get_info(idx)
        return int(free * 0.8)
    if device.type == "mps":
        try:
            rec = torch.mps.recommended_max_memory() - torch.mps.current_allocated_memory()
            if rec > 0:
                return int(rec)
        except Exception:
            pass
        return int(psutil.virtual_memory().available * 0.6)
    return int(psutil.virtual_memory().available * 0.5)


def plan_chunks(n_pixels, bytes_per_pixel, device, safety=0.5):
    """Number of pixels to process per chunk given the device memory budget."""
    budget = memory_budget(device) * safety
    chunk = int(budget // max(1, bytes_per_pixel))
    return max(1, min(n_pixels, chunk))


def compute_dtype(device):
    """float32 on GPU (MPS has no float64), float64 on CPU."""
    return torch.float32 if device.type in ("cuda", "mps") else torch.float64
