"""Global determinism plumbing. Audit ref: §4.2 "seed: 13 present in configs but
never applied"; v3 Step 2a Commit 2; gate criterion "consistent across >=3 seeds".
"""
from __future__ import annotations
import os, random
import numpy as np
import torch

def seed_everything(seed: int, deterministic_algorithms: bool = True) -> dict:
    """Seed Python/NumPy/Torch(+CUDA) and set deterministic flags.
    Returns a record dict — the caller MUST place it into run_config output."""
    if not isinstance(seed, int):
        raise TypeError(f"seed must be int, got {type(seed).__name__} — config not plumbed?")
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # needed for CUDA determinism
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)                    # seeds all CUDA devices too
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if deterministic_algorithms:
        torch.use_deterministic_algorithms(True, warn_only=True)
    return {
        "seed": seed,
        "pythonhashseed": os.environ["PYTHONHASHSEED"],
        "cudnn_deterministic": True, "cudnn_benchmark": False,
        "deterministic_algorithms_warn_only": deterministic_algorithms,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }

def make_generator(seed: int, offset: int = 0) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed + offset)
    return g

def worker_init_fn(worker_id: int) -> None:
    """DataLoader worker seeding: derive per-worker numpy/random state from torch's
    per-worker seed (torch handles its own). Windows spawn-safe: module-level fn."""
    ws = torch.initial_seed() % 2**31
    np.random.seed(ws + worker_id)
    random.seed(ws + worker_id)

def dataloader_seed_kwargs(seed: int) -> dict:
    """Kwargs to splat into every DataLoader for reproducible shuffling/workers."""
    return {"generator": make_generator(seed, offset=1), "worker_init_fn": worker_init_fn}
