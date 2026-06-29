"""Runtime reporting helpers for scripts."""

from __future__ import annotations

import torch


def describe_device(device: torch.device | str) -> None:
    """Print concise device/GPU details at script startup."""
    device = torch.device(device)
    print(f"[runtime] torch={torch.__version__} cuda_runtime={torch.version.cuda}")
    if device.type != "cuda" or not torch.cuda.is_available():
        print(f"[runtime] device={device} cuda_available={torch.cuda.is_available()}")
        return

    idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)
    mem_gb = props.total_memory / (1024 ** 3)
    capability = f"{props.major}.{props.minor}"
    print(
        f"[runtime] device={device} gpu_index={idx} gpu_name={props.name} "
        f"capability={capability} total_mem={mem_gb:.2f}GB"
    )
