"""
Device Management Module
Handles CUDA/CPU detection, GPU memory tracking, and device allocation for MONAI/PyTorch.
"""

import os
import torch
from typing import Dict, Any


def get_device(force_cpu: bool = False) -> torch.device:
    """
    Returns the appropriate PyTorch compute device.
    Respects FORCE_CPU environment variable or force_cpu parameter.
    """
    env_force_cpu = os.getenv("FORCE_CPU", "false").lower() in ("true", "1", "yes")
    if force_cpu or env_force_cpu:
        return torch.device("cpu")
    
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_device_info() -> Dict[str, Any]:
    """
    Returns hardware diagnostics and PyTorch/MONAI acceleration details.
    """
    cuda_available = torch.cuda.is_available()
    device = get_device()
    
    info = {
        "device_type": "CUDA" if device.type == "cuda" else "CPU",
        "device_name": "CPU",
        "cuda_available": cuda_available,
        "device_str": f"Device: {device.type.upper()}",
        "gpu_count": 0,
        "vram_allocated_mb": 0.0,
        "vram_total_mb": 0.0,
        "torch_version": torch.__version__,
    }

    if cuda_available and device.type == "cuda":
        try:
            current_idx = torch.cuda.current_device()
            info["device_name"] = torch.cuda.get_device_name(current_idx)
            info["gpu_count"] = torch.cuda.device_count()
            vram_total = torch.cuda.get_device_properties(current_idx).total_memory / (1024 ** 2)
            vram_alloc = torch.cuda.memory_allocated(current_idx) / (1024 ** 2)
            info["vram_total_mb"] = round(vram_total, 1)
            info["vram_allocated_mb"] = round(vram_alloc, 1)
        except Exception:
            pass

    return info
