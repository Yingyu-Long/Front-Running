"""CUDA / RTX 5070 Ti (Blackwell sm_120) device detection and VRAM lifecycle manager."""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]


@dataclass
class DeviceProfile:
    backend: str                # "cuda" | "mps" | "cpu"
    device_name: str
    vram_total_gb: float
    vram_free_gb: float
    cuda_version: str
    compute_capability: str
    supports_bf16: bool
    is_blackwell: bool
    torch_available: bool

    def summary_str(self) -> str:
        if self.backend == "cuda":
            return (
                f"{self.device_name} ({self.vram_total_gb:.1f}GB VRAM, "
                f"CC {self.compute_capability}, CUDA {self.cuda_version})"
            )
        if self.backend == "mps":
            return f"{self.device_name} (Apple MPS)"
        return f"{self.device_name} (CPU 模式)"


def get_device_profile(preference: str = "auto") -> DeviceProfile:
    """Inspect available compute hardware with special awareness for RTX 5070 Ti Blackwell."""
    if torch is None:
        return DeviceProfile(
            backend="cpu",
            device_name="NumPy/CPU (未安装 PyTorch)",
            vram_total_gb=0.0,
            vram_free_gb=0.0,
            cuda_version="N/A",
            compute_capability="N/A",
            supports_bf16=False,
            is_blackwell=False,
            torch_available=False,
        )

    pref = (preference or "auto").lower().strip()
    if pref != "cpu" and torch.cuda.is_available():
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        major, minor = props.major, props.minor
        total_gb = props.total_memory / (1024**3)
        try:
            free_bytes, _ = torch.cuda.mem_get_info(idx)
            free_gb = free_bytes / (1024**3)
        except Exception:
            free_gb = total_gb
        cc = f"{major}.{minor}"
        is_blackwell = major >= 12
        return DeviceProfile(
            backend="cuda",
            device_name=props.name,
            vram_total_gb=round(total_gb, 2),
            vram_free_gb=round(free_gb, 2),
            cuda_version=str(torch.version.cuda or "unknown"),
            compute_capability=cc,
            supports_bf16=bool(torch.cuda.is_bf16_supported()),
            is_blackwell=is_blackwell,
            torch_available=True,
        )

    if pref != "cpu" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return DeviceProfile(
            backend="mps",
            device_name="Apple Silicon GPU",
            vram_total_gb=0.0,
            vram_free_gb=0.0,
            cuda_version="N/A",
            compute_capability="MPS",
            supports_bf16=False,
            is_blackwell=False,
            torch_available=True,
        )

    return DeviceProfile(
        backend="cpu",
        device_name="CPU",
        vram_total_gb=0.0,
        vram_free_gb=0.0,
        cuda_version="N/A",
        compute_capability="N/A",
        supports_bf16=False,
        is_blackwell=False,
        torch_available=True,
    )


def get_torch_device(preference: str = "auto") -> Any:
    """Return a torch.device object (or None if torch is absent)."""
    if torch is None:
        return None
    profile = get_device_profile(preference)
    return torch.device(profile.backend)


def release_gpu_memory() -> None:
    """
    Aggressively release PyTorch CUDA cached VRAM.

    CRITICAL for 16GB RTX 5070 Ti workflows:
    Run PyTorch CUDA Quant/ML/Monte-Carlo FIRST, call `release_gpu_memory()`,
    and THEN invoke Ollama local LLM so neither engine OOMs on 16GB GDDR7.
    """
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass
