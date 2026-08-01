import os
import psutil
from dataclasses import dataclass
from typing import Dict, Any, Optional

try:
    import torch
except ImportError:
    torch = None

from .hardware import HardwareInfo, detect_hardware

@dataclass
class MemoryProfile:
    total_ram_gb: float
    available_ram_gb: float
    total_vram_gb: float
    available_vram_gb: float
    profile_level: str          # 'low' (<=12GB), 'medium' (12GB-32GB), 'high' (>32GB)
    recommended_max_seq_len: int
    recommended_micro_batch_size: int
    recommended_grad_accum_steps: int
    recommend_quantization: bool
    recommend_gradient_checkpointing: bool
    recommend_cpu_offload: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_ram_gb": round(self.total_ram_gb, 2),
            "available_ram_gb": round(self.available_ram_gb, 2),
            "total_vram_gb": round(self.total_vram_gb, 2),
            "available_vram_gb": round(self.available_vram_gb, 2),
            "profile_level": self.profile_level,
            "recommended_max_seq_len": self.recommended_max_seq_len,
            "recommended_micro_batch_size": self.recommended_micro_batch_size,
            "recommended_grad_accum_steps": self.recommended_grad_accum_steps,
            "recommend_quantization": self.recommend_quantization,
            "recommend_gradient_checkpointing": self.recommend_gradient_checkpointing,
            "recommend_cpu_offload": self.recommend_cpu_offload,
        }

def get_memory_budget(hw_info: Optional[HardwareInfo] = None) -> MemoryProfile:
    if hw_info is None:
        hw_info = detect_hardware()

    # System RAM
    vm = psutil.virtual_memory()
    total_ram_gb = vm.total / (1024 ** 3)
    available_ram_gb = vm.available / (1024 ** 3)

    # VRAM
    total_vram_gb = 0.0
    available_vram_gb = 0.0

    if torch is not None and hw_info.device_type in ("cuda", "rocm"):
        try:
            free_b, total_b = torch.cuda.mem_get_info()
            total_vram_gb = total_b / (1024 ** 3)
            available_vram_gb = free_b / (1024 ** 3)
        except Exception:
            total_vram_gb = 8.0
            available_vram_gb = 6.0
    elif hw_info.device_type == "mps":
        # Apple Silicon shares unified memory
        total_vram_gb = total_ram_gb * 0.75
        available_vram_gb = available_ram_gb * 0.75
    else:
        # CPU host
        total_vram_gb = available_ram_gb
        available_vram_gb = available_ram_gb

    effective_mem = max(available_vram_gb, available_ram_gb if hw_info.device_type in ("mps", "arm_cpu", "cpu") else available_vram_gb)

    # Memory tier decision logic
    if effective_mem <= 12.0 or total_ram_gb <= 12.0:
        profile_level = "low"
        max_seq_len = 512
        micro_batch_size = 1
        grad_accum = 8
        quantization = True
        grad_ckpt = True
        cpu_offload = True
    elif effective_mem <= 28.0 or total_ram_gb <= 32.0:
        profile_level = "medium"
        max_seq_len = 2048
        micro_batch_size = 2
        grad_accum = 4
        quantization = True
        grad_ckpt = True
        cpu_offload = False
    else:
        profile_level = "high"
        max_seq_len = 4096
        micro_batch_size = 4
        grad_accum = 2
        quantization = False
        grad_ckpt = False
        cpu_offload = False

    return MemoryProfile(
        total_ram_gb=total_ram_gb,
        available_ram_gb=available_ram_gb,
        total_vram_gb=total_vram_gb,
        available_vram_gb=available_vram_gb,
        profile_level=profile_level,
        recommended_max_seq_len=max_seq_len,
        recommended_micro_batch_size=micro_batch_size,
        recommended_grad_accum_steps=grad_accum,
        recommend_quantization=quantization,
        recommend_gradient_checkpointing=grad_ckpt,
        recommend_cpu_offload=cpu_offload,
    )

def configure_memory_guardrails(hw_info: HardwareInfo, mem_profile: MemoryProfile):
    """Configures PyTorch environment variables to prevent OOM errors."""
    if hw_info.device_type in ("cuda", "rocm"):
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128")
    elif hw_info.device_type == "mps":
        if mem_profile.profile_level == "low":
            os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.7")
        else:
            os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.9")
