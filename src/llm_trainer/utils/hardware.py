import os
import platform
import sys
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

try:
    import torch
except ImportError:
    torch = None

@dataclass
class HardwareInfo:
    device_type: str            # 'cuda', 'rocm', 'mps', 'arm_cpu', 'cpu', 'xpu'
    device_name: str
    arch: str                   # e.g., 'arm64', 'x86_64', 'grace_hopper', 'ampere', etc.
    is_nvidia_grace: bool = False
    is_dgx: bool = False
    is_apple_silicon: bool = False
    is_apu: bool = False
    device_count: int = 1
    supports_bf16: bool = False
    supports_fp16: bool = False
    supports_flash_attn: bool = False
    supports_fp8: bool = False
    optimal_dtype: Any = "float32"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device_type": self.device_type,
            "device_name": self.device_name,
            "arch": self.arch,
            "is_nvidia_grace": self.is_nvidia_grace,
            "is_dgx": self.is_dgx,
            "is_apple_silicon": self.is_apple_silicon,
            "is_apu": self.is_apu,
            "device_count": self.device_count,
            "supports_bf16": self.supports_bf16,
            "supports_fp16": self.supports_fp16,
            "supports_flash_attn": self.supports_flash_attn,
            "supports_fp8": self.supports_fp8,
            "optimal_dtype": str(self.optimal_dtype),
        }

def detect_hardware() -> HardwareInfo:
    machine_arch = platform.machine().lower()
    system_name = platform.system()
    
    # Check Apple Silicon MPS if torch available
    if torch is not None and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return HardwareInfo(
            device_type="mps",
            device_name=f"Apple Silicon ({machine_arch})",
            arch="arm64",
            is_apple_silicon=True,
            device_count=1,
            supports_bf16=hasattr(torch, "bfloat16"),
            supports_fp16=True,
            supports_flash_attn=False,
            supports_fp8=False,
            optimal_dtype=torch.float16 if hasattr(torch, "float16") else "float32",
        )

    # Check CUDA / ROCm / NVIDIA Grace / DGX if torch available
    if torch is not None and torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        device_count = torch.cuda.device_count()
        is_rocm = hasattr(torch.version, "hip") and torch.version.hip is not None
        
        device_type = "rocm" if is_rocm else "cuda"
        
        is_grace = "grace" in device_name.lower() or "gh200" in device_name.lower() or "gb200" in device_name.lower()
        is_dgx = "dgx" in device_name.lower() or "dgx" in os.environ.get("HOSTNAME", "").lower() or is_grace
        is_apu = "apu" in device_name.lower() or "strix" in device_name.lower() or "phoenix" in device_name.lower()
        
        supports_bf16 = torch.cuda.is_bf16_supported()
        supports_fp16 = True
        
        supports_fp8 = False
        if not is_rocm and torch.cuda.get_device_capability(0)[0] >= 8:
            if torch.cuda.get_device_capability(0)[0] >= 9:
                supports_fp8 = True
        
        supports_flash_attn = not is_rocm and torch.cuda.get_device_capability(0)[0] >= 8
        optimal_dtype = torch.bfloat16 if supports_bf16 else torch.float16
        
        return HardwareInfo(
            device_type=device_type,
            device_name=device_name,
            arch="rocm_hip" if is_rocm else f"cuda_sm{torch.cuda.get_device_capability(0)[0]}",
            is_nvidia_grace=is_grace,
            is_dgx=is_dgx,
            is_apple_silicon=False,
            is_apu=is_apu,
            device_count=device_count,
            supports_bf16=supports_bf16,
            supports_fp16=supports_fp16,
            supports_flash_attn=supports_flash_attn,
            supports_fp8=supports_fp8,
            optimal_dtype=optimal_dtype,
        )

    # Check Intel XPU if available
    if torch is not None and hasattr(torch, "xpu") and torch.xpu.is_available():
        return HardwareInfo(
            device_type="xpu",
            device_name="Intel XPU / Integrated Graphics",
            arch="xpu",
            is_apu=True,
            device_count=1,
            supports_bf16=True,
            supports_fp16=True,
            supports_flash_attn=False,
            supports_fp8=False,
            optimal_dtype=torch.bfloat16 if hasattr(torch, "bfloat16") else "float32",
        )

    # Check ARM CPU (Neoverse, Graviton, Ampere Altra, Apple CPU fallback)
    is_arm = "arm" in machine_arch or "aarch64" in machine_arch
    supports_cpu_bf16 = torch is not None and hasattr(torch, "bfloat16")
    
    return HardwareInfo(
        device_type="arm_cpu" if is_arm else "cpu",
        device_name=f"CPU Host ({machine_arch})",
        arch=machine_arch,
        is_apple_silicon="darwin" in system_name.lower() and is_arm,
        is_apu=False,
        device_count=1,
        supports_bf16=supports_cpu_bf16,
        supports_fp16=False,
        supports_flash_attn=False,
        supports_fp8=False,
        optimal_dtype=torch.bfloat16 if (torch is not None and supports_cpu_bf16) else "float32",
    )
