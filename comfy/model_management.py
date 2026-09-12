"""
    Stubbed model management for stripped-down ComfyUI (S3T_Nodes variant).
    Provides only the interfaces needed by execution.py, server.py, and main.py.
"""
from __future__ import annotations

import logging
from enum import Enum
import threading
import torch
import sys
import comfy.system_memory
import comfy.utils
from comfy.cli_args import args

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from comfy.model_patcher import ModelPatcher


class VRAMState(Enum):
    DISABLED = 0
    NO_VRAM = 1
    LOW_VRAM = 2
    NORMAL_VRAM = 3
    HIGH_VRAM = 4
    SHARED = 5

class CPUState(Enum):
    GPU = 0
    CPU = 1
    MPS = 2

vram_state = VRAMState.NORMAL_VRAM
set_vram_to = VRAMState.NORMAL_VRAM
cpu_state = CPUState.GPU
total_vram = 0
directml_enabled = False
in_training = False
training_fp8_bwd = False
DISABLE_SMART_MEMORY = False
TOTAL_PINNED_MEMORY = 0
NUM_STREAMS = 1
LARGEST_AIMDO_CASTED_WEIGHT = [None, 0]
torch_version = ""
torch_version_numeric = (0, 0)
rocm_version = (0, 0)
total_ram = 0

_xpu_available = False
_mps_available = False
_cudnn_available = False

try:
    torch_version = torch.version.__version__
    parts = torch_version.split("+")[0].split(".")
    torch_version_numeric = tuple(int(x) for x in parts[:3])
except Exception:
    pass

try:
    _xpu_available = hasattr(torch, "xpu") and torch.xpu.is_available()
except Exception:
    pass

try:
    _mps_available = hasattr(torch, "mps") and torch.mps.is_available()
except Exception:
    pass

try:
    _cudnn_available = hasattr(torch.backends, "cudnn") and torch.backends.cudnn.is_available()
except Exception:
    pass


class InterruptProcessingException(Exception):
    pass

_processing_interrupted = False
_processing_interrupted_lock = threading.Lock()

def throw_exception_if_processing_interrupted():
    if _processing_interrupted:
        raise InterruptProcessingException()

def interrupt_current_processing(value=True):
    global _processing_interrupted
    with _processing_interrupted_lock:
        _processing_interrupted = value

def is_oom(ex):
    return False

def debug_memory_summary():
    return ""

def unload_all_models():
    pass

def cleanup_models_gc():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def soft_empty_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def reset_cast_buffers():
    pass

def should_free_pins_for_ram_pressure(ram_shortfall):
    return False

def free_pins(size):
    pass

def set_cudnn_benchmark():
    pass

def get_torch_device():
    if torch.cuda.is_available():
        return torch.device("cuda", 0)
    if _mps_available:
        return torch.device("mps")
    return torch.device("cpu")

def get_torch_device_name(device=None):
    if device is None:
        device = get_torch_device()
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "mps":
        return "MPS"
    return "CPU"

def get_all_torch_devices():
    devices = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            devices.append(torch.device("cuda", i))
    if _mps_available:
        devices.append(torch.device("mps"))
    if not devices:
        devices.append(torch.device("cpu"))
    return devices

def get_total_memory(device, torch_total_too=False):
    if device.type == "cuda":
        total = torch.cuda.get_device_properties(device).total_mem
        if torch_total_too:
            return total, total
        return total
    mem = comfy.system_memory.virtual_memory_available()
    if torch_total_too:
        return mem, mem
    return mem

def get_free_memory(device, torch_free_too=False):
    if device.type == "cuda":
        free = torch.cuda.mem_get_info(device)[0]
        if torch_free_too:
            return free, free
        return free
    mem = comfy.system_memory.virtual_memory_available()
    if torch_free_too:
        return mem, mem
    return mem

def module_size(module):
    return 0

def cast_to(weight, dtype, device, non_blocking=False, copy=False, stream=None, r=None):
    if weight is None:
        return None
    try:
        return weight.to(device=device, dtype=dtype, non_blocking=non_blocking)
    except Exception:
        return weight

def cast_to_device(tensor, device, dtype, copy=False):
    if tensor is None:
        return None
    try:
        return tensor.to(device=device, dtype=dtype, non_blocking=False)
    except Exception:
        return tensor

def cast_to_gathered(*args, **kwargs):
    pass

def is_device_cuda(device):
    return hasattr(device, "type") and device.type == "cuda"

def is_device_cpu(device):
    return hasattr(device, "type") and device.type == "cpu"

def is_nvidia():
    return torch.cuda.is_available() and "nvidia" in torch.cuda.get_device_name(0).lower() if torch.cuda.is_available() else False

def is_amd():
    return torch.cuda.is_available() and "amd" in torch.cuda.get_device_name(0).lower() if torch.cuda.is_available() else False

def device_supports_non_blocking(device):
    if hasattr(device, "type"):
        return device.type in ("cuda", "mps")
    return False

def intermediate_device():
    if torch.cuda.is_available():
        return torch.device("cuda", 0)
    return torch.device("cpu")

def intermediate_dtype():
    return torch.float16

def lora_compute_dtype(device):
    return torch.float16

def minimum_inference_memory():
    return 0

def unload_model_and_clones(model):
    pass

def pin_memory(tensor):
    return False

def unpin_memory(tensor):
    pass

def pin_memory_supported():
    return False

def ensure_pin_registerable(size):
    pass

def ensure_pin_budget(size, loaded=False):
    return True

def free_registrations(size):
    pass

def discard_cuda_async_error():
    pass

def pinned_hostbuf_size(model_size):
    return 0

def sync_stream(device, stream):
    pass

def get_offload_stream(device):
    return None

def get_cast_buffer(stream, device, size, s):
    return None

def get_aimdo_cast_buffer(stream, device):
    return None

def supports_fp8_compute(device):
    return False

def supports_int8_compute(device):
    return False

def supports_nvfp4_compute(device):
    return False

def supports_mxfp8_compute(device):
    return False

def set_ram_cache_release_state(callback, headroom):
    pass
