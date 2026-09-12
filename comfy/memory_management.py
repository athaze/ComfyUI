"""
    Stubbed memory management for stripped-down ComfyUI.
"""
import math
import torch
from typing import NamedTuple

aimdo_enabled = False

RAM_CACHE_HEADROOM = 0

class TensorFileSlice(NamedTuple):
    file_ref: object
    lock: object
    offset: int
    size: int

def set_ram_cache_release_state(callback, headroom):
    pass

def vram_aligned_size(tensors):
    return 0

def extra_ram_release(headroom):
    pass

def read_tensor_file_slice_into(tensor, destination, stream=None, destination2=None):
    return False
