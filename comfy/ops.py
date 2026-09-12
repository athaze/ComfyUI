"""
    Stubbed ops for stripped-down ComfyUI (S3T_Nodes variant).
    Provides basic nn modules without quantization/AIMDO dependencies.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import comfy.model_management
from comfy.cli_args import args


class AttentionOp:
    pass

class Linear(torch.nn.Linear):
    def forward(self, input):
        comfy.model_management.throw_exception_if_processing_interrupted()
        return super().forward(input)

class Conv2d(torch.nn.Conv2d):
    def forward(self, input):
        comfy.model_management.throw_exception_if_processing_interrupted()
        return super().forward(input)

class GroupNorm(torch.nn.GroupNorm):
    def forward(self, input):
        comfy.model_management.throw_exception_if_processing_interrupted()
        return super().forward(input)

class LayerNorm(torch.nn.LayerNorm):
    def forward(self, input):
        comfy.model_management.throw_exception_if_processing_interrupted()
        return super().forward(input)

class Embedding(torch.nn.Embedding):
    pass

class RMSNorm(torch.nn.RMSNorm):
    pass

def conv_nd(*args, **kwargs):
    return Conv2d(*args, **kwargs)

def disabled_train(self=True):
    def func(x, *args, **kwargs):
        return x
    return func

cast_to = comfy.model_management.cast_to
