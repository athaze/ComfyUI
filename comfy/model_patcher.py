"""
    Stubbed model patcher for stripped-down ComfyUI (S3T_Nodes variant).
    Only provides PromptModelTracker needed by execution.py.
"""
from __future__ import annotations

import collections


def is_model_patcher_output(output):
    return False


class PromptModelTracker:
    def __init__(self):
        self.models = {}

    def start(self):
        self.end()

    def end(self):
        self.models = {}

    def add(self, outputs):
        pass
