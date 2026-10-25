from __future__ import annotations

import copy

import torch
from torch import nn


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999) -> None:
        if not (0.0 < decay < 1.0):
            raise ValueError("EMA decay must be in (0,1)")
        self.decay = float(decay)
        self.model = copy.deepcopy(model).eval()
        self.model.requires_grad_(False)

    @torch.no_grad()
    def update(self, source: nn.Module) -> None:
        source_state = source.state_dict()
        for name, value in self.model.state_dict().items():
            src = source_state[name].detach()
            if value.is_floating_point():
                value.mul_(self.decay).add_(src, alpha=1.0 - self.decay)
            else:
                value.copy_(src)

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict) -> None:
        self.model.load_state_dict(state_dict)
