import torch
import torch.nn as nn
import torch.nn.functional as F

class Extractor(nn.Module):
    def __init__(self, steps=5, out_dim=5, n_channels=1, tau_init=0.0, **kwargs):
        super().__init__()
        self.n_channels = n_channels
        table = torch.zeros(steps, 2, out_dim)
        table[..., 1:3] = tau_init
        self.table = nn.Parameter(table)

    def forward(self, inputs):
        step = inputs['step']
        # (1, 2, out_dim)
        out = self.table[step:step+1]
        return out, None