import torch
import torch.nn as nn
import torch.nn.functional as F

class Extractor(nn.Module):
    def __init__(self, steps=5, out_dim=5, n_channels=1, **kwargs):
        super().__init__()
        self.n_channels = n_channels
        self.table = nn.Parameter(torch.zeros(steps, 2, out_dim, n_channels))

    def forward(self, inputs):
        step = inputs['step']
        # (1, 2, out_dim, n_channels)
        out = self.table[step:step+1]
        return out, None