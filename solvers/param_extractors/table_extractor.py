import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class Extractor(nn.Module):
    def __init__(self, steps=5, out_dim=5, n_channels=1, **kwargs):
        super().__init__()
        self.n_channels = n_channels
        if n_channels == 1:
            self.table = nn.Parameter(torch.zeros(steps, 2, out_dim))
        else:
            self.table = nn.Parameter(torch.zeros(steps, 2, n_channels, out_dim))

    def forward(self, inputs):
        step = inputs['step']
        if self.n_channels == 1:
            out = self.table[step:step+1, :, :]
            # (B, p or c, n_params, 1, 1, 1)
            out = out.reshape(1, 2, -1, 1, 1, 1)
        else:
            out = self.table[step:step+1, :, :, :]
            # (B, p or c, n_params, 1, 1, 1)
            out = out.reshape(1, 2, -1, self.n_channels, 1, 1)

        return out, None