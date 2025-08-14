import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class Extractor(nn.Module):
    def __init__(self, steps=5, out_dim=5, **kwargs):
        super().__init__()
        self.table = nn.Parameter(torch.zeros(steps, 2, out_dim))

    def forward(self, inputs):
        step = inputs['step']
        out = self.table[step:step+1, :, :]
        # (B, p or c, n_params, 1, 1, 1)
        out = out.reshape(1, 2, -1, 1, 1, 1)
        return out, None