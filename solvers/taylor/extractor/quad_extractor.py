import torch
import torch.nn as nn
import torch.nn.functional as F

class Extractor(nn.Module):
    def __init__(self, steps=5, out_dim=5, n_channels=1, **kwargs):
        super().__init__()
        self.steps = steps
        self.n_channels = n_channels
        self.coeff = nn.Parameter(torch.zeros(3, 2, out_dim))

    def forward(self, inputs):
        s = inputs['step']
        s = s / self.steps
        # (3, 2, out_dim)
        c = self.coeff
        out = c[0:1] + c[1:2]*s + c[2:3]*(s**2)
        return out, None