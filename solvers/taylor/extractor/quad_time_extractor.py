import torch
import torch.nn as nn
import torch.nn.functional as F

class Extractor(nn.Module):
    def __init__(self, out_dim=5, **kwargs):
        super().__init__()
        self.coeff = nn.Parameter(torch.zeros(3, 2, out_dim))

    def forward(self, inputs):
        # (1,)
        t = inputs['t'][:1]
        # (3, 2, out_dim)
        c = self.coeff
        # (1, 2, out_dim)
        out = c[0:1] + c[1:2]*t + c[2:3]*(t**2)
        return out, None