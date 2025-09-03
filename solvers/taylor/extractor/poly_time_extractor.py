import torch
import torch.nn as nn
import torch.nn.functional as F

class Extractor(nn.Module):
    def __init__(self, poly_order=3, out_dim=5, **kwargs):
        super().__init__()
        self.poly_order = poly_order
        self.coeff = nn.Parameter(torch.zeros(poly_order, 2, out_dim))

    def forward(self, inputs):
        # (1,)
        t = inputs['t'][:1]
        # (3, 2, out_dim)
        c = self.coeff
        # (1, 2, out_dim)
        out = 0
        for p in range(self.poly_order):
            out = c[p:p+1]*(t**p)
        return out, None