import torch
import torch.nn as nn

class Extractor(nn.Module):
    def __init__(self, out_dim=5, **kwargs):
        super().__init__()
        # coeff[0]: P0, coeff[1]: P1, coeff[2]: P2
        self.coeff = nn.Parameter(torch.zeros(3, 2, out_dim))

    def forward(self, inputs):
        # (1,)
        t = inputs['t'][:1]
        # Legendre basis on [0,1]
        P0 = 1.0
        P1 = 2*t - 1
        P2 = 6*t**2 - 6*t + 1
        c = self.coeff  # (3, 2, out_dim)
        out = c[0:1]*P0 + c[1:2]*P1 + c[2:3]*P2  # (1, 2, out_dim)

        return out, None