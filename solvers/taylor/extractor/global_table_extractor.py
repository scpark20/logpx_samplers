import torch
import torch.nn as nn
import torch.nn.functional as F

class Extractor(nn.Module):
    def __init__(self, steps=5, out_dim=5, n_channels=1, global_gamma=False, global_tau=False, global_kappa=False, **kwargs):
        super().__init__()
        assert out_dim == 5

        self.n_channels = n_channels
        self.table = nn.Parameter(torch.zeros(steps, 2, out_dim))
        
        self.global_gamma = global_gamma
        self.global_tau = global_tau
        self.global_kappa = global_kappa

    def forward(self, inputs):
        step = inputs['step']
        # (1, 2, out_dim)
        out = self.table[step:step+1]
        if self.global_gamma:
            out = torch.cat([self.table[0:1, :, 0:1], out[:, :, 1:]], dim=-1)
        if self.global_tau:
            out = torch.cat([out[:, :, :1], self.table[0:1, :, 1:3], out[:, :, 3:]], dim=-1)
        if self.global_kappa:
            out = torch.cat([out[:, :, :3], self.table[0:1, :, 3:]], dim=-1)

        return out, None