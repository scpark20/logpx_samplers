import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class Linear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        torch.nn.init.normal_(self.linear.weight, mean=0.0, std=0.02)
        torch.nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x)

class Extractor(nn.Module):
    def __init__(self, hidden_dim=128, out_dim=5, input_shape=(4, 32, 32), dropout=0.0, hidden=False, **kwargs):
        super().__init__()
        # feature -> hidden
        self.feat = nn.Sequential(OrderedDict([
            ("gap",  nn.AdaptiveAvgPool2d(1)),  # [B,C,H,W] -> [B,C,1,1]
            ("flat", nn.Flatten(1)),            # [B,C,1,1] -> [B,C]
            ("proj", nn.Sequential(Linear(input_shape[0], hidden_dim),
                                   nn.Tanh(),
                                   Linear(hidden_dim, hidden_dim))
            ),
        ]))
        self.hidden = hidden
        self.hidden_dim = hidden_dim
        self.time_proj = nn.Sequential(Linear(2, hidden_dim),
                                       nn.Tanh(),
                                       Linear(hidden_dim, hidden_dim))
        self.hidden_proj = Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.hidden_init = nn.Parameter(torch.zeros(1, hidden_dim))
        self.act = nn.Tanh()

        self.out_dim = out_dim
        self.out = nn.Linear(hidden_dim, 2*out_dim)
        with torch.no_grad():
            self.out.weight.zero_()
            self.out.bias.zero_()

    def forward(self, inputs):
        
        if inputs['h'] is None:
            inputs['h'] = torch.zeros(len(inputs['x']), self.hidden_dim).to(inputs['x'].device)
        h = self.feat(inputs['x'])
        h = h + self.time_proj(inputs['t'][None, :])
        h = h + self.hidden_proj(inputs['h'])
        h = self.dropout(h)
        h = self.act(h)
        out = self.out(h)
        # (B, 2, n_params)
        out = out.reshape(len(inputs['x']), 2, self.out_dim)
        if self.hidden:
            return out, h
        else:
            return out, None