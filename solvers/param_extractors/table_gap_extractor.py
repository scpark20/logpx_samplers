import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class Extractor(nn.Module):
    def __init__(self, steps=5, hidden_dim=128, out_dim=5, input_shape=(4, 32, 32), dropout=0.0):
        super().__init__()
        self.table = nn.Parameter(torch.zeros(steps, 2, out_dim))
        # feature -> hidden
        self.feat = nn.Sequential(OrderedDict([
            ("gap",  nn.AdaptiveAvgPool2d(1)),  # [B,C,H,W] -> [B,C,1,1]
            ("flat", nn.Flatten(1)),            # [B,C,1,1] -> [B,C]
            ("proj", nn.Linear(input_shape[0], hidden_dim)),
        ]))
        self.hidden_dim = hidden_dim
        self.time_proj = nn.Linear(2, hidden_dim)
        self.hidden_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(p=dropout)
        #self.hidden_init = nn.Parameter(torch.randn(1, hidden_dim))
        self.hidden_init = nn.Parameter(torch.zeros(1, hidden_dim))
        self.act = nn.GELU()

        self.out_dim = out_dim
        self.out = nn.Linear(hidden_dim, 2*out_dim)
        with torch.no_grad():
            self.out.weight.zero_()
            self.out.bias.zero_()

    def forward(self, inputs):
        
        if inputs['h'] is None:
            inputs['h'] = self.hidden_init
        h = self.feat(inputs['x']) + self.time_proj(inputs['t'][None, :]) + self.hidden_proj(inputs['h'])
        h = self.dropout(h)
        h = self.act(h)
        out = self.out(h)
        out = out.reshape(len(inputs['x']), 2, self.out_dim)
        scale = torch.exp(out)
        step = inputs['step']
        out = self.table[step:step+1, :, :] * scale
        out = out.reshape(len(out), 2, -1, 1, 1, 1)
        return out, h