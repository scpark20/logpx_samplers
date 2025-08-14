import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class Extractor(nn.Module):
    def __init__(self, hidden_dim=128, out_dim=5, input_shape=(4, 32, 32), dropout=0.0):
        super().__init__()
        # feature -> hidden
        self.feat = nn.Sequential(OrderedDict([
            ("gap",  nn.AdaptiveAvgPool2d(1)),  # [B,C,H,W] -> [B,C,1,1]
            ("flat", nn.Flatten(1)),            # [B,C,1,1] -> [B,C]
            ("proj", nn.Linear(input_shape[0], hidden_dim)),
        ]))
        self._set_zero(self.feat[-1])
        
        self.hidden_dim = hidden_dim
        self.time_proj = nn.Linear(2, hidden_dim)
        self._set_zero(self.time_proj)

        self.hidden_proj = nn.Linear(hidden_dim, hidden_dim)
        self._set_zero(self.hidden_proj)

        self.dropout = nn.Dropout(p=dropout)
        self.hidden_init = nn.Parameter(torch.zeros(1, hidden_dim))
        self.act = nn.GELU()

        self.out_dim = out_dim
        self.out = nn.Linear(hidden_dim, 2*out_dim)
        self._set_zero(self.out)

    def _set_zero(self, layer):
        with torch.no_grad():
            if hasattr(layer, "weight"): layer.weight.zero_()
            if hasattr(layer, "bias") and layer.bias is not None: layer.bias.zero_()

    def forward(self, inputs):
        
        if inputs['h'] is None:
            inputs['h'] = self.hidden_init
        h = self.feat(inputs['x']) + self.time_proj(inputs['t'][None, :]) + self.hidden_proj(inputs['h'])
        h = self.dropout(h)
        h = self.act(h)
        out = self.out(h)
        # (B, 2, n_params, C=1, H=1, W=1)
        out = out.reshape(len(inputs['x']), 2, self.out_dim, 1, 1, 1)
        return out, h