import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class Extractor(nn.Module):
    def __init__(self, hidden_dim=256, out_dim=5, input_shape=(4, 32, 32), std=0.02):
        super().__init__()
        # feature -> hidden
        self.feat = nn.Sequential(OrderedDict([
            ("gap",  nn.AdaptiveAvgPool2d(1)),  # [B,C,H,W] -> [B,C,1,1]
            ("flat", nn.Flatten(1)),            # [B,C,1,1] -> [B,C]
            ("proj", nn.Linear(input_shape[0], hidden_dim)),
        ]))
        
        self.hidden_dim = hidden_dim
        self.time_proj   = nn.Linear(2, hidden_dim)
        self.hidden_proj = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.GELU()

        self.out_dim = out_dim
        self.out = nn.Linear(hidden_dim, 2 * out_dim)

        # ---- init: N(0, 0.02) for weights, 0 for bias ----
        self._init_normal(self.feat[-1], std=std)   # Linear in feat
        self._init_normal(self.time_proj, std=std)
        self._init_normal(self.hidden_proj, std=std)
        self._init_normal(self.out, std=0.0)

        # hidden_init도 동일 규칙
        self.hidden_init = nn.Parameter(torch.empty(1, hidden_dim))
        nn.init.normal_(self.hidden_init, mean=0.0, std=std)

    def _init_normal(self, layer, std=0.02):
        with torch.no_grad():
            if hasattr(layer, "weight") and layer.weight is not None:
                nn.init.normal_(layer.weight, mean=0.0, std=std)
            if hasattr(layer, "bias") and layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, inputs):
        if inputs['h'] is None:
            inputs['h'] = self.hidden_init
        h = (
            self.feat(inputs['x']) +
            self.time_proj(inputs['t'][None, :]) +
            self.hidden_proj(inputs['h'])
        )
        h = self.act(h)
        out = self.out(h)
        out = out.reshape(len(inputs['x']), 2, self.out_dim, 1, 1, 1)
        return out, h
