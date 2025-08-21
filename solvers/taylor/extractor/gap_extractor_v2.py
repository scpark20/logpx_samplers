import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

# --- core stable extractor (drop-in) ---
class Linear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, orthogonal=False):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        if orthogonal:
            nn.init.orthogonal_(self.linear.weight, gain=1.0)
        else:
            nn.init.normal_(self.linear.weight, mean=0.0, std=0.02)
        if bias: nn.init.zeros_(self.linear.bias)
    def forward(self, x): return self.linear(x)

class ExtractorStable(nn.Module):
    def __init__(
        self,
        hidden_dim=128,
        out_dim=5,
        input_shape=(4, 32, 32),
        t_dim=2,
        t_fourier_feats=0,
        dropout=0.1,
        hidden=True,
        softplus_scale=False,
        **kwargs
    ):
        super().__init__()
        C = input_shape[0]
        self.hidden, self.hidden_dim = hidden, hidden_dim
        self.out_dim, self.softplus_scale = out_dim, softplus_scale

        self.feat = nn.Sequential(OrderedDict([
            ("gap",  nn.AdaptiveAvgPool2d(1)),
            ("flat", nn.Flatten(1)),
            ("proj", nn.Sequential(
                Linear(C, hidden_dim, orthogonal=True),
                nn.SiLU(),
                Linear(hidden_dim, hidden_dim, orthogonal=True),
            )),
        ]))
        self.norm = nn.LayerNorm(hidden_dim)

        # time embedding
        self.register_buffer("_t_B", torch.tensor(t_fourier_feats, dtype=torch.long), persistent=False)
        t_embed_dim = t_dim
        if t_fourier_feats > 0:
            self.register_buffer("W_t", torch.randn(t_dim, t_fourier_feats) * 2.0, persistent=False)
            t_embed_dim = 2 * t_dim * t_fourier_feats

        self.t_mlp = nn.Sequential(
            Linear(t_embed_dim, hidden_dim, orthogonal=True),
            nn.SiLU(),
            Linear(hidden_dim, hidden_dim, orthogonal=True),
        )
        self.h_mlp = nn.Sequential(
            Linear(hidden_dim, hidden_dim, orthogonal=True),
            nn.SiLU(),
            Linear(hidden_dim, hidden_dim, orthogonal=True),
        )
        self.film = Linear(2 * hidden_dim, 2 * hidden_dim, orthogonal=True)
        self.rezero_t    = nn.Parameter(torch.zeros(1))
        self.rezero_h    = nn.Parameter(torch.zeros(1))
        self.rezero_film = nn.Parameter(torch.zeros(1))

        self.gru = nn.GRUCell(input_size=hidden_dim, hidden_size=hidden_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.hidden_init = nn.Parameter(torch.zeros(1, hidden_dim))

        self.head = Linear(hidden_dim, 2 * out_dim, bias=True)
        with torch.no_grad():
            self.head.linear.weight.zero_()
            self.head.linear.bias.zero_()

    def _embed_t(self, t):
        if self._t_B.item() == 0:
            return t
        # [B, t_dim] -> [B, 2 * t_dim * Bfreq]
        proj = [t[:, j:j+1] @ self.W_t[j] for j in range(t.shape[1])]
        proj = torch.cat(proj, dim=1)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=1)

    def forward(self, inputs):
        x = inputs['x']
        device, B = x.device, x.size(0)

        h_prev = inputs.get('h', None)
        if h_prev is None: h_prev = self.hidden_init.expand(B, -1)
        elif h_prev.dim() == 1: h_prev = h_prev.unsqueeze(0)
        h_prev = h_prev.to(device)

        t = inputs.get('t', None)
        if t is None: raise ValueError("inputs['t'] must be provided with shape [B, t_dim] or [t_dim].")
        if t.dim() == 1: t = t.unsqueeze(0).expand(B, -1)
        elif t.size(0) == 1 and B > 1: t = t.expand(B, -1)
        t = t.to(device).float()

        f = self.norm(self.feat(x))
        t_emb = self.t_mlp(self._embed_t(t)) * self.rezero_t
        h_emb = self.h_mlp(h_prev)            * self.rezero_h

        gamma, beta = (self.film(torch.cat([t_emb, h_emb], -1)) * self.rezero_film).chunk(2, -1)
        f_mod = self.dropout(F.silu((1 + gamma) * f + beta))

        h_new = self.gru(f_mod, h_prev)
        out = self.head(h_new).view(B, 2, self.out_dim)
        if self.softplus_scale:
            out = torch.stack([out[:, 0], F.softplus(out[:, 1]) + 1e-6], dim=1)

        return (out, h_new) if self.hidden else (out, None)

# --- 4 presets (light -> heavy) ---
class ExtractorTiny(ExtractorStable):
    def __init__(self, out_dim=5, input_shape=(4,32,32), **kwargs):
        super().__init__(
            hidden_dim=64, out_dim=out_dim, input_shape=input_shape,
            t_dim=2, t_fourier_feats=0, dropout=0.0,
            hidden=True, softplus_scale=False, **kwargs
        )

class ExtractorSmall(ExtractorStable):
    def __init__(self, out_dim=5, input_shape=(4,32,32), **kwargs):
        super().__init__(
            hidden_dim=128, out_dim=out_dim, input_shape=input_shape,
            t_dim=2, t_fourier_feats=4, dropout=0.05,
            hidden=True, softplus_scale=False, **kwargs
        )

class ExtractorBase(ExtractorStable):
    def __init__(self, out_dim=5, input_shape=(4,32,32), **kwargs):
        super().__init__(
            hidden_dim=256, out_dim=out_dim, input_shape=input_shape,
            t_dim=2, t_fourier_feats=8, dropout=0.10,
            hidden=True, softplus_scale=True, **kwargs
        )

class ExtractorLarge(ExtractorStable):
    def __init__(self, out_dim=5, input_shape=(4,32,32), **kwargs):
        super().__init__(
            hidden_dim=512, out_dim=out_dim, input_shape=input_shape,
            t_dim=2, t_fourier_feats=16, dropout=0.12,
            hidden=True, softplus_scale=True, **kwargs
        )

# 선택형 팩토리
def build_extractor(version: str = "base", **overrides) -> nn.Module:
    version = version.lower()
    cls = {
        "tiny": ExtractorTiny,
        "small": ExtractorSmall,
        "base": ExtractorBase,
        "large": ExtractorLarge,
    }.get(version, ExtractorBase)
    return cls(**overrides)
