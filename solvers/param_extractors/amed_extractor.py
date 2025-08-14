import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# from https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, theta = 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.theta) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

# from https://github.com/zju-pi/diff-sampler/blob/main/amed-solver-main/training/networks.py
class Extractor(nn.Module):
    def __init__(self, hidden_dim=128, out_dim=5, input_shape=(4, 32, 32)):
        super().__init__()
        self.time_embedding = nn.Sequential(SinusoidalPosEmb(hidden_dim),
                                            nn.Linear(hidden_dim, hidden_dim),
                                            nn.GELU(),
                                            nn.Linear(hidden_dim, hidden_dim))
        self.x_linear = nn.Sequential(nn.Linear(input_shape[0], hidden_dim),
                                     nn.GELU(),
                                     nn.Linear(hidden_dim, hidden_dim))
        self.out = nn.Linear(hidden_dim, 2*out_dim)
        
    def forward(self, inputs):
        x = inputs['x'].mean(dim=[2, 3])
        #print(1, x.shape)
        x = self.x_linear(x)
        #print(2, x.shape)
        tc = inputs['t'][0:1]
        #print(3, tc.shape)
        tc = self.time_embedding(tc * 1000)
        #print(4, tc.shape)
        tn = inputs['t'][1:2]
        #print(5, tn.shape)
        tn = self.time_embedding(tn * 1000)
        #print(6, tn.shape)
        h = x + tc + tn
        #print(7, h.shape)
        out = self.out(h).reshape(len(h), 2, -1, 1, 1, 1)
        #print(8, out.shape)
        return out, None
