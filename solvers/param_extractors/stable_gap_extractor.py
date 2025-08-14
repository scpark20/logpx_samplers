import torch
import torch.nn as nn
from collections import OrderedDict

class Extractor(nn.Module):
    def __init__(self, hidden_dim=128, out_dim=5, input_shape=(4, 32, 32)):
        super().__init__()
        C = input_shape[0]

        # [B,C,H,W] -> [B,C] -> [B,H]
        self.feat = nn.Sequential(OrderedDict([
            ("gap",  nn.AdaptiveAvgPool2d(1)),
            ("flat", nn.Flatten(1)),
            ("lnc",  nn.LayerNorm(C)),                 # 채널 정규화: 분기 스케일 안정화
            ("proj", nn.Linear(C, hidden_dim)),
        ]))
        self.time_proj   = nn.Linear(2, hidden_dim)    # [t_i, t_{i+1}] → [B,H]
        self.hidden_proj = nn.Linear(hidden_dim, hidden_dim)

        # 게이팅(초기 0 → 학습으로 점진 활성). tanh로 범위 제한(-1,1)
        self.g_feat   = nn.Parameter(torch.zeros(1))
        self.g_time   = nn.Parameter(torch.zeros(1))
        self.g_hidden = nn.Parameter(torch.zeros(1))

        # hidden init을 0으로 (랜덤 오프셋 제거)
        self.hidden_init = nn.Parameter(torch.zeros(1, hidden_dim))

        self.act = nn.GELU()
        
        self.out_dim = out_dim
        self.out = nn.Linear(hidden_dim, 2 * out_dim)
        # 초기에 정확히 0 출력 → 베이스 스킴 유지
        with torch.no_grad():
            self.out.weight.zero_()
            self.out.bias.zero_()

        # 가중치 초기화(분산 너무 크지 않게)
        self._init_linear(self.feat[-1])      # proj
        self._init_linear(self.time_proj)
        self._init_linear(self.hidden_proj)

    def _init_linear(self, layer, std=1.0):
        with torch.no_grad():
            nn.init.normal_(layer.weight, mean=0.0, std=std)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def forward(self, inputs):
        x = inputs['x']                # [B,C,H,W]
        t = inputs['t']                # [2]
        h_in = inputs.get('h', None)   # [B,H] or None
        if h_in is None:
            # 배치에 브로드캐스트
            h_in = self.hidden_init.expand(x.size(0), -1)

        # 분기별 임베딩
        h_feat   = self.feat(x)                           # [B,H]
        h_time   = self.time_proj(t[None, :].expand(x.size(0), -1)) # [B,H]
        h_hidden = self.hidden_proj(h_in)               # [B,H]

        h = h_feat + h_time + h_hidden
        h = self.act(h)

        out = self.out(h)
        out = out.reshape(x.size(0), 2, self.out_dim, 1, 1, 1)
        return out, h
