import torch
import torch.nn as nn
import torch.nn.functional as F

class Extractor(nn.Module):
    """
    - x/e: [B,C,H,W], t: [2], log_alpha: [2], log_sigma: [2]
    - 내부에서 스칼라 피처(스케줄/통계)를 생성하여 결합
    - 출력: (B, 2, out_dim, 1, 1, 1), hidden: (B, hidden_dim)
    """
    def _set_zero(self, layer):
        with torch.no_grad():
            if hasattr(layer, "weight"): layer.weight.zero_()
            if hasattr(layer, "bias") and layer.bias is not None: layer.bias.zero_()

    def __init__(
        self,
        hidden_dim=128,
        out_dim=5,
        input_shape=(4, 32, 32),
        dropout=0.0,
        x_input=True,
        v_input=False,
        e_input=False,
        t_input=True,
        a_input=False,  # alpha/sigma 입력 사용 여부
        rho_input=False, # delta로 작동 안하고 있음 그냥 log rho...
        drho_input=False,
        n_out_layers=1
    ):
        super().__init__()
        C = input_shape[0]

        # ---- 토글 보관
        self.x_input = x_input
        self.v_input = v_input
        self.e_input = e_input
        self.t_input = t_input
        self.a_input = a_input
        self.rho_input = rho_input
        self.drho_input = drho_input

        # --- 이미지 임베딩(브랜치별 프로젝션) ---
        if self.x_input:
            self.x_proj = nn.Linear(C, hidden_dim)
        if self.v_input:
            self.v_proj = nn.Linear(C, hidden_dim)
        if self.e_input:
            self.e_proj = nn.Linear(C, hidden_dim)

        # --- 시간/스케줄 임베딩 ---
        if self.t_input:
            self.time_proj  = nn.Linear(2, hidden_dim)  # [t_i, t_{i+1}]
        if self.a_input:
            self.alpha_proj = nn.Linear(2, hidden_dim)  # [log_alpha_i, log_alpha_{i+1}]
            self.sigma_proj = nn.Linear(2, hidden_dim)  # [log_sigma_i, log_sigma_{i+1}]
        if self.rho_input:
            self.rho_proj  = nn.Linear(2, hidden_dim)
        if self.drho_input:
            self.drho_proj  = nn.Linear(1, hidden_dim)

        # --- 히든/출력 ---
        self.hidden_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.act = nn.GELU()
        self.hidden_init = nn.Parameter(torch.randn(1, hidden_dim))

        self.out_dim = out_dim
        if n_out_layers==1:
            self.out = nn.Linear(hidden_dim, 2 * out_dim)
            self._set_zero(self.out)  # 초기엔 0 출력(베이스 스킴 유지)
        elif n_out_layers==2:
            self.out = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                     nn.GELU(),
                                     nn.Linear(hidden_dim, 2 * out_dim))
            self._set_zero(self.out[-1])  # 초기엔 0 출력(베이스 스킴 유지)
        

    def forward(self, inputs):
        """
        inputs:
            - 'x','e'       : [B,C,H,W]
            - 't'           : [2]
            - 'log_alpha'   : [2]
            - 'log_sigma'   : [2]
            - 'h'           : [B,H] or None
        """
        x  = inputs['x']
        e  = inputs['e']
        t_pair  = inputs['t']
        la_pair = inputs['log_alpha']
        ls_pair = inputs['log_sigma']

        B, C = x.size(0), x.size(1)
        device, dtype = x.device, x.dtype
        t_pair  = t_pair.to(device=device, dtype=dtype)
        la_pair = la_pair.to(device=device, dtype=dtype)
        ls_pair = ls_pair.to(device=device, dtype=dtype)

        # 히든 상태
        h_prev = inputs.get('h', None)
        if h_prev is None:
            h_prev = self.hidden_init
        h = self.hidden_proj(h_prev)  # [B,H]

        # --- 이미지 통계(공간 평균) + 브랜치별 proj 후 합산 ---
        if self.x_input:
            x_mean = x.mean(dim=(2, 3))              # [B,C]
            h = h + self.x_proj(x_mean)              # [B,H]
        if self.v_input:
            # 이산 차분 기반 velocity 근사: v = dα * x + dσ * e
            d_alpha = torch.exp(la_pair[1]) - torch.exp(la_pair[0])
            d_sigma = torch.exp(ls_pair[1]) - torch.exp(ls_pair[0])
            v = d_alpha * x + d_sigma * e
            v_mean = v.mean(dim=(2, 3))              # [B,C]
            h = h + self.v_proj(v_mean)              # [B,H]
        if self.e_input:
            e_mean = e.mean(dim=(2, 3))              # [B,C]
            h = h + self.e_proj(e_mean)              # [B,H]

        # --- 시간/스케줄 임베딩 ---
        if self.t_input:
            t_embed = self.time_proj(t_pair[None, :])   # [B,2] -> [B,H]
            h = h + t_embed
        if self.a_input:
            a_embed = self.alpha_proj(la_pair[None, :]) # [B,H]
            s_embed = self.sigma_proj(ls_pair[None, :]) # [B,H]
            h = h + a_embed + s_embed
        if self.rho_input:
            rho_embed = self.rho_proj((la_pair - ls_pair)[None, :]) # [B,H]
            h = h + rho_embed
        if self.drho_input:
            lrho_pair = la_pair - ls_pair
            drho_embed = self.drho_proj((lrho_pair[1:2] - lrho_pair[0:1])[None, :]) # [B,H]
            h = h + drho_embed
            
        # --- 활성/드롭아웃 & 출력 ---
        h = self.act(self.dropout(h))
        out = self.out(h).reshape(B, 2, self.out_dim, 1, 1, 1)        # (B,2,P,1,1,1)
        return out, h
