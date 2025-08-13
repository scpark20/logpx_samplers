import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from ...solver import Solver
from collections import OrderedDict


class GDual_Dynamic_RNN_Time_PC_Box_Solver(Solver):
    """
    GDual PC with Box–Cox
      - L_tau(y) = (y^tau - 1)/tau  (tau->0에서는 log y)
      - u/v는 로그 도메인 계산( pow 대신 add/mul/exp )
      - L, Δ는 expm1/log1p 기반의 안정형 계산
      - RNN-like: feat(xc) + time_info([t_i, t_{i+1}]) + hidden_info(h_{i-1})로 스텝별 파라미터 예측
    """
    def __init__(
        self,
        noise_schedule,
        steps,
        skip_type="time_uniform_flow",
        flow_shift=1.0,
        order=2,
        lower_order_final=True,
        eps=1e-8,
        algorithm_type="dual_prediction",
        time_learning=True,
        use_corrector=True,
        hidden_dim=128,
    ):
        assert algorithm_type == "dual_prediction"
        assert order <= 2
        super().__init__(noise_schedule, algorithm_type)

        self.steps = steps
        self.skip_type = skip_type
        self.order = order
        self.flow_shift = flow_shift
        self.lower_order_final = lower_order_final
        self.eps = eps
        self.time_learning = time_learning
        self.use_corrector = use_corrector
        self.hidden_dim = hidden_dim

        # learned timesteps (learnable deltas)
        t_0 = 1.0 / noise_schedule.total_N
        t_T = noise_schedule.T
        ts0 = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=steps, device='cpu', shift=flow_shift)
        self.log_deltas = nn.Parameter(torch.log(ts0[:-1] - ts0[1:]))

        # feature -> hidden
        self.feat = nn.Sequential(OrderedDict([
            ("gap",  nn.AdaptiveAvgPool2d(1)),  # [B,C,H,W] -> [B,C,1,1]
            ("flat", nn.Flatten(1)),            # [B,C,1,1] -> [B,C]
            ("proj", nn.LazyLinear(hidden_dim)),
        ]))
        self.time_info   = nn.Linear(2, hidden_dim)
        self.hidden_info = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_init = nn.Parameter(torch.zeros(1, hidden_dim))
        self.act = nn.GELU()

        # hidden -> params([B,10]=[B,2*5]), 마지막 레이어 0으로 초기화
        self.out = nn.Linear(hidden_dim, 10)
        with torch.no_grad():
            self.out.weight.zero_()
            if self.out.bias is not None:
                self.out.bias.zero_()

    # ---------- time steps ----------
    def learned_timesteps(self, device=None, dtype=None):
        if device is None: device = self.log_deltas.device
        if dtype  is None: dtype  = self.log_deltas.dtype
        T     = torch.as_tensor(self.noise_schedule.T, device=device, dtype=dtype)
        t_eps = torch.as_tensor(1.0 / self.noise_schedule.total_N, device=device, dtype=dtype)
        w = F.softmax(self.log_deltas, dim=0)    # (S,)
        deltas = (T - t_eps) * w                 # (S,)
        c = torch.cumsum(deltas, dim=0)          # (S,)
        ts = torch.cat([T[None], T - c], dim=0)  # (S+1,)
        return ts

    # ---------- Box–Cox (안정형) ----------
    @staticmethod
    def _box_from_logy(logy: torch.Tensor, tau: torch.Tensor, tau_eps: float = 1e-6) -> torch.Tensor:
        """L_tau(e^{logy}) = (e^{tau·logy} - 1)/tau,  tau→0 에서 logy"""
        small = tau.abs() <= tau_eps
        L_big   = torch.expm1(tau * logy) / tau
        L_small = logy
        return torch.where(small, L_small, L_big)

    # ---------- u, v 로그-도메인 ----------
    @staticmethod
    def _logy_u(la: torch.Tensor, ls: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        # y = { alpha * sigma^{-γ}  (γ≥0),  alpha^{1+γ} (γ<0) }
        pos = (gamma >= 0)
        return torch.where(pos, la - gamma * ls, (1.0 + gamma) * la)

    @staticmethod
    def _logy_v(la: torch.Tensor, ls: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        # y = { sigma^{1-γ} (γ≥0),  sigma * alpha^{γ} (γ<0) }
        pos = (gamma >= 0)
        return torch.where(pos, (1.0 - gamma) * ls, ls + gamma * la)

    @staticmethod
    def _O_delta_square(delta, kappa):
        return kappa * (delta ** 2)

    # ---------- params (RNN-like) ----------
    def get_params(self, xc, hidden, t_pair):
        """
        xc: [B,C,H,W]
        hidden: [B,H]
        t_pair: [2] or [B,2]  (예: [t_i, t_{i+1}])
        return: params [B,2,5], new_hidden [B,H]
        """
        B = xc.size(0)

        # 헤드 dtype으로 통일(AMP에서 dtype mismatch 방지)
        head_dtype = self.out.weight.dtype

        h = self.feat(xc).to(head_dtype)  # [B,H]
        if t_pair.dim() == 1:
            t_pair = t_pair.view(1, -1).expand(B, -1)
        t_pair = t_pair.to(xc.device, dtype=head_dtype)
        hidden = hidden.to(xc.device, dtype=head_dtype)

        h = self.act(h + self.time_info(t_pair) + self.hidden_info(hidden))
        out = self.out(h)                 # [B,10]
        return out.view(B, 2, 5), h       # params, new_hidden

    # ---------- one step ----------
    def get_next_sample(
        self,
        sample,
        xs, es, i,
        log_alpha, log_sigma,
        log_alpha_ratio, log_sigma_ratio, log_alpha_next, log_sigma_next,
        gamma, tau_x, tau_e, kappa_x, kappa_e,
        order, eps=1e-8, corrector=False
    ):
        xn, xc, xp = xs
        en, ec, ep = es

        la_c, la_n = log_alpha[i],   log_alpha[i + 1]
        ls_c, ls_n = log_sigma[i],   log_sigma[i + 1]

        # log y
        logy_uc = self._logy_u(la_c, ls_c, gamma)
        logy_un = self._logy_u(la_n, ls_n, gamma)
        logy_vc = self._logy_v(la_c, ls_c, gamma)
        logy_vn = self._logy_v(la_n, ls_n, gamma)

        # y
        y_uc = torch.exp(logy_uc).clamp_min(self.eps)
        y_un = torch.exp(logy_un).clamp_min(self.eps)
        y_vc = torch.exp(logy_vc).clamp_min(self.eps)
        y_vn = torch.exp(logy_vn).clamp_min(self.eps)

        # L & Δ (Box–Cox)
        val_uc = self._box_from_logy(logy_uc, tau_x)
        val_un = self._box_from_logy(logy_un, tau_x)
        val_vc = self._box_from_logy(logy_vc, tau_e)
        val_vn = self._box_from_logy(logy_vn, tau_e)
        delta_u = val_un - val_uc
        delta_v = val_vn - val_vc

        # base(euler)
        X = xc * (y_un - y_uc)
        E = ec * (y_vn - y_vc)

        if order == 2:
            # y^{1-τ} = exp((1-τ)·log y)
            y_uc_1m = torch.exp((1.0 - tau_x) * logy_uc)
            y_vc_1m = torch.exp((1.0 - tau_e) * logy_vc)

            core_u = y_uc_1m * delta_u + self._O_delta_square(delta_u, kappa_x)
            core_v = y_vc_1m * delta_v + self._O_delta_square(delta_v, kappa_e)

            if corrector:
                X += 0.5 * (xn - xc) * core_u
                E += 0.5 * (en - ec) * core_v
            else:
                if i > 0:
                    la_p, ls_p = log_alpha[i - 1], log_sigma[i - 1]
                    logy_up = self._logy_u(la_p, ls_p, gamma)
                    logy_vp = self._logy_v(la_p, ls_p, gamma)
                    val_up  = self._box_from_logy(logy_up, tau_x)
                    val_vp  = self._box_from_logy(logy_vp, tau_e)
                    r_u = (val_uc - val_up) / (delta_u + eps)
                    r_v = (val_vc - val_vp) / (delta_v + eps)
                else:
                    r_u = r_v = torch.as_tensor(1.0, device=sample.device, dtype=sample.dtype)
                X += 0.5 * (xc - xp) * core_u / r_u.clamp_min(eps)
                E += 0.5 * (ec - ep) * core_v / r_v.clamp_min(eps)

        # coeffs
        ls, la   = log_sigma_ratio[i], log_alpha_ratio[i]
        lsn, lan = log_sigma_next[i],  log_alpha_next[i]
        sample_coeff = torch.where(gamma >= 0, torch.exp(gamma * ls),  torch.exp(-gamma * la))
        grad_coeff   = torch.where(gamma >= 0, torch.exp(gamma * lsn), torch.exp(-gamma * lan))

        return sample_coeff * sample + grad_coeff * (X + E)

    # ---------- sampling ----------
    def sample(self, x, model_fn, **kwargs):
        self.set_model_fn(model_fn)

        device, dtype = x.device, x.dtype
        timesteps = self.learned_timesteps(device=device, dtype=dtype)
        if not self.time_learning:
            timesteps = timesteps.detach()

        # schedule
        alphas = self.noise_schedule.marginal_alpha(timesteps)
        sigmas = self.noise_schedule.marginal_std(timesteps)

        # logs/ratios
        log_alpha = torch.log(alphas)
        log_sigma = torch.log(sigmas)
        log_alpha_ratio = log_alpha[1:] - log_alpha[:-1]
        log_sigma_ratio = log_sigma[1:] - log_sigma[:-1]
        log_alpha_next  = log_alpha[1:]
        log_sigma_next  = log_sigma[1:]

        # states
        x_pred = x
        x_corr = x
        xn, en = None, None
        xp, ep = None, None

        # first eval + RNN init
        xc, ec = self.checkpoint_model_fn(x_pred, timesteps[0])

        # hidden init -> 헤드 dtype 맞춤
        head_dtype = self.out.weight.dtype
        hidden = self.hidden_init.to(device=device, dtype=head_dtype).expand(xc.size(0), -1).contiguous()

        params, hidden = self.get_params(xc, hidden, timesteps[0:2])

        use_tqdm = os.getenv("DPM_TQDM", "0") not in ("0", "False", "false", "")
        for i in tqdm(range(self.steps), disable=not use_tqdm):
            p_order = min(i + 1, self.steps - i, self.order) if self.lower_order_final else min(i + 1, self.order)

            # Predictor params (B,2,5) -> (B,1,1,1)
            gamma, tau_x, tau_e, kx, ke = (t.view(t.size(0), 1, 1, 1) for t in params[:, 0].unbind(dim=1))
            # Box–Cox는 τ>0 권장 → exp로 양수화
            tau_x = torch.exp(tau_x)
            tau_e = torch.exp(tau_e)

            x_pred = self.get_next_sample(
                x_corr, (xn, xc, xp), (en, ec, ep), i,
                log_alpha, log_sigma,
                log_alpha_ratio, log_sigma_ratio, log_alpha_next, log_sigma_next,
                gamma, tau_x, tau_e, kx, ke,
                p_order, self.eps, corrector=False
            )

            if i < self.steps - 1:
                xn, en = self.checkpoint_model_fn(x_pred, timesteps[i + 1])
                params, hidden = self.get_params(xn, hidden, timesteps[i+1:i+3])
            else:
                break

            # Corrector
            if self.use_corrector:
                gamma, tau_x, tau_e, kx, ke = (t.view(t.size(0), 1, 1, 1) for t in params[:, 1].unbind(dim=1))
                tau_x = torch.exp(tau_x)
                tau_e = torch.exp(tau_e)
                x_corr = self.get_next_sample(
                    x_corr, (xn, xc, xp), (en, ec, ep), i,
                    log_alpha, log_sigma,
                    log_alpha_ratio, log_sigma_ratio, log_alpha_next, log_sigma_next,
                    gamma, tau_x, tau_e, kx, ke,
                    2, self.eps, corrector=True
                )
            else:
                x_corr = x_pred

            # shift
            xp, ep = xc, ec
            xc, ec = xn, en

        return x_pred
