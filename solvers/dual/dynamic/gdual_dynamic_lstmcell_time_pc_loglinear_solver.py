import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from ...solver import Solver
from collections import OrderedDict


class GDual_Dynamic_LSTMCell_Time_PC_LogLinear_Solver(Solver):
    """
    GDual PC with Log-Linear L(y, τ)=τ·log y + (1-τ)·y
    - u/v는 로그-도메인 계산으로 pow 제거
    - L, Δ 등은 로그/exp 조합으로 계산 (LambertW 불필요)
    - 시간 특징(t_i, t_{i+1}) + LSTMCell로 스텝별 파라미터 예측
    - AMP 안전 캐스팅
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

        # learned timesteps
        t_0 = 1.0 / noise_schedule.total_N
        t_T = noise_schedule.T
        ts0 = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=steps, device='cpu', shift=flow_shift)
        self.log_deltas = nn.Parameter(torch.log(ts0[:-1] - ts0[1:]))

        # feature -> hidden input
        self.feat = nn.Sequential(OrderedDict([
            ("gap",  nn.AdaptiveAvgPool2d(1)),  # [B,C,H,W] -> [B,C,1,1]
            ("flat", nn.Flatten(1)),            # [B,C,1,1] -> [B,C]
            ("proj", nn.LazyLinear(hidden_dim)),
        ]))
        self.time_info   = nn.Linear(2, hidden_dim)                # [B,2] -> [B,H]
        self.lstm        = nn.LSTMCell(hidden_dim, hidden_dim)     # input:[B,H], (h,c):[B,H]
        self.hidden_h_init = nn.Parameter(torch.zeros(1, hidden_dim))
        self.hidden_c_init = nn.Parameter(torch.zeros(1, hidden_dim))

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

    # ---------- transforms ----------
    @staticmethod
    def _L_from_logy(logy: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        # L(y,τ) = τ log y + (1-τ) y,  y = exp(logy)
        return tau * logy + (1.0 - tau) * torch.exp(logy)

    @staticmethod
    def _logy_u(la: torch.Tensor, ls: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        pos = (gamma >= 0)
        return torch.where(pos, la - gamma * ls, (1.0 + gamma) * la)

    @staticmethod
    def _logy_v(la: torch.Tensor, ls: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        pos = (gamma >= 0)
        return torch.where(pos, (1.0 - gamma) * ls, ls + gamma * la)

    @staticmethod
    def _O_delta_square(delta, kappa):
        return kappa * (delta ** 2)

    # ---------- params via LSTMCell ----------
    def get_params(self, xc, hidden, t_pair):
        """
        xc: [B,C,H,W]
        hidden: (h, c) where each [B,H]
        t_pair: [2] or [B,2]  (예: [t_i, t_{i+1}])
        return: params [B,2,5], new_hidden (h,c)
        """
        B = xc.size(0)
        h, c = hidden
        head_dtype = self.out.weight.dtype

        x_in = self.feat(xc).to(head_dtype)  # [B,H]
        if t_pair.dim() == 1:
            t_pair = t_pair.view(1, -1).expand(B, -1)
        t_pair = t_pair.to(xc.device, dtype=head_dtype)

        x_in = x_in + self.time_info(t_pair)  # [B,H]

        h = h.to(xc.device, dtype=head_dtype)
        c = c.to(xc.device, dtype=head_dtype)
        new_h, new_c = self.lstm(x_in, (h, c))  # [B,H], [B,H]

        out = self.out(new_h)                   # [B,10]
        return out.view(B, 2, 5), (new_h, new_c)

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

        logy_uc = self._logy_u(la_c, ls_c, gamma)
        logy_un = self._logy_u(la_n, ls_n, gamma)
        logy_vc = self._logy_v(la_c, ls_c, gamma)
        logy_vn = self._logy_v(la_n, ls_n, gamma)

        y_uc = torch.exp(logy_uc).clamp_min(self.eps)
        y_un = torch.exp(logy_un).clamp_min(self.eps)
        y_vc = torch.exp(logy_vc).clamp_min(self.eps)
        y_vn = torch.exp(logy_vn).clamp_min(self.eps)

        val_uc = self._L_from_logy(logy_uc, tau_x)
        val_un = self._L_from_logy(logy_un, tau_x)
        val_vc = self._L_from_logy(logy_vc, tau_e)
        val_vn = self._L_from_logy(logy_vn, tau_e)
        delta_u = val_un - val_uc
        delta_v = val_vn - val_vc

        if order == 1:
            denom_u = (1.0 - tau_x) * y_uc + tau_x
            denom_v = (1.0 - tau_e) * y_vc + tau_e
            X = xc * (y_uc / denom_u * delta_u + self._O_delta_square(delta_u, kappa_x))
            E = ec * (y_vc / denom_v * delta_v + self._O_delta_square(delta_v, kappa_e))
        else:
            X = xc * (y_un - y_uc)
            E = ec * (y_vn - y_vc)
            core_u = (delta_u / ((1.0 - tau_x) + tau_x / y_uc)) + self._O_delta_square(delta_u, kappa_x)
            core_v = (delta_v / ((1.0 - tau_e) + tau_e / y_vc)) + self._O_delta_square(delta_v, kappa_e)
            if corrector:
                X += 0.5 * (xn - xc) * core_u
                E += 0.5 * (en - ec) * core_v
            else:
                if i > 0:
                    la_p, ls_p = log_alpha[i - 1], log_sigma[i - 1]
                    logy_up = self._logy_u(la_p, ls_p, gamma)
                    logy_vp = self._logy_v(la_p, ls_p, gamma)
                    val_up  = self._L_from_logy(logy_up, tau_x)
                    val_vp  = self._L_from_logy(logy_vp, tau_e)
                    r_u = (val_uc - val_up) / (delta_u + eps)
                    r_v = (val_vc - val_vp) / (delta_v + eps)
                else:
                    r_u = r_v = torch.as_tensor(1.0, device=sample.device, dtype=sample.dtype)
                X += 0.5 * (xc - xp) * core_u / r_u.clamp_min(eps)
                E += 0.5 * (ec - ep) * core_v / r_v.clamp_min(eps)

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

        alphas = self.noise_schedule.marginal_alpha(timesteps)
        sigmas = self.noise_schedule.marginal_std(timesteps)

        log_alpha = torch.log(alphas)
        log_sigma = torch.log(sigmas)
        log_alpha_ratio = log_alpha[1:] - log_alpha[:-1]
        log_sigma_ratio = log_sigma[1:] - log_sigma[:-1]
        log_alpha_next  = log_alpha[1:]
        log_sigma_next  = log_sigma[1:]

        x_pred = x
        x_corr = x
        xn, en = None, None
        xp, ep = None, None

        xc, ec = self.checkpoint_model_fn(x_pred, timesteps[0])

        # hidden init (학습 가능 파라미터) -> 배치 확장
        head_dtype = self.out.weight.dtype
        h = self.hidden_h_init.to(device=xc.device, dtype=head_dtype).expand(xc.size(0), -1).contiguous()
        c = self.hidden_c_init.to(device=xc.device, dtype=head_dtype).expand(xc.size(0), -1).contiguous()

        params, (h, c) = self.get_params(xc, (h, c), timesteps[0:2])

        use_tqdm = os.getenv("DPM_TQDM", "0") not in ("0", "False", "false", "")
        for i in tqdm(range(self.steps), disable=not use_tqdm):
            p_order = min(i + 1, self.steps - i, self.order) if self.lower_order_final else min(i + 1, self.order)

            gamma, tau_x, tau_e, kx, ke = (t.view(t.size(0), 1, 1, 1) for t in params[:, 0].unbind(dim=1))
            tau_x = torch.sigmoid(tau_x)
            tau_e = torch.sigmoid(tau_e)

            x_pred = self.get_next_sample(
                x_corr, (xn, xc, xp), (en, ec, ep), i,
                log_alpha, log_sigma,
                log_alpha_ratio, log_sigma_ratio, log_alpha_next, log_sigma_next,
                gamma, tau_x, tau_e, kx, ke,
                p_order, self.eps, corrector=False
            )

            if i < self.steps - 1:
                xn, en = self.checkpoint_model_fn(x_pred, timesteps[i + 1])
                params, (h, c) = self.get_params(xn, (h, c), timesteps[i+1:i+3])
            else:
                break

            if self.use_corrector:
                gamma, tau_x, tau_e, kx, ke = (t.view(t.size(0), 1, 1, 1) for t in params[:, 1].unbind(dim=1))
                tau_x = torch.sigmoid(tau_x)
                tau_e = torch.sigmoid(tau_e)
                x_corr = self.get_next_sample(
                    x_corr, (xn, xc, xp), (en, ec, ep), i,
                    log_alpha, log_sigma,
                    log_alpha_ratio, log_sigma_ratio, log_alpha_next, log_sigma_next,
                    gamma, tau_x, tau_e, kx, ke,
                    2, self.eps, corrector=True
                )
            else:
                x_corr = x_pred

            xp, ep = xc, ec
            xc, ec = xn, en

        return x_pred
