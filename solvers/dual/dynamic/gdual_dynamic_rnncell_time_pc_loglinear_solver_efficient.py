import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from ...solver import Solver
from collections import OrderedDict

# from GDual_Dynamic_RNNCell_Time_PC_LogLinear_Solver, Efficient Version by ChatGPT
class GDual_Dynamic_RNNCell_Time_PC_LogLinear_Solver_Efficient(Solver):
    """
    GDual PC with Log-Linear L(y, τ)=τ·log y + (1-τ)·y
    - u/v는 로그-도메인 계산 (pow 제거)
    - ΔL = τΔlog y + (1-τ)Δy 를 직접 계산 (L 두 번 평가 X)
    - Δy = y_c * expm1(Δlog y) 로 한 번에 (수치안정/속도)
    - ratio(prev->cur)도 ΔL 직접식으로 (y_p exp 불필요)
    - γ 분기 제거(γ⁺, γ⁻ 분해)로 브로드캐스트 효율↑
    - RNNCell로 hidden 업데이트(입력: feat(xc)+time_embed)
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
        dlog_clip=None,   # 예: 40.0 사용 시 expm1 오버플로우 방지
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
        self.dlog_clip = dlog_clip

        # learned timesteps (learnable deltas)
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
        self.time_info   = nn.Linear(2, hidden_dim)           # [B,2] -> [B,H]
        self.rnn         = nn.RNNCell(hidden_dim, hidden_dim) # input:[B,H], hidden:[B,H]
        self.hidden_init = nn.Parameter(torch.zeros(1, hidden_dim))

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

    # ---------- transforms / utils ----------
    @staticmethod
    def _O_delta_square(delta, kappa):
        return kappa * (delta ** 2)

    # ---------- params via RNNCell ----------
    def get_params(self, xc, hidden, t_pair):
        """
        xc: [B,C,H,W]
        hidden: [B,H]
        t_pair: [2] or [B,2] (예: [t_i, t_{i+1}])
        return: params [B,2,5], new_hidden [B,H]
        """
        B = xc.size(0)
        head_dtype = self.out.weight.dtype

        x_in = self.feat(xc).to(head_dtype)  # [B,H]
        if t_pair.dim() == 1:
            t_pair = t_pair.view(1, -1).expand(B, -1)
        t_pair = t_pair.to(xc.device, dtype=head_dtype)

        x_in = x_in + self.time_info(t_pair)  # [B,H], RNNCell input
        hidden = hidden.to(xc.device, dtype=head_dtype)

        new_hidden = self.rnn(x_in, hidden)  # [B,H]
        out = self.out(new_hidden)           # [B,10]
        return out.view(B, 2, 5), new_hidden

    # ---------- one step (optimized) ----------
    def get_next_sample(
        self,
        sample,
        xs, es, i,
        log_alpha, log_sigma,
        log_alpha_ratio, log_sigma_ratio, log_alpha_next, log_sigma_next,
        gamma, tau_x, tau_e, kappa_x, kappa_e,
        order, eps=1e-8, corrector=False
    ):
        """
        ΔL = τ Δlog y + (1-τ) Δy
        Δy = y_c * expm1(Δlog y)
        invJ(y_c) = 1 / ((1-τ) + τ / y_c) = y_c / ((1-τ) y_c + τ)
        """
        xn, xc, xp = xs
        en, ec, ep = es

        # step-wise logs
        la_c, la_n = log_alpha[i],   log_alpha[i + 1]
        ls_c, ls_n = log_sigma[i],   log_sigma[i + 1]
        dla = la_n - la_c                      # == log_alpha_ratio[i]
        dls = ls_n - ls_c                      # == log_sigma_ratio[i]

        # sign-split of gamma (no branch)
        gp = torch.clamp(gamma, min=0)         # gamma^+ >= 0
        gn = torch.clamp(gamma, max=0)         # gamma^- <= 0

        # log y at current (u,v)
        logy_uc = la_c*(1.0 + gn) - gp*ls_c
        logy_vc = ls_c*(1.0 - gp) + gn*la_c

        # Δlog y to next (u,v)
        dlogy_u =  (1.0 + gn)*dla - gp*dls
        dlogy_v =  (1.0 - gp)*dls + gn*dla

        # (optional) stabilize very large deltas
        if self.dlog_clip is not None:
            dlogy_u = dlogy_u.clamp(-self.dlog_clip, self.dlog_clip)
            dlogy_v = dlogy_v.clamp(-self.dlog_clip, self.dlog_clip)

        # y_c only (exp once), and delta-y via expm1
        y_uc = torch.exp(logy_uc).clamp_min(self.eps)
        y_vc = torch.exp(logy_vc).clamp_min(self.eps)
        dy_u = y_uc * torch.expm1(dlogy_u)     # y_un - y_uc
        dy_v = y_vc * torch.expm1(dlogy_v)     # y_vn - y_vc

        # exact ΔL using Δlog y and Δy
        du = tau_x * dlogy_u + (1.0 - tau_x) * dy_u
        dv = tau_e * dlogy_v + (1.0 - tau_e) * dy_v

        # Jacobian inverse at current
        invJ_u = 1.0 / ((1.0 - tau_x) + tau_x / y_uc)
        invJ_v = 1.0 / ((1.0 - tau_e) + tau_e / y_vc)

        if order == 1:
            X = xc * (invJ_u * du + self._O_delta_square(du, kappa_x))
            E = ec * (invJ_v * dv + self._O_delta_square(dv, kappa_e))
        else:
            # predictor base: x += xc*(y_un - y_uc) = xc * dy_u (same for E)
            X = xc * dy_u
            E = ec * dy_v

            core_u = invJ_u * du + self._O_delta_square(du, kappa_x)
            core_v = invJ_v * dv + self._O_delta_square(dv, kappa_e)

            if corrector:
                X = X + 0.5 * (xn - xc) * core_u
                E = E + 0.5 * (en - ec) * core_v
            else:
                if i > 0:
                    la_p, ls_p = log_alpha[i - 1], log_sigma[i - 1]
                    # prev->cur Δlog
                    dlog_u_pc =  (1.0 + gn)*(la_c - la_p) - gp*(ls_c - ls_p)
                    dlog_v_pc =  (1.0 - gp)*(ls_c - ls_p) + gn*(la_c - la_p)

                    if self.dlog_clip is not None:
                        dlog_u_pc = dlog_u_pc.clamp(-self.dlog_clip, self.dlog_clip)
                        dlog_v_pc = dlog_v_pc.clamp(-self.dlog_clip, self.dlog_clip)

                    # exact ΔL(prev->cur) without y_p:
                    # y_c - y_p = - y_c * expm1(-Δlog(prev->cur))
                    du_pc = tau_x * dlog_u_pc + (1.0 - tau_x) * (- y_uc * torch.expm1(-dlog_u_pc))
                    dv_pc = tau_e * dlog_v_pc + (1.0 - tau_e) * (- y_vc * torch.expm1(-dlog_v_pc))

                    r_u = du_pc / (du + eps)
                    r_v = dv_pc / (dv + eps)
                else:
                    r_u = r_v = torch.as_tensor(1.0, device=sample.device, dtype=sample.dtype)

                X = X + 0.5 * (xc - xp) * (core_u / r_u.clamp_min(eps))
                E = E + 0.5 * (ec - ep) * (core_v / r_v.clamp_min(eps))

        # sample/grad coeffs (log-domain)
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

        # noise schedule (벡터화)
        alphas = self.noise_schedule.marginal_alpha(timesteps)
        sigmas = self.noise_schedule.marginal_std(timesteps)

        # logs & ratios (scalars per step)
        log_alpha = torch.log(alphas)
        log_sigma = torch.log(sigmas)
        log_alpha_ratio = log_alpha[1:] - log_alpha[:-1]
        log_sigma_ratio = log_sigma[1:] - log_sigma[:-1]
        log_alpha_next  = log_alpha[1:]
        log_sigma_next  = log_sigma[1:]

        # init
        x_pred = x
        x_corr = x
        xn, en = None, None
        xp, ep = None, None

        # first eval + hidden init
        xc, ec = self.checkpoint_model_fn(x_pred, timesteps[0])

        head_dtype = self.out.weight.dtype
        hidden = self.hidden_init.to(device=xc.device, dtype=head_dtype).expand(xc.size(0), -1).contiguous()

        params, hidden = self.get_params(xc, hidden, timesteps[0:2])

        use_tqdm = os.getenv("DPM_TQDM", "0") not in ("0", "False", "false", "")
        for i in tqdm(range(self.steps), disable=not use_tqdm):
            p_order = min(i + 1, self.steps - i, self.order) if self.lower_order_final else min(i + 1, self.order)

            # Predictor params -> [B,1,1,1]
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
                params, hidden = self.get_params(xn, hidden, timesteps[i+1:i+3])
            else:
                break  # 마지막 스텝

            # Corrector (항상 2차)
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

            # shift
            xp, ep = xc, ec
            xc, ec = xn, en

        return x_pred
