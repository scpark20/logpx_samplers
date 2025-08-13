import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from ...solver import Solver
from collections import OrderedDict

class GDual_Dynamic_Time_PC_Box_Solver(Solver):
    """
    GDual PC with Box–Cox (효율화):
      - u/v를 로그-도메인으로 계산해 pow 제거
      - Box–Cox는 expm1/log1p 기반으로 수치안정 + 속도
      - log(alpha/sigma) 및 비율, 파라미터(τ 등) 루프 밖 선계산
      - ratio(i-1,i,i+1) 필요한 시점에만 계산 (i=0 가드)
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
        latent_shape=(4, 32, 32),
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

        t_0 = 1.0 / noise_schedule.total_N
        t_T = noise_schedule.T
        timesteps = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=steps, device='cpu', shift=flow_shift)
        self.log_deltas = nn.Parameter(torch.log(timesteps[:-1] - timesteps[1:]))

        self.feat = nn.Sequential(OrderedDict([
            ("gap",  nn.AdaptiveAvgPool2d(1)),  # [B,C,H,W] -> [B,C,1,1]
            ("flat", nn.Flatten(1)),            # [B,C,1,1] -> [B,C]
            ("proj", nn.LazyLinear(hidden_dim)),# [B,hidden]
        ]))

        self.act = nn.GELU()
        self.time_info = nn.Linear(2, hidden_dim)

        # 마지막 레이어는 0으로 초기화
        self.out = nn.Linear(hidden_dim, 10)    # 2*5
        with torch.no_grad():
            self.out.weight.zero_()
            if self.out.bias is not None:
                self.out.bias.zero_()

    def learned_timesteps(self, device=None, dtype=None):
        """
        log_deltas (length = steps)  ->  timesteps (length = steps+1, strictly decreasing)
        anchors: t[0] = T, t[-1] = t_eps
        """
        if device is None: device = self.log_deltas.device
        if dtype  is None: dtype  = self.log_deltas.dtype

        T     = torch.as_tensor(self.noise_schedule.T, device=device, dtype=dtype)
        t_eps = torch.as_tensor(1.0 / self.noise_schedule.total_N, device=device, dtype=dtype)

        # 1) 양수 간격 + 총합 고정: softmax로 비율을 만들고 전체 스팬에 맞춤
        w = F.softmax(self.log_deltas, dim=0)            # (S,)
        deltas = (T - t_eps) * w                         # (S,), sum(deltas) = T - t_eps

        # 2) 누적합으로 감소하는 시간축 복원
        c = torch.cumsum(deltas, dim=0)                  # (S,)
        ts = torch.cat([T[None], T - c], dim=0)          # (S+1,)
        return ts

    # ---------- Box–Cox (안정형) ----------
    @staticmethod
    def _box_from_logy(logy: torch.Tensor, tau: torch.Tensor, tau_eps: float = 1e-6) -> torch.Tensor:
        """L_tau(e^{logy}) = (e^{tau·logy} - 1)/tau,  tau→0 에서 logy"""
        small = tau.abs() <= tau_eps
        L_big   = torch.expm1(tau * logy) / tau
        L_small = logy
        return torch.where(small, L_small, L_big)

    # (참고) 역변환의 log형. 여기선 직접 y=exp(logy)로 써서 호출 안함.
    @staticmethod
    def _boxinv_logy(L: torch.Tensor, tau: torch.Tensor, tau_eps: float = 1e-6, eps: float = 1e-12) -> torch.Tensor:
        """log y = log1p(tau·L)/tau,  tau→0 에서 L"""
        small = tau.abs() <= tau_eps
        logy_big   = torch.log1p((tau * L).clamp_min(-1 + eps)) / tau
        logy_small = L
        return torch.where(small, logy_small, logy_big)

    # ---------- u, v 로그-도메인 ----------

    @staticmethod
    def _logy_u(la: torch.Tensor, ls: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        """u: y = alpha * sigma^{-γ} (γ≥0) / alpha^{-(1+γ)} (γ<0) → log y"""
        pos = (gamma >= 0)
        return torch.where(pos, la - gamma * ls, (1.0 + gamma) * la)

    @staticmethod
    def _logy_v(la: torch.Tensor, ls: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        """v: y = sigma^{1-γ} (γ≥0) / (sigma * alpha^{-γ}) (γ<0) → log y"""
        pos = (gamma >= 0)
        return torch.where(pos, (1.0 - gamma) * ls, ls + gamma * la)

    # ---------- 한 스텝 ----------

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

        # 1) u/v의 log y 계산 (pow 제거)
        logy_uc = self._logy_u(la_c, ls_c, gamma)
        logy_un = self._logy_u(la_n, ls_n, gamma)
        logy_vc = self._logy_v(la_c, ls_c, gamma)
        logy_vn = self._logy_v(la_n, ls_n, gamma)

        # 2) Box–Cox 값 (L) 계산 (안정형)
        val_uc = self._box_from_logy(logy_uc, tau_x)
        val_un = self._box_from_logy(logy_un, tau_x)
        val_vc = self._box_from_logy(logy_vc, tau_e)
        val_vn = self._box_from_logy(logy_vn, tau_e)

        delta_u = val_un - val_uc
        delta_v = val_vn - val_vc

        # 3) L^{-1}: 이미 logy_*가 있으므로 y = exp(logy_*) 바로 사용
        y_uc = torch.exp(logy_uc)
        y_un = torch.exp(logy_un)
        y_vc = torch.exp(logy_vc)
        y_vn = torch.exp(logy_vn)

        X = xc * (y_un - y_uc)
        E = ec * (y_vn - y_vc)

        if order == 2:
            # y^{1-τ} = exp((1-τ)·log y)  (pow 제거)
            y_uc_1m = torch.exp((1.0 - tau_x) * logy_uc)
            y_vc_1m = torch.exp((1.0 - tau_e) * logy_vc)

            core_u = y_uc_1m * delta_u + kappa_x * (delta_u ** 2)
            core_v = y_vc_1m * delta_v + kappa_e * (delta_v ** 2)

            if corrector:
                X += 0.5 * (xn - xc) * core_u
                E += 0.5 * (en - ec) * core_v
            else:
                # ratio: i>0에서만 계산
                if i > 0:
                    la_p, ls_p = log_alpha[i - 1], log_sigma[i - 1]
                    logy_up = self._logy_u(la_p, ls_p, gamma)
                    val_up  = self._box_from_logy(logy_up, tau_x)
                    delta_up = val_uc - val_up
                    r_u = delta_up / (delta_u + eps)

                    logy_vp = self._logy_v(la_p, ls_p, gamma)
                    val_vp  = self._box_from_logy(logy_vp, tau_e)
                    delta_vp = val_vc - val_vp
                    r_v = delta_vp / (delta_v + eps)
                else:
                    r_u = r_v = torch.as_tensor(1.0, device=sample.device, dtype=sample.dtype)

                X += 0.5 * (xc - xp) * core_u / r_u.clamp_min(eps)
                E += 0.5 * (ec - ep) * core_v / r_v.clamp_min(eps)

        # 4) sample/grad 계수 (로그-도메인)
        ls, la   = log_sigma_ratio[i], log_alpha_ratio[i]
        lsn, lan = log_sigma_next[i],  log_alpha_next[i]

        sample_coeff = torch.where(gamma >= 0, torch.exp(gamma * ls),  torch.exp(-gamma * la))
        grad_coeff   = torch.where(gamma >= 0, torch.exp(gamma * lsn), torch.exp(-gamma * lan))

        return sample_coeff * sample + grad_coeff * (X + E)

    # get_params만 간단히 변경
    def get_params(self, xc, t_pair):
        """
        xc: [B,C,H,W]
        t_pair: [2] 또는 [B,2] (예: [t_i, t_{i+1}])
        """
        B = xc.size(0)
        h = self.feat(xc)                   # [B,hidden]

        if t_pair.dim() == 1:               # [2] -> [B,2]
            t_pair = t_pair.view(1, -1).expand(B, -1)
        t_pair = t_pair.to(h.device, dtype=h.dtype)

        h = self.act(h + self.time_info(t_pair))  # time을 hidden에서 add
        out = self.out(h)                         # [B,10] = [B,2*5]
        return out.view(B, 2, 5)

    # ---------- 샘플링 ----------

    def sample(self, x, model_fn, **kwargs):
        self.set_model_fn(model_fn)

        device, dtype = x.device, x.dtype
        timesteps = self.learned_timesteps(device=device, dtype=dtype)  # <-- 학습된 ts
        if not self.time_learning:
            timesteps = timesteps.detach()
        # noise_schedule이 텐서를 받아들일 수 있어야 자동미분이 유지됩니다.
        alphas = self.noise_schedule.marginal_alpha(timesteps)          # 벡터화된 구현 권장
        sigmas = self.noise_schedule.marginal_std(timesteps)

        # 로그/비율 선계산
        log_alpha = torch.log(alphas)
        log_sigma = torch.log(sigmas)
        log_alpha_ratio = log_alpha[1:] - log_alpha[:-1]   # len S
        log_sigma_ratio = log_sigma[1:] - log_sigma[:-1]
        log_alpha_next  = log_alpha[1:]
        log_sigma_next  = log_sigma[1:]

        # 초기 상태
        x_pred = x
        x_corr = x
        xn, en = None, None
        xp, ep = None, None

        # 첫 모델 평가
        xc, ec = self.checkpoint_model_fn(x_pred, timesteps[0])
        params = self.get_params(xc, timesteps[0:2])

        use_tqdm = os.getenv("DPM_TQDM", "0") not in ("0", "False", "false", "")
        for i in tqdm(range(self.steps), disable=not use_tqdm):
            p_order = min(i + 1, self.steps - i, self.order) if self.lower_order_final else min(i + 1, self.order)

            # Predictor
            gamma, tau_x, tau_e, kx, ke = (t.view(t.size(0),1,1,1) for t in params[:,0].unbind(dim=1))
            tau_x, tau_e = torch.exp(tau_x), torch.exp(tau_e)
            x_pred = self.get_next_sample(
                x_corr, (xn, xc, xp), (en, ec, ep), i,
                log_alpha, log_sigma,
                log_alpha_ratio, log_sigma_ratio, log_alpha_next, log_sigma_next,
                gamma, tau_x, tau_e, kx, ke,
                p_order, self.eps, corrector=False
            )

            if i < self.steps - 1:
                xn, en = self.checkpoint_model_fn(x_pred, timesteps[i + 1])
                params = self.get_params(xn, timesteps[i+1:i+3])
                
            else:
                break  # 마지막 스텝: 코렉터 없이 종료
            
            if self.use_corrector:
                # Corrector (항상 2차)
                gamma, tau_x, tau_e, kx, ke = (t.view(t.size(0),1,1,1) for t in params[:,1].unbind(dim=1))
                tau_x, tau_e = torch.exp(tau_x), torch.exp(tau_e)
                x_corr = self.get_next_sample(
                    x_corr, (xn, xc, xp), (en, ec, ep), i,
                    log_alpha, log_sigma,
                    log_alpha_ratio, log_sigma_ratio, log_alpha_next, log_sigma_next,
                    gamma, tau_x, tau_e, kx, ke,
                    2, self.eps, corrector=True
                )
            else:
                x_corr = x_pred

            # 시프트
            xp, ep = xc, ec
            xc, ec = xn, en

        return x_pred
