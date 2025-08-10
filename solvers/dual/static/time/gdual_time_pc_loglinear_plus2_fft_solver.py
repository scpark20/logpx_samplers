import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from ....solver import Solver

def lambertw_autograd(x: torch.Tensor, max_iter: int = 10) -> torch.Tensor:
    w = torch.log1p(x).clamp(min=-10, max=10)
    for _ in range(max_iter):
        ew = torch.exp(w)
        wew = w * ew
        num = wew - x
        den = ew * (w + 1) - (w + 2) * num / (2 * w + 2)
        w = w - num / den
    return w

class GDual_Time_PC_LogLinear_Plus2_FFT_Solver(Solver):
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
        param_dim=(),
        # --- 추가 ---
        cutoff=0.15,          # cycles/pixel, 0.5=Nyquist(axis)
        transition=0.02       # 부드러운 경계 폭 (0이면 하드 컷)
    ):
        assert algorithm_type == 'dual_prediction'
        assert order <= 2
        super().__init__(noise_schedule, algorithm_type)

        self.steps = steps
        self.skip_type = skip_type
        self.order = order
        self.flow_shift = flow_shift
        self.lower_order_final = lower_order_final
        self.eps = eps

        # params: [steps, phase(2), {gamma, tau_x, tau_e, kappa_x, kappa_e}, ...]
        if len(param_dim) > 0:
            init_params = torch.zeros(steps, 2, 5, 1, *param_dim)
        else:
            init_params = torch.zeros(steps, 2, 5)

        # --- 밴드별 파라미터 세트 ---
        self.params_low  = nn.Parameter(init_params.clone())
        self.params_high = nn.Parameter(init_params.clone())

        # 시간 그리드 학습
        t_0 = 1.0 / noise_schedule.total_N
        t_T = noise_schedule.T
        timesteps = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=steps, device='cpu', shift=flow_shift)
        self.log_deltas = nn.Parameter(torch.log(timesteps[:-1] - timesteps[1:]))

        # --- FFT 밴드 설정 ---
        self.cutoff = float(cutoff)
        self.transition = float(transition)
        self._cached_mask = None   # (1,1,H,W//2+1) rFFT용 마스크 캐시

    # ====== 시간 그리드 ======
    def learned_timesteps(self, device=None, dtype=None):
        if device is None: device = self.log_deltas.device
        if dtype  is None: dtype  = self.log_deltas.dtype
        T     = torch.as_tensor(self.noise_schedule.T, device=device, dtype=dtype)
        t_eps = torch.as_tensor(1.0 / self.noise_schedule.total_N, device=device, dtype=dtype)
        w = F.softmax(self.log_deltas, dim=0)
        deltas = (T - t_eps) * w
        c = torch.cumsum(deltas, dim=0)
        ts = torch.cat([T[None], T - c], dim=0)
        return ts

    # ====== 도메인 변환 ======
    def L(self, y, tau, eps=1e-8):
        return tau*torch.log(y) + (1-tau)*y

    def L_inv(self, x, tau, eps=1e-8):
        a = 1.0 - tau
        z = (a / tau) * torch.exp(x / tau)
        y = (tau / a) * lambertw_autograd(z).real
        return y.clamp_min(eps)

    def u(self, alpha, sigma, gamma, tau):
        y_pos = alpha * sigma.pow(-gamma)
        y_neg = alpha.pow(1 + gamma)
        y = torch.where(gamma >= 0, y_pos, y_neg)
        return self.L(y, tau)

    def v(self, alpha, sigma, gamma, tau):
        y_pos = sigma.pow(1 - gamma)
        y_neg = sigma * alpha.pow(gamma)
        y = torch.where(gamma >= 0, y_pos, y_neg)
        return self.L(y, tau)

    def O_delta_square(self, delta, kappa):
        return kappa * delta**2

    def compute_delta_and_ratio(self, fn, alphas, sigmas, i, gamma, tau, eps=1e-8):
        c, n, p = i, i + 1, i - 1
        val_c = fn(alphas[c], sigmas[c], gamma, tau)
        val_n = fn(alphas[n], sigmas[n], gamma, tau)
        delta_c = val_n - val_c
        L_inv_c = self.L_inv(val_c, tau)
        L_inv_n = self.L_inv(val_n, tau)
        if p >= 0:
            val_p = fn(alphas[p], sigmas[p], gamma, tau)
            delta_p = val_c - val_p
            ratio = delta_p / (delta_c + eps)
        else:
            ratio = None
        return delta_c, ratio, L_inv_c, L_inv_n

    # --- (신규) 밴드별 코어 업데이트: sample은 쓰지 않고, X+E와 coeff들을 반환 ---
    def _band_core(self, xs, es, i, alphas, sigmas,
                   gamma, tau_x, tau_e, kappa_x, kappa_e,
                   order, corrector, eps=1e-8):
        xn, xc, xp = xs
        en, ec, ep = es
        delta_u, r_u, L_uc, L_un = self.compute_delta_and_ratio(self.u, alphas, sigmas, i, gamma, tau_x, eps)
        delta_v, r_v, L_vc, L_vn = self.compute_delta_and_ratio(self.v, alphas, sigmas, i, gamma, tau_e, eps)

        if order == 1:
            X = xc * (L_uc / ((1-tau_x)*L_uc + tau_x) * delta_u + self.O_delta_square(delta_u, kappa_x))
            E = ec * (L_vc / ((1-tau_e)*L_vc + tau_e) * delta_v + self.O_delta_square(delta_v, kappa_e))
        else:
            X = xc * (L_un - L_uc)
            E = ec * (L_vn - L_vc)
            if corrector:
                X += 0.5 * (xn - xc) * (delta_u/((1-tau_x)+tau_x/L_uc) + self.O_delta_square(delta_u, kappa_x))
                E += 0.5 * (en - ec) * (delta_v/((1-tau_e)+tau_e/L_vc) + self.O_delta_square(delta_v, kappa_e))
            else:
                # ratio는 첫 스텝에서 None이 되지 않도록 order=1로 제한됨(상위 로직)
                X += 0.5 * (xc - xp) * (delta_u/((1-tau_x)+tau_x/L_uc) + self.O_delta_square(delta_u, kappa_x)) / r_u.clamp_min(eps)
                E += 0.5 * (ec - ep) * (delta_v/((1-tau_e)+tau_e/L_vc) + self.O_delta_square(delta_v, kappa_e)) / r_v.clamp_min(eps)

        pos_sample_coeff = (sigmas[i + 1] / sigmas[i]) ** gamma
        neg_sample_coeff = (alphas[i + 1] / alphas[i]) ** (-gamma)
        pos_grad_coeff   = sigmas[i + 1] ** gamma
        neg_grad_coeff   = alphas[i + 1] ** (-gamma)
        sample_coeff = torch.where(gamma >= 0, pos_sample_coeff, neg_sample_coeff)
        grad_coeff   = torch.where(gamma >= 0, pos_grad_coeff,   neg_grad_coeff)

        return sample_coeff, grad_coeff, (X + E)   # 모두 (B,C,H,W) 브로드캐스트 가능 스칼라/텐서

    # --- (신규) rFFT 저역 마스크 ---
    def _get_low_mask(self, H, W, device, dtype):
        if (self._cached_mask is not None
            and self._cached_mask.shape[-2:] == (H, W//2+1)
            and self._cached_mask.device == device):
            return self._cached_mask

        fy = torch.fft.fftfreq(H, device=device, dtype=dtype).view(H, 1)         # [-0.5,0.5)
        fx = torch.fft.rfftfreq(W, device=device, dtype=dtype).view(1, W//2+1)   # [0,0.5]
        r = torch.sqrt(fx**2 + fy**2)  # cycles/pixel

        if self.transition > 0.0:
            low = torch.sigmoid((self.cutoff - r) / self.transition)
        else:
            low = (r <= self.cutoff).to(dtype)
        low = low[None, None, ...]  # (1,1,H,W//2+1)
        self._cached_mask = low
        return low

    # --- (신규) 밴드 결합(선형): sample 캐리오버와 잔차를 주파수에서 합성 ---
    def _combine_bands_freq(self, sample, upd_low, upd_high,
                            sc_low, sc_high):
        """
        sample: (B,C,H,W)
        upd_low/upd_high: (B,C,H,W)   [이미 grad_coeff까지 곱해진 잔차]
        sc_low/sc_high:   scalar or (B,C,1,1)  [sample carryover coeffs]
        """
        B, C, H, W = sample.shape
        M = self._get_low_mask(H, W, sample.device, sample.dtype)       # (1,1,H,W//2+1)

        # 1) sample 캐리오버를 밴드별 스케일
        Ux = torch.fft.rfft2(sample, dim=(-2, -1))                      # (B,C,H,W//2+1) complex
        # 실수 마스크/계수는 복소수와 곱해져도 OK
        Hs = M * sc_low + (1.0 - M) * sc_high                           # 브로드캐스트

        # 2) 잔차 업데이트를 밴드별로 분리
        U_low  = torch.fft.rfft2(upd_low,  dim=(-2, -1)) * M
        U_high = torch.fft.rfft2(upd_high, dim=(-2, -1)) * (1.0 - M)

        Y = Ux * Hs + U_low + U_high
        y = torch.fft.irfft2(Y, s=(H, W), dim=(-2, -1)).real
        return y

    # ====== 메인 샘플링 ======
    def sample(self, x, model_fn, **kwargs):
        self.set_model_fn(model_fn)
        device, dtype = x.device, x.dtype

        timesteps = self.learned_timesteps(device=device, dtype=dtype)
        alphas = self.noise_schedule.marginal_alpha(timesteps)
        sigmas = self.noise_schedule.marginal_std(timesteps)

        x_pred = x
        x_corr = x
        xn, en = None, None
        xp, ep = None, None

        # t0 모델 출력
        xc, ec = self.checkpoint_model_fn(x_pred, timesteps[0])

        for i in tqdm(range(self.steps), disable=os.getenv("TQDM", "False")):
            p = min(i+1, self.steps - i, self.order) if self.lower_order_final else min(i+1, self.order)

            # ===== Predictor (phase=0) : 밴드별 파라미터로 두 세트 계산 =====
            gL, txL, teL, kxL, keL = self.params_low[i][0]
            gH, txH, teH, kxH, keH = self.params_high[i][0]
            txL, teL = torch.sigmoid(txL), torch.sigmoid(teL)
            txH, teH = torch.sigmoid(txH), torch.sigmoid(teH)

            scL, gcL, upd_core_L = self._band_core((xn, xc, xp), (en, ec, ep),
                                                   i, alphas, sigmas,
                                                   gL, txL, teL, kxL, keL,
                                                   p, False, self.eps)
            scH, gcH, upd_core_H = self._band_core((xn, xc, xp), (en, ec, ep),
                                                   i, alphas, sigmas,
                                                   gH, txH, teH, kxH, keH,
                                                   p, False, self.eps)

            updL = gcL * upd_core_L
            updH = gcH * upd_core_H

            # 주파수에서 밴드별로 합성
            x_pred = self._combine_bands_freq(
                sample=x_corr,
                upd_low=updL, upd_high=updH,
                sc_low=scL, sc_high=scH
            )

            if i < self.steps - 1:
                xn, en = self.checkpoint_model_fn(x_pred, timesteps[i + 1])

            # ===== Corrector (phase=1) : 항상 2차 보정(corrector=True) =====
            gL, txL, teL, kxL, keL = self.params_low[i][1]
            gH, txH, teH, kxH, keH = self.params_high[i][1]
            txL, teL = torch.sigmoid(txL), torch.sigmoid(teL)
            txH, teH = torch.sigmoid(txH), torch.sigmoid(teH)

            scL, gcL, upd_core_L = self._band_core((xn, xc, xp), (en, ec, ep),
                                                   i, alphas, sigmas,
                                                   gL, txL, teL, kxL, keL,
                                                   2, True, self.eps)
            scH, gcH, upd_core_H = self._band_core((xn, xc, xp), (en, ec, ep),
                                                   i, alphas, sigmas,
                                                   gH, txH, teH, kxH, keH,
                                                   2, True, self.eps)

            updL = gcL * upd_core_L
            updH = gcH * upd_core_H

            x_corr = self._combine_bands_freq(
                sample=x_corr,
                upd_low=updL, upd_high=updH,
                sc_low=scL, sc_high=scH
            )

            # 롤링 버퍼 업데이트
            xp = xc; ep = ec; xc = xn; ec = en

        return x_pred
