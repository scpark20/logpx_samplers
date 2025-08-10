import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from ....solver import Solver


# ===== Lambert W (autograd-friendly) =====
def lambertw_autograd(x: torch.Tensor, max_iter: int = 10) -> torch.Tensor:
    """
    Newton-Raphson approximation of Lambert W with backprop support.
    Solves w * exp(w) = x for w.
    """
    w = torch.log1p(x).clamp(min=-10, max=10)
    for _ in range(max_iter):
        ew = torch.exp(w)
        wew = w * ew
        num = wew - x
        den = ew * (w + 1) - (w + 2) * num / (2 * w + 2)
        w = w - num / den
    return w


class GDual_Time_PC_LogLinear_Plus2_Learnable_FFT_Solver(Solver):
    """
    FFT-band (low/high) split sampler with learnable cutoff/transition per step/phase.
    - Nonlinear model calls remain in spatial domain.
    - Linear carryover/residual updates are combined in frequency domain per band.
    """

    def __init__(
        self,
        noise_schedule,
        steps,
        skip_type: str = "time_uniform_flow",
        flow_shift: float = 1.0,
        order: int = 2,
        lower_order_final: bool = True,
        eps: float = 1e-8,
        algorithm_type: str = "dual_prediction",
        param_dim=(),
        # --- Frequency band options ---
        cutoff: float = 0.15,        # cycles/pixel (axis Nyquist = 0.5)
        transition: float = 0.02,    # smooth boundary width (>0 for soft mask)
        learnable_cutoff: bool = True,
        learnable_transition: bool = True,
        per_step_cutoff: bool = True,
        per_phase_cutoff: bool = True,
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

        # ---- Per-band parameters: [steps, phase(2), {gamma, tau_x, tau_e, kappa_x, kappa_e}, ...] ----
        if len(param_dim) > 0:
            init_params = torch.zeros(steps, 2, 5, 1, *param_dim)
        else:
            init_params = torch.zeros(steps, 2, 5)
        self.params_low = nn.Parameter(init_params.clone())
        self.params_high = nn.Parameter(init_params.clone())

        # ---- Learnable time grid (strictly decreasing; anchors T -> t_eps) ----
        t_0 = 1.0 / noise_schedule.total_N
        t_T = noise_schedule.T
        ts0 = self.get_time_steps(
            skip_type=skip_type, t_T=t_T, t_0=t_0, N=steps, device="cpu", shift=flow_shift
        )
        self.log_deltas = nn.Parameter(torch.log(ts0[:-1] - ts0[1:]))

        # ---- Frequency band (learnable cutoff/transition) ----
        self.learnable_cutoff = learnable_cutoff
        self.learnable_transition = learnable_transition
        self.per_step_cutoff = per_step_cutoff
        self.per_phase_cutoff = per_phase_cutoff

        # rFFT max radial frequency (diagonal Nyquist)
        self.r_max = float((0.5**2 + 0.5**2) ** 0.5)

        def inv_sigmoid(p: float) -> float:
            p = float(max(1e-6, min(1 - 1e-6, p)))
            return math.log(p) - math.log(1 - p)

        raw_c0 = inv_sigmoid(cutoff / self.r_max)           # cutoff = r_max * sigmoid(raw_c)
        raw_w0 = math.log(math.exp(transition) - 1.0)       # transition = softplus(raw_w)

        S = steps if per_step_cutoff else 1
        P = 2 if per_phase_cutoff else 1

        if learnable_cutoff:
            self.raw_cutoff = nn.Parameter(torch.full((S, P), raw_c0))
        else:
            self.register_buffer("raw_cutoff", torch.full((S, P), raw_c0), persistent=False)

        if learnable_transition:
            self.raw_transition = nn.Parameter(torch.full((S, P), raw_w0))
        else:
            self.register_buffer("raw_transition", torch.full((S, P), raw_w0), persistent=False)

        # Caches
        self._freq_grid = {}  # key=(device, dtype, H, W) -> r-grid (H, W//2+1)

    # ===== Time grid =====
    def learned_timesteps(self, device=None, dtype=None):
        if device is None:
            device = self.log_deltas.device
        if dtype is None:
            dtype = self.log_deltas.dtype
        T = torch.as_tensor(self.noise_schedule.T, device=device, dtype=dtype)
        t_eps = torch.as_tensor(1.0 / self.noise_schedule.total_N, device=device, dtype=dtype)
        w = F.softmax(self.log_deltas, dim=0)
        deltas = (T - t_eps) * w
        c = torch.cumsum(deltas, dim=0)
        ts = torch.cat([T[None], T - c], dim=0)  # (S+1,)
        return ts

    # ===== Domain transforms (log-affine + LambertW inverse) =====
    def L(self, y, tau, eps: float = 1e-8):
        return tau * torch.log(y) + (1 - tau) * y

    def L_inv(self, x, tau, eps: float = 1e-8):
        a = 1.0 - tau
        z = (a / tau) * torch.exp(x / tau)
        y = (tau / a) * lambertw_autograd(z).real
        return y.clamp_min(eps)

    def u(self, alpha, sigma, gamma, tau):
        y_pos = alpha * sigma.pow(-gamma)  # gamma >= 0
        y_neg = alpha.pow(1 + gamma)       # gamma < 0
        y = torch.where(gamma >= 0, y_pos, y_neg)
        return self.L(y, tau)

    def v(self, alpha, sigma, gamma, tau):
        y_pos = sigma.pow(1 - gamma)       # gamma >= 0
        y_neg = sigma * alpha.pow(gamma)   # gamma < 0
        y = torch.where(gamma >= 0, y_pos, y_neg)
        return self.L(y, tau)

    def O_delta_square(self, delta, kappa):
        return kappa * delta**2

    # ===== Δ and ratio =====
    def compute_delta_and_ratio(self, fn, alphas, sigmas, i, gamma, tau, eps: float = 1e-8):
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

    # ===== Core update per band (returns sample_coeff, grad_coeff, residual) =====
    def _band_core(
        self,
        xs, es, i, alphas, sigmas,
        gamma, tau_x, tau_e, kappa_x, kappa_e,
        order, corrector, eps: float = 1e-8
    ):
        xn, xc, xp = xs
        en, ec, ep = es

        delta_u, r_u, L_uc, L_un = self.compute_delta_and_ratio(self.u, alphas, sigmas, i, gamma, tau_x, eps)
        delta_v, r_v, L_vc, L_vn = self.compute_delta_and_ratio(self.v, alphas, sigmas, i, gamma, tau_e, eps)

        if order == 1:
            X = xc * (L_uc / ((1 - tau_x) * L_uc + tau_x) * delta_u + self.O_delta_square(delta_u, kappa_x))
            E = ec * (L_vc / ((1 - tau_e) * L_vc + tau_e) * delta_v + self.O_delta_square(delta_v, kappa_e))
        else:
            X = xc * (L_un - L_uc)
            E = ec * (L_vn - L_vc)
            if corrector:
                X += 0.5 * (xn - xc) * (delta_u / ((1 - tau_x) + tau_x / L_uc) + self.O_delta_square(delta_u, kappa_x))
                E += 0.5 * (en - ec) * (delta_v / ((1 - tau_e) + tau_e / L_vc) + self.O_delta_square(delta_v, kappa_e))
            else:
                X += 0.5 * (xc - xp) * (delta_u / ((1 - tau_x) + tau_x / L_uc) + self.O_delta_square(delta_u, kappa_x)) / r_u.clamp_min(eps)
                E += 0.5 * (ec - ep) * (delta_v / ((1 - tau_e) + tau_e / L_vc) + self.O_delta_square(delta_v, kappa_e)) / r_v.clamp_min(eps)

        pos_sample_coeff = (sigmas[i + 1] / sigmas[i]) ** gamma
        neg_sample_coeff = (alphas[i + 1] / alphas[i]) ** (-gamma)
        pos_grad_coeff   = sigmas[i + 1] ** gamma
        neg_grad_coeff   = alphas[i + 1] ** (-gamma)

        sample_coeff = torch.where(gamma >= 0, pos_sample_coeff, neg_sample_coeff)
        grad_coeff   = torch.where(gamma >= 0, pos_grad_coeff,   neg_grad_coeff)

        return sample_coeff, grad_coeff, (X + E)

    # ===== r-grid (cached) =====
    def _get_r_grid(self, H, W, device, dtype):
        key = (device, dtype, H, W)
        if key in self._freq_grid:
            return self._freq_grid[key]
        fy = torch.fft.fftfreq(H, device=device, dtype=dtype).view(H, 1)           # [-0.5, 0.5)
        fx = torch.fft.rfftfreq(W, device=device, dtype=dtype).view(1, W // 2 + 1) # [0, 0.5]
        r = torch.sqrt(fx**2 + fy**2)  # (H, W//2+1)
        self._freq_grid[key] = r
        return r

    # ===== Learnable low mask per (step i, phase) =====
    def _get_low_mask(self, i: int, phase: int, H: int, W: int, device, dtype):
        r = self._get_r_grid(H, W, device, dtype)  # (H, W//2+1)
        S, P = self.raw_cutoff.shape
        raw_c = self.raw_cutoff[i % S, phase % P]
        raw_w = self.raw_transition[i % S, phase % P]
        c = self.r_max * torch.sigmoid(raw_c)              # cutoff
        w = F.softplus(raw_w) + 1e-6                       # transition (>0)
        low = torch.sigmoid((c - r) / w)                   # (H, W//2+1)
        return low[None, None, ...]                        # (1,1,H,W//2+1)

    # ===== Frequency-domain linear combine (single pass) =====
    def _combine_bands_freq(self, sample, upd_low, upd_high, sc_low, sc_high, i: int, phase: int):
        """
        sample:   (B,C,H,W)
        upd_low:  (B,C,H,W)  [already scaled by grad_coeff]
        upd_high: (B,C,H,W)  [already scaled by grad_coeff]
        sc_low/sc_high: scalar or (B,C,1,1) carryover coeffs
        """
        B, C, H, W = sample.shape
        M = self._get_low_mask(i, phase, H, W, sample.device, sample.dtype)     # (1,1,H,W//2+1)

        Ux = torch.fft.rfft2(sample, dim=(-2, -1))                               # (B,C,H,W//2+1)
        Hs = M * sc_low + (1.0 - M) * sc_high                                    # broadcast

        U_low  = torch.fft.rfft2(upd_low,  dim=(-2, -1)) * M
        U_high = torch.fft.rfft2(upd_high, dim=(-2, -1)) * (1.0 - M)

        Y = Ux * Hs + U_low + U_high
        y = torch.fft.irfft2(Y, s=(H, W), dim=(-2, -1)).real
        return y

    # ===== Main sampling =====
    def sample(self, x, model_fn, **kwargs):
        self.set_model_fn(model_fn)

        device, dtype = x.device, x.dtype
        timesteps = self.learned_timesteps(device=device, dtype=dtype)
        alphas = self.noise_schedule.marginal_alpha(timesteps)  # (S+1,)
        sigmas = self.noise_schedule.marginal_std(timesteps)    # (S+1,)

        x_pred = x
        x_corr = x
        xn, en = None, None
        xp, ep = None, None

        # t0 model evaluations
        xc, ec = self.checkpoint_model_fn(x_pred, timesteps[0])

        for i in tqdm(range(self.steps), disable=os.getenv("TQDM", "False")):
            p = min(i + 1, self.steps - i, self.order) if self.lower_order_final else min(i + 1, self.order)

            # ---------- Predictor (phase=0) ----------
            gL, txL, teL, kxL, keL = self.params_low[i][0]
            gH, txH, teH, kxH, keH = self.params_high[i][0]
            txL, teL = torch.sigmoid(txL), torch.sigmoid(teL)
            txH, teH = torch.sigmoid(txH), torch.sigmoid(teH)

            scL, gcL, upd_core_L = self._band_core(
                (xn, xc, xp), (en, ec, ep), i, alphas, sigmas,
                gL, txL, teL, kxL, keL, p, False, self.eps
            )
            scH, gcH, upd_core_H = self._band_core(
                (xn, xc, xp), (en, ec, ep), i, alphas, sigmas,
                gH, txH, teH, kxH, keH, p, False, self.eps
            )

            updL = gcL * upd_core_L
            updH = gcH * upd_core_H

            x_pred = self._combine_bands_freq(
                sample=x_corr, upd_low=updL, upd_high=updH,
                sc_low=scL, sc_high=scH, i=i, phase=0
            )

            if i < self.steps - 1:
                xn, en = self.checkpoint_model_fn(x_pred, timesteps[i + 1])

            # ---------- Corrector (phase=1; 2nd order, corrector=True) ----------
            gL, txL, teL, kxL, keL = self.params_low[i][1]
            gH, txH, teH, kxH, keH = self.params_high[i][1]
            txL, teL = torch.sigmoid(txL), torch.sigmoid(teL)
            txH, teH = torch.sigmoid(txH), torch.sigmoid(teH)

            scL, gcL, upd_core_L = self._band_core(
                (xn, xc, xp), (en, ec, ep), i, alphas, sigmas,
                gL, txL, teL, kxL, keL, 2, True, self.eps
            )
            scH, gcH, upd_core_H = self._band_core(
                (xn, xc, xp), (en, ec, ep), i, alphas, sigmas,
                gH, txH, teH, kxH, keH, 2, True, self.eps
            )

            updL = gcL * upd_core_L
            updH = gcH * upd_core_H

            x_corr = self._combine_bands_freq(
                sample=x_corr, upd_low=updL, upd_high=updH,
                sc_low=scL, sc_high=scH, i=i, phase=1
            )

            # roll buffers
            xp, ep = xc, ec
            xc, ec = xn, en

        return x_pred
