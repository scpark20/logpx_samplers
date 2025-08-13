import torch
import torch.nn.functional as F

class TemperedLogLinearTransform:
    """
    L(y; τ, T) = (1-τ) * y + τ * T * log(y)
      - τ \in (0,1) via sigmoid
      - T > 0 via softplus
    Notes
      g(y) = d(L^{-1})/du = 1 / L'(y) = y / ((1-τ) y + τ T)
      ΔL   = (1-τ) * y_c * expm1(Δlog) + τ * T * Δlog
    """
    _DENOM_EPS = 1e-12
    _T_MIN, _T_MAX = 1e-4, 1e4

    @staticmethod
    def _unpack(params):
        # params shape after selecting predictor/corrector slice:
        # [B, 7, 1, 1, 1] -> (γ, τ_x, τ_e, κ_x, κ_e, T_x, T_e)
        gamma = params[:, 0]
        tau_x = params[:, 1]
        tau_e = params[:, 2]
        kx    = params[:, 3]
        ke    = params[:, 4]
        Tx    = params[:, 5]
        Te    = params[:, 6]
        return gamma, tau_x, tau_e, kx, ke, Tx, Te

    @staticmethod
    def _tau_T(params, x_side: bool):
        gamma, tau_x, tau_e, kx, ke, Tx, Te = TemperedLogLinearTransform._unpack(params)
        # τ ∈ (0,1)
        tau = torch.sigmoid(tau_x) if x_side else torch.sigmoid(tau_e)
        # T > 0 (softplus + clamp for stability)
        T_raw = Tx if x_side else Te
        T = F.softplus(T_raw) + 1e-6
        T = torch.clamp(T, TemperedLogLinearTransform._T_MIN, TemperedLogLinearTransform._T_MAX)
        kappa = kx if x_side else ke
        return tau, T, gamma, kappa

    # ---------- core maps ----------
    @staticmethod
    def L_from_logy(logy, params, x_side: bool = True):
        # L(y;τ,T) with y = exp(logy)
        tau, T, _, _ = TemperedLogLinearTransform._tau_T(params, x_side)
        return (1.0 - tau) * torch.exp(logy) + tau * T * logy

    @staticmethod
    def delta_from_logy(logy_n, logy_c, params, x_side: bool = True):
        # ΔL = (1-τ) * y_c * expm1(Δlog) + τ * T * Δlog
        tau, T, _, _ = TemperedLogLinearTransform._tau_T(params, x_side)
        dlog = logy_n - logy_c
        return (1.0 - tau) * torch.exp(logy_c) * torch.expm1(dlog) + tau * T * dlog

    @staticmethod
    def logy_u(log_alpha, log_sigma, params):
        # y_u = α σ^{-γ} (γ>=0) | α^{1+γ} (γ<0)
        _, _, gamma, _ = TemperedLogLinearTransform._tau_T(params, True)
        return torch.where(gamma >= 0, log_alpha - gamma * log_sigma,
                           (1.0 + gamma) * log_alpha)

    @staticmethod
    def logy_v(log_alpha, log_sigma, params):
        # y_v = σ^{1-γ} (γ>=0) | σ α^{γ} (γ<0)
        _, _, gamma, _ = TemperedLogLinearTransform._tau_T(params, False)
        return torch.where(gamma >= 0, (1.0 - gamma) * log_sigma,
                           log_sigma + gamma * log_alpha)

    @staticmethod
    def g_from_y(y, params, x_side: bool = True):
        # g(y) = y / ((1-τ) y + τ T)
        tau, T, _, _ = TemperedLogLinearTransform._tau_T(params, x_side)
        denom = (1.0 - tau) * y + tau * T
        return y / denom.clamp_min(TemperedLogLinearTransform._DENOM_EPS)

    @staticmethod
    def O_delta_square(delta, params, x_side: bool = True):
        _, _, _, kappa = TemperedLogLinearTransform._tau_T(params, x_side)
        return kappa * (delta ** 2)

    @staticmethod
    def get_sample_grad_coeff(i, log_alpha, log_sigma,
                              log_alpha_ratio, log_sigma_ratio, params):
        # Only γ affects these coefficients
        _, _, gamma, _ = TemperedLogLinearTransform._tau_T(params, True)
        sample_coeff = torch.where(gamma >= 0,
                                   torch.exp(gamma * log_sigma_ratio[i]),
                                   torch.exp(-gamma * log_alpha_ratio[i]))
        grad_coeff   = torch.where(gamma >= 0,
                                   torch.exp(gamma * log_sigma[i+1]),
                                   torch.exp(-gamma * log_alpha[i+1]))
        return sample_coeff, grad_coeff
