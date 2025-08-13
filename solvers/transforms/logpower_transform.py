import torch
import torch.nn.functional as F

class LogPowerMixtureTransform:
    """L(y; τ, T, p) = (1-τ) * T * log y + τ * y^p   (y>0)
       - τ in (0,1) via sigmoid
       - T>0, p>0 via softplus (with floors)
       API는 기존 transform과 동일
    """
    _DENOM_EPS = 1e-12
    _LOG_EPS   = 1e-20
    _T_MIN     = 1e-4
    _P_MIN     = 1e-4

    @staticmethod
    def _unpack(params):
        # params: [B, 9, 1, 1, 1] (predictor/corrector slice 선택 후)
        gamma = params[:, 0]
        tau_x = params[:, 1]
        tau_e = params[:, 2]
        kx    = params[:, 3]
        ke    = params[:, 4]
        Tx    = params[:, 5]
        Te    = params[:, 6]
        px    = params[:, 7]
        pe    = params[:, 8]
        return gamma, tau_x, tau_e, kx, ke, Tx, Te, px, pe

    @staticmethod
    def _tau_T_p(params, x_side: bool):
        gamma, tau_x, tau_e, kx, ke, Tx, Te, px, pe = LogPowerMixtureTransform._unpack(params)
        tau = torch.sigmoid(tau_x) if x_side else torch.sigmoid(tau_e)
        Traw = Tx if x_side else Te
        praw = px if x_side else pe
        # enforce positivity with softplus + floor
        T = F.softplus(Traw) + LogPowerMixtureTransform._T_MIN
        p = F.softplus(praw) + LogPowerMixtureTransform._P_MIN
        kappa = kx if x_side else ke
        return tau, T, p, gamma, kappa

    # ------------ core maps ------------
    @staticmethod
    def L_from_logy(logy, params, x_side=True):
        tau, T, p, _, _ = LogPowerMixtureTransform._tau_T_p(params, x_side)
        # y^p = exp(p*logy)
        return (1.0 - tau) * T * logy + tau * torch.exp(p * logy)

    @staticmethod
    def delta_from_logy(logy_n, logy_c, params, x_side=True):
        tau, T, p, _, _ = LogPowerMixtureTransform._tau_T_p(params, x_side)
        dlog = logy_n - logy_c
        # (1-τ) T * Δlog + τ * y_c^p * (exp(p Δlog) - 1)
        yc_p = torch.exp(p * logy_c)
        return (1.0 - tau) * T * dlog + tau * yc_p * torch.expm1(p * dlog)

    @staticmethod
    def logy_u(log_alpha, log_sigma, params):
        _, _, _, gamma, _ = LogPowerMixtureTransform._tau_T_p(params, True)
        return torch.where(gamma >= 0, log_alpha - gamma * log_sigma,
                           (1.0 + gamma) * log_alpha)

    @staticmethod
    def logy_v(log_alpha, log_sigma, params):
        _, _, _, gamma, _ = LogPowerMixtureTransform._tau_T_p(params, False)
        return torch.where(gamma >= 0, (1.0 - gamma) * log_sigma,
                           log_sigma + gamma * log_alpha)

    @staticmethod
    def g_from_y(y, params, x_side=True):
        tau, T, p, _, _ = LogPowerMixtureTransform._tau_T_p(params, x_side)
        # L'(y) = (1-τ)T / y + τ p y^{p-1}  →  g = 1 / L'(y)
        y = y.clamp_min(LogPowerMixtureTransform._LOG_EPS)
        y_pow = torch.exp((p - 1.0) * torch.log(y))
        denom = (1.0 - tau) * T / y + tau * p * y_pow
        return 1.0 / denom.clamp_min(LogPowerMixtureTransform._DENOM_EPS)

    @staticmethod
    def O_delta_square(delta, params, x_side=True):
        _, _, _, _, kappa = LogPowerMixtureTransform._tau_T_p(params, x_side)
        return kappa * (delta ** 2)

    @staticmethod
    def get_sample_grad_coeff(i, log_alpha, log_sigma, log_alpha_ratio, log_sigma_ratio, params):
        # γ만 영향
        _, _, _, gamma, _ = LogPowerMixtureTransform._tau_T_p(params, True)
        sample_coeff = torch.where(gamma >= 0,
                                   torch.exp(gamma * log_sigma_ratio[i]),
                                   torch.exp(-gamma * log_alpha_ratio[i]))
        grad_coeff   = torch.where(gamma >= 0,
                                   torch.exp(gamma * log_sigma[i+1]),
                                   torch.exp(-gamma * log_alpha[i+1]))
        return sample_coeff, grad_coeff
